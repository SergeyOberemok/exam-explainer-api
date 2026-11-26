"""Round rules for the Unity game. Raises `ApiError`s from `game.errors`; never builds HTTP.

No response here carries a question's answer key before that question is answered in a
round: the key stays in the round document, and answering returns it, once.
"""

import random

from bson import ObjectId

from claude_exam.api.game import errors
from claude_exam.api.game.domain import (
    FIFTY_FIFTY_CHARGES,
    FIFTY_FIFTY_ELIMINATES,
    MAX_ROUND_QUESTIONS,
    offered_rephrase,
    OVERSAMPLE_FACTOR,
    dedupe,
    is_playable,
    pick_playable,
    playable_query,
    round_entry,
    story_of,
    to_game_question,
    usable_simplified,
)
from claude_exam.api.game.repository import QuestionRepository, RoundRepository
from claude_exam.api.objective_catalog import OBJECTIVES
from claude_exam.api.scenario_catalog import COMPANY, COMPANY_INTRO, SCENARIOS, STAGES
from claude_exam.api.schemas import (
    GameAnswerResult,
    GameChapter,
    GameFiftyFiftyResult,
    GameObjective,
    GameObjectiveList,
    GameQuestion,
    GameQuestionList,
    GameReview,
    GameRound,
    GameRoundBatch,
    GameResolution,
    GameRoundQuestion,
    GameStage,
    GameStory,
)


def answer_result(question_id: str, answer_index: int, entry: dict) -> GameAnswerResult:
    # Entries stored before objectives existed have no misconceptions.
    misconceptions = entry.get("misconceptions") or []
    is_correct = answer_index == entry["correct"]
    # ... and entries stored before the story existed have no outcomes.
    outcomes = entry.get("outcomes") or {}
    return GameAnswerResult(
        question_id=question_id,
        answer_index=answer_index,
        is_correct=is_correct,
        correct_index=entry["correct"],
        original=GameReview(**entry["original"]),
        misconception=misconceptions[answer_index] if answer_index < len(misconceptions) else "",
        outcome=outcomes.get("win" if is_correct else "fail") or "",
        # ... nor, before the decomposition existed, a resolution.
        resolution=GameResolution(**(entry.get("resolution") or {})),
    )


class RoundRules:
    """What a round does with the entries it stores, whatever the questions look like:
    answering once, reading a recorded answer and the 50/50. An entry has at least
    `correct` and `count`; `_result` turns it into the answer result of the API version."""

    rounds: RoundRepository

    def _result(self, question_id: str, answer_index: int, entry: dict):
        raise NotImplementedError

    @staticmethod
    def _raise_round_error(
        round_doc: dict | None, question_id: str, answer_index: int | None = None
    ) -> None:
        """Explain why a round update didn't apply; `answer_index` is set for an answer."""
        if round_doc is None:
            raise errors.RoundNotFound()
        question = (round_doc.get("questions") or {}).get(question_id)
        if question is None:
            raise errors.QuestionNotInRound()
        if question_id in (round_doc.get("answers") or {}):
            raise errors.AlreadyAnswered()
        if answer_index is not None:
            if answer_index >= question["count"]:
                raise errors.AnswerOutOfRange()
            if answer_index in (round_doc.get("eliminated") or {}).get(question_id, []):
                raise errors.AnswerEliminated()
        else:
            if round_doc.get("fifty_fifty_remaining", 0) <= 0:
                raise errors.NoFiftyFiftyCharges()
            if question["count"] - 1 < FIFTY_FIFTY_ELIMINATES:
                raise errors.TooFewAnswers()
        raise errors.RoundChanged()

    async def answer(
        self, round_id: ObjectId, question_id: str, answer_index: int
    ):
        """Answers once; the same answer again repeats the result. A different answer is
        `AlreadyAnswered`, after which the client reads the recorded one (`recorded_answer`)."""
        entry = await self.rounds.record_answer(round_id, question_id, answer_index)
        if entry is None:
            doc = await self.rounds.get_for_question(round_id, question_id)
            # The same answer again is a retry after a lost response: repeat the result.
            if (doc or {}).get("answers", {}).get(question_id) != answer_index:
                self._raise_round_error(doc, question_id, answer_index)
            entry = doc["questions"][question_id]
        return self._result(question_id, answer_index, entry)

    async def recorded_answer(self, round_id: ObjectId, question_id: str):
        """The answer the round has recorded for a question, with its result and reviews."""
        doc = await self.rounds.get_for_question(round_id, question_id)
        if doc is None:
            raise errors.RoundNotFound()
        entry = (doc.get("questions") or {}).get(question_id)
        if entry is None:
            raise errors.QuestionNotInRound()
        answered = (doc.get("answers") or {}).get(question_id)
        if answered is None:
            raise errors.QuestionNotAnswered()
        return self._result(question_id, answered, entry)

    async def fifty_fifty(self, round_id: ObjectId, question_id: str) -> GameFiftyFiftyResult:
        """Removes two random wrong answers from an unanswered question, once per question
        (asking again repeats the result), within the round's charges."""
        doc = await self.rounds.get_for_question(round_id, question_id)
        question = ((doc or {}).get("questions") or {}).get(question_id)
        already = ((doc or {}).get("eliminated") or {}).get(question_id)
        if question is not None and already is not None:
            # A retry after a lost response: repeat the result without using a charge.
            return GameFiftyFiftyResult(
                question_id=question_id,
                eliminated_indices=already,
                fifty_fifty_remaining=doc.get("fifty_fifty_remaining", 0),
            )
        usable = (
            question is not None
            and question_id not in (doc.get("answers") or {})
            and doc.get("fifty_fifty_remaining", 0) > 0
            and question["count"] - 1 >= FIFTY_FIFTY_ELIMINATES
        )
        if not usable:
            self._raise_round_error(doc, question_id)

        wrong = [i for i in range(question["count"]) if i != question["correct"]]
        eliminated = sorted(random.sample(wrong, FIFTY_FIFTY_ELIMINATES))
        remaining = await self.rounds.spend_fifty_fifty(round_id, question_id, eliminated)
        if remaining is None:
            self._raise_round_error(
                await self.rounds.get_for_question(round_id, question_id), question_id
            )
        return GameFiftyFiftyResult(
            question_id=question_id,
            eliminated_indices=eliminated,
            fifty_fifty_remaining=remaining,
        )


class GameService(RoundRules):
    def __init__(self, questions: QuestionRepository, rounds: RoundRepository) -> None:
        self.questions = questions
        self.rounds = rounds

    def _result(self, question_id: str, answer_index: int, entry: dict) -> GameAnswerResult:
        return answer_result(question_id, answer_index, entry)

    # ---------- rounds ----------

    async def _entries_for(
        self, docs: list[dict]
    ) -> tuple[dict[str, dict], list[GameQuestion], list[GameQuestion]]:
        """Round entries for playable `docs`, the questions to send and the rephrases to
        offer."""
        ids = [str(d["_id"]) for d in docs]
        simplified_by_id = await self.questions.get_simplified_many(ids)
        tags_by_id = await self.questions.get_tags_many(ids)
        story_by_id = await self.questions.get_story_many(ids)
        decomposition_by_id = await self.questions.get_decomposition_many(ids)
        entries: dict[str, dict] = {}
        items: list[GameQuestion] = []
        simplified_items: list[GameQuestion] = []
        for doc in docs:
            question_id = str(doc["_id"])
            tags = tags_by_id.get(question_id)
            record = story_by_id.get(question_id)
            story = story_of(doc, record)
            simplified = usable_simplified(doc, simplified_by_id.get(question_id))
            decomposition = decomposition_by_id.get(question_id)
            entries[question_id] = round_entry(
                doc, simplified is not None, tags, record, decomposition
            )
            items.append(to_game_question(doc, tags=tags, story=story, decomposition=decomposition))
            if simplified is not None:
                simplified_items.append(
                    to_game_question(
                        simplified, original=doc, tags=tags, story=story, decomposition=decomposition
                    )
                )
        return entries, items, simplified_items

    async def _only_ids(self, filters: dict) -> list[str]:
        """The questions a round may hold: those with a rephrase record (the game serves no
        other), narrowed by the round's `objective`, `scenario` and `stage` filters, in
        story order when the story filters are set."""
        rephrased = await self.questions.rephrased_ids()
        only: list[str] | None = None
        if filters.get("scenario") or filters.get("stage"):
            records = await self.questions.story_records(
                filters.get("scenario") or "", filters.get("stage") or ""
            )
            only = [str(r["question_id"]) for r in records]
        if filters.get("objective"):
            tagged = set(await self.questions.ids_with_objective(filters["objective"]))
            only = list(tagged) if only is None else [i for i in only if i in tagged]
        if only is None:
            return rephrased
        allowed = set(rephrased)
        return [i for i in only if i in allowed]

    async def _pick(
        self, filters: dict, exclude: list[str], count: int
    ) -> tuple[list[dict], list[str]]:
        """Up to `count` playable questions the round's filters allow and `exclude` does
        not hold: the next ones in story order for an ordered story round, else a random
        pick. Also returns the ids the filters allow (see `_only_ids`)."""
        only = await self._only_ids(filters)
        query = playable_query(
            filters.get("section"), filters.get("assessment_type"), exclude, only
        )
        story_filter = filters.get("scenario") or filters.get("stage")
        if filters.get("ordered") and story_filter:
            position = {question_id: i for i, question_id in enumerate(only)}
            docs = sorted(
                await self.questions.find_all(query), key=lambda d: position[str(d["_id"])]
            )
        else:
            docs = await self.questions.sample(query, count * OVERSAMPLE_FACTOR)
        return pick_playable(docs, count), only

    async def create_round(
        self,
        count: int,
        section: str,
        assessment_type: str,
        objective: str = "",
        scenario: str = "",
        stage: str = "",
        ordered: bool = False,
    ) -> GameRound:
        """Playable questions and their simplified versions; the key stays stored. Random,
        or in story order for an `ordered` round of a chapter or stage."""
        filters = {
            "section": section or "",
            "assessment_type": assessment_type or "",
            "objective": objective or "",
            "scenario": scenario or "",
            "stage": stage or "",
            "ordered": bool(ordered),
        }
        originals, only = await self._pick(filters, [], count)
        total = await self.questions.count(
            playable_query(section, assessment_type, only_ids=only)
        )
        if not originals:
            # Nothing to play: don't store an empty round.
            raise errors.NoQuestions()

        entries, items, simplified_items = await self._entries_for(originals)
        round_id = await self.rounds.create(filters, entries)
        return GameRound(
            round_id=round_id,
            items=items,
            simplified=simplified_items,
            total=total,
            fifty_fifty_charges=FIFTY_FIFTY_CHARGES,
        )

    async def next_batch(
        self, round_id: ObjectId, batch_index: int, count: int
    ) -> GameRoundBatch:
        """More questions for the same round, none already in it (including questions added
        from search). Asking for a batch that exists returns it again (a retry)."""
        layout = await self.rounds.get_layout(round_id)
        if layout is None:
            raise errors.RoundNotFound()
        batches = layout.get("batches") or []
        filters = layout.get("filters") or {}
        in_round = list(layout.get("question_ids") or [])
        if batch_index < len(batches):
            return await self._stored_batch(round_id, batch_index, batches[batch_index], filters, in_round)
        if batch_index > len(batches):
            raise errors.BatchOutOfOrder()

        room = MAX_ROUND_QUESTIONS - len(in_round)
        if room <= 0:
            raise errors.RoundFull()
        count = min(count, room)
        originals, only = await self._pick(filters, in_round, count)
        if not originals:
            return GameRoundBatch(batch_index=batch_index, items=[], simplified=[], remaining=0)

        entries, items, simplified_items = await self._entries_for(originals)
        if not await self.rounds.add_batch(round_id, batch_index, entries):
            # Another request changed the round first (a concurrent batch or search add).
            # A retry either gets the stored batch or samples again without the new questions.
            raise errors.RoundChanged()
        remaining = await self.questions.count(
            playable_query(
                filters.get("section"),
                filters.get("assessment_type"),
                in_round + list(entries),
                only,
            )
        )
        return GameRoundBatch(
            batch_index=batch_index,
            items=items,
            simplified=simplified_items,
            remaining=remaining,
        )

    async def _stored_batch(
        self,
        round_id: ObjectId,
        batch_index: int,
        question_ids: list[str],
        filters: dict,
        in_round: list[str],
    ) -> GameRoundBatch:
        docs = await self.questions.get_many(question_ids)
        stored = await self.rounds.get_entries(round_id, question_ids) or {}
        simplified_by_id = await self.questions.get_simplified_many(question_ids)
        tags_by_id = await self.questions.get_tags_many(question_ids)
        story_by_id = await self.questions.get_story_many(question_ids)
        decomposition_by_id = await self.questions.get_decomposition_many(question_ids)
        items, simplified_items = [], []
        for question_id in question_ids:
            doc = docs.get(question_id)
            if doc is None:  # Deleted from `examination` since; the round can't show it.
                continue
            tags = tags_by_id.get(question_id)
            story = story_of(doc, story_by_id.get(question_id))
            # Offer the rephrase again only if the round offered it then.
            simplified = simplified_by_id.get(question_id)
            if not offered_rephrase(stored.get(question_id)):
                simplified = None
            decomposition = decomposition_by_id.get(question_id)
            items.append(to_game_question(doc, tags=tags, story=story, decomposition=decomposition))
            if simplified is not None:
                simplified_items.append(
                    to_game_question(
                        simplified, original=doc, tags=tags, story=story, decomposition=decomposition
                    )
                )
        remaining = await self.questions.count(
            playable_query(
                filters.get("section"),
                filters.get("assessment_type"),
                in_round,
                await self._only_ids(filters),
            )
        )
        return GameRoundBatch(
            batch_index=batch_index, items=items, simplified=simplified_items, remaining=remaining
        )

    async def add_question(self, round_id: ObjectId, question_id: str) -> GameRoundQuestion:
        """Adds a question (a search result) to the round. Adding one the round already has
        changes nothing and returns it again."""
        doc = await self.questions.get(question_id)
        if doc is None:
            raise errors.QuestionNotFound()
        record = await self.questions.get_simplified(question_id)
        # The game serves only questions with a rephrase record.
        if not is_playable(doc) or record is None:
            raise errors.QuestionNotPlayable()
        simplified = usable_simplified(doc, record)
        tags = (await self.questions.get_tags_many([question_id])).get(question_id)
        record = (await self.questions.get_story_many([question_id])).get(question_id)
        story = story_of(doc, record)
        decomposition = (await self.questions.get_decomposition_many([question_id])).get(question_id)

        if not await self.rounds.add_question(
            round_id, question_id, round_entry(doc, simplified is not None, tags, record, decomposition)
        ):
            stored = await self.rounds.get_entries(round_id, [question_id])
            if stored is None:
                raise errors.RoundNotFound()
            entry = stored.get(question_id)
            if entry is None:
                raise errors.RoundFull()
            # Offer the rephrase only if the round offered it when the question was added.
            if not offered_rephrase(entry):
                simplified = None

        return GameRoundQuestion(
            item=to_game_question(doc, tags=tags, story=story, decomposition=decomposition),
            simplified=(
                [
                    to_game_question(
                        simplified, original=doc, tags=tags, story=story, decomposition=decomposition
                    )
                ]
                if simplified
                else []
            ),
        )

    # ---------- browsing, without answer keys ----------

    async def list_questions(
        self,
        count: int,
        section: str | None,
        assessment_type: str | None,
        shuffle: bool,
        *,
        simplified: bool = False,
    ) -> GameQuestionList:
        """Playable questions only (the same pre-filter and check as a round), so the game
        can show any of them. A simplified version is checked against its original only
        when a round uses it."""
        key = "question_id" if simplified else "_id"
        if simplified:
            # Rephrase records are the browse itself; leave out those of held questions.
            query = {
                **playable_query(section, assessment_type),
                "question_id": {"$nin": await self.questions.held_ids()},
            }
        else:
            # The originals need a rephrase record.
            query = playable_query(
                section, assessment_type, only_ids=await self.questions.rephrased_ids()
            )
        docs, total = await self.questions.fetch_with_total(
            query,
            count,
            shuffle=shuffle,
            simplified=simplified,
        )
        items = [
            to_game_question(d, simplified=simplified)
            for d in dedupe(docs, key)
            if is_playable(d)
        ]
        return GameQuestionList(items=items, total=total)

    async def get_question(self, question_id: ObjectId) -> GameQuestion:
        doc = await self.questions.get(question_id)
        if doc is None:
            raise errors.QuestionNotFound()
        return to_game_question(doc)

    async def get_simplified(self, question_id: str) -> GameQuestion:
        doc = await self.questions.get_simplified(question_id)
        if doc is None:
            # Distinct from "question not found" so the game can fall back to the raw question.
            raise errors.SimplifiedNotFound()
        return to_game_question(doc, simplified=True)

    async def sections(self, *, simplified: bool = False) -> list[str]:
        return sorted(s for s in await self.questions.sections(simplified=simplified) if s)

    async def objectives(self) -> GameObjectiveList:
        """The objective catalog, in catalog order, with how many playable questions
        each objective has."""
        counts = await self.questions.objective_counts()
        return GameObjectiveList(
            items=[
                GameObjective(
                    key=o.key,
                    section=o.section,
                    title=o.title,
                    statement=o.statement,
                    count=counts.get(o.key, 0),
                )
                for o in OBJECTIVES
            ]
        )

    async def story(self) -> GameStory:
        """The main scenario: chapters in play order, each with every stage and how many
        playable questions with a story record it has."""
        records = await self.questions.story_records()
        playable = await self.questions.playable_ids([str(r["question_id"]) for r in records])
        counts: dict[tuple, int] = {}
        for record in records:
            if str(record["question_id"]) in playable:
                key = (record.get("scenario"), record.get("stage"))
                counts[key] = counts.get(key, 0) + 1
        chapters = []
        for scenario in SCENARIOS:
            stages = [
                GameStage(key=s.key, title=s.title, count=counts.get((scenario.key, s.key), 0))
                for s in STAGES
            ]
            chapters.append(
                GameChapter(
                    key=scenario.key,
                    title=scenario.title,
                    district=scenario.district,
                    intro=scenario.intro,
                    outro=scenario.outro,
                    count=sum(s.count for s in stages),
                    stages=stages,
                )
            )
        return GameStory(company=COMPANY, intro=COMPANY_INTRO, chapters=chapters)
