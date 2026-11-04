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
    OVERSAMPLE_FACTOR,
    dedupe,
    is_playable,
    pick_playable,
    playable_query,
    round_entry,
    to_game_question,
    usable_simplified,
)
from claude_exam.api.game.repository import QuestionRepository, RoundRepository
from claude_exam.api.schemas import (
    GameAnswerResult,
    GameFiftyFiftyResult,
    GameQuestion,
    GameQuestionList,
    GameReview,
    GameRound,
    GameRoundBatch,
    GameRoundQuestion,
)


def answer_result(question_id: str, answer_index: int, entry: dict) -> GameAnswerResult:
    simplified = entry.get("simplified")
    return GameAnswerResult(
        question_id=question_id,
        answer_index=answer_index,
        is_correct=answer_index == entry["correct"],
        correct_index=entry["correct"],
        original=GameReview(**entry["original"]),
        has_simplified=simplified is not None,
        simplified=GameReview(**simplified) if simplified else GameReview(),
    )


class GameService:
    def __init__(self, questions: QuestionRepository, rounds: RoundRepository) -> None:
        self.questions = questions
        self.rounds = rounds

    # ---------- rounds ----------

    async def _entries_for(
        self, docs: list[dict]
    ) -> tuple[dict[str, dict], list[GameQuestion]]:
        """Round entries for playable `docs` and the simplified versions to offer."""
        simplified_by_id = await self.questions.get_simplified_many([str(d["_id"]) for d in docs])
        entries: dict[str, dict] = {}
        simplified_items: list[GameQuestion] = []
        for doc in docs:
            question_id = str(doc["_id"])
            simplified = usable_simplified(doc, simplified_by_id.get(question_id))
            entries[question_id] = round_entry(doc, simplified)
            if simplified is not None:
                simplified_items.append(to_game_question(simplified, simplified=True))
        return entries, simplified_items

    async def create_round(
        self, count: int, section: str, assessment_type: str
    ) -> GameRound:
        """Random playable questions and their simplified versions; the key stays stored."""
        query = playable_query(section, assessment_type)
        docs, total = await self.questions.fetch_with_total(query, count * OVERSAMPLE_FACTOR)
        originals = pick_playable(docs, count)
        if not originals:
            # Nothing to play: don't store an empty round.
            raise errors.NoQuestions()

        entries, simplified_items = await self._entries_for(originals)
        round_id = await self.rounds.create(
            {"section": section or "", "assessment_type": assessment_type or ""}, entries
        )
        return GameRound(
            round_id=round_id,
            items=[to_game_question(d) for d in originals],
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
        query = playable_query(filters.get("section"), filters.get("assessment_type"), in_round)
        originals = pick_playable(
            await self.questions.sample(query, count * OVERSAMPLE_FACTOR), count
        )
        if not originals:
            return GameRoundBatch(batch_index=batch_index, items=[], simplified=[], remaining=0)

        entries, simplified_items = await self._entries_for(originals)
        if not await self.rounds.add_batch(round_id, batch_index, entries):
            # Another request changed the round first (a concurrent batch or search add).
            # A retry either gets the stored batch or samples again without the new questions.
            raise errors.RoundChanged()
        remaining = await self.questions.count(
            playable_query(
                filters.get("section"), filters.get("assessment_type"), in_round + list(entries)
            )
        )
        return GameRoundBatch(
            batch_index=batch_index,
            items=[to_game_question(d) for d in originals],
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
        items, simplified_items = [], []
        for question_id in question_ids:
            doc = docs.get(question_id)
            if doc is None:  # Deleted from `examination` since; the round can't show it.
                continue
            items.append(to_game_question(doc))
            # Offer the simplified text only if the stored entry can review it.
            simplified = simplified_by_id.get(question_id)
            if simplified is not None and (stored.get(question_id) or {}).get("simplified"):
                simplified_items.append(to_game_question(simplified, simplified=True))
        remaining = await self.questions.count(
            playable_query(filters.get("section"), filters.get("assessment_type"), in_round)
        )
        return GameRoundBatch(
            batch_index=batch_index, items=items, simplified=simplified_items, remaining=remaining
        )

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
    ) -> GameAnswerResult:
        """Answers once; the same answer again repeats the result. A different answer is
        `AlreadyAnswered`, after which the client reads the recorded one (`recorded_answer`)."""
        entry = await self.rounds.record_answer(round_id, question_id, answer_index)
        if entry is None:
            doc = await self.rounds.get_for_question(round_id, question_id)
            # The same answer again is a retry after a lost response: repeat the result.
            if (doc or {}).get("answers", {}).get(question_id) != answer_index:
                self._raise_round_error(doc, question_id, answer_index)
            entry = doc["questions"][question_id]
        return answer_result(question_id, answer_index, entry)

    async def recorded_answer(self, round_id: ObjectId, question_id: str) -> GameAnswerResult:
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
        return answer_result(question_id, answered, entry)

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

    async def add_question(self, round_id: ObjectId, question_id: str) -> GameRoundQuestion:
        """Adds a question (a search result) to the round. Adding one the round already has
        changes nothing and returns it again."""
        doc = await self.questions.get(question_id)
        if doc is None:
            raise errors.QuestionNotFound()
        if not is_playable(doc):
            raise errors.QuestionNotPlayable()
        simplified = usable_simplified(doc, await self.questions.get_simplified(question_id))

        if not await self.rounds.add_question(round_id, question_id, round_entry(doc, simplified)):
            stored = await self.rounds.get_entries(round_id, [question_id])
            if stored is None:
                raise errors.RoundNotFound()
            entry = stored.get(question_id)
            if entry is None:
                raise errors.RoundFull()
            # Offer the simplified text only if the stored entry can review it.
            if entry.get("simplified") is None:
                simplified = None

        return GameRoundQuestion(
            item=to_game_question(doc),
            simplified=[to_game_question(simplified, simplified=True)] if simplified else [],
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
        docs, total = await self.questions.fetch_with_total(
            playable_query(section, assessment_type), count, shuffle=shuffle, simplified=simplified
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

