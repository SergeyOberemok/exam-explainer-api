"""Round rules for `/v2/game`, over the composed `game_question` documents (`compose.py`).

The rules are those of v1 (`GameService`): a round holds at most `MAX_ROUND_QUESTIONS`,
batches come in order and never repeat a question, an answer counts once, the 50/50 is
used once per question. What differs is where a question comes from: one stored
document, already shaped for the game and already checked to be complete, instead of
five collections joined per request. Its `public` part is sent; its `key` goes into the
round entry and comes back, for the answer given, in the answer result.
"""

from bson import ObjectId

from claude_exam.api.game import errors
from claude_exam.api.game.card_repository import CardRepository, card_query
from claude_exam.api.game.domain import FIFTY_FIFTY_CHARGES, MAX_ROUND_QUESTIONS, OVERSAMPLE_FACTOR, dedupe
from claude_exam.api.game.repository import RoundRepository
from claude_exam.api.game.service import RoundRules
from claude_exam.api.objective_catalog import OBJECTIVES
from claude_exam.api.scenario_catalog import COMPANY, COMPANY_INTRO, SCENARIOS, STAGES
from claude_exam.api.schemas import (
    GameCard,
    GameCardAnswerResult,
    GameCardBatch,
    GameCardKey,
    GameCardList,
    GameCardRound,
    GameCardRoundQuestion,
    GameChapter,
    GameObjective,
    GameObjectiveList,
    GameStage,
    GameStory,
)


def card_of(doc: dict) -> GameCard:
    return GameCard.model_validate(doc["public"])


def card_entry(doc: dict) -> dict:
    """What a round keeps for one question: the fields the round rules read (`correct`,
    `count`) and the answer key."""
    key = GameCardKey.model_validate(doc["key"])
    return {"correct": key.correct, "count": doc["count"], "key": key.model_dump()}


def card_answer_result(question_id: str, answer_index: int, entry: dict) -> GameCardAnswerResult:
    key = GameCardKey.model_validate(entry["key"])
    is_correct = answer_index == key.correct
    return GameCardAnswerResult(
        question_id=question_id,
        answer_index=answer_index,
        is_correct=is_correct,
        correct_index=key.correct,
        outcome=key.outcomes[answer_index],
        story_line=key.win if is_correct else key.fail,
        review=key.review,
        rule=key.rule,
        answer_requirements=key.answer_requirements,
        resolution_flow=key.resolution_flow,
    )


class CardService(RoundRules):
    def __init__(self, cards: CardRepository, rounds: RoundRepository) -> None:
        self.cards = cards
        self.rounds = rounds

    def _result(self, question_id: str, answer_index: int, entry: dict) -> GameCardAnswerResult:
        return card_answer_result(question_id, answer_index, entry)

    # ---------- rounds ----------

    async def _pick(self, filters: dict, exclude: list[str], count: int) -> list[dict]:
        """Up to `count` documents the filters allow and `exclude` does not hold: the next
        ones in story order for an ordered story round, else a random pick."""
        query = card_query(filters, exclude)
        if filters.get("ordered") and (filters.get("scenario") or filters.get("stage")):
            return (await self.cards.in_story_order(query))[:count]
        docs = await self.cards.sample(query, count * OVERSAMPLE_FACTOR)
        return dedupe(docs, "question_id")[:count]

    async def create_round(
        self,
        count: int,
        section: str,
        assessment_type: str,
        objective: str = "",
        scenario: str = "",
        stage: str = "",
        ordered: bool = False,
    ) -> GameCardRound:
        filters = {
            "section": section or "",
            "assessment_type": assessment_type or "",
            "objective": objective or "",
            "scenario": scenario or "",
            "stage": stage or "",
            "ordered": bool(ordered),
        }
        docs = await self._pick(filters, [], count)
        if not docs:
            raise errors.NoQuestions()
        total = await self.cards.count(card_query(filters))
        round_id = await self.rounds.create(filters, {d["question_id"]: card_entry(d) for d in docs})
        return GameCardRound(
            round_id=round_id,
            items=[card_of(d) for d in docs],
            total=total,
            fifty_fifty_charges=FIFTY_FIFTY_CHARGES,
        )

    async def next_batch(self, round_id: ObjectId, batch_index: int, count: int) -> GameCardBatch:
        layout = await self.rounds.get_layout(round_id)
        if layout is None:
            raise errors.RoundNotFound()
        batches = layout.get("batches") or []
        filters = layout.get("filters") or {}
        in_round = list(layout.get("question_ids") or [])
        if batch_index < len(batches):
            stored = await self.cards.get_many(batches[batch_index])
            return GameCardBatch(
                batch_index=batch_index,
                items=[card_of(stored[i]) for i in batches[batch_index] if i in stored],
                remaining=await self.cards.count(card_query(filters, in_round)),
            )
        if batch_index > len(batches):
            raise errors.BatchOutOfOrder()

        room = MAX_ROUND_QUESTIONS - len(in_round)
        if room <= 0:
            raise errors.RoundFull()
        docs = await self._pick(filters, in_round, min(count, room))
        if not docs:
            return GameCardBatch(batch_index=batch_index, items=[], remaining=0)
        entries = {d["question_id"]: card_entry(d) for d in docs}
        if not await self.rounds.add_batch(round_id, batch_index, entries):
            raise errors.RoundChanged()
        return GameCardBatch(
            batch_index=batch_index,
            items=[card_of(d) for d in docs],
            remaining=await self.cards.count(card_query(filters, in_round + list(entries))),
        )

    async def add_question(self, round_id: ObjectId, question_id: str) -> GameCardRoundQuestion:
        """Adds a served question (a search result) to the round; adding one it already
        has changes nothing and returns it again."""
        doc = await self.cards.get(question_id)
        if doc is None:
            if await self.cards.question_exists(question_id):
                raise errors.QuestionNotPlayable()
            raise errors.QuestionNotFound()
        if not await self.rounds.add_question(round_id, question_id, card_entry(doc)):
            stored = await self.rounds.get_entries(round_id, [question_id])
            if stored is None:
                raise errors.RoundNotFound()
            if stored.get(question_id) is None:
                raise errors.RoundFull()
        return GameCardRoundQuestion(item=card_of(doc))

    # ---------- browsing, without answer keys ----------

    async def list_questions(
        self, count: int, section: str | None, assessment_type: str | None, shuffle: bool
    ) -> GameCardList:
        query = card_query({"section": section, "assessment_type": assessment_type})
        docs, total = await self.cards.browse(query, count, shuffle)
        return GameCardList(items=[card_of(d) for d in dedupe(docs, "question_id")], total=total)

    async def get_question(self, question_id: str) -> GameCard:
        doc = await self.cards.get(question_id)
        if doc is None:
            raise errors.QuestionNotFound()
        return card_of(doc)

    async def sections(self) -> list[str]:
        return sorted(s for s in await self.cards.sections() if s)

    async def objectives(self) -> GameObjectiveList:
        counts: dict[str, int] = {}
        for doc in await self.cards.summaries():
            for key in set(doc.get("objectives") or []):
                counts[key] = counts.get(key, 0) + 1
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
        counts: dict[tuple, int] = {}
        for doc in await self.cards.summaries():
            key = (doc.get("scenario"), doc.get("stage"))
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
