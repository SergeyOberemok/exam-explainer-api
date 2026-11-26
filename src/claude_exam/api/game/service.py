"""Round rules shared by every version of the game routes. Raises `ApiError`s from
`game.errors`; never builds HTTP.

No response here carries a question's answer key before that question is answered in a
round: the key stays in the round document, and answering returns it, once.
"""

import random

from bson import ObjectId

from claude_exam.api.game import errors
from claude_exam.api.game.domain import FIFTY_FIFTY_ELIMINATES
from claude_exam.api.game.repository import RoundRepository
from claude_exam.api.schemas import GameFiftyFiftyResult


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
