"""Why a game request failed. Each `code` is part of the API contract with the Unity client."""

from claude_exam.api.errors import ApiError


class RoundNotFound(ApiError):
    status_code, code, detail = 404, "round_not_found", "round not found"


class QuestionNotFound(ApiError):
    status_code, code, detail = 404, "question_not_found", "question not found"


class SimplifiedNotFound(ApiError):
    status_code, code, detail = 404, "simplified_not_found", "simplified question not found"


class QuestionNotInRound(ApiError):
    status_code, code, detail = 404, "question_not_in_round", "question is not in this round"


class QuestionNotAnswered(ApiError):
    status_code, code, detail = 404, "question_not_answered", "question has not been answered"


class NoQuestions(ApiError):
    status_code, code, detail = 404, "no_questions", "no playable questions match"


class AlreadyAnswered(ApiError):
    # The client recovers with GET /rounds/{round_id}/answers/{question_id}.
    status_code, code, detail = 409, "already_answered", "question already answered"


class NoFiftyFiftyCharges(ApiError):
    status_code, code, detail = 409, "no_fifty_fifty_charges", "no 50/50 charges left"


class TooFewAnswers(ApiError):
    status_code, code, detail = 409, "too_few_answers", "too few answers for 50/50"


class RoundFull(ApiError):
    status_code, code, detail = 409, "round_full", "the round has the most questions it can hold"


class BatchOutOfOrder(ApiError):
    status_code, code, detail = 409, "batch_out_of_order", "batches must be requested in order"


class RoundChanged(ApiError):
    # The state changed between a conditional write and the read that explains it.
    status_code, code, detail = 409, "round_changed", "round changed, try again"


class AnswerOutOfRange(ApiError):
    status_code, code, detail = 422, "answer_out_of_range", "answer index out of range"


class AnswerEliminated(ApiError):
    status_code, code, detail = 422, "answer_eliminated", "answer was eliminated by 50/50"


class QuestionNotPlayable(ApiError):
    status_code, code, detail = 422, "question_not_playable", "question is not playable"
