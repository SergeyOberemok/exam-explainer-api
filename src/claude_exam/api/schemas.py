from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ImportResult(BaseModel):
    source: str
    inserted: int
    # Set when a background simplification job was started for the import.
    job_id: str | None = None


class QuestionSummary(BaseModel):
    id: str
    section: str
    assessment_type: str
    source: str
    question: str


class Question(QuestionSummary):
    answers: list[str]
    feedbacks: list[str]
    correct_response: list[str]
    explanation: str
    related_lectures: list[Any]


# Question search. Unlike the /game models, these DO carry the answer key: they back
# the study skills (look up the answer / explanation for a question), not the game.
class SearchHit(BaseModel):
    id: str
    # MongoDB textScore; higher is a better match.
    score: float
    section: str
    assessment_type: str
    question: str
    answers: list[str]
    # Text of the correct answer(s), resolved from `correct_response` letters.
    correct_answers: list[str]
    explanation: str


class SearchResponse(BaseModel):
    items: list[SearchHit]
    total: int


class QuestionPage(BaseModel):
    items: list[QuestionSummary]
    total: int
    offset: int
    limit: int


# Game models are flat and null-free so Unity's JsonUtility can parse them. Nothing a
# game client can read says which answer is correct: it learns that, with the
# feedback and explanation, only by answering inside a round (`/game/rounds`).

# Examination ids are ObjectId hex strings; the pattern also keeps them safe to use
# as keys in a round document's field paths.
GameQuestionId = Annotated[str, Field(pattern=r"^[0-9a-f]{24}$")]


class GameAnswer(BaseModel):
    """One answer option, without whether it is correct."""

    index: int
    text: str


class GameQuestion(BaseModel):
    # Always the examination question id, also for simplified questions.
    id: str
    # True when text comes from `examination_simplified`.
    simplified: bool = False
    section: str
    assessment_type: str
    question: str
    answers: list[GameAnswer]


class GameQuestionList(BaseModel):
    # JsonUtility needs an object at the JSON root, so lists are wrapped.
    items: list[GameQuestion]
    total: int


class GameRoundRequest(BaseModel):
    count: int = Field(10, ge=1, le=50)
    # Empty means any: JsonUtility writes "" for an unset string, never null.
    section: str = Field("", max_length=200)
    assessment_type: str = Field("", max_length=100)


class GameRound(BaseModel):
    round_id: str
    # The round's questions, in play order.
    items: list[GameQuestion]
    # Simplified versions of the round's questions that have one, matched by `id`.
    simplified: list[GameQuestion]
    # Playable questions matching the filters.
    total: int
    fifty_fifty_charges: int


class GameAnswerRequest(BaseModel):
    question_id: GameQuestionId
    answer_index: int = Field(ge=0)


class GameReview(BaseModel):
    """What one version of a question shows once it is answered."""

    explanation: str = ""
    # One per answer, index-aligned with the question's answers.
    feedbacks: list[str] = []


class GameAnswerResult(BaseModel):
    question_id: str
    answer_index: int
    is_correct: bool
    correct_index: int
    original: GameReview
    has_simplified: bool
    # Empty when `has_simplified` is false.
    simplified: GameReview


class GameFiftyFiftyRequest(BaseModel):
    question_id: GameQuestionId


class GameFiftyFiftyResult(BaseModel):
    question_id: str
    eliminated_indices: list[int]
    fifty_fifty_remaining: int


class GameRoundQuestionRequest(BaseModel):
    question_id: GameQuestionId


class GameRoundQuestion(BaseModel):
    """A question added to a round, without its answer key."""

    item: GameQuestion
    # Its simplified version: one item, or none (a list, since JsonUtility can't read null).
    simplified: list[GameQuestion]


class GameSectionList(BaseModel):
    items: list[str]


class IcaChatRequest(BaseModel):
    prompt: str
    assistant: str | None = None
    max_tokens: int | None = None


class IcaChatResponse(BaseModel):
    reply: str
    finish_reason: str | None
    total_tokens: int
    assistant_id: str


class IcaAssistant(BaseModel):
    id: str
    name: str
    model: str | None = None


class IcaAgentList(BaseModel):
    items: list[dict[str, Any]]


# Models available via /chat-models/chat/completions.
IcaModelId = Literal["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]


class ExplainRequest(BaseModel):
    """Send one practice-exam item to the ICA chat model for simplification."""

    question_id: int
    # Model called via /chat-models/chat/completions.
    # Defaults to claude-haiku-5-5 — the same model the Exam Explainer assistant uses.
    model_id: IcaModelId = "claude-haiku-5-5"
    max_tokens: int | None = None


class SimplifyRequest(BaseModel):
    """Options for simplifying a stored question (the question id is in the path)."""

    model_id: IcaModelId = "claude-haiku-5-5"
    max_tokens: int | None = None


class SimplifiedPrompt(BaseModel):
    """Mirrors the original item's prompt sub-object with simplified content."""

    question: str
    answers: list[str]
    feedbacks: list[str]
    explanation: str


class IcaMeta(BaseModel):
    """ICA call metadata, kept out of the item's top-level keys."""

    model_id: IcaModelId
    finish_reason: str | None
    total_tokens: int


class SimplifiedQuestion(BaseModel):
    """Record stored in `examination_simplified`, keyed by the examination `_id`."""

    question_id: str
    section: str
    assessment_type: str
    source: str
    correct_response: list[str]
    related_lectures: list[Any]
    prompt: SimplifiedPrompt
    ica: IcaMeta
    created_at: datetime
    updated_at: datetime


class SimplificationError(BaseModel):
    question_id: str
    status_code: int
    detail: str


class SimplificationJob(BaseModel):
    """Progress of the batch started after `/examination/import`."""

    id: str
    source: str
    model_id: IcaModelId
    status: Literal["queued", "running", "completed", "failed"]
    total: int
    processed: int
    succeeded: int
    failed: int
    errors: list[SimplificationError]
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ExplainResponse(BaseModel):
    """
    Simplified item — top-level keys match an original practice-exam item
    (_class, id, assessment_type, prompt, correct_response, section,
    question_plain, related_lectures). Call metadata lives under `ica`, so
    dropping that one key yields a record storable beside the source record.
    """

    # populate_by_name lets internal code set the field as class_.
    # serialization_alias on the field + model_config serialize_by_alias=True
    # ensures FastAPI sends "_class" (not "class_") over the wire.
    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(alias="_class", serialization_alias="_class")
    id: int
    assessment_type: str
    prompt: SimplifiedPrompt
    correct_response: list[str]
    section: str
    question_plain: str
    related_lectures: list[Any]
    # Not part of the original item; strip before storing.
    ica: IcaMeta
