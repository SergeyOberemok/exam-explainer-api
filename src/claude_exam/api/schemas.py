from datetime import datetime
from typing import Any, Literal

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


class QuestionPage(BaseModel):
    items: list[QuestionSummary]
    total: int
    offset: int
    limit: int


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
