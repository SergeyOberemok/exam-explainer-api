import logging
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


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
# feedback and explanation, only by answering inside a round (`/v1/game/rounds`).

# Examination ids are ObjectId hex strings; the pattern also keeps them safe to use
# as keys in a round document's field paths.
OBJECT_ID_PATTERN = r"^[0-9a-f]{24}$"
GameQuestionId = Annotated[str, Field(pattern=OBJECT_ID_PATTERN)]


EntityKind = Literal["actor", "object", "action"]


class VisualEntity(BaseModel):
    """A reusable thing a diagram can show: who acts, what is handled, what is done.
    The same entity is repeated wherever it is mentioned; match across questions by
    normalized `name`."""

    name: str
    kind: EntityKind
    aliases: list[str] = []


class AnswerEntities(BaseModel):
    """Entities of one answer (an object, since lists of lists are awkward for clients)."""

    entities: list[VisualEntity] = []


class PromptEntities(BaseModel):
    """Entities of a simplified question, resolved per place they are mentioned."""

    question: list[VisualEntity] = []
    # One item per answer, index-aligned with `prompt.answers`.
    answers: list[AnswerEntities] = []

    @model_validator(mode="before")
    @classmethod
    def _from_catalog(cls, data: Any) -> Any:
        """Read the first stored shape: a catalog plus id references. A malformed one
        raises ValueError, which pydantic reports as a ValidationError."""
        if not (isinstance(data, dict) and isinstance(data.get("entities"), list)):
            return data
        catalog: dict[str, dict] = {}
        for entity in data["entities"]:
            if not isinstance(entity, dict) or not isinstance(entity.get("id"), str):
                raise ValueError("every catalog entity needs a string 'id'")
            catalog[entity["id"]] = {k: v for k, v in entity.items() if k != "id"}

        def pick(ids: Any) -> list[dict]:
            if not isinstance(ids, list):
                raise ValueError("entity references must be a list of ids")
            return [catalog[i] for i in ids if isinstance(i, str) and i in catalog]

        answers = data.get("answers", [])
        if not isinstance(answers, list):
            raise ValueError("'answers' must be a list of id lists")
        return {
            "question": pick(data.get("question", [])),
            "answers": [{"entities": pick(ids)} for ids in answers],
        }


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
    # Visual entities of a simplified question (index-aligned with `answers`); empty
    # lists for originals, search results and questions extracted before entities.
    entities: PromptEntities = PromptEntities()


class GameQuestionList(BaseModel):
    # JsonUtility needs an object at the JSON root, so lists are wrapped.
    items: list[GameQuestion]
    total: int


GameBatchSize = Annotated[int, Field(ge=1, le=50)]


class GameRoundRequest(BaseModel):
    count: GameBatchSize = 10
    # Empty means any: JsonUtility writes "" for an unset string, never null.
    section: str = Field(default="", max_length=200)
    assessment_type: str = Field(default="", max_length=100)


class GameRound(BaseModel):
    round_id: str
    # The round's questions, in play order.
    items: list[GameQuestion]
    # Simplified versions of the round's questions that have one, matched by `id`.
    simplified: list[GameQuestion]
    # Playable questions matching the filters.
    total: int
    fifty_fifty_charges: int


class GameRoundBatchRequest(BaseModel):
    """Ask for the round's next batch. `batch_index` is the batch wanted (the start is batch 0,
    so the first call sends 1); sending the same index again returns that batch again."""

    batch_index: int = Field(ge=1)
    count: GameBatchSize = 10


class GameRoundBatch(BaseModel):
    """More questions for the same round, none of them already in it. Empty `items` means
    every playable question matching the round's filters is already in the round."""

    batch_index: int
    items: list[GameQuestion]
    # Simplified versions of `items` that have one, matched by `id`.
    simplified: list[GameQuestion]
    # Playable questions matching the round's filters that are not in the round yet.
    remaining: int


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
    # Absent on records simplified before entity extraction existed (and empty when
    # extraction failed); `entities_ica` is the metadata of that second call.
    entities: PromptEntities = PromptEntities()
    entities_ica: IcaMeta | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("entities", mode="wrap")
    @classmethod
    def _tolerate_malformed_entities(cls, value: Any, handler) -> PromptEntities:
        """Entities are decoration: a malformed stored value reads as none instead of
        failing every route that returns the record (re-extract to repair it)."""
        try:
            return handler(value if value is not None else {})
        except ValidationError as exc:
            logger.warning("ignoring malformed stored entities: %s", exc)
            return PromptEntities()


class EntitiesBackfill(BaseModel):
    """Result of starting an entity backfill; `job_id` is None when nothing needs it."""

    job_id: str | None = None
    total: int


# The pipeline is import -> simplify -> extract entities; entities are added to an
# existing simplified record, so they always come second.
# "simplify": simplify each question, then extract its entities.
# "simplify_only": only simplify (entities are left for a later "entities" job).
# "entities": only extract entities for questions that are already simplified.
JobMode = Literal["simplify", "simplify_only", "entities"]
# Which step a per-question job error comes from.
JobStage = Literal["simplify", "entities"]


class SimplificationError(BaseModel):
    question_id: str
    # Absent on errors recorded before stages existed.
    stage: JobStage = "simplify"
    status_code: int
    detail: str


class SimplificationJob(BaseModel):
    """Progress of a simplification or entity job (see `JobMode`)."""

    id: str
    source: str
    model_id: IcaModelId
    # Defaults cover jobs recorded before these fields existed.
    mode: JobMode = "simplify"
    only_missing: bool = False
    status: Literal["queued", "running", "completed", "failed"]
    total: int
    processed: int
    succeeded: int
    # Every question with an error; `entities_failed` of them still got their
    # simplified text stored (only the entity step failed).
    failed: int
    entities_failed: int = 0
    # Entity jobs with `only_missing`: questions that already had entities when reached.
    skipped: int = 0
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
