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


class ImportCount(BaseModel):
    name: str
    count: int


class ImportResult(BaseModel):
    source: str
    # Questions stored by this call (0 for a dry run).
    inserted: int
    # Set when a background pipeline job was started for the import.
    job_id: str | None = None
    # True when nothing was stored: the rest of the report says what an import would do.
    dry_run: bool = False
    # Questions that would be (or were) stored: in the file, and not a duplicate.
    new: int = 0
    # Left out: the same question and answers are already stored, or occur earlier in
    # the file.
    duplicates: int = 0
    # New questions the game cannot play (not pick-one with 2-4 answers). They are
    # stored and processed, for the study routes.
    unplayable: int = 0
    # New questions per chapter the text names, and how many have no scenario
    # paragraph (the story step picks their chapter).
    chapters: list[ImportCount] = []
    without_scenario: int = 0
    # Scenario paragraphs no chapter claims (the start of each): these questions get
    # no story until `scenario_catalog.py` has a chapter for them.
    unknown_scenarios: list[ImportCount] = []
    # Sections without objectives in `objective_catalog.py`: not tagged until added.
    unknown_sections: list[ImportCount] = []


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
    # Key from `icon_catalog.ICONS`, set by the icon step (None before it has run).
    icon: str | None = None


class AnswerEntities(BaseModel):
    """Entities of one answer (an object, since lists of lists are awkward for clients)."""

    entities: list[VisualEntity] = []
    # Icon key for what this answer proposes; it need not be one of its entities' icons.
    icon: str | None = None


class PromptEntities(BaseModel):
    """Entities of a simplified question, resolved per place they are mentioned."""

    # Icon key for the question as a whole.
    question_icon: str | None = None
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
    # Learning-objective keys from `objective_catalog.OBJECTIVES`, the main one first;
    # empty until the question is tagged.
    objectives: list[str] = []
    # 1 (easy) to 3 (hard); 0 until the question is tagged.
    difficulty: int = 0
    # Chapter key from `scenario_catalog.SCENARIOS`; "" when the question's scenario
    # paragraph is missing or unknown.
    scenario: str = ""
    # Stage key from `scenario_catalog.STAGES`; "" until the question has a story record.
    stage: str = ""
    # `question` without its scenario paragraph, for a client that shows the chapter's
    # intro instead. "" when there is no chapter, and on simplified questions.
    stem: str = ""
    # One line to show before the question; "" until the question has a story record.
    beat: str = ""


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
    # An objective key: only questions tagged with it (as main or secondary objective).
    objective: str = Field(default="", max_length=100)
    # A chapter key and a stage key: only questions whose story record has them.
    scenario: str = Field(default="", max_length=100)
    stage: str = Field(default="", max_length=100)
    # With `scenario` or `stage`: serve the questions in story order instead of at random.
    ordered: bool = False


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
    # What picking `answer_index` suggests the player got wrong; "" for the correct
    # answer and for a question that is not tagged.
    misconception: str = ""
    # What happens in the story after this answer; "" without a story record.
    outcome: str = ""


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


class GameObjective(BaseModel):
    key: str
    # Section name without its "Domain N: " prefix.
    section: str
    title: str
    statement: str
    # Playable questions tagged with it (as main or secondary objective).
    count: int


class GameObjectiveList(BaseModel):
    items: list[GameObjective]


class GameStage(BaseModel):
    key: str
    title: str
    # Playable questions of the chapter in this stage.
    count: int


class GameChapter(BaseModel):
    key: str
    title: str
    # The chapter's place on the game's map.
    district: str
    # Shown when the player enters the chapter, and when it is finished.
    intro: str
    outro: str
    # Playable questions with a story record.
    count: int
    stages: list[GameStage]


class GameStory(BaseModel):
    """The game's main scenario: one company, chapters in play order, the same stages
    in each. Send a chapter and stage key as `scenario` and `stage` to start a round."""

    company: str
    intro: str
    chapters: list[GameChapter]


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


class SimplifiedCheck(BaseModel):
    """What the check step found for a game text (`check.py`)."""

    # "same": a model answering the game text without the key picked the right answer.
    # "changed": it picked another one there but the right one on the original, so the
    # rewrite is at fault and the game does not offer this text.
    # "hard": it missed both; the question is hard, the rewrite is not shown to be wrong.
    verdict: Literal["same", "changed", "hard"]
    # Index of the answer picked on the game text, and on the original (None when the
    # original was not asked: only a disagreement asks it).
    picked: int
    picked_original: int | None = None
    prompt_version: str
    ica: IcaMeta


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
    # Metadata of the icon call; None until icons were picked for the stored entities
    # (the icons themselves are the `icon` fields inside `entities`).
    icons_ica: IcaMeta | None = None
    # What the text was written with (`simplify.PROMPT_VERSION`, and
    # `scenario_catalog.frame_version` of the question's chapter); None on records
    # written before the game text existed. A pipeline backfill redoes other versions.
    prompt_version: str | None = None
    frame_version: str | None = None
    # Result of the check step; None until it has run for this text.
    check: SimplifiedCheck | None = None
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


ObjectiveLevel = Literal["remember", "understand", "apply", "analyze"]


class Distractor(BaseModel):
    """What one wrong answer reveals about a learner who picks it."""

    # The objective this answer would be right for, when there is one.
    confused_with: str | None = None
    misconception: str


class QuestionObjectives(BaseModel):
    """Record stored in `question_objectives`, keyed by the examination `_id` (as a
    string, like `examination_simplified`). Written by `objectives.tag_and_store`."""

    question_id: str
    section: str
    # Keys from `objective_catalog.OBJECTIVES`; the first is the main objective.
    objectives: list[str]
    level: ObjectiveLevel
    # 1-3, derived from `level` and `tempting` (`objectives.difficulty_of`).
    difficulty: int
    # Indices of the wrong answers that would partly work. Absent on records tagged
    # before prompt version 4.
    tempting: list[int] = []
    # One per answer, index-aligned with `prompt.answers`; None for a correct answer.
    distractors: list[Distractor | None]
    # What the record was tagged with; a backfill redoes records of another version.
    catalog_version: str
    prompt_version: str
    ica: IcaMeta
    created_at: datetime
    updated_at: datetime


class QuestionStory(BaseModel):
    """Record stored in `question_story`, keyed by the examination `_id` (as a string).
    Written by `story.story_and_store` or loaded from `data/story_seed.json`."""

    question_id: str
    # `scenario_catalog.text_hash` of the question the record was written for.
    text_hash: str
    scenario: str
    # "text": the chapter the question's scenario paragraph names. "model": picked by
    # the model for a question without a paragraph.
    scenario_origin: Literal["text", "model"] = "text"
    stage: str
    # Position inside the stage, lowest first.
    order: int
    beat: str
    win: str
    fail: str
    origin: Literal["seed", "ica"]
    catalog_version: str
    prompt_version: str
    # None for a seed record.
    ica: IcaMeta | None = None
    created_at: datetime
    updated_at: datetime


class EntitiesBackfill(BaseModel):
    """Result of starting an entity, icon, objectives or story backfill; `job_id` is None when
    nothing needs it."""

    job_id: str | None = None
    total: int


# The pipeline is import -> simplify -> extract entities -> pick icons; each step adds
# to the simplified record the one before it stored.
# "simplify": simplify each question, extract its entities, pick its icons.
# "simplify_only": only simplify (entities and icons are left for a later job).
# "entities": extract entities for already simplified questions, then pick icons.
# "icons": only pick icons for already simplified questions.
# "objectives": tag questions with learning objectives. It reads the original question
# and writes `question_objectives`, so it needs no simplified record and is not part
# of the import job: the objectives of a section must be in the catalog first.
# "story": place questions in their chapter's stages and write their story lines. Like
# "objectives" it reads the original question and writes its own collection
# (`question_story`); the question's scenario must be in the scenario catalog first.
# "check": test that the game text of already simplified questions keeps the right answer.
# "full": everything for a question, in the order objectives -> story -> simplify ->
# entities -> icons -> check. This is what an import starts. With `only_missing` each
# step is skipped where its result is current, so the same job resumes a broken one.
JobMode = Literal[
    "full", "simplify", "simplify_only", "entities", "icons", "objectives", "story", "check"
]
# Which step a per-question job error comes from.
JobStage = Literal["simplify", "entities", "icons", "objectives", "story", "check"]


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
    # Questions whose earlier steps were stored but whose icon step failed.
    icons_failed: int = 0
    # Entity and icon jobs with `only_missing`: questions that already had them when reached.
    skipped: int = 0
    # Failures per step, every one counted (a question can fail more than one step of
    # a "full" job).
    stages_failed: dict[str, int] = {}
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
