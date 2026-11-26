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


# Long enough for a whole pasted question (with its scenario paragraph, ~1.2k chars).
SEARCH_TEXT_MAX = 2000


class SearchRequest(BaseModel):
    """The body of `POST /v1/search/...`: the same fields as the GET query, for texts
    too long for a URL (a pasted question, percent-encoded, can pass a proxy's limit)."""

    q: str = Field(min_length=2, max_length=SEARCH_TEXT_MAX)
    # Empty means any: JsonUtility writes "" for an unset string, never null.
    section: str = Field(default="", max_length=200)
    assessment_type: str = Field(default="", max_length=100)
    limit: int = Field(default=5, ge=1, le=50)


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
    # The id of this entity in the question's decomposition ("" when the entities were
    # extracted without one), so parts and flow nodes can point at its icon.
    ref: str = ""
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
    # True for the question's rephrase (`examination_simplified`).
    simplified: bool = False
    section: str
    assessment_type: str
    question: str
    answers: list[GameAnswer]
    # Visual entities of the question (from its decomposition, index-aligned with
    # `answers`; the same on its rephrase); empty lists without a decomposition and on
    # search results.
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
    # The question as semantic parts, built from its decomposition (originals of a
    # decomposed question; [] on rephrases). A part's `entity_ids` match `entities[].ref`
    # and the flow's node `entity_ids`.
    parts: list["GamePart"] = []
    # The pre-answer flow of the question (key-free, `flow.py`); empty when it has none.
    flow: "GameFlow" = Field(default_factory=lambda: GameFlow())


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
    # Rephrases of the round's questions that have one, matched by `id`.
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
    # Rephrases of `items` that have one, matched by `id`.
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
    # What picking `answer_index` suggests the player got wrong; "" for the correct
    # answer and for a question that is not tagged.
    misconception: str = ""
    # What happens in the story after this answer; "" without a story record.
    outcome: str = ""
    # How the correct answer resolves the question, and how each option measures up
    # (`flow.py`); empty when the question has none. Part of the answer key.
    resolution: "GameResolution" = Field(default_factory=lambda: GameResolution())


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


PartRole = Literal[
    "context", "current_state", "problem", "consequence", "goal", "constraint", "ask"
]


class SemanticPart(BaseModel):
    """One sentence of the question (built from its decomposition) with what it does in
    the question, and the
    decomposition entities (`QuestionDecomposition.entities` ids) it mentions."""

    role: PartRole
    text: str
    entity_ids: list[str] = []


class SimplifiedPrompt(BaseModel):
    """Mirrors the original item's prompt sub-object with simplified content."""

    question: str
    answers: list[str]
    feedbacks: list[str]
    explanation: str


class Rephrase(BaseModel):
    """A question and its answers in other words (`rephrase.py`), index-aligned with
    the original's answers. Written without the key, so it has no feedback."""

    question: str
    answers: list[str]


class IcaMeta(BaseModel):
    """ICA call metadata, kept out of the item's top-level keys."""

    model_id: IcaModelId
    finish_reason: str | None
    total_tokens: int
    # Replies asked for (the first one plus repairs, `ica_json.ask_validated`).
    attempts: int = 1


class SimplifiedCheck(BaseModel):
    """What the check step found for a rephrase (`check.py`)."""

    # "same": a model answering the rephrase without the key picked the right answer.
    # "changed": it picked another one there but the right one on the original, so the
    # rewrite is at fault and the game does not offer this text.
    # "hard": it missed both; the question is hard, the rewrite is not shown to be wrong.
    verdict: Literal["same", "changed", "hard"]
    # Index of the answer picked on the rephrase, and on the original (None when the
    # original was not asked: only a disagreement asks it).
    picked: int
    picked_original: int | None = None
    # `step_key.py`: what the check was asked from (None on older records).
    input_key: str | None = None
    ica: IcaMeta


class SimplifiedQuestion(BaseModel):
    """Record stored in `examination_simplified`, keyed by the examination `_id`: the
    question's rephrase (`rephrase.py`) and its check."""

    question_id: str
    section: str
    assessment_type: str
    source: str
    correct_response: list[str]
    related_lectures: list[Any]
    prompt: Rephrase
    ica: IcaMeta
    # `step_key.py`: what the rephrase was written from (None on older records).
    input_key: str | None = None
    # Result of the check step; None until it has run for this text.
    check: SimplifiedCheck | None = None
    created_at: datetime
    updated_at: datetime


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
    # `step_key.py`: what the record was tagged from (None on older records).
    input_key: str | None = None
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
    # Shown before the question: hand-written in a seed record, else written by the
    # key-free beat step (`beat.py`); "" until then (writing the story clears it).
    beat: str = ""
    win: str
    fail: str
    origin: Literal["seed", "ica"]
    # `step_key.py`: what a model-written record was written from (None for a seed).
    input_key: str | None = None
    # None for a seed record.
    ica: IcaMeta | None = None
    # The beat step's call and input key (None for a seed beat and before it ran).
    beat_ica: IcaMeta | None = None
    beat_input_key: str | None = None
    created_at: datetime
    updated_at: datetime


class EntitiesBackfill(BaseModel):
    """Result of starting an entity, icon, objectives or story backfill; `job_id` is None when
    nothing needs it."""

    job_id: str | None = None
    total: int


# Job modes (`batch.STAGES` lists the steps of each):
# "full": everything for a question, in the order objectives -> story -> decompose
# (with its parts and entities) -> icons -> flow -> resolve -> rephrase -> check. This
# is what an import starts. With `only_missing` each step is skipped where its result is
# current, so the same job resumes a broken one.
# "decompose": the decomposition and what is built on it (icons, flow, resolution).
# "icons", "flow", "resolve": one step of those (the flow also redoes its resolution).
# "rephrase": write the rephrase and check it; "check": only check stored rephrases.
# "objectives", "story": tag questions with learning objectives, or place them in their
# chapter's stages; each writes its own collection and needs its catalog first.
JobMode = Literal[
    "full",
    "rephrase",
    "icons",
    "objectives",
    "story",
    "beat",
    "check",
    "decompose",
    "flow",
    "resolve",
]
# Which step a per-question job error comes from.
JobStage = Literal[
    "objectives", "story", "beat", "decompose", "icons", "flow", "resolve", "rephrase", "check"
]


class SimplificationError(BaseModel):
    question_id: str
    # A `JobStage` (older jobs also name steps that no longer exist).
    stage: str = "simplify"
    status_code: int
    detail: str


class SimplificationJob(BaseModel):
    """Progress of a simplification or entity job (see `JobMode`)."""

    id: str
    source: str
    model_id: IcaModelId
    # Defaults cover jobs recorded before these fields existed.
    # A `JobMode` (older jobs also name modes that no longer exist).
    mode: str = "simplify"
    only_missing: bool = False
    status: Literal["queued", "running", "completed", "failed"]
    total: int
    processed: int
    succeeded: int
    # Every question with an error. `entities_failed` is kept for jobs recorded before
    # the entity step was built from the decomposition.
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


# ---------------------------------------------------------------------------------------
# Question decomposition, decision flow and resolution (`decompose.py`, `flow.py`).
# The model replies in these shapes (strict `json_schema`, see `ica_json.strict_schema`)
# and they are stored as they are in `question_decomposition`, one record per question.
# Field names follow `docs/question_decomposition_decision_flow_spec.md`, except that a
# flow edge says `source` / `target` (not `from` / `to`): `from` is a keyword in the
# game's C#.

SourceStatus = Literal["explicit", "inferred", "unknown"]
DecompEntityType = Literal[
    "actor", "system", "object", "knowledge_source", "artifact", "external_service", "concept"
]
QuestionIntentKind = Literal[
    "decision_design",
    "architecture_design",
    "process_design",
    "troubleshooting",
    "root_cause_analysis",
    "comparison",
    "classification",
    "sequence_explanation",
    "policy_interpretation",
    "other",
]
RelationshipType = Literal[
    "performs",
    "acts_on",
    "uses",
    "references",
    "produces",
    "communicates_with",
    "causes",
    "leads_to",
    "depends_on",
    "routes_to",
    "reviews",
    "contests",
    "associated_with",
]
ConditionType = Literal["policy", "evidence", "confidence", "permission", "state", "time", "other"]
ConditionEvaluation = Literal["true_false", "multi_value", "unknown", "not_evaluable"]
VisualizationKind = Literal[
    "decision_flow",
    "process_flow",
    "sequence_diagram",
    "architecture_diagram",
    "cause_effect_diagram",
    "comparison_matrix",
    "entity_relationship_diagram",
    "timeline",
]
FlowNodeType = Literal["start", "process", "decision", "input", "output", "human_review", "end"]
FlowEdgeType = Literal["next", "yes", "no", "unknown", "approved", "rejected", "custom"]


class Fact(BaseModel):
    """One statement of the scenario. `evidence` is the phrase of the question it rests
    on, copied word for word ("" when it is inferred or unknown)."""

    text: str
    source_status: SourceStatus
    evidence: str


class DecompEntity(BaseModel):
    id: str
    name: str
    type: DecompEntityType
    aliases: list[str]
    description: str
    source_status: SourceStatus
    evidence: str


class DecompEvent(BaseModel):
    id: str
    name: str
    actor_id: str | None
    target_id: str | None
    input_ids: list[str]
    output_ids: list[str]
    sequence: int | None
    source_status: SourceStatus
    evidence: str


class DecompRelationship(BaseModel):
    source_id: str
    type: RelationshipType
    target_id: str
    source_status: SourceStatus


class ProblemStructure(BaseModel):
    symptoms: list[Fact]
    root_causes: list[Fact]
    consequences: list[Fact]
    affected_entities: list[str]


class DecompCondition(BaseModel):
    id: str
    description: str
    condition_type: ConditionType
    evaluation: ConditionEvaluation
    source_status: SourceStatus
    evidence: str


class GoalsAndRequirements(BaseModel):
    business_goals: list[Fact]
    behavioral_requirements: list[Fact]
    quality_requirements: list[Fact]
    constraints: list[Fact]


class QuestionIntent(BaseModel):
    primary: QuestionIntentKind
    secondary: list[QuestionIntentKind]
    question_task: str
    target_of_question: str
    success_criteria: list[str]


class ScenarioContext(BaseModel):
    description: str
    current_state: list[Fact]
    desired_state: list[Fact]


class AnswerOptionMechanism(BaseModel):
    """What one answer proposes, described without judging it."""

    index: int
    mechanism: str
    entity_ids: list[str]


class QuestionDecomposition(BaseModel):
    """Stage 1 of the spec: the meaning of a question, written WITHOUT its answer key,
    so everything here may be shown before the player answers."""

    question_summary: str
    question_intent: QuestionIntent
    context: ScenarioContext
    entities: list[DecompEntity]
    events: list[DecompEvent]
    relationships: list[DecompRelationship]
    problem: ProblemStructure
    conditions: list[DecompCondition]
    goals_and_requirements: GoalsAndRequirements
    answer_options: list[AnswerOptionMechanism]
    ambiguities_and_unknowns: list[str]


class Visualization(BaseModel):
    primary: VisualizationKind
    secondary: list[VisualizationKind]
    reason: str


class FlowNode(BaseModel):
    id: str
    type: FlowNodeType
    label: str
    entity_ids: list[str]
    # The decomposition condition a decision node tests ("" when none).
    condition_id: str
    source_status: SourceStatus
    # Pre-answer flow: the one decision node that is the question itself; the answer
    # cards hang on it, so it has no outgoing edges.
    is_question: bool
    # Resolution flow: added by the correct answer's mechanism.
    added: bool


class FlowEdge(BaseModel):
    source: str
    target: str
    type: FlowEdgeType
    label: str
    source_status: SourceStatus
    added: bool


class DecisionFlow(BaseModel):
    """Stage 2 of the spec: a flow of nodes and edges, with the visualization that fits
    the question (a flow without decisions is drawn as a process or cause-effect
    flow)."""

    visualization: Visualization
    trigger: str
    nodes: list[FlowNode]
    edges: list[FlowEdge]


class OptionEvaluation(BaseModel):
    index: int
    verdict: Literal["correct", "incorrect"]
    supported_requirements: list[str]
    gaps: list[str]
    reason: str


class Resolution(BaseModel):
    """After answering: what a right answer has to do, how the correct mechanism
    changes the flow, and how each option measures up. Part of the answer key."""

    answer_requirements: list[str]
    flow: DecisionFlow
    options: list[OptionEvaluation]
    rule: str


class DecompositionRecord(BaseModel):
    """A `question_decomposition` document: the decomposition and everything built on
    it, so all of it names the same entity ids. Writing the decomposition clears the
    icons, flow and resolution; writing the flow clears the resolution."""

    question_id: str
    text_hash: str
    semantic: QuestionDecomposition
    semantic_ica: IcaMeta
    # Built from `semantic` in code when it is written (`decompose.parts_from`,
    # `entities.derive_entities`); the icons are the `icon` fields inside `entities`.
    parts: list[SemanticPart] = []
    entities: PromptEntities = PromptEntities()
    icons_ica: IcaMeta | None = None
    icons_input_key: str | None = None
    # `step_key.py` input keys of the decomposition, the flow and the resolution.
    input_key: str | None = None
    flow: DecisionFlow | None = None
    flow_ica: IcaMeta | None = None
    flow_input_key: str | None = None
    resolution: Resolution | None = None
    resolution_ica: IcaMeta | None = None
    resolution_input_key: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("flow", "resolution", mode="wrap")
    @classmethod
    def _outdated_part_reads_as_none(cls, value, handler):
        # A flow or resolution of an older shape (edges once said from/to) is outdated
        # anyway (its version differs); it must not make the whole record unreadable.
        try:
            return handler(value)
        except ValidationError as exc:
            logger.warning("ignoring a malformed stored flow or resolution: %s", exc)
            return None


class GamePart(BaseModel):
    role: str
    text: str
    entity_ids: list[str] = []


class GameFlow(BaseModel):
    """A flow for the game, flat and null-free (`DecisionFlow` with its visualization
    spread out). All lists are empty when the question has no flow."""

    primary: str = ""
    secondary: list[str] = []
    reason: str = ""
    trigger: str = ""
    nodes: list[FlowNode] = []
    edges: list[FlowEdge] = []


class GameResolution(BaseModel):
    answer_requirements: list[str] = []
    flow: GameFlow = GameFlow()
    options: list[OptionEvaluation] = []
    rule: str = ""


GameQuestion.model_rebuild()
GameAnswerResult.model_rebuild()
