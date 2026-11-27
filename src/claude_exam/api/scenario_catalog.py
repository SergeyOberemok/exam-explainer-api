"""The game's main scenario: one company, one chapter per exam scenario, as a closed,
hand-approved list.

Most source questions start with `SCENARIO: <paragraph> QUESTION: <stem>`, and the
paragraph is one of a few that repeat word for word. A chapter (`ScenarioDef`) claims a
paragraph by its `marker`, so the chapter of such a question is read from its text and
needs no model call. A question without a paragraph has no chapter of its own: the story
step lets the model pick one of the chapters for it. Inside a chapter every question
belongs to one stage of `STAGES`, the same list for all chapters; that part is decided
per question (`story.py`, or the seed file `data/story_seed.json`).

Edit this file by hand and review the change. A paragraph no chapter claims leaves its
questions without a chapter; the import's dry run and `scripts/propose_scenarios.py`
list them.

A chapter's intro and people are in the prompts written from it, so changing them
outdates (by input key, `step_key.py`) only that chapter's questions; adding a chapter
costs nothing for the others. Titles, districts and outros can change freely.

The casts are empty for now (no named people until the characters are designed): the
prompts then tell the model to name no person. Giving a chapter a cast puts its names
back into the story, beat and resolution prompts and outdates that chapter's records.
"""

import hashlib
import re
from typing import NamedTuple


class ScenarioDef(NamedTuple):
    key: str
    title: str
    # The place on the game's map.
    district: str
    # Shown once when the player enters the chapter; replaces the source's paragraph.
    intro: str
    # Shown when the chapter is finished.
    outro: str
    # Lower-case words found in the source's scenario paragraph, and in no other.
    marker: str
    # The people of the chapter, as (name, role). The model uses these names and no
    # others, so every question of a chapter is brought by the same few people. Empty
    # for now: the prompts then name no person.
    cast: tuple[tuple[str, str], ...] = ()


class StageDef(NamedTuple):
    key: str
    title: str
    # What the model reads when it picks a stage.
    description: str


COMPANY = "Copperline"
COMPANY_INTRO = (
    "Copperline sells home goods online and runs on a billing system older than most of "
    "its engineers. You are its first Claude architect. Its teams are waiting for you, "
    "each with a system to build on Claude, and each will bring you its decisions and "
    "its incidents."
)

SCENARIOS: tuple[ScenarioDef, ...] = (
    ScenarioDef(
        "dev-tools", "Know the Legacy", "The Archive",
        "The platform team wants an agent that helps engineers explore the legacy billing "
        "code, generate boilerplate and automate chores, using the built-in tools and MCP "
        "servers. Nobody fully understands that code any more. Start here.",
        "Engineers now ask the agent before they ask each other, and the billing code "
        "has a map.",
        "developer productivity tools",
    ),
    ScenarioDef(
        "support", "The Front Desk", "Front Desk",
        "The support team wants an agent that resolves returns, billing disputes and "
        "account issues on first contact, using get_customer, lookup_order, "
        "process_refund and escalate_to_human, and that hands over to a person when it "
        "should. The target is 80% resolved at first contact.",
        "The agent closes most cases alone, and the people behind it get cases they can "
        "act on.",
        "customer support resolution agent",
    ),
    ScenarioDef(
        "pipeline", "Guard the Pipeline", "The Pipeline",
        "The delivery team wants Claude Code in CI/CD: reviewing pull requests, "
        "generating tests and commenting where it matters, without burying developers in "
        "false positives.",
        "Reviews arrive before the coffee does, and developers read them.",
        "continuous integration",
    ),
    ScenarioDef(
        "research", "The Research Desk", "Research Desk",
        "The strategy team wants a research system: a coordinator that delegates to "
        "subagents for web search, document analysis, synthesis and report writing, and "
        "produces reports whose every claim has a source.",
        "Leadership reads the reports and, more to the point, trusts the citations.",
        "multi-agent research system",
    ),
)

STAGES: tuple[StageDef, ...] = (
    StageDef(
        "design", "Design",
        "choosing the architecture or approach before or while building: how agents, "
        "loops, sessions, subagents or passes are arranged",
    ),
    StageDef(
        "build", "Build",
        "making one part work well: a tool, its description, a prompt, examples, a "
        "schema, a hook",
    ),
    StageDef(
        "team", "Team",
        "sharing the work with the team or the pipeline: repository configuration, "
        "CLAUDE.md, rules, skills, MCP server scopes, CI settings",
    ),
    StageDef(
        "incident", "Incident",
        "something already running fails: production logs, traces, audits or reports "
        "from users show a wrong or unreliable result",
    ),
    StageDef(
        "scale", "Scale",
        "growth or cost: more volume, more tools, longer sessions, larger inputs, "
        "batch work, or less human review",
    ),
)

KEY_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
BY_KEY: dict[str, ScenarioDef] = {s.key: s for s in SCENARIOS}
STAGE_KEYS: tuple[str, ...] = tuple(s.key for s in STAGES)

assert len(BY_KEY) == len(SCENARIOS), "scenario keys must be unique"
assert all(KEY_PATTERN.match(k) for k in (*BY_KEY, *STAGE_KEYS)), "keys are kebab-case"

_SPLIT = re.compile(r"^\s*SCENARIO:\s*(.*?)\s*QUESTION:\s*(.*?)\s*$", re.DOTALL)


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def split_scenario(question: object) -> tuple[str, str]:
    """The scenario paragraph and the question itself. A text without the source's
    `SCENARIO:` / `QUESTION:` markers has no paragraph: ("", the text)."""
    text = question if isinstance(question, str) else ""
    found = _SPLIT.match(text)
    return (found.group(1), found.group(2)) if found else ("", text.strip())


def scenario_of(doc: dict) -> ScenarioDef | None:
    """The chapter an `examination` question belongs to, or None when its scenario
    paragraph is missing or no chapter claims it."""
    paragraph = _squash(split_scenario((doc.get("prompt") or {}).get("question"))[0])
    if not paragraph:
        return None
    return next((s for s in SCENARIOS if s.marker in paragraph), None)


def unknown_paragraph(doc: dict) -> str:
    """The question's scenario paragraph when no chapter claims it, else "". Such a
    question needs a new chapter in this file before it can get a story."""
    paragraph = split_scenario((doc.get("prompt") or {}).get("question"))[0]
    return paragraph if paragraph and scenario_of(doc) is None else ""


def stem_of(doc: dict) -> str:
    """The question without its scenario paragraph; "" when it has no known chapter (the
    paragraph is then the only context the player gets, so it must stay)."""
    if scenario_of(doc) is None:
        return ""
    return split_scenario((doc.get("prompt") or {}).get("question"))[1]


def text_hash(doc: dict) -> str:
    """Identifies a question by its text and answers, so the story seed still finds it
    after the bank is imported again under new ids."""
    prompt = doc.get("prompt") or {}
    parts = [prompt.get("question") or "", *(a or "" for a in prompt.get("answers") or [])]
    return hashlib.sha1("\n".join(_squash(p) for p in parts).encode()).hexdigest()[:16]


def known_scenario(key: object) -> str | None:
    return key if isinstance(key, str) and key in BY_KEY else None


def known_stage(key: object) -> str | None:
    return key if isinstance(key, str) and key in STAGE_KEYS else None
