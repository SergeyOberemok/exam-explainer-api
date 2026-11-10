"""The learning objectives a question can be tagged with, as a closed, hand-approved list.

The tagging step (`objectives.py`) only ever picks keys from here, so questions can be
grouped, counted and (later) laid out on the game's map by objective. This file is the
place where learning objectives are enforced: edit it by hand and review the change.
`scripts/propose_objectives.py` suggests objectives for a section from its questions;
`scripts/objective_coverage.py` shows how many questions each objective has.

`statement` is what the model reads when it picks, and what a learner should be able to
do. `title` is the short name a client shows. Changing a key, section or statement
changes `CATALOG_VERSION`, and the next objectives backfill tags every question again;
changing only a title does not. Removing a key is safe: stored uses of it are dropped
when served.
"""

import hashlib
import re
from typing import NamedTuple


class ObjectiveDef(NamedTuple):
    key: str
    # The question section it belongs to, without the "Domain N: " prefix.
    section: str
    title: str
    statement: str


KEY_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_DOMAIN_PREFIX = re.compile(r"^Domain \d+:\s*")

AGENTS = "Agentic Architecture & Orchestration"
TOOLS = "Tool Design & MCP Integration"
CLAUDE_CODE = "Claude Code Configuration & Workflows"
PROMPTS = "Prompt Engineering & Structured Output"
CONTEXT = "Context Management & Reliability"

OBJECTIVES: tuple[ObjectiveDef, ...] = (
    # --- Agentic Architecture & Orchestration ---
    ObjectiveDef(
        "agents-loop-control", AGENTS, "Agent loop control",
        "Drive and stop an agentic loop from the API's signals (stop_reason, tool results) "
        "and sensible turn limits, not from text markers or arbitrary caps",
    ),
    ObjectiveDef(
        "agents-orchestration", AGENTS, "Coordinator and subagents",
        "Design a coordinator with subagents: what to delegate, in which order, what can "
        "run in parallel and what must wait for an earlier result",
    ),
    ObjectiveDef(
        "agents-subagent-context", AGENTS, "Subagent context",
        "Define a subagent (description, prompt, tools) and pass it the context it needs "
        "explicitly, because it does not inherit the main conversation",
    ),
    ObjectiveDef(
        "agents-hooks", AGENTS, "Hooks and hard rules",
        "Enforce rules that must always hold with hooks or code (PreToolUse, PostToolUse, "
        "prerequisite checks) instead of prompt instructions",
    ),
    ObjectiveDef(
        "agents-sessions", AGENTS, "Sessions and state",
        "Keep, resume or fork session state across headless runs, steps and machines",
    ),
    ObjectiveDef(
        "agents-decomposition", AGENTS, "Task decomposition",
        "Split large or open-ended work into passes or steps (explore first, per-file then "
        "cross-file) instead of one pass over everything",
    ),
    ObjectiveDef(
        "agents-handoff", AGENTS, "Handoff to people",
        "Hand work over to a person or another agent with a complete, structured summary "
        "of the case and what was already tried",
    ),
    # --- Tool Design & MCP Integration ---
    ObjectiveDef(
        "tools-descriptions", TOOLS, "Tool descriptions",
        "Write tool names and descriptions (purpose, inputs, when to use, limits) so the "
        "model picks the right one among similar tools",
    ),
    ObjectiveDef(
        "tools-scoping", TOOLS, "Tool sets per agent",
        "Keep each agent's tool set small and role-specific, splitting tools across "
        "specialized agents and steering choice with tool_choice when needed",
    ),
    ObjectiveDef(
        "tools-errors", TOOLS, "Tool error responses",
        "Return structured tool errors (category, retryable or not, what to do next) so "
        "the agent can recover or escalate correctly",
    ),
    ObjectiveDef(
        "tools-mcp-config", TOOLS, "MCP server setup",
        "Configure MCP servers in the right scope (project, user, local), share them with "
        "the team and keep secrets out of the repository",
    ),
    ObjectiveDef(
        "tools-builtin", TOOLS, "Built-in tools",
        "Choose the right built-in tool (Read, Grep, Glob, Edit, Write, Bash) for a job, "
        "and know when a built-in tool beats an MCP tool",
    ),
    # --- Claude Code Configuration & Workflows ---
    ObjectiveDef(
        "claude-code-memory", CLAUDE_CODE, "CLAUDE.md hierarchy",
        "Put instructions at the right level of the CLAUDE.md hierarchy (user, project, "
        "directory) so the right people and directories get them",
    ),
    ObjectiveDef(
        "claude-code-rules", CLAUDE_CODE, "Path-scoped rules",
        "Use .claude/rules files with path patterns for conventions that apply to file "
        "types spread across directories",
    ),
    ObjectiveDef(
        "claude-code-skills", CLAUDE_CODE, "Skills and commands",
        "Create and scope skills and slash commands (project or personal) and configure "
        "them with frontmatter such as allowed tools and a forked context",
    ),
    ObjectiveDef(
        "claude-code-planning", CLAUDE_CODE, "Plan or execute",
        "Decide between plan mode, direct execution and exploring in a subagent from the "
        "size, risk and uncertainty of the change",
    ),
    ObjectiveDef(
        "claude-code-ci", CLAUDE_CODE, "Claude Code in CI",
        "Run Claude Code non-interactively in CI (-p, JSON output and schemas, "
        "permissions, the GitHub Action) with the project context it needs",
    ),
    # --- Prompt Engineering & Structured Output ---
    ObjectiveDef(
        "prompts-criteria", PROMPTS, "Explicit criteria",
        "Replace vague instructions with explicit criteria, categories and thresholds for "
        "what to flag, skip or escalate",
    ),
    ObjectiveDef(
        "prompts-examples", PROMPTS, "Few-shot examples",
        "Choose few-shot examples that are varied and cover the hard cases, so the model "
        "generalizes instead of copying one pattern",
    ),
    ObjectiveDef(
        "prompts-structured-output", PROMPTS, "Structured output",
        "Get schema-valid output with tool use and JSON schemas, designing required, "
        "optional and nullable fields so the model does not invent values",
    ),
    ObjectiveDef(
        "prompts-validation", PROMPTS, "Validate and retry",
        "Validate model output and retry with the specific error, and know which failures "
        "a retry cannot fix",
    ),
    ObjectiveDef(
        "prompts-batch", PROMPTS, "Batch processing",
        "Decide when the Message Batches API fits (no one is waiting) and handle its "
        "results, failures and request ids",
    ),
    ObjectiveDef(
        "prompts-review", PROMPTS, "Independent review",
        "Check output with an independent instance or a focused second pass instead of "
        "the generator reviewing itself in the same context",
    ),
    # --- Context Management & Reliability ---
    ObjectiveDef(
        "context-facts", CONTEXT, "Keeping key facts",
        "Keep exact facts (amounts, ids, dates, decisions) through a long conversation or "
        "a summary instead of letting them blur",
    ),
    ObjectiveDef(
        "context-long-sessions", CONTEXT, "Long sessions",
        "Keep long explorations reliable with notes on disk, subagents with their own "
        "context, compaction and a summary to resume from",
    ),
    ObjectiveDef(
        "context-errors", CONTEXT, "Errors across agents",
        "Report failures between agents with what failed, what was tried and partial "
        "results, without hiding them or failing the whole run",
    ),
    ObjectiveDef(
        "context-human-review", CONTEXT, "Human review",
        "Decide what a person must review from measured accuracy per segment and "
        "confidence, not from one overall number",
    ),
    ObjectiveDef(
        "context-provenance", CONTEXT, "Sources and conflicts",
        "Keep where each claim came from through synthesis, and show conflicting sources "
        "instead of silently picking one",
    ),
)

BY_KEY: dict[str, ObjectiveDef] = {o.key: o for o in OBJECTIVES}

assert len(BY_KEY) == len(OBJECTIVES), "objective keys must be unique"
assert all(KEY_PATTERN.match(o.key) for o in OBJECTIVES), "objective keys are kebab-case"

CATALOG_VERSION: str = hashlib.sha1(
    "\n".join(f"{o.key}|{o.section}|{o.statement}" for o in OBJECTIVES).encode()
).hexdigest()[:12]


def normalize_section(section: object) -> str:
    """A stored section name without its "Domain N: " prefix."""
    return _DOMAIN_PREFIX.sub("", section).strip() if isinstance(section, str) else ""


def for_section(section: object) -> list[ObjectiveDef]:
    """The objectives of a question's section (empty when the catalog has none for it)."""
    name = normalize_section(section)
    return [o for o in OBJECTIVES if o.section == name]


def known(key: object) -> str | None:
    """`key` if the catalog has it, else None (a stored key can outlive its entry)."""
    return key if isinstance(key, str) and key in BY_KEY else None
