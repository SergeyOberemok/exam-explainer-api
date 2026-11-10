"""The icons a game client can draw, as provider-neutral keys.

Stored data and the game API only ever carry a `key`. What a key looks like is the
client's business: today each key is drawn with the Lucide glyph in `lucide`
(`scripts/export_game_icons.py` copies them into the game as `<key>.svg`). Moving to
another icon set or to custom art means changing that column (or the script), not the
stored questions.

`meaning` is what the model reads when it picks (see `icons.py`), so keep it short and
concrete. Removing a key is safe: stored uses of it are served as "no icon".
"""

import re
from typing import NamedTuple


class IconDef(NamedTuple):
    key: str
    meaning: str
    lucide: str


KEY_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# The icon of an entity whose pick is missing or unknown, by its kind.
KIND_FALLBACK = {"actor": "actor", "object": "object", "action": "action"}

ICONS: tuple[IconDef, ...] = (
    # --- fallbacks, one per entity kind ---
    IconDef("actor", "someone or something that acts, when nothing below fits", "circle-user-round"),
    IconDef("object", "a thing, when nothing below fits", "box"),
    IconDef("action", "an operation, when nothing below fits", "zap"),
    # --- who ---
    IconDef("user", "customer, end user, client, caller", "user"),
    IconDef("developer", "developer, engineer, admin, operator", "user-cog"),
    IconDef("human-reviewer", "a human who reviews, approves or takes over", "user-check"),
    IconDef("person", "a person or account in general, teammate, team lead", "user-round"),
    IconDef("team", "team, group of people, organization unit", "users"),
    IconDef("organization", "company, enterprise, business", "building"),
    IconDef("support", "customer support, help desk, escalation to support", "headset"),
    IconDef("model", "Claude, an LLM, an AI model, a model instance", "sparkles"),
    IconDef("agent", "an AI agent or assistant in general", "bot-message-square"),
    IconDef("main-agent", "the main or parent agent that drives the work", "brain-circuit"),
    IconDef("subagent", "a subagent, a specialized or child agent", "bot"),
    IconDef("coordinator", "coordinator, orchestrator, hub that hands out work", "network"),
    # --- software and platforms ---
    IconDef("terminal", "Claude Code, CLI, shell, bash, a command line", "square-terminal"),
    IconDef("command", "a slash command, a named command, a menu of commands", "square-slash"),
    IconDef("api", "an API, endpoint, integration, interface", "plug"),
    IconDef("sdk", "an SDK, library, package, framework", "boxes"),
    IconDef("server", "a server, MCP server, backend, host, service", "server"),
    IconDef("protocol", "MCP or another protocol, a standard way to connect", "cable"),
    IconDef("database", "database, table, data store", "database"),
    IconDef("migration", "database or code migration", "database-zap"),
    IconDef("cloud", "cloud service or hosting", "cloud"),
    IconDef("compute", "CPU, runtime, machine, compute resources", "cpu"),
    IconDef("container", "container, isolated instance, sandboxed runner", "container"),
    IconDef("worker", "background worker, job, CI runner, automated process", "cog"),
    IconDef("web", "website, web page, browser, URL, the internet", "globe"),
    IconDef("link", "link, reference, connection between two things", "link"),
    IconDef("screen", "screen, display, UI, interactive view", "monitor"),
    IconDef("workflow", "workflow, pipeline, CI/CD, chain of steps", "workflow"),
    IconDef("hook", "hook, webhook, event handler", "webhook"),
    IconDef("plugin", "skill, plugin, extension, add-on", "puzzle"),
    IconDef("tool", "a tool the model can call, tooling", "wrench"),
    # --- files and code ---
    IconDef("file", "a file or document, Markdown file, CLAUDE.md", "file-text"),
    IconDef("config-file", "a configuration file such as JSON or YAML", "file-braces"),
    IconDef("folder", "folder, directory, workspace", "folder"),
    IconDef("repository", "code repository, codebase, monorepo", "folder-git-2"),
    IconDef("legacy", "legacy code, old system, archive", "archive"),
    IconDef("code", "source code, script, snippet", "code"),
    IconDef("function", "function, method, callable", "square-function"),
    IconDef("schema", "schema, structured data, JSON structure", "braces"),
    IconDef("tags", "XML tags, labels, markup, frontmatter fields", "tags"),
    IconDef("variable", "variable, environment variable", "variable"),
    IconDef("branch", "git branch, version control", "git-branch"),
    IconDef("commit", "a commit", "git-commit-horizontal"),
    IconDef("merge", "merge of branches or results", "git-merge"),
    IconDef("diff", "a diff, changed lines, comparison of two versions", "git-compare"),
    IconDef("pull-request", "pull request, code review request", "git-pull-request"),
    IconDef("test", "test, test suite, evaluation, experiment", "flask-conical"),
    IconDef("bug", "bug, error, exception, failure detail", "bug"),
    IconDef("dependency-graph", "dependency graph, relations between modules", "waypoints"),
    # --- settings ---
    IconDef("settings", "settings, configuration, options", "settings"),
    IconDef("parameter", "a parameter, field or numeric option of a call", "sliders-horizontal"),
    IconDef("mode", "a mode or switch that changes behaviour (plan mode, headless)", "toggle-left"),
    IconDef("feature-flag", "feature flag, marker", "flag"),
    IconDef("scope", "scope or level where something applies (user, project, local)", "target"),
    IconDef("permission", "permission, authorization, login, access control", "lock"),
    IconDef("credential", "API key, access token, secret, credential", "key-round"),
    IconDef("security", "security, safety, sandbox, protection", "shield"),
    IconDef("policy", "policy, guardrail, enforced business rule", "shield-check"),
    IconDef("rule", "a rule, a rule file, a legal-style requirement", "gavel"),
    IconDef("conventions", "conventions, standards, guidelines, style", "ruler"),
    # --- prompts and model context ---
    IconDef("system-prompt", "system prompt, standing instructions", "scroll"),
    IconDef("prompt", "a prompt, message, instruction, reply", "message-square"),
    IconDef("conversation", "conversation, chat, thread, message history", "messages-square"),
    IconDef("session", "a session, resumable state, history of one run", "history"),
    IconDef("context", "context window, layers of context", "layers"),
    IconDef("cache", "cache, prompt caching, stored copy for reuse", "hard-drive"),
    IconDef("memory", "memory, remembered facts", "memory-stick"),
    IconDef("thinking", "extended thinking, reasoning steps", "brain"),
    IconDef("example", "examples, few-shot examples, samples", "shapes"),
    IconDef("token-budget", "tokens, cost, price, budget", "coins"),
    IconDef("rate-limit", "rate limit, quota, cap, performance gauge", "gauge"),
    IconDef("score", "score, confidence, percentage, statistic", "percent"),
    IconDef("batch", "batch of requests or items processed together", "combine"),
    # --- documents and results ---
    IconDef("documentation", "documentation, guide, book, lecture", "book-open"),
    IconDef("source", "source document, reference material being looked up", "file-search"),
    IconDef("article", "article, publication, news", "newspaper"),
    IconDef("citation", "citation, quote, attributed claim", "quote"),
    IconDef("report", "report, written analysis", "file-chart-column"),
    IconDef("chart", "chart, metrics, dashboard", "chart-bar"),
    IconDef("result", "result, output, findings, tool result", "clipboard-list"),
    IconDef("summary", "summary, condensed notes, handoff summary", "notepad-text"),
    IconDef("note", "note, scratchpad, notebook", "notebook-pen"),
    IconDef("log", "logs, transcript, recorded history", "scroll-text"),
    IconDef("checklist", "checklist, criteria, list of checks", "list-checks"),
    IconDef("task", "task, to-do, subtask, ticket of work", "list-todo"),
    IconDef("steps", "ordered steps, sequence, numbered list", "list-ordered"),
    IconDef("plan", "plan, roadmap, plan mode", "map"),
    IconDef("idea", "idea, hint, tip", "lightbulb"),
    IconDef("image", "image, screenshot, picture", "image"),
    IconDef("input", "input, form field, typed text", "text-cursor-input"),
    # --- business objects ---
    IconDef("order", "order, product, parcel", "package"),
    IconDef("shipment", "shipment, delivery, shipping", "truck"),
    IconDef("refund", "refund, invoice, receipt", "receipt"),
    IconDef("payment", "payment, billing, charge, card", "credit-card"),
    IconDef("cart", "cart, purchase, checkout", "shopping-cart"),
    IconDef("reward", "loyalty points, reward, gift, subscription perk", "gift"),
    IconDef("ticket", "support ticket, case, issue", "ticket"),
    IconDef("email", "email, mail", "mail"),
    IconDef("notification", "notification, alert", "bell"),
    IconDef("calculation", "calculation, balance, total", "calculator"),
    # --- time ---
    IconDef("time", "time, timestamp, timeout, latency", "clock"),
    IconDef("date", "date, deadline, schedule", "calendar"),
    IconDef("wait", "waiting, delay", "hourglass"),
    IconDef("timer", "timer, countdown, time limit", "timer"),
    # --- operations ---
    IconDef("search", "search, find, look up, query", "search"),
    IconDef("inspect", "inspect, analyze, scan, explore", "scan-search"),
    IconDef("read", "read, view, observe", "eye"),
    IconDef("review", "review, check someone's work, audit", "scan-eye"),
    IconDef("write", "write, edit, update, modify", "pencil"),
    IconDef("delete", "delete, remove, drop", "trash"),
    IconDef("copy", "copy, duplicate, replica", "copy"),
    IconDef("generate", "generate, create, produce", "wand-sparkles"),
    IconDef("run", "run, execute, start, launch", "play"),
    IconDef("stop", "stop, cancel, block, deny, stop reason", "ban"),
    IconDef("retry", "retry, repeat, loop, iterate", "repeat"),
    IconDef("undo", "undo, revert, roll back", "undo-2"),
    IconDef("reset", "reset, restart, start fresh", "rotate-ccw"),
    IconDef("upload", "upload, push, deploy, publish", "upload"),
    IconDef("download", "download, fetch, pull, install", "download"),
    IconDef("send", "send, request, call, invoke", "send"),
    IconDef("transfer", "transfer, exchange, pass data both ways", "arrow-right-left"),
    IconDef("delegate", "delegate, hand off work to another", "handshake"),
    IconDef("escalate", "escalate, raise to a higher level", "circle-arrow-up"),
    IconDef("route", "route, routing, dispatch to the right place", "route"),
    IconDef("split", "split, run in parallel, fork", "split"),
    IconDef("filter", "filter, classify, narrow down", "funnel"),
    IconDef("compress", "compact, summarize, shrink", "shrink"),
    IconDef("validate", "validate, verify, check against rules", "check-check"),
    IconDef("approve", "approve, approval, sign-off", "badge-check"),
    IconDef("success", "success, pass, done, correct result", "circle-check"),
    IconDef("failure", "failure, reject, failed result", "circle-x"),
)

ICON_KEYS: frozenset[str] = frozenset(icon.key for icon in ICONS)

assert len(ICON_KEYS) == len(ICONS), "icon keys must be unique"
assert all(KEY_PATTERN.match(icon.key) for icon in ICONS), "icon keys are kebab-case"
assert set(KIND_FALLBACK.values()) <= ICON_KEYS


def known(key: object) -> str | None:
    """`key` if the catalog has it, else None (a stored key can outlive its entry)."""
    return key if isinstance(key, str) and key in ICON_KEYS else None
