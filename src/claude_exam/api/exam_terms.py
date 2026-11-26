"""Exact terms of the exam that a rewrite must keep.

The rephrase of a question is written by a model in plain words. Plain words are the
point, except for the names the exam tests: a learner who never sees `dontAsk`,
`PreToolUse` or `.mcp.json` in the game cannot recognise them in the exam. This module
finds those names in a text, so `rephrase.py` can reject a rewrite that lost one.

Two kinds are found, and `rephrase.fidelity_errors` holds them to different rules. An
identifier must stay in the very answer it came from: it is what tells that option from
the others. A named concept must stay somewhere in the question and its answers: a
rewrite may drop a clause that repeats "the subagent", but may not turn every subagent
into a "helper".

The two kinds:
- identifiers, by their form (`identifiers`): anything in backticks, snake_case,
  camelCase and PascalCase names, command-line flags, file names and paths under
  `.claude/`, slash commands, MCP tool names;
- named concepts, from the closed list `TERMS`: a canonical name and the spellings that
  count as the same thing ("subagent" / "sub-agents"; "pull request" / "PR").

`TERMS` is hand-approved, like the catalogs. Its names were checked against the Claude
Code and Claude Agent SDK documentation (permission modes, hook events, option names,
settings files). Add a name when the report shows a rewrite replaced it with a
paraphrase; a common word that is no exam concept does not belong here, because every
entry is a reason to reject a reply.
"""

import re

_BACKTICKED = re.compile(r"`([^`\n]+)`")
_FORMS = (
    # snake_case: tool_choice, get_customer, stop_reason
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b"),
    # MCP tool names: mcp__server__tool
    re.compile(r"\bmcp__[A-Za-z0-9_*]+"),
    # camelCase: acceptEdits, allowedTools
    re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b"),
    # PascalCase with two humps or more: PreToolUse, WebFetch, AgentDefinition
    re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b"),
    # Command-line flags: -p, --output-format
    re.compile(r"(?<![\w@-])--?[a-z][a-z-]*\b"),
    # Files and paths: CLAUDE.md, .mcp.json, .claude/rules/api.md, ~/.claude.json
    re.compile(r"(?<![\w/])~?/?\.claude(?:\.json)?(?:/[\w.*-]+)*"),
    re.compile(r"(?<![\w/.~-])[\w.~-]*\w\.(?:md|json|jsonl|yaml|yml|toml)\b"),
    # Slash commands: /compact, /memory (not a path segment, not "and/or")
    re.compile(r"(?<![\w/.])/[a-z][a-z-]{2,}\b(?!/)"),
)

# (canonical name, spellings that count as that name). Matched ignoring case.
TERMS: tuple[tuple[str, str], ...] = (
    ("Claude Code", r"\bclaude code\b"),
    ("Agent SDK", r"\bagent sdk\b"),
    ("MCP", r"\bmcp\b|\bmodel context protocol\b"),
    ("subagent", r"\bsub-?agents?\b"),
    ("coordinator", r"\bcoordinators?\b"),
    ("hook", r"\bhooks?\b"),
    ("skill", r"\bskills?\b"),
    ("slash command", r"\bslash commands?\b"),
    ("plan mode", r"\bplan mode\b"),
    ("permission mode", r"\bpermission modes?\b"),
    ("print mode", r"\bprint mode\b"),
    ("headless", r"\bheadless\b"),
    ("system prompt", r"\bsystem prompts?\b"),
    ("tool description", r"\btool descriptions?\b"),
    ("few-shot", r"\bfew[- ]shot\b"),
    ("JSON schema", r"\bjson[- ]schemas?\b"),
    ("structured output", r"\bstructured outputs?\b"),
    ("extended thinking", r"\bextended thinking\b"),
    ("prompt caching", r"\bprompt cach(?:e|ing)\b"),
    ("Message Batches API", r"\b(?:message )?batch(?:es)? api\b"),
    ("stop reason", r"\bstop[_ ]reasons?\b"),
    ("context window", r"\bcontext windows?\b"),
    ("compaction", r"\bcompact(?:ion|ing|ed|s)?\b"),
    ("session", r"\bsessions?\b"),
    ("worktree", r"\bworktrees?\b"),
    ("stdio", r"\bstdio\b"),
    ("temperature", r"\btemperature\b"),
    ("pull request", r"\bpull requests?\b|\bprs?\b"),
    ("GitHub Action", r"\bgithub actions?\b"),
)
_TERMS = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in TERMS)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def identifiers(text: str) -> list[str]:
    """The identifiers in `text`, as written, each once, in order of appearance."""
    found: dict[str, str] = {}
    for match in _BACKTICKED.finditer(text):
        term = match.group(1).strip()
        if term:
            found.setdefault(term.lower(), term)
    plain = _BACKTICKED.sub(" ", text)
    for form in _FORMS:
        for match in form.finditer(plain):
            term = match.group(0).rstrip(".")
            found.setdefault(term.lower(), term)
    return list(found.values())


def lost_identifiers(original: str, rewrite: str) -> list[str]:
    """The identifiers of `original` that `rewrite` does not contain (ignoring case)."""
    lowered = rewrite.lower()
    return [term for term in identifiers(original) if term.lower() not in lowered]


def lost_names(original: str, rewrite: str) -> list[str]:
    """The named concepts of `TERMS` in `original` that `rewrite` names in none of
    their spellings."""
    return [
        name
        for name, pattern in _TERMS
        if pattern.search(original) and not pattern.search(rewrite)
    ]


def lost_terms(original: str, rewrite: str) -> list[str]:
    """Both: the identifiers and the named concepts `rewrite` lost of `original`."""
    return lost_identifiers(original, rewrite) + lost_names(original, rewrite)


def lost_numbers(original: str, rewrite: str) -> list[str]:
    """The numbers of `original` (as digits) that `rewrite` does not contain."""
    have = set(_NUMBER.findall(rewrite))
    return [n for n in dict.fromkeys(_NUMBER.findall(original)) if n not in have]


# Words that turn "most" or "these" into "all": a text a model writes from the question
# may not use one the question does not (the rephrase said the agent posts "every"
# finding as definitive; the question says it posts "these", the false positives).
# "never" and "none" are left out: a rewrite says "never edits" for "does not edit".
_SWEEPING = re.compile(r"\b(every|everything|all|always)\b", re.IGNORECASE)


def widened(original: str, rewrite: str) -> list[str]:
    """The sweeping words (`_SWEEPING`) of `rewrite` that `original` does not use."""
    have = {w.lower() for w in _SWEEPING.findall(original)}
    return [w for w in dict.fromkeys(w.lower() for w in _SWEEPING.findall(rewrite)) if w not in have]
