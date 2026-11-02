import re

SECTION_TO_FOLDER: dict[str, str] = {
    # architect-practice ("Domain N: <topic>") section names
    "Agentic Architecture & Orchestration": "Agents and Workflows",
    "Tool Design & MCP Integration": "Tools and MCPs",
    "Claude Code Configuration & Workflows": "Claude",
    "Prompt Engineering & Structured Output": "Prompt and Context Engineering",
    "Context Management & Reliability": "Context Management & Reliability",
    # developer-practice (no "Domain N:" prefix) section names
    "Agents and Workflows": "Agents and Workflows",
    "Applications and Integration": "Applications & Integration & Orchestration",
    "Claude Code": "Claude",
    "Eval, Testing, and Debugging": "Eval Testing Debug",
    "Model Selection and Optimization": "Model selection and Optimization",
    "Prompt and Context Engineering": "Prompt and Context Engineering",
    "Security and Safety": "Security and Safety",
    "Tools and MCPs": "Tools and MCPs",
}

_DOMAIN_PREFIX = re.compile(r"^Domain \d+:\s*")


def folder_for_section(section: str) -> str:
    key = _DOMAIN_PREFIX.sub("", section).strip()
    try:
        return SECTION_TO_FOLDER[key]
    except KeyError:
        raise KeyError(
            f"No vault folder mapped for section {section!r} "
            f"(stripped key {key!r}). Add it to SECTION_TO_FOLDER."
        ) from None
