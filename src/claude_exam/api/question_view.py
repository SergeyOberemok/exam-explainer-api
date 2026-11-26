"""The question as the steps that write pre-answer content may read it: without the key.

Everything a player can see before answering (the decomposition, its parts and
entities, the icons, the flow, the beat) is written from a `KeyFreeQuestion`. It has no field for the correct
answer, the feedback or the explanation, so a prompt builder that takes it cannot leak
them: only objectives, the story and the resolution read the examination record itself.
"""

import re
from dataclasses import dataclass

from claude_exam.api.scenario_catalog import scenario_of, split_scenario
from claude_exam.html_clean import clean_plain, clean_question

_SENTENCE_END = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'`(])")


@dataclass(frozen=True)
class KeyFreeQuestion:
    id: str
    section: str
    # The scenario paragraph ("" without one) and the question without it, cleaned.
    paragraph: str
    stem: str
    # The whole question as written (paragraph and stem), cleaned.
    question: str
    answers: tuple[str, ...]
    # Whether the paragraph is a chapter's (shared by several questions).
    shared: bool


def key_free(doc: dict) -> KeyFreeQuestion:
    """The key-free view of an `examination` record."""
    prompt = doc.get("prompt") or {}
    question = clean_question(prompt.get("question", ""))
    paragraph, stem = split_scenario(question)
    return KeyFreeQuestion(
        id=str(doc.get("_id", "")),
        section=doc.get("section") or "",
        paragraph=paragraph,
        stem=stem,
        question=question,
        answers=tuple(clean_plain(a) for a in prompt.get("answers", [])),
        shared=scenario_of(doc) is not None,
    )


def as_view(question: "KeyFreeQuestion | dict") -> KeyFreeQuestion:
    """`question` as a view; a record is reduced to one first, so a helper that accepts
    either still reads only what the view holds."""
    return question if isinstance(question, KeyFreeQuestion) else key_free(question)


def the_ask(question: "KeyFreeQuestion | dict") -> str:
    """The question's own ask: the last sentence of the stem that ends with "?" ("" when
    it has none)."""
    sentences = _SENTENCE_END.split(" ".join(as_view(question).stem.split()))
    asks = [s for s in sentences if s.endswith("?")]
    return asks[-1] if asks else ""
