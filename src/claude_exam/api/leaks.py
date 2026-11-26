"""Does a text shown before answering give an answer away? One definition for every step.

A text a player reads before answering (a decomposition's scenario facts, a semantic
part, a flow label, the story beat, the rephrased question) leaks when it

- names an entity only an answer option names (`AnswerOnly.entities`: what the answers
  propose, such as "a separate review instance"; one the scenario paragraph or the stem
  names is part of the situation and may be shown), or
- contains the text of an answer (at least `MIN_ANSWER_CHARS` long, and not already in
  the scenario or stem, which the player reads anyway).

`answer_only` computes those terms once per question, from the key-free view and, when
there is one, its decomposition; `leak` tests one text and `leak_errors` turns the
findings into gate messages for `ica_json.ask_validated`. Nothing here reads the answer
key: `points_at_correct`, the soft check for reports, gets the correct index passed in.
"""

import re
from dataclasses import dataclass

from claude_exam.api.question_view import KeyFreeQuestion, as_view
from claude_exam.api.schemas import QuestionDecomposition

# An answer shorter than this is a word or two ("Opus", "Use hooks"): finding it in a
# text says nothing.
MIN_ANSWER_CHARS = 20
# Names and aliases under this length match inside other words.
MIN_TERM_CHARS = 4

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "`": ""})


def normalize(text: str) -> str:
    """Text as the checks compare it: lower case, straight quotes without backticks or
    double quotes, whitespace collapsed, no trailing punctuation."""
    text = text.translate(_QUOTES).replace('"', "").lower()
    return re.sub(r"\s+", " ", text).strip(" .,;:")


def names_in(entity, text: str) -> bool:
    """Whether `text` (normalized) names this entity, by its evidence, its name or one
    of its aliases."""
    terms = [entity.evidence, entity.name, *entity.aliases]
    return any(len(normalize(t)) >= 3 and normalize(t) in text for t in terms)


def names_entity(text: str, entities) -> str:
    """The first name or alias of `entities` that `text` contains ("" when none); terms
    under `MIN_TERM_CHARS` characters are ignored."""
    normalized = normalize(text)
    for entity in entities:
        for term in (entity.name, *entity.aliases):
            if len(normalize(term)) >= MIN_TERM_CHARS and normalize(term) in normalized:
                return term
    return ""


def _scenario(question: KeyFreeQuestion) -> str:
    return normalize("\n".join([question.paragraph, question.stem]))


def answer_only_entities(
    decomposition: QuestionDecomposition, question: "KeyFreeQuestion | dict"
) -> list:
    """The entities the scenario paragraph and the stem do not name: what the answers
    propose (a separate instance, a personal skills folder), or the model's own words."""
    scenario = _scenario(as_view(question))
    return [e for e in decomposition.entities if not names_in(e, scenario)]


@dataclass(frozen=True)
class AnswerOnly:
    """What a pre-answer text of one question must not contain."""

    # Entities no scenario text names (`answer_only_entities`).
    entities: tuple = ()
    # Normalized answer texts long enough to test, by answer index; "" for one that is
    # too short or that the scenario already contains.
    answers: tuple[str, ...] = ()
    # The `entities` an answer actually names (one no text names at all is the model's
    # own word, not an answer's).
    proposed: tuple = ()

    def narrow(self) -> "AnswerOnly":
        """Only what an answer names. For a text written from the raw question (the
        decomposition, its parts, the beat, the rephrase), which may well reuse the
        model's own word for something; the flow, drawn from `scenario_view` where such
        entities are already gone, keeps the wide set."""
        return AnswerOnly(entities=self.proposed, answers=self.answers, proposed=self.proposed)


def answer_only(
    question: "KeyFreeQuestion | dict", decomposition: QuestionDecomposition | None = None
) -> AnswerOnly:
    """The leak terms of a question (entities only with a decomposition)."""
    q = as_view(question)
    scenario = _scenario(q)
    answers = tuple(
        a if len(a) >= MIN_ANSWER_CHARS and a not in scenario else ""
        for a in (normalize(a) for a in q.answers)
    )
    if decomposition is None:
        return AnswerOnly(answers=answers)
    entities = tuple(answer_only_entities(decomposition, q))
    answer_text = normalize(" ".join(q.answers))
    proposed = tuple(e for e in entities if names_in(e, answer_text))
    return AnswerOnly(entities=entities, answers=answers, proposed=proposed)


def leak(text: str, terms: AnswerOnly) -> str:
    """Why `text` gives an answer away ("" when it does not)."""
    named = names_entity(text, terms.entities)
    if named:
        return f"names {named!r}, which only an answer option mentions"
    normalized = normalize(text)
    for i, answer in enumerate(terms.answers):
        if answer and answer in normalized:
            return f"repeats answer {chr(65 + i)}"
    return ""


def leak_errors(texts, terms: AnswerOnly, why: str) -> list[str]:
    """Gate messages, one per leaking `(where, text)`: "<where> <reason>: <why>"."""
    errors = []
    for where, text in texts:
        reason = leak(text, terms)
        if reason:
            errors.append(f"{where} {reason}: {why}")
    return errors


_WORD = re.compile(r"[a-z][a-z0-9_-]{3,}")


def points_at_correct(
    texts, question: "KeyFreeQuestion | dict", correct: list[int], what: str = "the text"
) -> str:
    """A soft check (never rejects): when `texts` share clearly more words with the
    correct answer than with any other (words of the stem do not count), they may point
    at it. "" when nothing stands out or the question has not exactly one correct
    answer. The caller passes `correct`, so this module never reads the key."""
    q = as_view(question)
    words = set(_WORD.findall(" ".join(texts).lower()))
    stem = set(_WORD.findall(q.stem.lower()))
    overlap = [len((set(_WORD.findall(a.lower())) - stem) & words) for a in q.answers]
    if len(correct) != 1 or len(overlap) < 2:
        return ""
    best = overlap[correct[0]]
    others = max(o for i, o in enumerate(overlap) if i != correct[0])
    if best >= 2 and best >= 2 * max(others, 1):
        return (
            f"{what} shares {best} words with the correct answer "
            f"{chr(65 + correct[0])} and at most {others} with the others"
        )
    return ""
