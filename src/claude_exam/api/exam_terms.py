"""Checks on a text a model writes from a question: it keeps the question's numbers
(`lost_numbers`, the decomposition gate) and does not widen its claims (`widened`, the
beat gate).
"""

import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def lost_numbers(original: str, rewrite: str) -> list[str]:
    """The numbers of `original` (as digits) that `rewrite` does not contain."""
    have = set(_NUMBER.findall(rewrite))
    return [n for n in dict.fromkeys(_NUMBER.findall(original)) if n not in have]


# Words that turn "most" or "these" into "all": a text a model writes from the question
# may not use one the question does not (a rewrite said the agent posts "every" finding
# as definitive; the question says it posts "these", the false positives).
# "never" and "none" are left out: a rewrite says "never edits" for "does not edit".
_SWEEPING = re.compile(r"\b(every|everything|all|always)\b", re.IGNORECASE)


def widened(original: str, rewrite: str) -> list[str]:
    """The sweeping words (`_SWEEPING`) of `rewrite` that `original` does not use."""
    have = {w.lower() for w in _SWEEPING.findall(original)}
    return [w for w in dict.fromkeys(w.lower() for w in _SWEEPING.findall(rewrite)) if w not in have]
