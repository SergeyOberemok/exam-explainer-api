import html
import re

from claude_exam.api.scenario_catalog import text_hash

_HTML_TAG_NAMES = (
    "a|b|i|u|p|br|hr|code|pre|strong|em|img|"
    "ul|ol|li|span|div|blockquote|table|thead|tbody|tr|td|th|h[1-6]"
)
_TAG_RE = re.compile(rf"</?(?:{_HTML_TAG_NAMES})(?:\s+[^>]*)?/?>", re.IGNORECASE)


def strip_html(text: str) -> str:
    for _ in range(3):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_result(item: dict, source: str) -> dict:
    item = dict(item)
    item.pop("_class", None)
    item.pop("id", None)
    item.pop("question_plain", None)
    item["source"] = source

    prompt = dict(item.get("prompt") or {})
    if "question" in prompt:
        prompt["question"] = strip_html(prompt["question"])
    if "answers" in prompt:
        prompt["answers"] = [strip_html(a) for a in prompt["answers"]]
    if "feedbacks" in prompt:
        prompt["feedbacks"] = [strip_html(f) for f in prompt["feedbacks"]]
    if "explanation" in prompt:
        prompt["explanation"] = strip_html(prompt["explanation"])
    item["prompt"] = prompt
    # Identifies the question by its text and answers: an import skips a question the
    # bank already holds, and the story seed finds its question under any id.
    item["text_hash"] = text_hash(item)

    return item
