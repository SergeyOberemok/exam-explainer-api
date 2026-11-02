import html
import re

_P_TAG = re.compile(r"</?p>")
_PRE_TAG = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL)
_CODE_TAG = re.compile(r"<code>(.*?)</code>", re.DOTALL)
# Fallback: strip any remaining HTML tags not handled by name above.
_ANY_TAG = re.compile(r"<[^>]+>")


def clean_question(question_html: str) -> str:
    # Named conversions first: preserve fenced code blocks and inline code.
    text = _PRE_TAG.sub(lambda m: f"\n```\n{m.group(1)}\n```\n", question_html)
    text = _CODE_TAG.sub(lambda m: f"`{m.group(1)}`", text)
    text = _P_TAG.sub("", text).strip()
    parts = [html.unescape(p).strip() for p in text.split("<br><br>")]
    # Strip any residual tags (<b>, <i>, <a href=...>, etc.) from each part.
    parts = [_ANY_TAG.sub("", p) for p in parts]
    return "  \n  \n".join(parts)


def clean_plain(text: str) -> str:
    # Unescape entities, then strip any HTML tags that slipped through.
    return _ANY_TAG.sub("", html.unescape(text)).strip()
