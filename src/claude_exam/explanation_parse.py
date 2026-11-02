import html
import re

_INCORRECT_MARKER = re.compile(r"Incorrect option")
_REFERENCES_MARKER = re.compile(r"References:")
_CORRECT_PREFIX = re.compile(r"^.*?Correct options?:", re.DOTALL)
_STRONG = re.compile(r"<strong>(.*?)</strong>", re.DOTALL)
_IMG_TAG = re.compile(r"<img[^>]*>")
_VIA_CITATION = re.compile(r"via\s*-\s*<a[^>]*>.*?</a>", re.DOTALL)
_P_TAG = re.compile(r"</?p>")
_HREF = re.compile(r'<a href="(.*?)"')


def is_explanation_based(result: dict) -> bool:
    return all(not f.strip() for f in result["prompt"]["feedbacks"])


def _clean_paragraph_chunk(chunk: str) -> str:
    chunk = _IMG_TAG.sub("", chunk)
    chunk = _VIA_CITATION.sub("", chunk)
    chunk = _P_TAG.sub("", chunk)
    paragraphs = [html.unescape(p).strip() for p in chunk.split("<br><br>")]
    return "  \n  \n".join(p for p in paragraphs if p)


def parse_explanation(explanation_html: str, correct_texts: list[str]) -> tuple[str, list[str]]:
    incorrect_match = _INCORRECT_MARKER.search(explanation_html)
    references_match = _REFERENCES_MARKER.search(explanation_html)

    correct_segment = explanation_html[:incorrect_match.start()]
    correct_segment = _CORRECT_PREFIX.sub("", correct_segment)
    parts = _STRONG.split(correct_segment)

    blocks = [_clean_paragraph_chunk(parts[i + 1]) for i in range(1, len(parts), 2)]
    if len(correct_texts) > 1:
        blocks = [f"**{text}**  \n  \n{block}" for text, block in zip(correct_texts, blocks)]
    explanation = "  \n  \n".join(blocks)

    references_segment = explanation_html[references_match.end():]
    urls = []
    for url in _HREF.findall(references_segment):
        if url not in urls:
            urls.append(url)

    return explanation, urls
