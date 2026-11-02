"""Generate the Obsidian roadmap/ note hierarchy from public/roadmap.yaml.

Re-run this whenever roadmap.yaml changes:
    uv run python -m claude_exam.roadmap_gen
"""
import shutil
import sys
from pathlib import Path

import yaml

from claude_exam.vault_scan import REPO_ROOT

ROADMAP_YAML = REPO_ROOT / "public" / "roadmap.yaml"
ROADMAP_DIR = REPO_ROOT / "roadmap"
ROOT_TITLE = "Roadmap"

_INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def safe_filename(title: str) -> str:
    cleaned = "".join(ch for ch in title if ch not in _INVALID_FILENAME_CHARS)
    cleaned = cleaned.strip()
    return cleaned or "untitled"


def note_text(title: str, parent_title: str | None, children: list[dict] | None) -> str:
    filename = safe_filename(title)
    lines = []
    if filename != title:
        lines += ["---", "aliases:", f'  - "{title}"', "---", ""]
    lines.append(f"# {title}")
    lines.append("")
    if parent_title is not None:
        lines.append(f"Part of: [[{parent_title}]]")
        lines.append("")
    if children:
        lines.append("## Steps")
        for child in children:
            lines.append(f"- [[{child['title']}]]")
        lines.append("")
    return "\n".join(lines)


def write_node(node: dict, parent_title: str, parent_dir: Path) -> None:
    title = node["title"]
    children = node.get("children")
    filename = safe_filename(title)

    if children:
        node_dir = parent_dir / filename
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / f"{filename}.md").write_text(
            note_text(title, parent_title, children), encoding="utf-8"
        )
        for child in children:
            write_node(child, title, node_dir)
    else:
        (parent_dir / f"{filename}.md").write_text(
            note_text(title, parent_title, None), encoding="utf-8"
        )


def main() -> int:
    nodes = yaml.safe_load(ROADMAP_YAML.read_text(encoding="utf-8"))

    if ROADMAP_DIR.exists():
        shutil.rmtree(ROADMAP_DIR)
    ROADMAP_DIR.mkdir(parents=True)

    (ROADMAP_DIR / f"{ROOT_TITLE}.md").write_text(
        note_text(ROOT_TITLE, None, nodes), encoding="utf-8"
    )
    for node in nodes:
        write_node(node, ROOT_TITLE, ROADMAP_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
