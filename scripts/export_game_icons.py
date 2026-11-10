"""Write one SVG per icon-catalog key into the game.

The catalog (`claude_exam.api.icon_catalog`) names icons by provider-neutral keys; this
script is where a key becomes a picture. Today that is the Lucide glyph in the entry's
`lucide` column, saved as `<key>.svg` with a white stroke (the game tints it).
To move to another icon set, change where `fetch` reads from.

    uv run python scripts/export_game_icons.py            # write missing icons
    uv run python scripts/export_game_icons.py --force    # rewrite all
    uv run python scripts/export_game_icons.py --check    # only report differences

Unity imports new files as VectorImage (`Assets/Editor/EntityIconImport.cs`). Files of
keys that left the catalog are reported, not deleted: remove them in the Unity Editor
so their `.meta` goes too.
"""

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from claude_exam.api.icon_catalog import ICONS

LUCIDE_VERSION = "1.69.0"
SOURCE = "https://unpkg.com/lucide-static@{version}/icons/{name}.svg"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[2] / "exam-cards-game" / "Assets" / "Resources" / "EntityIcons"
)


def fetch(name: str, version: str) -> str:
    """The Lucide SVG as the game wants it: no licence comment or class attribute
    (the licence ships once, as LICENSE.txt), stroke in white instead of currentColor."""
    with urllib.request.urlopen(SOURCE.format(version=version, name=name), timeout=30) as response:
        svg = response.read().decode("utf-8")
    svg = re.sub(r"<!--.*?-->\s*", "", svg, flags=re.DOTALL)
    svg = re.sub(r'\n\s*class="[^"]*"', "", svg)
    if 'stroke="currentColor"' not in svg:
        raise ValueError(f"{name}: no stroke=\"currentColor\" to replace")
    return svg.replace('stroke="currentColor"', 'stroke="#FFFFFF"').strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the game's icon folder")
    parser.add_argument("--version", default=LUCIDE_VERSION, help="lucide-static version")
    parser.add_argument("--force", action="store_true", help="rewrite icons that already exist")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 on a difference")
    args = parser.parse_args()

    if not args.out.is_dir():
        print(f"{args.out} is not a folder", file=sys.stderr)
        return 2

    problems = written = 0
    for icon in ICONS:
        path = args.out / f"{icon.key}.svg"
        if path.exists() and not (args.force or args.check):
            continue
        try:
            svg = fetch(icon.lucide, args.version)
        except (urllib.error.URLError, ValueError) as exc:
            print(f"FAILED {icon.key} (lucide {icon.lucide}): {exc}", file=sys.stderr)
            problems += 1
            continue
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == svg:
            continue
        if args.check:
            print(f"{'differs' if current else 'missing'}: {path.name} (lucide {icon.lucide})")
            problems += 1
        else:
            path.write_text(svg, encoding="utf-8", newline="\n")
            written += 1
            print(f"wrote {path.name} (lucide {icon.lucide})")

    keys = {icon.key for icon in ICONS}
    for path in sorted(args.out.glob("*.svg")):
        if path.stem not in keys:
            print(f"not in the catalog: {path.name}")
            problems += args.check

    print(f"{written} written, {problems} problem(s), {len(ICONS)} icons in the catalog")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
