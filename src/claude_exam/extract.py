import json
import sys
from collections import Counter
from pathlib import Path

from claude_exam.explanation_parse import is_explanation_based, parse_explanation
from claude_exam.html_clean import clean_plain, clean_question
from claude_exam.section_map import folder_for_section
from claude_exam.vault_scan import QUESTIONS_DIR, REPO_ROOT, find_existing, load_vault

STAGING_DIR = REPO_ROOT / ".staging"


def build_record_feedback_based(result: dict) -> dict:
    prompt = result["prompt"]
    idx = ord(result["correct_response"][0]) - ord("a")
    return {
        "id": result["id"],
        "section": result["section"],
        "folder": folder_for_section(result["section"]),
        "question": clean_question(prompt["question"]),
        "answer": clean_plain(prompt["answers"][idx]),
        "explanation": clean_plain(prompt["feedbacks"][idx]),
        "references": [],
    }


def build_record_explanation_based(result: dict) -> dict:
    prompt = result["prompt"]
    idx_list = [ord(c) - ord("a") for c in result["correct_response"]]
    answer_texts = [clean_plain(prompt["answers"][i]) for i in idx_list]
    answer = (
        answer_texts[0]
        if len(answer_texts) == 1
        else "\n".join(f"{n}. {t}" for n, t in enumerate(answer_texts, 1))
    )
    explanation, references = parse_explanation(prompt["explanation"], answer_texts)
    return {
        "id": result["id"],
        "section": result["section"],
        "folder": folder_for_section(result["section"]),
        "question": clean_question(prompt["question"]),
        "answer": answer,
        "explanation": explanation,
        "references": references,
    }


def build_record(result: dict) -> dict:
    if is_explanation_based(result):
        return build_record_explanation_based(result)
    return build_record_feedback_based(result)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: python -m claude_exam.extract <path-to-practice-json>", file=sys.stderr)
        return 2

    input_path = Path(argv[1])
    data = json.loads(input_path.read_text(encoding="utf-8"))
    results = data["results"]

    vault_files = load_vault(QUESTIONS_DIR)

    pending = []
    already_present = 0
    for result in results:
        if find_existing(result["question_plain"], vault_files):
            already_present += 1
            continue
        pending.append(build_record(result))

    STAGING_DIR.mkdir(exist_ok=True)
    out_path = STAGING_DIR / f"{input_path.stem}.pending.json"
    out_path.write_text(json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8")

    folder_counts = Counter(record["folder"] for record in pending)

    print(f"total questions: {len(results)}")
    print(f"already present: {already_present}")
    print(f"pending: {len(pending)}")
    for folder, count in sorted(folder_counts.items()):
        print(f"  {folder}: {count}")
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
