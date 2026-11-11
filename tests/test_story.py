"""The story step: the scenario catalog, the prompt, parsing, and the seed file."""

import importlib.util
import json
from pathlib import Path

import pytest
from bson import ObjectId
from fastapi import HTTPException
from test_entities import FakeIca, reply

from claude_exam.api.scenario_catalog import (
    BY_KEY,
    CATALOG_VERSION,
    SCENARIOS,
    chapter_version,
    STAGE_KEYS,
    scenario_of,
    split_scenario,
    stem_of,
    text_hash,
)
from claude_exam.api.story import (
    MAX_LINE_WORDS,
    SEED_LINE_WORDS,
    _build_system_message,
    _build_user_message,
    parse_story,
    record_fields,
    write_story,
)

pytestmark = pytest.mark.anyio
ROOT = Path(__file__).resolve().parents[1]

PARAGRAPH = "You are building a customer support resolution agent using the Claude Agent SDK."
DOC = {
    "_id": ObjectId("6ae4891ec11925a6a80c076b"),
    "correct_response": ["b"],
    "prompt": {
        "question": f"SCENARIO: {PARAGRAPH}\n\nQUESTION: The agent escalates too often.\nWhat should change?",
        "answers": ["Add more tools to the agent", "Write explicit escalation criteria", "Lower the temperature"],
    },
}
GOOD = {
    "stage": "incident",
    "beat": "Support leads complain that easy cases keep landing on their desks.",
    "win": "Escalations follow the case.",
    "fail": "Easy cases keep landing on the wrong desk.",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def with_question(text):
    return {**DOC, "prompt": {**DOC["prompt"], "question": text}}


def test_catalog_is_consistent():
    assert len(BY_KEY) == len(SCENARIOS) >= 4 and len(STAGE_KEYS) == 5
    assert all(s.title and s.district and s.intro and s.outro for s in SCENARIOS)
    assert all(s.marker == s.marker.lower() for s in SCENARIOS)
    assert len(CATALOG_VERSION) == 12


def test_split_takes_the_scenario_paragraph_off_the_question():
    assert split_scenario(DOC["prompt"]["question"]) == (
        PARAGRAPH, "The agent escalates too often.\nWhat should change?",
    )
    assert split_scenario("  Just a question?  ") == ("", "Just a question?")
    assert split_scenario(None) == ("", "")


def test_chapter_comes_from_the_paragraph():
    assert scenario_of(DOC).key == "support"
    assert stem_of(DOC) == "The agent escalates too often.\nWhat should change?"
    # Line breaks and case in the source do not matter.
    assert scenario_of(with_question("SCENARIO: a Customer  Support\nResolution Agent. QUESTION: q")).key == "support"
    for text in ("no markers", "SCENARIO: You run a bakery. QUESTION: q"):
        unknown = with_question(text)
        assert scenario_of(unknown) is None and stem_of(unknown) == ""


def test_text_hash_follows_the_text_not_the_id_or_the_spacing():
    spaced = with_question(DOC["prompt"]["question"].replace("\n", "  ").upper())
    assert text_hash({**spaced, "_id": ObjectId()}) == text_hash(DOC) and len(text_hash(DOC)) == 16
    assert text_hash(with_question("another")) != text_hash(DOC)
    other_answers = {**DOC, "prompt": {**DOC["prompt"], "answers": ["x", "y", "z"]}}
    assert text_hash(other_answers) != text_hash(DOC)


def test_parse_returns_the_fields_to_store():
    assert parse_story({**GOOD, "beat": "  two   spaces  ", "extra": 1}, DOC["prompt"]["answers"]) == {
        **GOOD, "beat": "two spaces",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "finale"},
        {"stage": None},
        {"beat": ""},
        {"win": None},
        {"fail": 5},
        {"fail": " ".join(["word"] * (MAX_LINE_WORDS + 1))},
        {"win": "x" * 300},
        # The beat must not give an answer away.
        {"beat": "Someone suggests: write explicit  ESCALATION criteria."},
        {"beat": "Someone suggests to lower the temperature."},
    ],
)
def test_parse_rejects_what_would_break_the_story(changes):
    with pytest.raises(HTTPException) as exc:
        parse_story({**GOOD, **changes}, DOC["prompt"]["answers"], correct=[1])
    assert exc.value.status_code == 502


def test_lines_after_the_answer_may_name_a_wrong_answer_but_not_the_correct_one():
    answers = DOC["prompt"]["answers"]
    wrong = {**GOOD, "fail": "Someone decided to lower the temperature, and nothing changed."}
    assert parse_story(wrong, answers, correct=[1])["fail"] == wrong["fail"]
    for name in ("win", "fail"):
        with pytest.raises(HTTPException):
            parse_story({**GOOD, name: "They write explicit escalation criteria."}, answers, correct=[1])
    # Hand-written seed lines may be longer than the model's.
    long = {**GOOD, "win": " ".join(["word"] * (MAX_LINE_WORDS + 1))}
    assert parse_story(long, answers, max_words=SEED_LINE_WORDS)["win"] == long["win"]


def test_messages_carry_the_chapter_and_the_stem_only():
    system = _build_system_message(DOC)
    assert "{{" not in system and BY_KEY["support"].title in system and BY_KEY["support"].intro in system
    assert all(f"- {key}: " in system for key in STAGE_KEYS)
    user = _build_user_message(DOC, [1])
    assert "The agent escalates too often." in user and PARAGRAPH not in user
    assert "CORRECT ANSWER: B" in user and "  C. Lower the temperature" in user


def test_record_fields_identify_the_question_and_the_versions():
    fields = record_fields(DOC, GOOD, "seed", 3)
    assert (fields["scenario"], fields["order"], fields["origin"]) == ("support", 3, "seed")
    assert fields["text_hash"] == text_hash(DOC) and fields["scenario_origin"] == "text"
    assert fields["catalog_version"] == chapter_version("support") != chapter_version("research")
    assert fields["beat"] == GOOD["beat"]


async def test_write_story_returns_fields_and_meta():
    ica = FakeIca(reply(GOOD))
    fields, meta = await write_story(ica, DOC, "claude-haiku-5-5")
    assert fields == GOOD and meta.total_tokens == 42
    assert "story of a quiz game" in ica.calls[0]["system"]


async def test_write_story_rejects_a_truncated_reply():
    with pytest.raises(HTTPException) as exc:
        await write_story(FakeIca(reply(GOOD, finish_reason="length")), DOC, "claude-haiku-5-5")
    assert exc.value.status_code == 502


@pytest.mark.parametrize(
    "doc",
    [with_question("SCENARIO: You run a bakery. QUESTION: q"), {**DOC, "correct_response": []}],
)
async def test_question_of_an_unknown_scenario_or_without_a_key_is_refused_without_calling_the_model(doc):
    ica = FakeIca(reply(GOOD))
    with pytest.raises(HTTPException) as exc:
        await write_story(ica, doc, "claude-haiku-5-5")
    assert exc.value.status_code == 422 and ica.calls == []


# --- the seed file -----------------------------------------------------------------


def loader():
    spec = importlib.util.spec_from_file_location("load_story_seed", ROOT / "scripts" / "load_story_seed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_file_is_well_formed():
    items = json.loads((ROOT / "data" / "story_seed.json").read_text(encoding="utf-8"))["items"]
    assert len({i["hash"] for i in items}) == len(items) > 0
    for item in items:
        assert item["scenario"] in BY_KEY, item["hash"]
        assert parse_story(item, [], max_words=SEED_LINE_WORDS)["stage"] in STAGE_KEYS


def test_seed_updates_order_items_inside_a_stage_and_report_problems():
    first, second = DOC, with_question(f"SCENARIO: {PARAGRAPH} QUESTION: Another one?")
    second = {**second, "_id": ObjectId()}
    items = [
        {**GOOD, "hash": text_hash(second), "scenario": "support"},
        {**GOOD, "hash": text_hash(first), "scenario": "support"},
        {**GOOD, "hash": text_hash(first), "scenario": "support"},
        {**GOOD, "hash": "nobody", "scenario": "support"},
    ]
    updates, problems = loader().seed_updates(items, [first, second])
    assert [(qid, f["order"], f["origin"]) for qid, f in updates] == [
        (str(second["_id"]), 1, "seed"), (str(first["_id"]), 2, "seed"),
    ]
    assert len(problems) == 2 and "listed twice" in problems[0] and "no question" in problems[1]

    wrong_chapter = [{**GOOD, "hash": text_hash(first), "scenario": "research"}]
    bad_line = [{**GOOD, "hash": text_hash(first), "scenario": "support", "beat": ""}]
    for seed in (wrong_chapter, bad_line):
        updates, problems = loader().seed_updates(seed, [first])
        assert updates == [] and len(problems) == 1
