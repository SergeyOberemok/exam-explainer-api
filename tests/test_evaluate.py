"""The prompt evaluation page: its static file, the question view and the dry run's
JSON Lines stream (which must write nothing)."""

import json

import pytest
from bson import ObjectId
from fastapi import HTTPException
from test_decompose import review_question
from test_jobs import FakeMongo, ScriptedIca, client_for, question

from claude_exam.api import batch, ica_json
from claude_exam.api.scenario_catalog import text_hash


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(batch, "BACKOFF_SECONDS", 0)
    monkeypatch.setattr(ica_json, "_NO_SCHEMA", set())


def lines(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def written(mongo):
    db = mongo.get_default_database()
    return {name: len(coll.docs) for name, coll in db.items() if name != "examination"}


def test_the_page_and_its_options_are_served():
    client = client_for(FakeMongo(), ScriptedIca())
    page = client.get("/evaluate/ui/")
    assert page.status_code == 200 and "Prompt evaluation" in page.text
    options = client.get("/evaluate/options").json()
    assert options["dry_stages"][0] == "decompose" and "objectives" in options["stages"]
    assert options["default_model"] in options["models"]


def test_an_unknown_question_is_404():
    client = client_for(FakeMongo(), ScriptedIca())
    assert client.get(f"/evaluate/questions/{ObjectId()}").status_code == 404
    assert client.get("/evaluate/questions/nope").status_code == 404
    assert client.post(f"/evaluate/questions/{ObjectId()}/run", json={}).status_code == 404


def test_the_view_shows_the_original_with_its_key_and_nothing_stored_yet():
    q = review_question()
    client = client_for(FakeMongo([q]), ScriptedIca())
    got = client.get(f"/evaluate/questions/{q['_id']}").json()
    answers = got["original"]["answers"]
    assert [a["correct"] for a in answers] == [False, True, False, False]
    assert answers[1]["feedback"] == "Correct. b"
    assert got["original"]["explanation"] == "Missing guidance needs human judgment."
    assert got["stored"]["rephrase"] is None and got["stored"]["decomposition"] is None


def test_a_dry_run_streams_one_line_per_step_and_writes_nothing():
    q = question()
    mongo, ica = FakeMongo([q]), ScriptedIca()
    client = client_for(mongo, ica)
    before = written(mongo)
    stages = ["decompose", "icons", "flow", "resolve", "rephrase", "objectives"]
    response = client.post(f"/evaluate/questions/{q['_id']}/run", json={"stages": stages})
    assert response.status_code == 200
    events = lines(response)
    assert [e["stage"] for e in events] == [*stages, "done"]
    assert [e["status"] for e in events[:5]] == ["ok"] * 5
    assert events[5]["status"].startswith("skipped")
    result = events[-1]["result"]
    assert result["decomposition"] and result["parts"][-1]["role"] == "ask"
    assert result["resolution"]["rule"] and result["rephrase"]["answers"]
    assert result["entities"]["question_icon"] and result["mermaid"]["flow"].startswith("flowchart")
    # The experiment: the flow built in code, with the flow gate's verdict.
    assert result["code_flow"]["nodes"][-1]["is_question"] and result["code_flow_errors"] == []
    assert result["mermaid"]["code_flow"].startswith("flowchart")
    assert written(mongo) == before


def test_a_dry_beat_needs_the_stored_story_and_feeds_the_rephrase():
    q = question()
    q["prompt"]["question"] = (
        "SCENARIO: You are integrating Claude Code into your Continuous Integration pipeline. "
        "QUESTION: Which mode?"
    )
    mongo, ica = FakeMongo([q]), ScriptedIca()
    client = client_for(mongo, ica)
    run = lambda: lines(client.post(f"/evaluate/questions/{q['_id']}/run", json={"stages": ["beat", "rephrase"]}))  # noqa: E731
    # No story record: no stage to write the beat for.
    assert run()[0]["status"] == "failed 422"
    db = mongo.get_default_database()
    db["question_story"].docs.append(
        {"question_id": str(q["_id"]), "text_hash": text_hash(q), "scenario": "pipeline", "stage": "build",
         "order": 1, "beat": "", "win": "w", "fail": "f", "origin": "ica"}
    )
    before = written(mongo)
    events = run()
    assert [e["status"] for e in events[:2]] == ["ok", "ok"]
    assert events[-1]["result"]["beat"] == "The team cannot agree on a working mode."
    rephrase_prompt = [p for kind, p in zip(ica.calls, ica.prompts) if kind == "rephrase"][-1]
    assert "The team cannot agree on a working mode." in rephrase_prompt
    assert written(mongo) == before


def test_a_failed_step_is_reported_and_the_run_goes_on():
    q = question()
    ica = ScriptedIca(decompose=[HTTPException(status_code=502, detail="bad reply")] * 5)
    client = client_for(FakeMongo([q]), ica)
    events = lines(
        client.post(f"/evaluate/questions/{q['_id']}/run", json={"stages": ["decompose", "flow", "rephrase"]})
    )
    by_stage = {e["stage"]: e for e in events}
    assert by_stage["decompose"]["status"] == "failed 502"
    assert by_stage["flow"]["status"] == "failed 409" and "decomposition" in by_stage["flow"]["detail"]
    assert by_stage["rephrase"]["status"] == "ok"
    assert events[-1]["stage"] == "done"


def test_the_run_needs_the_api_key_when_one_is_set():
    from claude_exam.api.main import app

    q = question()
    client = client_for(FakeMongo([q]), ScriptedIca())
    app.state.api_key = "secret"
    assert client.get(f"/evaluate/questions/{q['_id']}").status_code == 401
    assert client.get(f"/evaluate/questions/{q['_id']}", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/evaluate/ui/").status_code == 200


def test_a_persisted_run_writes_the_records_and_streams_what_is_stored():
    q = question()
    mongo = FakeMongo([q])
    client = client_for(mongo, ScriptedIca())
    stages = ["decompose", "flow", "rephrase"]
    response = client.post(
        f"/evaluate/questions/{q['_id']}/run", json={"stages": stages, "persist": True}
    )
    assert response.status_code == 200
    events = lines(response)
    assert [e["stage"] for e in events] == [*stages, "done"]
    assert [e["status"] for e in events[:3]] == ["ok or current"] * 3
    result = events[-1]["result"]
    assert result["decomposition"] and result["flow"] and result["rephrase"]["answers"]
    # Written, unlike a dry run: the view now shows the same records.
    stored = client.get(f"/evaluate/questions/{q['_id']}").json()["stored"]
    assert stored["decomposition"] and stored["rephrase"]
    assert written(mongo)["question_decomposition"] == 1


def test_a_persisted_step_that_fails_is_reported():
    q = question()
    ica = ScriptedIca(decompose=[HTTPException(status_code=502, detail="bad reply")] * 5)
    client = client_for(FakeMongo([q]), ica)
    events = lines(
        client.post(
            f"/evaluate/questions/{q['_id']}/run",
            json={"stages": ["decompose"], "persist": True},
        )
    )
    assert events[0]["status"] == "failed 502" and events[-1]["stage"] == "done"


def test_a_persisted_run_is_refused_while_a_job_runs():
    import asyncio

    from claude_exam.api.main import app

    q = question()
    client = client_for(FakeMongo([q]), ScriptedIca())
    from claude_exam.api.dependencies import get_simplification_lock

    locked = asyncio.Semaphore(0)
    app.dependency_overrides[get_simplification_lock] = lambda: locked
    response = client.post(
        f"/evaluate/questions/{q['_id']}/run", json={"stages": ["decompose"], "persist": True}
    )
    assert response.status_code == 409
    # A dry run does not take the lock.
    assert client.post(f"/evaluate/questions/{q['_id']}/run", json={"stages": ["decompose"]}).status_code == 200
