"""Tests demo/app.py and the shape of demo/snapshot.json.

The properties worth pinning here are the demo's non-negotiables, not the
styling: the page must render with no API key and no model reachable, the
snapshot must actually contain the things the page claims to show, and
every live-mode failure must come back as a rendered notice rather than a
broken page.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import demo.app as demo_app

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = json.loads((ROOT / "demo" / "snapshot.json").read_text(encoding="utf-8"))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    demo_app._state.update(snapshot=None, index=None, llm=None, live_calls=0)
    return TestClient(demo_app.app)


# --- works with no key at all ---------------------------------------------

def test_page_and_queue_render_without_any_api_key(client):
    assert client.get("/").status_code == 200
    body = client.get("/api/snapshot").json()
    assert len(body["queue"]) >= 10
    assert "error" not in body


def test_health_reports_live_off_but_queue_present(client):
    h = client.get("/api/health").json()
    assert h["live_enabled"] is False
    assert h["queue_size"] >= 10
    assert h["snapshot_error"] is None


def test_serving_the_snapshot_makes_no_model_calls(client, monkeypatch):
    """The main content must not depend on a live key. If anything on the
    default path constructed an LLMClient, this would raise."""
    import src.llm

    def explode(*a, **k):
        raise AssertionError("serving the console must not construct an LLM client")

    monkeypatch.setattr(src.llm.LLMClient, "__init__", explode)
    assert client.get("/").status_code == 200
    assert client.get("/api/snapshot").status_code == 200
    assert client.get("/api/health").status_code == 200


def test_missing_snapshot_degrades_to_a_reason_not_a_500(client, monkeypatch, tmp_path):
    monkeypatch.setattr(demo_app, "SNAPSHOT_PATH", tmp_path / "nope.json")
    demo_app._state["snapshot"] = None
    body = client.get("/api/snapshot").json()
    assert body["queue"] == []
    assert "snapshot.json" in body["error"]


# --- live mode degrades, never breaks -------------------------------------

def test_live_without_a_key_returns_ok_false_with_a_reason(client):
    r = client.post("/api/diagnose", json={"body": "why is my settlement late"})
    assert r.status_code == 200
    assert r.json() == {"ok": False, "reason": demo_app.live_available()[1]}


def test_live_rejects_empty_and_oversized_input(client):
    assert client.post("/api/diagnose", json={"body": "   "}).json()["ok"] is False
    long = "x" * (demo_app.MAX_LIVE_CHARS + 1)
    out = client.post("/api/diagnose", json={"body": long}).json()
    assert out["ok"] is False and "too long" in out["reason"]


def test_live_budget_is_enforced(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    demo_app._state["live_calls"] = demo_app.LIVE_CALL_BUDGET
    out = client.post("/api/diagnose", json={"body": "hello"}).json()
    assert out["ok"] is False and "budget" in out["reason"]


def test_live_exception_is_caught_and_reported(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr("src.retrieval.build_index",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429 quota exhausted")))
    out = client.post("/api/diagnose", json={"body": "why is my settlement late"}).json()
    assert out["ok"] is False
    assert "unaffected" in out["reason"]


# --- the snapshot contains what the page claims to show -------------------

def test_queue_covers_all_four_gate_lanes():
    lanes = {r["gate"]["outcome"] for r in SNAPSHOT["queue"]}
    assert lanes == {"AUTO_RESOLVE", "DRAFT_FOR_HUMAN", "REQUEST_IDENTIFICATION", "ESCALATE"}


def test_the_three_pinned_scenarios_are_present():
    pinned = {r["pinned"]: r for r in SNAPSHOT["queue"] if r.get("pinned")}
    assert set(pinned) == {"state_informed_retrieval", "identity_mismatch", "evidence_gap_failure"}
    assert pinned["state_informed_retrieval"]["id"] == "ticket_035"
    assert pinned["identity_mismatch"]["identity"]["outcome"] == "MISMATCH"
    gap = pinned["evidence_gap_failure"]
    assert gap["labels"]["refusal_type"] == "EVIDENCE_GAP"
    assert gap["known_failure"] is True
    assert gap["labels"]["adjudication"] == "WRONG"


def test_every_known_failure_is_labelled_as_one():
    """A ticket the adjudication calls WRONG must be flagged in the UI, not
    presented as a working example."""
    for row in SNAPSHOT["queue"]:
        if row["labels"].get("adjudication") == "WRONG":
            assert row.get("known_failure") is True, f"{row['id']} is wrong but unlabelled"


def test_state_informed_retrieval_contrast_is_actually_visible():
    """The pinned scenario only makes its point if the constructed query
    differs from the raw ticket text and names the real mechanism."""
    row = next(r for r in SNAPSHOT["queue"] if r["id"] == "ticket_035")
    raw = row["retrieval"]["raw_ticket"]
    built = " ".join(row["retrieval"]["constructed_queries"])
    assert built and built != raw
    assert "webhook" in built.lower()
    assert "webhook" not in raw.lower()


def test_mismatch_ticket_read_no_merchant_state():
    row = next(r for r in SNAPSHOT["queue"] if r["identity"]["outcome"] == "MISMATCH")
    assert row["tool_calls"] == []
    assert row["gate"]["outcome"] == "ESCALATE"


def test_every_cited_evidence_id_resolves_to_content():
    """Claims are clickable in the UI; a chip that opens an empty box would
    be worse than no chip."""
    for row in SNAPSHOT["queue"]:
        d = row["diagnosis"]
        parts = list(d.get("claims") or [])
        if d.get("root_cause"):
            parts.append(d["root_cause"])
        for part in parts:
            for eid in part.get("evidence", []):
                assert eid in row["evidence_content"], f"{row['id']}: {eid} has no content"


def test_agent_visible_ticket_carries_no_eval_labels():
    """The page shows the ticket as the agent saw it. Answer-key fields
    live in a separate object, never mixed into it."""
    leak = {"true_root_cause", "correct_routing", "refusal_type", "evidence_gap_fact",
            "true_merchant_id", "identity_note"}
    for row in SNAPSHOT["queue"]:
        assert not (leak & set(row["ticket"])), row["id"]
        assert not any(k.startswith("_") for k in row["ticket"]), row["id"]


def test_both_checks_are_recorded_for_every_diagnosed_ticket():
    for row in SNAPSHOT["queue"]:
        if not row["diagnosis"].get("claims"):
            continue  # refusals have nothing to check
        assert row["verification"]["ran"] is True, row["id"]
        assert row["responsiveness"]["verdict"] in (
            "RESPONSIVE", "PARTIALLY_RESPONSIVE", "NON_RESPONSIVE"), row["id"]


def test_metrics_keep_frozen_and_posthoc_separate_and_match_the_eval_files():
    m = SNAPSHOT["metrics"]
    assert m["frozen"] != m["posthoc"]
    assert m["frozen"]["false_auto"] == [6, 40]
    assert m["posthoc"]["false_auto"] == [5, 40]
    assert m["frozen"]["refusal"]["EVIDENCE_GAP"] == [1, 8]
    assert m["accuracy"]["correct"] == 24 and m["accuracy"]["wrong"] == 13
    # the delta-zero ablation finding must survive into the UI
    assert m["verifier_off"]["false_auto"] == m["frozen"]["false_auto"]


def test_the_page_references_every_api_it_needs():
    html = (ROOT / "demo" / "static" / "index.html").read_text(encoding="utf-8")
    for path in ("/api/snapshot", "/api/health", "/api/diagnose"):
        assert path in html
