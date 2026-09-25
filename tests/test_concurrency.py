"""The checks run concurrently. These pin the two things that makes safe:
the results are unchanged, and a stalled request can no longer hang forever.

Concurrency was added because a cold diagnosis spent ~88s of ~230s running
independent checks one after another. It is only defensible if the output
is identical to the sequential version - otherwise it silently invalidates
the frozen held-out numbers.
"""

import json
import threading
import time

import pytest

import src.verifier as verifier_module
from src.llm import LLMClient, _is_retryable_error, _is_timeout_error
from src.verifier import verify_diagnosis


class ContentLLM:
    """Answers by claim content rather than call order, because concurrent
    callers arrive in a non-deterministic order."""

    def __init__(self, verdict_for, delay=0.0):
        self.verdict_for = verdict_for
        self.delay = delay
        self.calls = []
        self.concurrent = 0
        self.peak = 0
        self._lock = threading.Lock()

    def generate_turn(self, turns, model=None, tools=None, response_json_schema=None, temperature=0.0):
        text = turns[0]["text"]
        with self._lock:
            self.calls.append(text)
            self.concurrent += 1
            self.peak = max(self.peak, self.concurrent)
        try:
            time.sleep(self.delay)
            for needle, verdict in self.verdict_for.items():
                if needle in text:
                    return {"text": json.dumps({"verdict": verdict, "reason": f"re {needle}"})}
            return {"text": json.dumps({"verdict": "SUPPORTED", "reason": "ok"})}
        finally:
            with self._lock:
                self.concurrent -= 1


def _diagnosis(n_claims=5):
    return {
        "category": "settlement_timing",
        "risk_class": "money_movement",
        "root_cause": {"text": "ROOT the batch is held", "evidence": ["e0"]},
        "claims": [{"text": f"CLAIM{i} fact number {i}", "evidence": [f"e{i}"]}
                   for i in range(n_claims)],
        "recommended_action": "auto_resolve",
    }


EVIDENCE = {f"e{i}": {"v": i} for i in range(8)}


def _run(workers, verdicts, n_claims=5, delay=0.0):
    llm = ContentLLM(verdicts, delay=delay)
    out = verify_diagnosis(_diagnosis(n_claims), EVIDENCE, llm, max_workers=workers)
    return out, llm


@pytest.mark.parametrize("verdicts", [
    {},                                                    # everything supported
    {"CLAIM2": "UNSUPPORTED"},                             # one stripped
    {"CLAIM0": "UNSUPPORTED", "CLAIM4": "PARTIALLY_SUPPORTED"},
    {"CLAIM1": "PARTIALLY_SUPPORTED", "CLAIM3": "UNSUPPORTED"},
])
def test_concurrent_verification_is_byte_identical_to_sequential(verdicts):
    seq, _ = _run(1, verdicts)
    par, _ = _run(4, verdicts)
    assert json.dumps(seq, sort_keys=True) == json.dumps(par, sort_keys=True)


def test_verdict_and_claim_order_is_preserved():
    """Order is what keeps recorded runs comparable; a thread pool that
    returned results as they completed would scramble it."""
    _, _ = _run(1, {})
    par, _ = _run(4, {"CLAIM2": "UNSUPPORTED"})
    claim_verdicts = [v for v in par["_verification"]["verdicts"] if not v["is_root_cause"]]
    assert [v["claim"] for v in claim_verdicts] == [f"CLAIM{i} fact number {i}" for i in range(5)]
    assert [c["text"] for c in par["claims"]] == [
        f"CLAIM{i} fact number {i}" for i in range(5) if i != 2]


def test_a_rejected_root_cause_still_short_circuits_without_checking_claims():
    """The root cause stays sequential and first: its rejection discards
    every claim verdict, so spending those calls would be pure waste."""
    par, llm = _run(4, {"ROOT": "UNSUPPORTED"})
    assert par["claims"] == []
    assert par["recommended_action"] == "escalate"
    assert len(llm.calls) == 1, "claims were checked despite a rejected root cause"


def test_claims_actually_overlap():
    _, llm = _run(4, {}, delay=0.05)
    assert llm.peak > 1, "claims did not run concurrently"


def test_max_workers_one_restores_fully_sequential_behaviour():
    _, llm = _run(1, {}, delay=0.02)
    assert llm.peak == 1


def test_pool_never_exceeds_the_configured_width():
    _, llm = _run(3, {}, n_claims=6, delay=0.05)
    assert llm.peak <= 3


def test_no_claims_makes_no_pool_and_no_calls():
    llm = ContentLLM({})
    d = _diagnosis(0)
    out = verify_diagnosis(d, EVIDENCE, llm, max_workers=4)
    assert out["claims"] == []
    assert len(llm.calls) == 1  # the root cause only


def test_default_pool_width_is_configured_not_hardcoded():
    from src.config import CHECK_MAX_WORKERS
    assert verifier_module.CHECK_MAX_WORKERS == CHECK_MAX_WORKERS
    assert CHECK_MAX_WORKERS >= 1


# --- the timeout: a stalled request must become an error ------------------

def test_client_sets_an_explicit_request_timeout():
    """Without http_options the SDK waits on the socket forever. Observed:
    a single call hung for 18 minutes with no response and no error."""
    from src.llm import DEFAULT_TIMEOUT_MS
    assert DEFAULT_TIMEOUT_MS >= 60_000    # above the slowest healthy call seen (45.6s)
    assert DEFAULT_TIMEOUT_MS <= 120_000   # and far below "forever"
    c = LLMClient(api_key="x", timeout_ms=1234)
    assert c.timeout_ms == 1234


def test_the_timeout_is_actually_passed_to_the_sdk(monkeypatch):
    captured = {}

    class FakeGenai:
        @staticmethod
        def Client(**kw):
            captured.update(kw)
            return object()

    import sys
    import types as pytypes
    fake_types = pytypes.SimpleNamespace(HttpOptions=lambda timeout: {"timeout": timeout})
    monkeypatch.setitem(sys.modules, "google.genai", pytypes.SimpleNamespace(types=fake_types))
    monkeypatch.setitem(sys.modules, "google", pytypes.SimpleNamespace(genai=FakeGenai))
    LLMClient(api_key="k", timeout_ms=77_000)._get_client()
    assert captured["http_options"] == {"timeout": 77_000}


@pytest.mark.parametrize("exc", [
    TimeoutError("read timed out"),
    RuntimeError("499 CANCELLED. The operation was cancelled"),
    RuntimeError("Request timed out after 90s"),
])
def test_timeouts_are_treated_as_transient_and_retried(exc):
    assert _is_timeout_error(exc)
    assert _is_retryable_error(exc), "a hung request must back off, not kill the run"


def test_a_bad_request_is_still_not_retried():
    assert not _is_retryable_error(ValueError("400 INVALID_ARGUMENT"))


def test_cache_writes_are_atomic():
    """Concurrent readers must never see a half-written cache entry."""
    import inspect
    from src import llm as llm_module
    src = inspect.getsource(llm_module)
    assert "cache_path.write_text" not in src
    assert src.count("_atomic_write(") >= 4
    assert "_os.replace(tmp, path)" in src


def test_client_construction_is_locked():
    import inspect
    from src import llm as llm_module
    assert "self._client_lock" in inspect.getsource(llm_module.LLMClient._get_client)
