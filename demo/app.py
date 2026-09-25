"""Anchor ops console: FastAPI serving one page.

Design constraint that drives everything here: the console must render
fully on first click after a cold start, with no API key present and no
model reachable. So the queue and every trace it shows are read from
demo/snapshot.json, pre-computed by demo/build_snapshot.py. This server
makes no model calls to serve its main content.

Live mode (POST /api/diagnose) is the one exception. It is lazily
initialised - the policy index is never built at startup, so a missing
key cannot break the page load - and every failure path returns a 200
with {"ok": false, "reason": ...} that the UI renders as a notice. A
quota-exhausted demo shows a disabled live box and a queue that still
works, never a stack trace.

The live budget is per visitor session (a cookie), not per process, so
one visitor cannot spend the allowance for everyone who arrives after
them. A generous process-wide backstop sits behind it.

    uvicorn demo.app:app --reload
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

load_dotenv()  # local dev reads .env; a deployment sets real env vars

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "demo" / "snapshot.json"
INDEX_HTML = ROOT / "demo" / "static" / "index.html"

# Live mode guard rails. A demo link is public; an unbounded box that
# spends someone's free-tier quota on arbitrary text is not something to
# leave open.
MAX_LIVE_CHARS = 1200

# Measured: one uncached live run is ~170s over 12 model calls (median 16s,
# slowest 46s), with no retries at all - that is just Gemini latency times
# the length of the agent loop. So the deadline has to sit above it, and a
# 60s ceiling would abort essentially every legitimate run. The client is
# told this value by /api/health so both ends agree rather than one cutting
# the other off mid-flight.
LIVE_DEADLINE_SECONDS = int(os.environ.get("ANCHOR_DEMO_LIVE_DEADLINE", "210"))

# Retry budget for LIVE requests only. The batch runner uses 8 retries at a
# 4s base (~17 minutes) because an overnight eval should out-wait an outage.
# Someone watching a spinner should not: two retries at a 1s base is ~4s of
# backoff, then a readable failure.
LIVE_MAX_RETRIES = 2
LIVE_BASE_DELAY = 1.0


def api_key() -> str | None:
    """The key LLMClient will actually use.

    LLMClient reads GEMINI_API_KEY only, but plenty of deployments set
    GOOGLE_API_KEY instead. Accepting one and using the other is how live
    mode ends up advertised as available and then failing on every
    submission, so whatever is found here is passed to the client
    explicitly rather than left to the environment.
    """
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class LiveDeadlineExceeded(RuntimeError):
    """The run outlasted LIVE_DEADLINE_SECONDS."""


def _deadline_client(deadline: float):
    """An LLMClient that refuses to start another call past the deadline.

    Without this the server keeps working - and keeps spending quota - long
    after the browser has given up, which is how a visitor's abandoned
    request turns into everyone else's exhausted budget.
    """
    from src.llm import LLMClient

    class DeadlineLLMClient(LLMClient):
        def generate_turn(self, *a, **k):
            if time.monotonic() > deadline:
                raise LiveDeadlineExceeded(
                    f"the run passed its {LIVE_DEADLINE_SECONDS}s limit"
                )
            return super().generate_turn(*a, **k)

    return DeadlineLLMClient(
        api_key=api_key(), max_retries=LIVE_MAX_RETRIES, base_delay=LIVE_BASE_DELAY
    )


def failure_reason(exc: Exception) -> str:
    """Turn an exception into something a visitor can act on.

    Every branch returns a sentence. An unclassified error still reports
    its type and message - a silent failure is the one outcome that makes
    the whole page look broken.
    """
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if isinstance(exc, LiveDeadlineExceeded):
        return (f"That run passed the {LIVE_DEADLINE_SECONDS}s limit and was stopped. "
                "A full diagnosis is around a dozen model calls, so a slow model "
                "day can exceed it. The pre-computed traces are unaffected.")
    if "resource_exhausted" in low or "quota" in low or "429" in low:
        return ("The model's free-tier quota is exhausted for now, so live mode "
                "cannot run. Everything else on this page still works - it was "
                "recorded ahead of time for exactly this reason.")
    if "503" in low or "unavailable" in low or "overloaded" in low:
        return ("The model is temporarily unavailable and did not recover within "
                "the retry budget. Worth trying again in a moment.")
    if "deadline" in low or "timeout" in low or "timed out" in low:
        return "The model call timed out before returning anything."
    if "api_key" in low or "api key" in low or "permission" in low or "401" in low:
        return "This deployment's API key is missing or rejected, so live mode cannot run."
    if "400" in low or "invalid" in low:
        return f"The model rejected the request. {text[:160]}"
    return f"Live run failed. {text[:200]}"

# Budgeted per visitor session, not per process. A process-wide counter
# lets the first person through the door spend the allowance for everyone
# who arrives after them, which is the opposite of what a shared demo
# link needs. Each visitor gets their own small budget, keyed to a cookie.
LIVE_BUDGET_PER_SESSION = int(os.environ.get("ANCHOR_DEMO_LIVE_BUDGET", "3"))

# A backstop on top of the per-session budget, so a scripted caller
# cycling cookies still cannot drain the key. Deliberately generous: it
# should never be what an ordinary visitor hits.
LIVE_BUDGET_PER_PROCESS = int(os.environ.get("ANCHOR_DEMO_LIVE_BUDGET_TOTAL", "120"))

SESSION_COOKIE = "anchor_sid"

app = FastAPI(title="Anchor ops console", docs_url=None, redoc_url=None)

_state: dict[str, Any] = {
    "snapshot": None, "index": None, "llm": None,
    "live_calls": 0,              # process-wide, for the backstop
    "sessions": {},               # sid -> runs used by that visitor
}


def _session_id(request: Request) -> str:
    return request.cookies.get(SESSION_COOKIE) or uuid.uuid4().hex


def _session_used(sid: str) -> int:
    return _state["sessions"].get(sid, 0)


def snapshot() -> dict:
    """Read once, cache in process. Missing snapshot is a real error - it
    means the deploy is broken - but it surfaces as an empty queue with a
    reason rather than a 500, so the page still renders and says why."""
    if _state["snapshot"] is None:
        if not SNAPSHOT_PATH.exists():
            _state["snapshot"] = {
                "queue": [], "metrics": None,
                "error": "demo/snapshot.json is missing - run python demo/build_snapshot.py",
            }
        else:
            _state["snapshot"] = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return _state["snapshot"]


def live_available(sid: str = "") -> tuple[bool, str]:
    """Whether this visitor may run one more live diagnosis, and why not.

    The reason string is rendered verbatim in the UI, so it says what is
    true without apologising for it: the traces on the page are real
    output either way.
    """
    if not api_key():
        return False, "No API key is configured on this deployment."
    if _state["live_calls"] >= LIVE_BUDGET_PER_PROCESS:
        return False, "This deployment's total live budget is spent."
    if sid and _session_used(sid) >= LIVE_BUDGET_PER_SESSION:
        return False, (
            f"You have used your {LIVE_BUDGET_PER_SESSION} live runs for this session."
        )
    return True, ""


class LiveRequest(BaseModel):
    body: str = Field(default="", max_length=20000)
    merchant_id: str | None = None


@app.get("/api/snapshot")
def api_snapshot() -> JSONResponse:
    return JSONResponse(snapshot())


@app.get("/api/health")
def api_health(request: Request, response: Response) -> dict:
    sid = _session_id(request)
    response.set_cookie(SESSION_COOKIE, sid, max_age=86400, httponly=True, samesite="lax")
    ok, reason = live_available(sid)
    snap = snapshot()
    return {
        "queue_size": len(snap.get("queue", [])),
        "snapshot_error": snap.get("error"),
        "live_enabled": ok,
        "live_reason": reason,
        "live_calls_used": _session_used(sid),
        "live_budget": LIVE_BUDGET_PER_SESSION,
        # the browser aborts a little after the server does, so the server's
        # own reason wins rather than both sides timing out independently
        "live_timeout_seconds": LIVE_DEADLINE_SECONDS + 15,
        "live_typical_seconds": 180,
    }


@app.post("/api/diagnose")
def api_diagnose(req: LiveRequest, request: Request, response: Response) -> dict:
    """Run one typed ticket through the real pipeline.

    Never raises to the client. Every failure - no key, spent budget,
    quota 429, a malformed fixture - comes back as ok:false with a reason
    the UI prints inline. The console keeps working either way.
    """
    body = (req.body or "").strip()
    if not body:
        return {"ok": False, "reason": "Empty ticket."}
    if len(body) > MAX_LIVE_CHARS:
        return {"ok": False, "reason": f"Ticket too long ({len(body)} chars, max {MAX_LIVE_CHARS})."}

    sid = _session_id(request)
    response.set_cookie(SESSION_COOKIE, sid, max_age=86400, httponly=True, samesite="lax")
    ok, reason = live_available(sid)
    if not ok:
        return {"ok": False, "reason": reason}

    started = time.monotonic()
    deadline = started + LIVE_DEADLINE_SECONDS
    try:
        from src.agent import diagnose_ticket
        from src.gate import evaluate
        from src.retrieval import build_index
        from src.schema import render_prose

        # A fresh client per request: the deadline is per-run, and the live
        # retry budget must not leak into anything else.
        llm = _deadline_client(deadline)
        if _state["index"] is None:
            # Lazy on purpose: building at startup would make a missing
            # key or an embedding hiccup break the page itself.
            _state["index"] = build_index(llm)

        _state["live_calls"] += 1
        _state["sessions"][sid] = _session_used(sid) + 1
        ticket = {"id": "live", "merchant_id": req.merchant_id, "body": body,
                  "subject": "(live)", "channel": "demo"}
        result = diagnose_ticket(ticket, llm, _state["index"],
                                 verify=True, check_responsive=True)
        diagnosis = result["diagnosis"]
        decision = evaluate(diagnosis)
        verification = diagnosis.get("_verification") or {}
        responsiveness = diagnosis.get("_responsiveness") or {}
        raw = next((e["raw_ticket"] for e in result["retrieval_log"] if "raw_ticket" in e), "")

        return {
            "ok": True,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "row": {
                "id": "live",
                "ticket": ticket,
                "labels": {},
                "identity": result["identity"],
                "retrieval": {
                    "raw_ticket": raw,
                    "constructed_queries": [e["constructed_query"] for e in result["retrieval_log"]
                                            if "constructed_query" in e],
                },
                "tool_calls": [{"tool": c["tool"], "args": c["args"], "result": c["result"]}
                               for c in result["tool_call_log"]],
                "diagnosis": {
                    "category": diagnosis.get("category"),
                    "risk_class": diagnosis.get("risk_class"),
                    "root_cause": diagnosis.get("root_cause"),
                    "claims": diagnosis.get("claims", []),
                    "recommended_action": diagnosis.get("recommended_action"),
                },
                "rendered_prose": render_prose(diagnosis) if diagnosis.get("root_cause") else "",
                "evidence_content": result["evidence_content"],
                "verification": {"ran": bool(verification),
                                 "root_cause_verdict": verification.get("root_cause_verdict"),
                                 "verdicts": verification.get("verdicts", [])},
                "responsiveness": {"ran": bool(responsiveness),
                                   "verdict": responsiveness.get("verdict"),
                                   "reason": responsiveness.get("reason")},
                "gate": decision.as_dict(),
            },
        }
    except Exception as exc:  # noqa: BLE001 - a demo never shows a traceback
        # The run was charged before it started, to stop a burst of parallel
        # submissions slipping past the budget. It failed for a reason that
        # was not the visitor's doing, so give the run back.
        if _state["sessions"].get(sid):
            _state["sessions"][sid] -= 1
        _state["live_calls"] = max(0, _state["live_calls"] - 1)
        return {
            "ok": False,
            "reason": failure_reason(exc),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)
