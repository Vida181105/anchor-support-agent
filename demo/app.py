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
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
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

    try:
        from src.agent import diagnose_ticket
        from src.gate import evaluate
        from src.llm import LLMClient
        from src.retrieval import build_index
        from src.schema import render_prose

        if _state["llm"] is None:
            _state["llm"] = LLMClient()
        if _state["index"] is None:
            # Lazy on purpose: building at startup would make a missing
            # key or an embedding hiccup break the page itself.
            _state["index"] = build_index(_state["llm"])

        _state["live_calls"] += 1
        _state["sessions"][sid] = _session_used(sid) + 1
        ticket = {"id": "live", "merchant_id": req.merchant_id, "body": body,
                  "subject": "(live)", "channel": "demo"}
        result = diagnose_ticket(ticket, _state["llm"], _state["index"],
                                 verify=True, check_responsive=True)
        diagnosis = result["diagnosis"]
        decision = evaluate(diagnosis)
        verification = diagnosis.get("_verification") or {}
        responsiveness = diagnosis.get("_responsiveness") or {}
        raw = next((e["raw_ticket"] for e in result["retrieval_log"] if "raw_ticket" in e), "")

        return {
            "ok": True,
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
        return {"ok": False, "reason": f"Live run failed ({type(exc).__name__}). "
                                       f"The pre-computed queue is unaffected. {exc}"[:400]}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)
