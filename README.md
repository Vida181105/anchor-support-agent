# Anchor

Anchor is a merchant support root-cause agent for a payments company: a merchant writes in with a problem, and the system diagnoses the root cause, drafts a resolution, and decides whether to auto-resolve, draft for a human, or escalate. Most root causes live in a merchant's account state (settlement schedules, holds, KYC status, reserves) rather than in policy documentation, so diagnosis requires fusing retrieved policy with structured account state and citing both.

Status: in progress.

## Demo — ops console

```bash
python demo/build_snapshot.py          # pre-compute every seeded run (cached; needs a key once)
uvicorn demo.app:app --port 8000       # serve; no key required
```

Open `/`. It lands on a populated ticket queue — no login, no key entry,
no empty input box.

**The console never calls a model to render its main content.** The queue,
every trace, and the metrics panel are read from `demo/snapshot.json`,
which `demo/build_snapshot.py` produces offline. A cold start on a free
tier with no `GEMINI_API_KEY` set serves the whole console; live mode
simply reports itself as off. `tests/test_demo_app.py` pins this by
patching `LLMClient.__init__` to raise and asserting the page still
renders.

Live mode (`POST /api/diagnose`) runs typed input through the real
pipeline. It is lazily initialised — the policy index is never built at
startup, so a missing key cannot break the page — capped by
`ANCHOR_DEMO_LIVE_BUDGET` (default 25 runs per process), and every failure
path returns `{"ok": false, "reason": ...}` that renders as an inline
notice. A 429 shows a message; it never shows a traceback.

Deploying: build the snapshot locally, commit it, and run
`uvicorn demo.app:app --host 0.0.0.0 --port $PORT`. Setting a key is
optional and only enables live mode.
