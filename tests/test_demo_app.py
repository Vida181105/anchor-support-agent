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
    demo_app._state.update(snapshot=None, index=None, llm=None, live_calls=0, sessions={})
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


def test_live_budget_is_per_session_not_per_process(client, monkeypatch):
    """One visitor spending their runs must not lock out the next one -
    the whole reason the budget is keyed to a cookie."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    demo_app._state["sessions"]["spent"] = demo_app.LIVE_BUDGET_PER_SESSION

    client.cookies.set(demo_app.SESSION_COOKIE, "spent")
    out = client.post("/api/diagnose", json={"body": "hello"}).json()
    assert out["ok"] is False and "live runs for this session" in out["reason"]

    client.cookies.set(demo_app.SESSION_COOKIE, "fresh")
    ok, reason = demo_app.live_available("fresh")
    assert ok is True and reason == ""


def test_per_session_budget_is_three(client):
    assert demo_app.LIVE_BUDGET_PER_SESSION == 3
    assert demo_app.LIVE_BUDGET_PER_PROCESS > demo_app.LIVE_BUDGET_PER_SESSION


def test_process_backstop_still_applies(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    demo_app._state["live_calls"] = demo_app.LIVE_BUDGET_PER_PROCESS
    out = client.post("/api/diagnose", json={"body": "hello"}).json()
    assert out["ok"] is False and "total live budget" in out["reason"]


def test_health_issues_a_session_cookie_and_reports_that_sessions_budget(client):
    r = client.get("/api/health")
    assert demo_app.SESSION_COOKIE in r.cookies
    assert r.json()["live_budget"] == demo_app.LIVE_BUDGET_PER_SESSION


def test_live_reason_is_not_duplicated_by_the_page_copy(client):
    """The UI supplies its own framing sentence, so the server's reason
    must not repeat 'live mode is off' back into it."""
    _, reason = demo_app.live_available("x")
    html = (ROOT / "demo" / "static" / "index.html").read_text(encoding="utf-8")
    assert "live mode is off" not in reason.lower()
    assert html.lower().count("live mode is off") <= 1


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


# --- presentation invariants ----------------------------------------------

def test_header_orients_a_cold_visitor_in_three_lines():
    html = (ROOT / "demo" / "static" / "index.html").read_text(encoding="utf-8")
    block = html[html.index('class="hlines"'):html.index('</header>')]
    assert block.count("<div") == 3, "the orientation header is three lines, not a hero section"
    assert "account state" in block and "not documentation" in block
    assert "Click any ticket" in block


def test_no_marketing_chrome_crept_in():
    html = (ROOT / "demo" / "static" / "index.html").read_text(encoding="utf-8")
    for banned in ("linear-gradient", "radial-gradient", "<img", "hero"):
        assert banned not in html.lower(), banned


def test_mobile_sidebar_is_collapsible_and_body_cannot_scroll_sideways():
    html = (ROOT / "demo" / "static" / "index.html").read_text(encoding="utf-8")
    assert "overflow-x:hidden" in html          # page body never scrolls sideways
    assert ".scroll{overflow-x:auto" in html    # wide tables scroll inside their own box
    assert "qtoggle" in html
    assert "#queue.collapsed" in html


def test_reading_column_is_capped():
    html = (ROOT / "demo" / "static" / "index.html").read_text(encoding="utf-8")
    assert "--read:" in html
    assert "max-width:var(--read)" in html


# --- information architecture ---------------------------------------------

HTML = (ROOT / "demo" / "static" / "index.html").read_text(encoding="utf-8")


def test_trace_uses_progressive_disclosure_not_six_stacked_sections():
    """A ticket opens on the answer; the six stages are collapsed rows."""
    assert '<details class="stage"' in HTML
    assert "function stage(" in HTML
    # exactly one stage auto-expands, chosen per ticket
    assert "PIN_FOCUS" in HTML and "RULE_FOCUS" in HTML and "function focusOf(" in HTML
    for pinned, stage_name in (("state_informed_retrieval", "retrieval"),
                               ("identity_mismatch", "identity"),
                               ("evidence_gap_failure", "checks")):
        assert f'{pinned}:"{stage_name}"' in HTML


def test_prose_is_sans_and_monospace_is_reserved():
    import re
    css = HTML[HTML.index("<style>"):HTML.index("</style>")]
    # the standalone `body{` rule, not the `html,body{` reset before it
    body = re.search(r"\nbody\{([^}]*)\}", css).group(1)
    assert "var(--sans)" in body and "var(--mono)" not in body
    assert "--sans:" in css and "--mono:" in css
    # mono survives for the things that are genuinely code-like
    for mono_class in (".mono", ".quote", ".eid", ".pre"):
        assert mono_class in css


def test_mobile_first_widens_rather_than_shrinks():
    css = HTML[HTML.index("<style>"):HTML.index("</style>")]
    assert "@media(max-width" not in css, "mobile-first means min-width queries only"
    assert "@media(min-width:900px)" in css


def _py_round(a, b):
    return format(a / b, ".0%")


def _js_round(a, b):
    """The page's pctRound, reimplemented: round half to even."""
    x = 100 * a / b
    fl = int(x // 1)
    d = x - fl
    if d > 0.5:
        return fl + 1
    if d < 0.5:
        return fl
    return fl if fl % 2 == 0 else fl + 1


def test_console_percentages_round_like_python():
    """5/40 rendered as 13% while score_posthoc.py printed 12% - a headline
    number disagreeing with itself. 1/8 had the identical half-value bug."""
    assert "function pctRound(" in HTML
    assert "Math.round(100" not in HTML, "Math.round rounds half up; Python does not"
    for a, b in ((5, 40), (1, 8), (25, 40), (6, 40), (24, 40), (2, 13), (8, 13)):
        assert f"{_js_round(a, b)}%" == _py_round(a, b), (a, b)


def test_evidence_opens_in_a_drawer_not_a_new_page():
    assert 'dialog id="ev"' in HTML
    assert "showModal()" in HTML
    assert "function openEvidence(" in HTML


# --- theming ---------------------------------------------------------------

def _tokens(theme: str) -> dict:
    """Pull one theme's custom properties straight out of the stylesheet, so
    these assertions test the shipped palette rather than a copy of it."""
    import re
    sel = r':root,:root\[data-theme="dark"\]\{' if theme == "dark" \
        else r':root\[data-theme="light"\]\{'
    block = re.search(sel + r"([^}]*)\}", HTML).group(1)
    return dict(re.findall(r"--([\w-]+):\s*([^;]+);", block))


def _lum(h):
    h = h.strip().lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [(x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4) for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a, b):
    l1, l2 = sorted((_lum(a), _lum(b)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def test_both_themes_define_the_same_tokens():
    dark, light = _tokens("dark"), _tokens("light")
    assert set(dark) == set(light), set(dark) ^ set(light)
    assert {"bg", "fg", "fg-quiet", "auto", "draft", "ident", "esc", "fail"} <= set(dark)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_text_and_lane_colours_meet_aa_on_their_own_background(theme):
    t = _tokens(theme)
    bg = t["bg"]
    for token in ("fg", "fg-strong", "fg-quiet", "auto", "draft", "ident", "esc", "fail"):
        ratio = _contrast(t[token], bg)
        assert ratio >= 4.5, f"{theme} --{token} is {ratio:.2f}:1 on --bg, below AA"


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_lane_colours_are_legible_on_a_raised_surface_too(theme):
    """Lane labels appear on cards, not only on the page background."""
    t = _tokens(theme)
    for token in ("auto", "draft", "ident", "esc"):
        assert _contrast(t[token], t["s1"]) >= 4.5, f"{theme} --{token} on --s1"


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_four_lanes_carry_equal_visual_weight(theme):
    """No lane should shout louder than another - they are peers."""
    t = _tokens(theme)
    ratios = [_contrast(t[k], t["bg"]) for k in ("auto", "draft", "ident", "esc")]
    assert max(ratios) / min(ratios) < 1.6, f"{theme} lanes uneven: {ratios}"


def test_light_is_a_document_not_an_inverted_dark_mode():
    light = _tokens("light")
    assert light["bg"].lower() != "#ffffff", "off-white, not pure white"
    assert _lum(light["bg"]) > 0.85
    assert _lum(light["fg"]) < 0.03, "near-black body text"


def test_retrieval_pane_is_hue_separated_not_lightness_separated():
    """The blue tint must sit at the same luminance as the pane beside it.
    Lighter and it glows; darker and it reads as a hole. Dark mode holds
    this relationship at 1.002:1; light has to match."""
    for theme in ("dark", "light"):
        t = _tokens(theme)
        ratio = _contrast(t["built"], t["s2"])
        assert ratio < 1.06, f"{theme} --built is {ratio:.3f}:1 from --s2, too far apart"


def test_theme_persists_and_respects_the_system_preference_first():
    assert "prefers-color-scheme: light" in HTML
    assert 'localStorage.getItem("anchor-theme")' in HTML
    assert 'store.set("anchor-theme"' in HTML
    # applied before first paint, so there is no flash of the wrong theme
    assert HTML.index("anchor-theme") < HTML.index("<style>")


def test_theme_has_a_keyboard_shortcut_that_does_not_fire_while_typing():
    assert 'e.key === "t"' in HTML
    assert "INPUT|TEXTAREA" in HTML


def test_card_edges_are_light_only_so_dark_keeps_its_borderless_layout():
    assert "--card-line:transparent" in HTML
    assert "--card-line:var(--hair)" in HTML


# --- stages, dividers, drawer ---------------------------------------------

def test_any_number_of_stages_can_be_open_and_the_reader_owns_the_state():
    assert "const stageState = {}" in HTML
    assert "function stagesFor(" in HTML and "function onStageToggle(" in HTML
    assert "ontoggle=" in HTML                       # a click updates the map
    assert "setAllStages(true)" in HTML and "setAllStages(false)" in HTML
    assert "new Set([focusOf(r)])" in HTML           # focus only SEEDS it


def test_dividers_are_bounded_persisted_and_resettable():
    for token in ("QW_MIN", "QW_MAX", "DW_MIN", "DW_MAX",
                  "anchor-qw", "anchor-dw", "dblclick", "pointerdown"):
        assert token in HTML, token
    css = HTML[HTML.index("<style>"):HTML.index("</style>")]
    # 1px line, ~9px grab area, and only inside the desktop breakpoint
    desktop = css[css.index("@media(min-width:900px)"):]
    assert "#split" in desktop and "cursor:col-resize" in desktop
    assert "left:-4px;right:-4px" in desktop
    # mobile-first: the base rule only hides it; nothing interactive there
    base = css[:css.index("@media(min-width:900px)")]
    assert "#split{display:none}" in base
    assert "col-resize" not in base


def test_reading_column_follows_the_panel_once_resized():
    assert "function syncRead(" in HTML
    assert 'root.style.setProperty("--read"' in HTML
    assert "!dragged" in HTML, "before any drag, the breakpoint cap wins"


def test_drawer_traps_focus_and_restores_it_on_close():
    assert "showModal()" in HTML                     # native focus trap + Escape
    assert '$("#ev-close").focus()' in HTML
    assert 'EV().addEventListener("close"' in HTML
    assert "lastFocus.focus()" in HTML


def test_layout_steps_up_on_large_monitors():
    css = HTML[HTML.index("<style>"):HTML.index("</style>")]
    for bp in ("@media(min-width:1400px)", "@media(min-width:1920px)", "@media(min-width:2400px)"):
        assert bp in css, bp
    assert "max-width:2200px" in css                 # the app centres, not stretches
