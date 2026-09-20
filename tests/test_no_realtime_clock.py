"""Every elapsed-time computation in this project must derive from
src.config.CORPUS_NOW, never the real wall clock - otherwise every
"is this dispute still within its normal window" answer silently rots
the day this code runs after 2026-09-17. Grep-based, but that's the
point: it should catch a `datetime.now()` slipped into any new file
under src/, not just ones some other test happens to exercise.
"""

import re
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

# datetime.now(), date.today(), and time.time() are the three common ways
# a "current real time" sneaks in. time.time() is fine for measuring a
# duration (e.g. LLMClient's backoff), so it's checked but allowed with an
# explicit nearby comment rather than banned outright - see the exemption
# list below.
_FORBIDDEN_PATTERNS = [
    re.compile(r"datetime\.now\("),
    re.compile(r"date\.today\("),
]

# Files allowed to use time.time()/similar for measuring elapsed wall-clock
# duration (e.g. retry backoff), which is not the same thing as treating
# "now" as a business-logic date.
_TIME_TIME_ALLOWED = {"llm.py"}


def test_no_forbidden_realtime_calls_in_src():
    violations = []
    for path in SRC_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.relative_to(SRC_DIR.parent)}: {pattern.pattern}")
    assert not violations, f"real-time clock call(s) found in src/: {violations}"


def test_time_time_only_used_where_explicitly_allowed():
    violations = []
    for path in SRC_DIR.rglob("*.py"):
        if path.name in _TIME_TIME_ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\btime\.time\(", text):
            violations.append(str(path.relative_to(SRC_DIR.parent)))
    assert not violations, f"time.time() used outside the allowed files: {violations}"


def test_corpus_now_is_importable_and_is_the_single_source_of_truth():
    from src.config import CORPUS_NOW
    from datetime import datetime

    assert isinstance(CORPUS_NOW, datetime)
    assert CORPUS_NOW.isoformat() == "2026-09-17T12:00:00+05:30"
