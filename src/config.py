"""Shared corpus-wide constants."""

import os
from datetime import datetime

# The single "now" against which every elapsed-time calculation in the
# corpus, evaluation harness, and agent is computed. Nothing else should
# hardcode a current-date assumption.
CORPUS_NOW = datetime.fromisoformat("2026-09-17T12:00:00+05:30")

# Pinned so retrieval results are reproducible: re-running the indexer next
# month must not silently start comparing vectors from two different model
# versions. Verified live against the API on 2026-09-20 (see src/llm.py's
# ALLOWED_MODELS note - the Gemini lineup moves fast; if this 404s, re-run
# `client.models.list()` and update it deliberately, not by trial and error.
EMBEDDING_MODEL = "gemini-embedding-001"

# gemini-embedding-001 supports Matryoshka truncation down from 3072. 768 is
# Google's documented recommended default for most retrieval use cases and
# is plenty for a ~70-chunk corpus - cosine search over a 70x768 matrix is
# microseconds either way, so this is picked for a stable, documented
# choice rather than because it matters at this scale.
EMBEDDING_DIMENSIONALITY = 768


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "")


# --------------------------------------------------------------------------
# Gate thresholds (src/gate.py). Every tunable number the routing gate uses
# lives here, so "what would change the routing behaviour" is one file to
# read rather than a hunt through conditionals.
#
# These are set from TRAINING tickets only (corpus/tickets/) and frozen
# before any held-out run. The frozen values and the timestamp are recorded
# in eval/README.md.
# --------------------------------------------------------------------------

# FROZEN 2026-09-23. Set from a 21-ticket stratified sample of TRAINING
# tickets only (eval/training_sample_results.json); no held-out ticket had
# been run at the time these were fixed. Provenance in eval/README.md.
#
# Fraction of verified claims that must come back SUPPORTED (as opposed to
# PARTIALLY_SUPPORTED) before a diagnosis may AUTO_RESOLVE.
#
# Set at the conservative end, and honestly: the training data did not
# discriminate. 20 of 21 sampled tickets scored exactly 1.0, leaving a
# single sub-threshold case (ticket_002, 11/12 = 0.92) - no basis to fit a
# looser bar to. The argument for 1.0 is therefore from the cost
# asymmetry, not from a measured optimum: PARTIALLY_SUPPORTED means, by
# the verifier's own rubric, that a claim overstates or goes beyond its
# evidence, and letting that into an unreviewed answer is precisely the
# failure this project exists to prevent. Blocking costs a human review;
# a confidently wrong answer costs a merchant.
#
# Only consulted when verification actually ran. When the verifier is off
# (the ablation), the gate has no verdicts and this rule is skipped rather
# than treated as failed - see src/gate.py for why that distinction
# decides whether the ablation measures anything real.
AUTO_RESOLVE_MIN_SUPPORTED_RATIO = {
    "money_movement": 1.0,
    "informational": 1.0,
}

# A diagnosis with fewer than this many surviving claims may not
# AUTO_RESOLVE, regardless of verdicts: an answer resting on almost nothing
# is not a confident answer, it is a thin one.
#
# This is where "money_movement gets a stricter bar" is actually
# implemented - with both ratios pinned at 1.0 there is nowhere else for
# the asymmetry to live. Both money_movement tickets that auto-resolved in
# the training sample carried >= 2 claims (ticket_014: 4, ticket_051: 2),
# so requiring 2 costs nothing observed while ruling out a settlement or
# refund answer resting on a single fact. The one-claim auto-resolve in
# the sample (ticket_059, "is the dispute window business or calendar
# days") was informational, where a single fact genuinely is the whole
# answer.
AUTO_RESOLVE_MIN_CLAIMS = {
    "money_movement": 2,
    "informational": 1,
}


# The ablation switch: whether src.agent.diagnose_ticket runs the grounding
# verifier (src/verifier.py) by default. This exists so measuring grounding
# precision with and without the verifier - the headline number of the
# project - is a config change, not a code change.
#
# diagnose_ticket's own `verify` parameter always takes precedence when
# passed explicitly (the primary mechanism: an eval script runs the same
# batch twice, verify=True then verify=False, in one process). This
# environment variable only sets what `verify=None` resolves to - for
# flipping the default without writing a loop, e.g.
# `ANCHOR_VERIFIER_ENABLED=0 python eval/run_something.py`.
VERIFIER_ENABLED_DEFAULT = _env_bool("ANCHOR_VERIFIER_ENABLED", True)


# The second ablation switch, same contract as VERIFIER_ENABLED_DEFAULT
# above: whether src.agent.diagnose_ticket runs the responsiveness check
# (src/responsiveness.py) by default.
#
# Two independent switches rather than one "checks on/off" flag, because
# the two checks answer different questions and can fail independently.
# The held-out run showed the verifier moving the false auto-resolve rate
# by exactly zero; that finding is only legible because the verifier could
# be switched off on its own. The same has to be possible here, and the
# 2x2 (verifier on/off x responsiveness on/off) has to be reachable.
#
# diagnose_ticket's `check_responsiveness` parameter takes precedence when
# passed explicitly; this only sets what None resolves to, e.g.
# `ANCHOR_RESPONSIVENESS_ENABLED=0 python eval/run_something.py`.
RESPONSIVENESS_ENABLED_DEFAULT = _env_bool("ANCHOR_RESPONSIVENESS_ENABLED", True)
