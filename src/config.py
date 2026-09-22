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
