"""Shared corpus-wide constants."""

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
