"""Policy retrieval: brute-force cosine search over the chunked policy KB.

Design choice, worth defending explicitly: there is no separate persisted
index file and no vector database. `load_policy_chunks()` re-parses the ~10
markdown files (milliseconds) on every startup, and each chunk's embedding
is fetched through `LLMClient.embed()`, which is disk-cached and
content-addressed. On a warm cache (i.e. every run after the first), that
means "loading the index from disk" is exactly: re-reading small markdown
files plus ~70 small cached JSON files - no network call, no server, no
ANN library. A ~70-chunk corpus makes a brute-force NxD cosine comparison
microseconds regardless of dimensionality, so FAISS/pgvector/etc. would be
solving a problem this corpus doesn't have. If the KB grows by an order of
magnitude or more, this is the first thing to revisit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.chunker import load_policy_chunks
from src.evidence import make_evidence
from src.llm import LLMClient


@dataclass
class PolicyIndex:
    chunks: list[dict]  # [{evidence_id, content}, ...]
    vectors: np.ndarray  # shape (n_chunks, dim), L2-normalized

    def search(self, query: str, llm: LLMClient, k: int = 5) -> list[dict]:
        """Return up to k evidence envelopes, most similar first."""
        query_vec = np.array(llm.embed(query, task_type="RETRIEVAL_QUERY"), dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) or 1.0)

        # vectors are already L2-normalized, so the dot product is cosine
        # similarity directly.
        sims = self.vectors @ query_vec
        top_k = np.argsort(-sims)[:k]

        return [
            make_evidence(
                self.chunks[i]["evidence_id"],
                self.chunks[i]["content"],
                "policy",
            )
            for i in top_k
        ]


def build_index(llm: LLMClient | None = None) -> PolicyIndex:
    """Load every policy chunk and embed it (cache-backed, so this is fast
    and free after the first run).
    """
    llm = llm or LLMClient()
    chunks = load_policy_chunks()

    raw_vectors = np.array(
        [llm.embed(c["content"], task_type="RETRIEVAL_DOCUMENT") for c in chunks],
        dtype=np.float32,
    )
    norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = raw_vectors / norms

    return PolicyIndex(chunks=chunks, vectors=vectors)
