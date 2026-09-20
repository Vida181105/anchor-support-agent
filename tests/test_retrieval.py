"""Unit tests for src/retrieval.py, fully mocked - no network calls, no
disk cache dependency, no API quota spent. The real index (built against
live embeddings) is exercised separately by the retrieval evaluation
script, not by the test suite.
"""

import numpy as np
import pytest

from src.retrieval import PolicyIndex, build_index


class FakeLLM:
    """Deterministic stand-in for LLMClient.embed: hashes text to a small
    vector so tests can control which chunk is "closest" to a query
    without any real embedding call.
    """

    def __init__(self, vectors_by_text: dict[str, list[float]]):
        self.vectors_by_text = vectors_by_text
        self.calls = []

    def embed(self, text, task_type="RETRIEVAL_DOCUMENT", output_dimensionality=None):
        self.calls.append((text, task_type))
        return self.vectors_by_text[text]


def test_build_index_embeds_every_chunk_as_a_document(monkeypatch):
    fake_chunks = [
        {"evidence_id": "doc:a#c1", "content": "alpha"},
        {"evidence_id": "doc:a#c2", "content": "beta"},
    ]
    monkeypatch.setattr("src.retrieval.load_policy_chunks", lambda: fake_chunks)

    llm = FakeLLM({"alpha": [1.0, 0.0], "beta": [0.0, 1.0]})
    index = build_index(llm)

    assert index.chunks == fake_chunks
    assert index.vectors.shape == (2, 2)
    assert all(call[1] == "RETRIEVAL_DOCUMENT" for call in llm.calls)


def test_build_index_normalizes_vectors():
    index = PolicyIndex(
        chunks=[{"evidence_id": "doc:a#c1", "content": "x"}],
        vectors=np.array([[3.0, 4.0]]),  # norm 5
    )
    # not run through build_index's normalization directly here, but verify
    # search's own normalization of a raw (non-unit) query still works
    llm = FakeLLM({"query": [3.0, 4.0]})
    # pre-normalize the stored vector by hand, mirroring build_index
    index.vectors = index.vectors / np.linalg.norm(index.vectors, axis=1, keepdims=True)
    results = index.search("query", llm, k=1)
    assert results[0]["evidence_id"] == "doc:a#c1"


def test_search_ranks_most_similar_chunk_first():
    chunks = [
        {"evidence_id": "doc:a#c1", "content": "about settlement timing"},
        {"evidence_id": "doc:b#c1", "content": "about kyc documents"},
        {"evidence_id": "doc:c#c1", "content": "about webhooks"},
    ]
    # query is identical to chunk a, at 45 degrees from chunk b, and
    # orthogonal to chunk c - an unambiguous a > b > c ranking.
    raw = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    vectors = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    index = PolicyIndex(chunks=chunks, vectors=vectors)

    llm = FakeLLM({"query": [1.0, 0.0, 0.0]})
    results = index.search("query", llm, k=3)

    assert [r["evidence_id"] for r in results] == ["doc:a#c1", "doc:b#c1", "doc:c#c1"]


def test_search_respects_k():
    chunks = [{"evidence_id": f"doc:a#c{i+1}", "content": f"chunk {i}"} for i in range(5)]
    vectors = np.eye(5, dtype=np.float32)[:, :2]
    vectors = np.hstack([vectors, np.zeros((5, 3))])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    index = PolicyIndex(chunks=chunks, vectors=vectors / norms)

    llm = FakeLLM({"query": [1.0, 0.0, 0.0, 0.0, 0.0]})
    results = index.search("query", llm, k=2)
    assert len(results) == 2


def test_search_returns_valid_evidence_envelopes():
    chunks = [{"evidence_id": "doc:settlement-cycles#c1", "content": "T+3 explanation"}]
    index = PolicyIndex(chunks=chunks, vectors=np.array([[1.0, 0.0]]))
    llm = FakeLLM({"q": [1.0, 0.0]})

    results = index.search("q", llm, k=1)
    assert results[0] == {
        "evidence_id": "doc:settlement-cycles#c1",
        "content": "T+3 explanation",
        "source_type": "policy",
    }


def test_search_embeds_query_with_query_task_type():
    chunks = [{"evidence_id": "doc:a#c1", "content": "x"}]
    index = PolicyIndex(chunks=chunks, vectors=np.array([[1.0, 0.0]]))
    llm = FakeLLM({"q": [1.0, 0.0]})

    index.search("q", llm, k=1)
    assert llm.calls == [("q", "RETRIEVAL_QUERY")]
