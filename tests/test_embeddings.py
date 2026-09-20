import tempfile
from pathlib import Path

from src.llm import LLMClient


def test_repeated_embed_hits_cache_without_network():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")

        call_count = {"n": 0}

        def fail_if_called(*args, **kwargs):
            call_count["n"] += 1
            raise AssertionError("network call should not happen on a cache hit")

        client._embed_with_backoff = lambda text, model, params: [0.1, 0.2, 0.3]
        first = client.embed("settlement cycles are T+3 by default")
        assert first == [0.1, 0.2, 0.3]

        client._embed_with_backoff = fail_if_called
        second = client.embed("settlement cycles are T+3 by default")

        assert second == [0.1, 0.2, 0.3]
        assert call_count["n"] == 0


def test_embed_and_generate_caches_never_collide():
    """A generate() call and an embed() call on the identical text string
    must not accidentally share a cache entry, even though they share one
    cache directory - the 'kind' discriminator in the cache key is what
    prevents this.
    """
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        client._call_with_backoff = lambda prompt, model, params: "a generated reply"
        client._embed_with_backoff = lambda text, model, params: [0.9, 0.9]

        text = "identical string used for both call types"
        gen_result = client.generate(text, model="gemini-3.5-flash-lite")
        embed_result = client.embed(text)

        assert gen_result == "a generated reply"
        assert embed_result == [0.9, 0.9]
        assert len(list(Path(tmp).glob("*.json"))) == 2


def test_query_and_document_task_type_produce_different_cache_entries():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        client._embed_with_backoff = lambda text, model, params: [
            1.0 if params["task_type"] == "RETRIEVAL_DOCUMENT" else 2.0
        ]

        doc_vec = client.embed("same text", task_type="RETRIEVAL_DOCUMENT")
        query_vec = client.embed("same text", task_type="RETRIEVAL_QUERY")

        assert doc_vec == [1.0]
        assert query_vec == [2.0]
        assert len(list(Path(tmp).glob("*.json"))) == 2
