import json
import tempfile
from pathlib import Path

import pytest

from src.llm import LLMClient


def test_repeated_call_hits_cache_without_network(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")

        call_count = {"n": 0}

        def fail_if_called(*args, **kwargs):
            call_count["n"] += 1
            raise AssertionError("network call should not happen on a cache hit")

        # First call: prime the cache directly (bypassing the network) by
        # writing what a real call would have produced.
        client._call_with_backoff = lambda prompt, model, params: "cached response"
        first = client.generate("hello world", model="gemini-2.5-flash")
        assert first == "cached response"

        # Second call, identical params: must be served from disk. Swap in a
        # version that would fail the test if the network path is hit.
        client._call_with_backoff = fail_if_called
        second = client.generate("hello world", model="gemini-2.5-flash")

        assert second == "cached response"
        assert call_count["n"] == 0


def test_different_params_produce_different_cache_entries():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        client._call_with_backoff = lambda prompt, model, params: f"temp={params['temperature']}"

        out_a = client.generate("hi", model="gemini-2.5-flash", temperature=0.0)
        out_b = client.generate("hi", model="gemini-2.5-flash", temperature=0.7)

        assert out_a == "temp=0.0"
        assert out_b == "temp=0.7"
        assert len(list(Path(tmp).glob("*.json"))) == 2


def test_rejects_non_free_tier_model():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        with pytest.raises(ValueError):
            client.generate("hi", model="gemini-2.5-pro")
