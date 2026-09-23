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
        first = client.generate("hello world", model="gemini-3.5-flash-lite")
        assert first == "cached response"

        # Second call, identical params: must be served from disk. Swap in a
        # version that would fail the test if the network path is hit.
        client._call_with_backoff = fail_if_called
        second = client.generate("hello world", model="gemini-3.5-flash-lite")

        assert second == "cached response"
        assert call_count["n"] == 0


def test_different_params_produce_different_cache_entries():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        client._call_with_backoff = lambda prompt, model, params: f"temp={params['temperature']}"

        out_a = client.generate("hi", model="gemini-3.5-flash-lite", temperature=0.0)
        out_b = client.generate("hi", model="gemini-3.5-flash-lite", temperature=0.7)

        assert out_a == "temp=0.0"
        assert out_b == "temp=0.7"
        assert len(list(Path(tmp).glob("*.json"))) == 2


def test_rejects_non_free_tier_model():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        with pytest.raises(ValueError):
            client.generate("hi", model="gemini-3.5-pro")


# --- transient API errors: retry 429 AND 503 -----------------------------

def test_retries_503_unavailable_not_just_429():
    """A 503 "model is experiencing high demand" is explicitly temporary.
    It was not being retried, and one such error killed a 19-ticket batch
    partway through - the whole run lost to a transient overload.
    """
    from src.llm import TransientAPIError, _is_retryable_error

    class Err503(Exception):
        status_code = 503

    assert _is_retryable_error(Err503()) is True

    client = LLMClient(cache_dir=tempfile.mkdtemp(), api_key="fake-key")
    client.base_delay = 0  # no real sleeping in tests
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Err503("503 UNAVAILABLE. high demand")
        return "recovered"

    assert client._retry_on_rate_limit(flaky) == "recovered"
    assert attempts["n"] == 3


def test_429_still_retried():
    from src.llm import _is_retryable_error

    class Err429(Exception):
        status_code = 429

    assert _is_retryable_error(Err429()) is True
    assert _is_retryable_error(Exception("429 RESOURCE_EXHAUSTED")) is True


def test_non_transient_errors_are_not_retried():
    """A 400 will never succeed on retry; retrying would waste quota and
    hide the bug."""
    from src.llm import _is_retryable_error

    class Err400(Exception):
        status_code = 400

    assert _is_retryable_error(Err400()) is False

    client = LLMClient(cache_dir=tempfile.mkdtemp(), api_key="fake-key")
    attempts = {"n": 0}

    def bad_request():
        attempts["n"] += 1
        raise Err400("400 INVALID_ARGUMENT")

    with pytest.raises(Err400):
        client._retry_on_rate_limit(bad_request)
    assert attempts["n"] == 1  # tried once, not retried


def test_exhausted_transient_retries_raise_transient_api_error():
    from src.llm import TransientAPIError

    class Err503(Exception):
        status_code = 503

    client = LLMClient(cache_dir=tempfile.mkdtemp(), api_key="fake-key", max_retries=2)
    client.base_delay = 0

    with pytest.raises(TransientAPIError):
        client._retry_on_rate_limit(lambda: (_ for _ in ()).throw(Err503("503")))
