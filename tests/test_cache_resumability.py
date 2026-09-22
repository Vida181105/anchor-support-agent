"""Proves the specific property a per-claim verifier batch depends on: a
mid-batch quota exhaustion must not cost the whole batch. Everything
already cached before the failure must be served from disk on the next
run, with the network only touched for what's still missing.

This matters because the exact free-tier daily quota for whichever model
the verifier uses is not something Google publishes as a static number
anymore (docs page checked directly: "Rate limits depend on a variety of
factors... can be viewed in Google AI Studio" - no table). What's
concretely known from this project's own usage this session:
gemini-3.5-flash's free tier returned a *daily* cap of exactly 20 requests
(observed directly in a live 429's error body: quotaId
"GenerateRequestsPerDayPerProjectPerModel-FreeTier", quotaValue "20") -
not just RPM, which is easy to assume and wrong. gemini-3.5-flash-lite has
not been exhausted (100+ successful calls this session with no 429), so
its ceiling is unconfirmed but evidently higher - deliberately exhausting
it to find the exact number would burn quota needed for the rest of this
work, so it's reported as "at least 100+, unconfirmed exact value" rather
than guessed. Given that uncertainty, resumability - not a specific
number - is the property actually worth guaranteeing in code.
"""

import tempfile

import pytest

from src.llm import LLMClient, RateLimitExceeded


def test_partial_batch_resumes_from_cache_after_simulated_quota_exhaustion():
    claims = ["claim 1", "claim 2", "claim 3", "claim 4"]
    responses = {c: {"text": f"verdict for {c!r}"} for c in claims}

    with tempfile.TemporaryDirectory() as tmp:
        # --- first pass: claims 1-2 succeed, claim 3 hits a simulated
        # quota exhaustion, claim 4 is never even attempted (as a real
        # batch loop would stop on the first failure) ---
        first_pass_calls = []
        client = LLMClient(cache_dir=tmp, api_key="fake-key")

        def backoff_then_exhausted(turns, model, tools, schema, temperature):
            text = turns[0]["text"]
            first_pass_calls.append(text)
            if text == "claim 3":
                raise RateLimitExceeded("simulated: daily quota exhausted")
            return responses[text]

        client._generate_turn_with_backoff = backoff_then_exhausted

        results = {}
        for claim in claims:
            try:
                results[claim] = client.generate_turn([{"role": "user", "text": claim}], model="gemini-3.5-flash-lite")
            except RateLimitExceeded:
                break  # a real batch loop stops here; claim 4 never attempted

        assert first_pass_calls == ["claim 1", "claim 2", "claim 3"]
        assert set(results) == {"claim 1", "claim 2"}

        # --- second pass: fresh LLMClient (simulating a new process
        # tomorrow, once quota resets), same cache_dir. Only claims 3 and
        # 4 should ever touch the network. ---
        second_pass_calls = []
        client2 = LLMClient(cache_dir=tmp, api_key="fake-key")

        def backoff_now_succeeds(turns, model, tools, schema, temperature):
            text = turns[0]["text"]
            second_pass_calls.append(text)
            return responses[text]

        client2._generate_turn_with_backoff = backoff_now_succeeds

        for claim in claims:
            results[claim] = client2.generate_turn([{"role": "user", "text": claim}], model="gemini-3.5-flash-lite")

        assert second_pass_calls == ["claim 3", "claim 4"], (
            "claims 1 and 2 must be served from cache, not re-fetched, on resume"
        )
        assert results == {c: responses[c] for c in claims}


def test_the_call_that_fails_is_never_cached_as_a_success():
    """The failing call itself (claim 3 above) must leave no cache entry -
    otherwise a resume would replay whatever partial/garbage state existed
    at the moment of failure instead of genuinely retrying it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        client._generate_turn_with_backoff = lambda *a, **kw: (_ for _ in ()).throw(
            RateLimitExceeded("simulated")
        )

        with pytest.raises(RateLimitExceeded):
            client.generate_turn([{"role": "user", "text": "will fail"}], model="gemini-3.5-flash-lite")

        import os

        assert os.listdir(tmp) == [], "a failed call must not leave a cache file behind"
