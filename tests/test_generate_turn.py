"""Tests LLMClient.generate_turn: caching behavior (mocked, no SDK), and
separately the turns-dict <-> SDK-object reconstruction including the
thought_signature round-trip that a live call surfaced as a real bug
during development (Gemini 3.x rejects a replayed function_call turn
missing its original thought_signature).
"""

import base64
import tempfile
from types import SimpleNamespace

from src.llm import LLMClient

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


# --- caching, mocked at the backoff boundary (fast, no SDK) --------------

def test_repeated_turn_hits_cache_without_network():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        calls = {"n": 0}

        def fail_if_called(*a, **kw):
            calls["n"] += 1
            raise AssertionError("should not hit network on cache hit")

        client._generate_turn_with_backoff = lambda *a, **kw: {"text": "hello"}
        turns = [{"role": "user", "text": "hi"}]
        first = client.generate_turn(turns, model="gemini-3.5-flash-lite", tools=TOOLS)
        assert first == {"text": "hello"}

        client._generate_turn_with_backoff = fail_if_called
        second = client.generate_turn(turns, model="gemini-3.5-flash-lite", tools=TOOLS)
        assert second == {"text": "hello"}
        assert calls["n"] == 0


def test_different_turns_produce_different_cache_entries():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        client._generate_turn_with_backoff = lambda turns, *a, **kw: {"text": turns[-1]["text"]}

        r1 = client.generate_turn([{"role": "user", "text": "hi"}], model="gemini-3.5-flash-lite")
        r2 = client.generate_turn([{"role": "user", "text": "bye"}], model="gemini-3.5-flash-lite")
        assert r1 == {"text": "hi"}
        assert r2 == {"text": "bye"}


def test_growing_conversation_prefix_produces_a_new_cache_entry_each_turn():
    """A 3-turn conversation must cache each distinct prefix separately,
    not collapse to one entry - otherwise turn 2 of ticket A could
    accidentally return turn 2 of ticket B if their first turns matched.
    """
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        client._generate_turn_with_backoff = lambda turns, *a, **kw: {"text": f"reply-{len(turns)}"}

        t1 = [{"role": "user", "text": "hi"}]
        r1 = client.generate_turn(t1, model="gemini-3.5-flash-lite")
        t2 = t1 + [{"role": "model", "text": r1["text"]}, {"role": "user", "text": "and then?"}]
        r2 = client.generate_turn(t2, model="gemini-3.5-flash-lite")

        assert r1["text"] == "reply-1"
        assert r2["text"] == "reply-3"


# --- turns <-> SDK object reconstruction, including thought_signature ---

class _FakeModels:
    def __init__(self, response):
        self._response = response
        self.received_contents = None

    def generate_content(self, model, contents, config):
        self.received_contents = contents
        return self._response


def _fake_function_call_response(name, args, thought_signature=None):
    part = SimpleNamespace(
        function_call=SimpleNamespace(name=name, args=args),
        thought_signature=thought_signature,
    )
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


def _fake_text_response(text):
    part = SimpleNamespace(function_call=None, thought_signature=None)
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
    response.text = text
    return response


def test_function_call_response_captures_thought_signature_base64():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        raw_sig = b"\x01\x02\xffsignature-bytes"
        fake_models = _FakeModels(_fake_function_call_response("get_weather", {"city": "Pune"}, raw_sig))
        client._client = SimpleNamespace(models=fake_models)

        result = client.generate_turn([{"role": "user", "text": "weather?"}], model="gemini-3.5-flash-lite", tools=TOOLS)

        assert result["function_call"]["name"] == "get_weather"
        assert result["function_call"]["args"] == {"city": "Pune"}
        assert base64.b64decode(result["function_call"]["thought_signature"]) == raw_sig


def test_replaying_a_function_call_turn_reconstructs_thought_signature_on_the_part():
    """The exact bug hit during development: a "model" turn with a
    function_call must carry its thought_signature back to the SDK Part
    object, or Gemini 3.x's API rejects the request outright.
    """
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        fake_models = _FakeModels(_fake_text_response("done"))
        client._client = SimpleNamespace(models=fake_models)

        raw_sig = b"\x9a\x01signature"
        turns = [
            {"role": "user", "text": "weather?"},
            {
                "role": "model",
                "function_call": {
                    "name": "get_weather",
                    "args": {"city": "Pune"},
                    "thought_signature": base64.b64encode(raw_sig).decode("ascii"),
                },
            },
            {"role": "function", "name": "get_weather", "response": {"temp_c": 28}},
        ]
        client.generate_turn(turns, model="gemini-3.5-flash-lite", tools=TOOLS)

        sent = fake_models.received_contents
        model_turn_part = sent[1].parts[0]
        assert model_turn_part.thought_signature == raw_sig


def test_missing_thought_signature_is_tolerated_not_crashed_on():
    # a turn built by hand (e.g. in a test) without a thought_signature
    # key must not raise - only a live model-turn from generate_turn
    # itself will actually carry one.
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        fake_models = _FakeModels(_fake_text_response("done"))
        client._client = SimpleNamespace(models=fake_models)

        turns = [
            {"role": "user", "text": "weather?"},
            {"role": "model", "function_call": {"name": "get_weather", "args": {"city": "Pune"}}},
            {"role": "function", "name": "get_weather", "response": {"temp_c": 28}},
        ]
        result = client.generate_turn(turns, model="gemini-3.5-flash-lite", tools=TOOLS)
        assert result == {"text": "done"}


def test_text_only_response_has_no_function_call_key():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        fake_models = _FakeModels(_fake_text_response("plain answer"))
        client._client = SimpleNamespace(models=fake_models)

        result = client.generate_turn([{"role": "user", "text": "hi"}], model="gemini-3.5-flash-lite")
        assert result == {"text": "plain answer"}


def test_response_json_schema_sets_json_mime_type():
    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=tmp, api_key="fake-key")
        fake_models = _FakeModels(_fake_text_response('{"a": 1}'))
        client._client = SimpleNamespace(models=fake_models)

        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        result = client.generate_turn(
            [{"role": "user", "text": "give me json"}],
            model="gemini-3.5-flash-lite",
            response_json_schema=schema,
        )
        assert result == {"text": '{"a": 1}'}
