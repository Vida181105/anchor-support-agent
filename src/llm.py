"""Disk-cached Gemini client.

Every call (text generation or embedding) is keyed by a hash of
(kind, model, text, params). A cache hit returns straight from disk with no
network request; a miss calls Gemini, retrying on 429s with exponential
backoff and jitter, and writes the result to disk before returning it. This
is what makes the evaluation harness - and, per Unit 2, the policy index -
re-runnable on a free API key without burning the daily quota.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from src.config import EMBEDDING_DIMENSIONALITY, EMBEDDING_MODEL

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Free-tier Flash / Flash-Lite models only. Never point this client at a Pro
# model. Verified live against the API on 2026-09-17; the Gemini model
# lineup moves fast, so if a model here starts 404ing, re-run
# `client.models.list()` and update this set rather than guessing new names.
ALLOWED_MODELS = {
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
}


class RateLimitExceeded(RuntimeError):
    """Raised when Gemini keeps returning 429 past the retry budget."""


def _cache_key(kind: str, model: str, text: str, params: dict[str, Any]) -> str:
    # `kind` ("generate" vs "embed") keeps the two call types from ever
    # colliding in the cache even though they share one directory - belt
    # and suspenders, since `model` alone already differs between a
    # generation model and an embedding model in practice.
    payload = json.dumps(
        {"kind": kind, "model": model, "text": text, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_rate_limit_error(exc: Exception) -> bool:
    # google-genai wraps HTTP errors; a 429 shows up either as a status code
    # attribute or in the string form of the exception.
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)


class LLMClient:
    """Thin, disk-cached wrapper around the Gemini API.

    A configurable model per call site lets, e.g., the composer and the
    verifier use different models without needing separate client instances.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        api_key: str | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._client = None  # lazy: built on first real API call

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set; cannot make a live call. "
                    "Set it in .env or pass api_key= explicitly."
                )
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        **extra_params: Any,
    ) -> str:
        """Return generated text for `prompt`, using the disk cache when possible."""
        if model not in ALLOWED_MODELS:
            raise ValueError(
                f"Model '{model}' is not an allowed free-tier model. "
                f"Allowed: {sorted(ALLOWED_MODELS)}"
            )

        params: dict[str, Any] = {"temperature": temperature}
        if max_output_tokens is not None:
            params["max_output_tokens"] = max_output_tokens
        params.update(extra_params)

        key = _cache_key("generate", model, prompt, params)
        cache_path = self._cache_path(key)

        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["text"]

        text = self._call_with_backoff(prompt, model, params)

        cache_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "prompt": prompt,
                    "params": params,
                    "text": text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return text

    def _retry_on_rate_limit(self, call):
        """Run `call()`, retrying with exponential backoff + jitter on 429s."""
        attempt = 0
        while True:
            try:
                return call()
            except Exception as exc:  # noqa: BLE001 - broad on purpose, retry logic below
                if not _is_rate_limit_error(exc) or attempt >= self.max_retries:
                    if _is_rate_limit_error(exc):
                        raise RateLimitExceeded(
                            f"Exceeded {self.max_retries} retries on 429s"
                        ) from exc
                    raise
                delay = self.base_delay * (2**attempt) + random.uniform(0, 1)
                time.sleep(delay)
                attempt += 1

    def _call_with_backoff(self, prompt: str, model: str, params: dict[str, Any]) -> str:
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            temperature=params.get("temperature", 0.0),
            max_output_tokens=params.get("max_output_tokens"),
        )

        def call():
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text

        return self._retry_on_rate_limit(call)

    def embed(
        self,
        text: str,
        model: str = EMBEDDING_MODEL,
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int = EMBEDDING_DIMENSIONALITY,
    ) -> list[float]:
        """Return an embedding vector for `text`, using the disk cache when possible.

        `task_type` should be "RETRIEVAL_DOCUMENT" when embedding something
        that will be searched over (e.g. a policy chunk) and
        "RETRIEVAL_QUERY" when embedding the search query itself - Gemini's
        embedding model is asymmetric and expects this distinction for
        good retrieval quality.
        """
        params = {"task_type": task_type, "output_dimensionality": output_dimensionality}
        key = _cache_key("embed", model, text, params)
        cache_path = self._cache_path(key)

        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["embedding"]

        vector = self._embed_with_backoff(text, model, params)

        cache_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "text": text,
                    "params": params,
                    "embedding": vector,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return vector

    def _embed_with_backoff(self, text: str, model: str, params: dict[str, Any]) -> list[float]:
        from google.genai import types

        client = self._get_client()
        config = types.EmbedContentConfig(
            task_type=params["task_type"],
            output_dimensionality=params["output_dimensionality"],
        )

        def call():
            response = client.models.embed_content(model=model, contents=text, config=config)
            return response.embeddings[0].values

        return self._retry_on_rate_limit(call)

    def generate_turn(
        self,
        turns: list[dict],
        model: str = DEFAULT_MODEL,
        tools: list[dict] | None = None,
        response_json_schema: dict | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Advance a multi-turn (optionally tool-calling) conversation by
        one model turn.

        `turns` is a plain, JSON-serializable list - never a google.genai
        SDK object - so the caller (src/agent.py) stays decoupled from the
        SDK and so this call can be cached exactly like generate()/embed().
        Each entry is one of:
          {"role": "user", "text": "..."}
          {"role": "model", "function_call": {"name": "...", "args": {...}}}
          {"role": "model", "text": "..."}
          {"role": "function", "name": "...", "response": {...}}

        Returns {"function_call": {"name", "args"}} or {"text": "..."}.

        `tools` and `response_json_schema` are mutually exclusive in
        practice (the underlying API does not reliably support forced
        JSON output in the same call as function declarations) - this
        method doesn't forbid combining them, but src/agent.py never does.
        """
        if model not in ALLOWED_MODELS:
            raise ValueError(
                f"Model '{model}' is not an allowed free-tier model. "
                f"Allowed: {sorted(ALLOWED_MODELS)}"
            )

        params: dict[str, Any] = {
            "temperature": temperature,
            "tools": tools,
            "response_json_schema": response_json_schema,
        }
        cache_input = json.dumps(turns, sort_keys=True, ensure_ascii=False)
        key = _cache_key("agent_turn", model, cache_input, params)
        cache_path = self._cache_path(key)

        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached["result"]

        result = self._generate_turn_with_backoff(turns, model, tools, response_json_schema, temperature)

        cache_path.write_text(
            json.dumps(
                {"turns": turns, "model": model, "params": params, "result": result},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    def _generate_turn_with_backoff(
        self,
        turns: list[dict],
        model: str,
        tools: list[dict] | None,
        response_json_schema: dict | None,
        temperature: float,
    ) -> dict:
        from google.genai import types

        client = self._get_client()
        contents = []
        for turn in turns:
            role = turn["role"]
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=turn["text"])]))
            elif role == "model" and "function_call" in turn:
                fc = turn["function_call"]
                part = types.Part(function_call=types.FunctionCall(name=fc["name"], args=fc["args"]))
                # Gemini 3.x rejects a replayed function_call turn that's
                # missing its original thought_signature ("required for
                # tools to work correctly") - discovered by hitting this
                # exact 400 error during development, not anticipated up
                # front. Round-tripped as base64 since it's opaque bytes.
                if fc.get("thought_signature"):
                    part.thought_signature = base64.b64decode(fc["thought_signature"])
                contents.append(types.Content(role="model", parts=[part]))
            elif role == "model":
                contents.append(types.Content(role="model", parts=[types.Part(text=turn["text"])]))
            elif role == "function":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=turn["name"], response=turn["response"]
                                )
                            )
                        ],
                    )
                )
            else:
                raise ValueError(f"unknown turn role: {role!r}")

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if tools:
            declarations = [
                types.FunctionDeclaration(
                    name=t["name"], description=t["description"], parameters=t["parameters"]
                )
                for t in tools
            ]
            config_kwargs["tools"] = [types.Tool(function_declarations=declarations)]
        if response_json_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = response_json_schema

        config = types.GenerateContentConfig(**config_kwargs)

        def call():
            response = client.models.generate_content(model=model, contents=contents, config=config)
            part = response.candidates[0].content.parts[0]
            if part.function_call is not None:
                fc_result = {"name": part.function_call.name, "args": dict(part.function_call.args)}
                if part.thought_signature:
                    fc_result["thought_signature"] = base64.b64encode(part.thought_signature).decode("ascii")
                return {"function_call": fc_result}
            return {"text": response.text}

        return self._retry_on_rate_limit(call)
