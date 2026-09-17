"""Disk-cached Gemini client.

Every call is keyed by a hash of (model, prompt, params). A cache hit returns
straight from disk with no network request; a miss calls Gemini, retrying on
429s with exponential backoff and jitter, and writes the result to disk before
returning it. This is what makes the evaluation harness re-runnable on a free
API key without burning the daily quota.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
DEFAULT_MODEL = "gemini-2.5-flash"

# Free-tier models only. Never point this client at a Pro model.
ALLOWED_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
}


class RateLimitExceeded(RuntimeError):
    """Raised when Gemini keeps returning 429 past the retry budget."""


def _cache_key(model: str, prompt: str, params: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "params": params},
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

        key = _cache_key(model, prompt, params)
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

    def _call_with_backoff(self, prompt: str, model: str, params: dict[str, Any]) -> str:
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            temperature=params.get("temperature", 0.0),
            max_output_tokens=params.get("max_output_tokens"),
        )

        attempt = 0
        while True:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                return response.text
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
