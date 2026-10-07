"""OpenAI-compatible chat client with retries and tolerant JSON parsing.

Works against api.openai.com, a local vLLM server, or anything else speaking
the same protocol - the endpoint comes from config, not a hardcoded
``http://ip:port/v1``.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Dict, Optional

from industrial_instruction.config import GenerateConfig
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when a completion cannot be obtained after all retries."""


class LLMClient:
    """Thread-safe wrapper around the chat-completions API."""

    def __init__(self, config: GenerateConfig) -> None:
        self.config = config
        self._client = None
        self._lock = threading.Lock()
        self._supports_json_mode = config.request_json_object

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:  # pragma: no cover
                    raise LLMError("pip install openai") from exc
                kwargs: Dict[str, Any] = {
                    "api_key": self.config.resolve_api_key(),
                    "timeout": self.config.timeout,
                }
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url
                self._client = OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> str:
        """Return raw completion text, retrying transient failures."""
        client = self._ensure_client()
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        last_error: Optional[Exception] = None

        for attempt in range(1, max(self.config.max_retries, 1) + 1):
            kwargs: Dict[str, Any] = {
                "model": model or self.config.model,
                "messages": messages,
                "temperature": (
                    self.config.temperature if temperature is None else temperature
                ),
            }
            if self.config.max_tokens:
                kwargs["max_tokens"] = self.config.max_tokens
            if self._supports_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - provider-specific errors
                last_error = exc
                if self._supports_json_mode and _is_json_mode_error(exc):
                    # Some servers reject response_format; drop it and retry.
                    logger.warning(
                        "endpoint rejected JSON mode, disabling it for this run"
                    )
                    self._supports_json_mode = False
                    continue
                if attempt >= max(self.config.max_retries, 1):
                    break
                delay = self.config.retry_backoff ** (attempt - 1)
                logger.warning(
                    "completion failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self.config.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise LLMError(f"completion failed after retries: {last_error}")

    def complete_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> tuple:
        """Return ``(parsed_dict, raw_text)``; raises on unparseable output."""
        raw = self.complete(prompt, temperature=temperature, model=model)
        return parse_json_object(raw), raw


def _is_json_mode_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_object" in text


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_object(text: str) -> Dict[str, Any]:
    """Best-effort extraction of one JSON object from a model reply.

    Handles the three failure modes seen in the original run logs: markdown
    code fences, leading prose before the object, and trailing commas. A
    single malformed reply should cost one sample, not the whole batch.
    """
    if not text or not text.strip():
        raise ValueError("empty completion")
    candidate = text.strip()

    try:
        return _as_dict(json.loads(candidate))
    except Exception:
        pass

    fenced = _FENCE.search(candidate)
    if fenced:
        inner = fenced.group(1).strip()
        try:
            return _as_dict(json.loads(inner))
        except Exception:
            candidate = inner

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        blob = candidate[start : end + 1]
        try:
            return _as_dict(json.loads(blob))
        except Exception:
            repaired = re.sub(r",\s*([}\]])", r"\1", blob)
            return _as_dict(json.loads(repaired))

    raise ValueError(f"no JSON object found in completion: {text[:200]!r}")


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
