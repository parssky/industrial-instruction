"""Built-in backend: any OpenAI-compatible vision model.

Covers the common self-hosted case with no code at all - start a VLM on
vLLM (``vllm serve Qwen/Qwen2.5-VL-7B-Instruct``, ``allenai/olmOCR-7B-0825``,
``nanonets/Nanonets-OCR-s``, ...) and set::

    extract:
      ocr:
        mode: auto
        backend: openai
        base_url: http://localhost:8000/v1
        model: Qwen/Qwen2.5-VL-7B-Instruct

Models trained on a specific instruction (olmOCR, Nanonets-OCR) work best
with their own prompt: set ``extract.ocr.prompt``. ``prompt: ""`` sends the
image alone, for models that take no instruction.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from industrial_instruction.config import OCRConfig
from industrial_instruction.ocr.base import OCRError, OCRPage, clean_markdown
from industrial_instruction.ocr.registry import ocr_backend
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PROMPT = (
    "Transcribe this page of a technical document into clean Markdown.\n"
    "- Reproduce all text exactly, in reading order. Do not summarize or translate.\n"
    "- Use #, ## and ### for headings, matching the document's hierarchy.\n"
    "- Write tables as Markdown tables, keeping every row, column and unit.\n"
    "- Keep lists, numbered steps, warnings and notes.\n"
    "- Omit running headers, footers and page numbers.\n"
    "- Skip pictures and diagrams; only transcribe text that appears in them.\n"
    "Return only the Markdown, with no code fences and no commentary."
)


class OpenAIVisionOCR:
    """Page function backed by a chat-completions endpoint that accepts images."""

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._client = None
        self._lock = threading.Lock()

    def _ensure_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        from openai import OpenAI
                    except ImportError as exc:  # pragma: no cover
                        raise OCRError("pip install openai") from exc
                    kwargs: Dict[str, Any] = {
                        "api_key": self.config.resolve_api_key(),
                        "timeout": self.config.timeout,
                        "max_retries": 0,  # retried below with backoff
                    }
                    if self.config.base_url:
                        kwargs["base_url"] = self.config.base_url
                    self._client = OpenAI(**kwargs)
        return self._client

    def messages(self, page: OCRPage) -> List[Dict[str, Any]]:
        prompt = DEFAULT_PROMPT if self.config.prompt is None else self.config.prompt
        content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": page.data_url()}}
        ]
        if prompt:
            content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def __call__(self, page: OCRPage) -> str:
        client = self._ensure_client()
        last: Optional[Exception] = None
        for attempt in range(1, max(self.config.max_retries, 1) + 1):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=self.messages(page),
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                return clean_markdown(response.choices[0].message.content)
            except Exception as exc:  # noqa: BLE001 - provider specific
                last = exc
                if attempt < self.config.max_retries:
                    delay = 2 ** (attempt - 1)
                    logger.warning(
                        "OCR page %d of %s failed (attempt %d): %s; retrying in %ds",
                        page.page_number, page.source_path, attempt, exc, delay,
                    )
                    time.sleep(delay)
        raise OCRError(f"OCR failed after {self.config.max_retries} attempts: {last}")


@ocr_backend("openai", factory=True)
def openai_vision(config: OCRConfig) -> OpenAIVisionOCR:
    return OpenAIVisionOCR(config)
