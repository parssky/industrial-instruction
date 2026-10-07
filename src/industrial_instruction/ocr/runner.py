"""Run an OCR backend over many pages: concurrency, caching, failures.

Backends only implement ``page -> markdown``; everything a real corpus
needs around that lives here, so a ten-line custom function gets it too:

- pages are sent concurrently (``max_workers``), which is what keeps a
  vLLM server's batch full
- results are cached on disk by page-image hash + backend settings, so
  re-running ``ii extract`` after a tweak doesn't re-OCR hundreds of pages
- one failing page is logged and falls back to the PDF's text layer; it
  never fails the document
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from industrial_instruction.config import OCRConfig
from industrial_instruction.ocr.base import OCRFunction, OCRPage, clean_markdown
from industrial_instruction.ocr.registry import get_ocr
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    page_number: int
    markdown: str = ""
    cached: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.markdown.strip())


class PageOCR:
    """Applies one OCR backend to pages, per `OCRConfig`."""

    def __init__(
        self,
        config: OCRConfig,
        root: Optional[Union[str, Path]] = None,
        fn: Optional[OCRFunction] = None,
    ) -> None:
        self.config = config
        self.root = Path(root) if root else Path.cwd()
        self._fn = fn  # resolved lazily: mode=auto may never need it
        cache_dir = Path(config.cache_dir)
        self.cache_dir = cache_dir if cache_dir.is_absolute() else self.root / cache_dir

    @property
    def enabled(self) -> bool:
        return self.config.mode in ("auto", "always")

    def needs_ocr(self, text_layer: str) -> bool:
        if self.config.mode == "always":
            return True
        if self.config.mode == "auto":
            return len(text_layer.strip()) < self.config.min_chars_per_page
        return False

    @property
    def fn(self) -> OCRFunction:
        if self._fn is None:
            self._fn = get_ocr(self.config, root=self.root)
        return self._fn

    # ------------------------------------------------------------- cache

    def _identity(self) -> str:
        """Everything about the backend that changes its output."""
        cfg = self.config
        return json.dumps(
            {
                "backend": cfg.backend,
                "model": cfg.model if cfg.backend == "openai" else None,
                "prompt": cfg.prompt if cfg.backend == "openai" else None,
                "dpi": cfg.dpi,
                "options": cfg.options,
            },
            sort_keys=True,
            default=str,
        )

    def _cache_path(self, page: OCRPage) -> Path:
        digest = hashlib.sha256(page.image + self._identity().encode("utf-8"))
        return self.cache_dir / f"{digest.hexdigest()[:32]}.md"

    # --------------------------------------------------------------- run

    def _one(self, page: OCRPage) -> OCRResult:
        path = self._cache_path(page) if self.config.cache else None
        if path is not None and path.exists():
            return OCRResult(page.page_number, path.read_text(encoding="utf-8"), cached=True)
        try:
            markdown = clean_markdown(self.fn(page))
        except Exception as exc:  # noqa: BLE001 - user code; isolate per page
            logger.warning(
                "OCR failed on page %d of %s: %s", page.page_number, page.source_path, exc
            )
            return OCRResult(page.page_number, error=repr(exc)[:500])
        if path is not None and markdown:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(markdown, encoding="utf-8")
        return OCRResult(page.page_number, markdown)

    def run(self, pages: Sequence[OCRPage]) -> Dict[int, OCRResult]:
        """OCR ``pages``; returns results keyed by page number."""
        if not pages:
            return {}
        workers = max(1, min(self.config.max_workers, len(pages)))
        if workers == 1:
            results: List[OCRResult] = [self._one(p) for p in pages]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self._one, pages))
        return {r.page_number: r for r in results}
