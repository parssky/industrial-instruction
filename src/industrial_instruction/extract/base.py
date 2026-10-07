"""Extractor protocol shared by all PDF backends."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, List

from industrial_instruction.config import ExtractConfig
from industrial_instruction.schemas import Document
from industrial_instruction.utils.io import sha256_file, sha256_text, stable_id


class ExtractionError(RuntimeError):
    """Raised when a PDF cannot be converted to usable markdown."""


class Extractor(abc.ABC):
    """Convert one PDF into an image-free `Document`.

    Implement `extract` in a subclass and register it to make it
    available via ``extract.backend`` in the config.
    """

    name: str = "base"

    def __init__(self, config: ExtractConfig | None = None) -> None:
        self.config = config or ExtractConfig()
        #: Base for relative OCR paths (backend file, cache); the extract
        #: stage sets it to ``paths.root``.
        self.root: Path = Path.cwd()
        self._ocr = None

    @property
    def ocr(self):
        """The `PageOCR` for this run."""
        if self._ocr is None:
            from industrial_instruction.ocr.runner import PageOCR

            self._ocr = PageOCR(self.config.ocr, root=self.root)
        return self._ocr

    def flush_ocr(
        self, pending: List, results: Dict[int, str], stats: Dict[str, int]
    ) -> None:
        """OCR the ``pending`` pages into ``results`` (page number -> markdown).

        Pages are flushed in batches so a 500-page scan never holds 500
        rendered images in memory. A page whose OCR fails or comes back
        empty is left out of ``results`` and keeps its text layer.
        """
        if not pending:
            return
        for number, result in self.ocr.run(pending).items():
            if result.error:
                stats["ocr_failed"] += 1
            elif result.ok:
                results[number] = result.markdown
                stats["ocr_pages"] += 1
                stats["ocr_cached"] += int(result.cached)
            else:
                stats["ocr_empty"] += 1
        pending.clear()

    @staticmethod
    def new_ocr_stats() -> Dict[str, int]:
        return {"ocr_pages": 0, "ocr_cached": 0, "ocr_failed": 0, "ocr_empty": 0}

    def ocr_meta(self, stats: Dict[str, int]) -> Dict[str, object]:
        if not self.ocr.enabled:
            return {}
        return {"ocr_backend": self.config.ocr.backend, **stats}

    def ocr_batch_size(self) -> int:
        """Pages held in memory before OCR runs (rendered PNGs are large)."""
        return max(self.config.ocr.max_workers, 1) * 4

    @abc.abstractmethod
    def extract(self, path: str | Path) -> Document:
        """Return a Document for ``path``."""

    # -- helpers available to every backend ---------------------------------

    def _new_document(self, path: str | Path, markdown: str, **kwargs) -> Document:
        p = Path(path)
        return Document(
            id=stable_id(p.name, sha256_file(p)),
            source_path=str(p),
            title=kwargs.pop("title", None) or p.stem,
            markdown=markdown,
            content_sha256=sha256_text(markdown),
            meta={"backend": self.name, **kwargs.pop("meta", {})},
            **kwargs,
        )

    def validate(self, doc: Document) -> Document:
        """Guard against scanned/empty PDFs producing junk downstream."""
        if len(doc.markdown.strip()) < self.config.min_chars_per_doc:
            raise ExtractionError(
                f"{doc.source_path}: only {len(doc.markdown.strip())} chars extracted "
                f"(min {self.config.min_chars_per_doc}). The PDF may be scanned; "
                "set extract.ocr.mode: auto and an extract.ocr.backend."
            )
        return doc
