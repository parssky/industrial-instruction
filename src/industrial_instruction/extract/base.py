"""Extractor protocol shared by all PDF backends."""

from __future__ import annotations

import abc
from pathlib import Path

from industrial_instruction.config import ExtractConfig
from industrial_instruction.schemas import Document
from industrial_instruction.utils.io import sha256_file, sha256_text, stable_id


class ExtractionError(RuntimeError):
    """Raised when a PDF cannot be converted to usable markdown."""


class Extractor(abc.ABC):
    """Convert one PDF into an image-free :class:`Document`.

    Implement :meth:`extract` in a subclass and register it to make it
    available via ``extract.backend`` in the config.
    """

    name: str = "base"

    def __init__(self, config: ExtractConfig | None = None) -> None:
        self.config = config or ExtractConfig()

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
                "enable OCR or use another backend."
            )
        return doc
