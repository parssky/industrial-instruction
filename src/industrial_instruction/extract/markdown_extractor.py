"""Pass-through extractor for corpora that are already markdown/text.

This keeps the paper's original workflow reproducible: the published results
started from pre-converted Panasonic markdowns rather than raw PDFs.
"""

from __future__ import annotations

from pathlib import Path

from industrial_instruction.extract.base import Extractor, ExtractionError
from industrial_instruction.extract.text_cleanup import normalize_whitespace
from industrial_instruction.schemas import Document

_IMAGE_MD = "!"


class MarkdownExtractor(Extractor):
    name = "markdown"

    def extract(self, path: str | Path) -> Document:
        p = Path(path)
        if not p.exists():
            raise ExtractionError(f"File not found: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        dropped = 0
        if self.config.drop_images:
            text, dropped = _strip_image_markdown(text)
        doc = self._new_document(
            p,
            normalize_whitespace(text),
            n_pages=0,
            n_images_dropped=dropped,
        )
        return self.validate(doc)


def _strip_image_markdown(text: str) -> tuple:
    """Remove ``![alt](src)`` image embeds, returning (text, n_removed)."""
    import re

    pattern = re.compile(r"!\[[^\]]*\]\([^)]*\)")
    n = len(pattern.findall(text))
    text = pattern.sub("", text)
    html_img = re.compile(r"<img\b[^>]*>", re.I)
    n += len(html_img.findall(text))
    return html_img.sub("", text), n
