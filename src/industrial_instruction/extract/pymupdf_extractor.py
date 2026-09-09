"""Default extractor: PyMuPDF text + tables, images discarded.

Why PyMuPDF is the default: pure-CPU, fast on large report sets, and its
``find_tables`` is good enough for the ruled tables typical of industrial
documentation. Heavier layout models (Docling, Marker) remain available as
extras for born-digital-but-messy or scanned corpora.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from industrial_instruction.extract.base import Extractor, ExtractionError
from industrial_instruction.extract.tables import (
    is_degenerate,
    rows_to_markdown,
    table_dimensions,
)
from industrial_instruction.extract.text_cleanup import clean_pages
from industrial_instruction.schemas import Document
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


class PyMuPDFExtractor(Extractor):
    name = "pymupdf"

    def _import_fitz(self):
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ExtractionError(
                "PyMuPDF is required for the 'pymupdf' backend. "
                "Install it with: pip install 'industrial-instruction[pdf]'"
            ) from exc
        return fitz

    def extract(self, path: str | Path) -> Document:
        fitz = self._import_fitz()
        p = Path(path)
        if not p.exists():
            raise ExtractionError(f"File not found: {p}")

        page_texts: List[str] = []
        page_tables: List[List[str]] = []
        n_tables = 0
        n_images = 0

        with fitz.open(p) as doc:
            title = (doc.metadata or {}).get("title") or p.stem
            for page in doc:
                # Images are never rendered or written out; we only count them
                # so the manifest records what was dropped.
                if self.config.drop_images:
                    try:
                        n_images += len(page.get_images(full=True))
                    except Exception:  # pragma: no cover - defensive
                        pass

                text = page.get_text("text") or ""
                tables_md: List[str] = []
                if self.config.extract_tables and self.config.table_backend != "none":
                    tables_md, found = self._page_tables(page)
                    n_tables += found
                page_texts.append(text)
                page_tables.append(tables_md)
            n_pages = doc.page_count

        cleaned = clean_pages(
            page_texts,
            strip_headers_footers=self.config.strip_headers_footers,
            do_dehyphenate=self.config.dehyphenate,
        )
        markdown = self._assemble(cleaned, page_tables)

        doc_obj = self._new_document(
            p,
            markdown,
            title=title,
            n_pages=n_pages,
            n_tables=n_tables,
            n_images_dropped=n_images,
        )
        return self.validate(doc_obj)

    # ------------------------------------------------------------------

    def _page_tables(self, page) -> Tuple[List[str], int]:
        """Extract tables from a page as markdown. Never raises."""
        out: List[str] = []
        try:
            finder = page.find_tables()
            tables = getattr(finder, "tables", []) or []
        except Exception as exc:  # pragma: no cover - older PyMuPDF
            logger.debug("table detection unavailable on page: %s", exc)
            return out, 0

        for table in tables:
            try:
                rows = table.extract()
            except Exception:
                continue
            if is_degenerate(rows):
                continue
            md = rows_to_markdown(rows)
            if md:
                n_rows, n_cols = table_dimensions(rows)
                out.append(f"<!-- table {n_rows}x{n_cols} -->\n{md}")
        return out, len(out)

    def _assemble(self, pages: List[str], page_tables: List[List[str]]) -> str:
        parts: List[str] = []
        for i, text in enumerate(pages):
            block: List[str] = []
            if self.config.page_markers:
                block.append(f"<!-- page {i + 1} -->")
            if text.strip():
                block.append(text.strip())
            for md in page_tables[i] if i < len(page_tables) else []:
                block.append(md)
            if len(block) > (1 if self.config.page_markers else 0):
                parts.append("\n\n".join(block))
        return "\n\n".join(parts).strip()
