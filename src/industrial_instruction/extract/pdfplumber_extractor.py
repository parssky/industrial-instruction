"""Optional extractor: pdfplumber. Slower, often better on unruled tables."""

from __future__ import annotations

from pathlib import Path
from typing import List

from industrial_instruction.extract.base import Extractor, ExtractionError
from industrial_instruction.extract.tables import is_degenerate, rows_to_markdown
from industrial_instruction.extract.text_cleanup import clean_pages
from industrial_instruction.schemas import Document


class PdfplumberExtractor(Extractor):
    name = "pdfplumber"

    def extract(self, path: str | Path) -> Document:
        try:
            import pdfplumber
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ExtractionError(
                "pdfplumber is required for this backend. Install with: "
                "pip install 'industrial-instruction[pdfplumber]'"
            ) from exc

        p = Path(path)
        if not p.exists():
            raise ExtractionError(f"File not found: {p}")

        page_texts: List[str] = []
        page_tables: List[List[str]] = []
        n_tables = 0
        n_images = 0

        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages:
                n_images += len(getattr(page, "images", []) or [])
                page_texts.append(page.extract_text() or "")
                tables_md: List[str] = []
                if self.config.extract_tables and self.config.table_backend != "none":
                    for rows in page.extract_tables() or []:
                        if is_degenerate(rows):
                            continue
                        md = rows_to_markdown(rows)
                        if md:
                            tables_md.append(md)
                n_tables += len(tables_md)
                page_tables.append(tables_md)
            n_pages = len(pdf.pages)

        cleaned = clean_pages(
            page_texts,
            strip_headers_footers=self.config.strip_headers_footers,
            do_dehyphenate=self.config.dehyphenate,
        )
        blocks: List[str] = []
        for i, text in enumerate(cleaned):
            block = []
            if self.config.page_markers:
                block.append(f"<!-- page {i + 1} -->")
            if text.strip():
                block.append(text.strip())
            block.extend(page_tables[i])
            if len(block) > (1 if self.config.page_markers else 0):
                blocks.append("\n\n".join(block))

        doc = self._new_document(
            p,
            "\n\n".join(blocks).strip(),
            n_pages=n_pages,
            n_tables=n_tables,
            n_images_dropped=n_images,
        )
        return self.validate(doc)
