"""Optional extractor: pdfplumber. Slower, often better on unruled tables."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List

from industrial_instruction.extract.base import Extractor, ExtractionError
from industrial_instruction.extract.tables import is_degenerate, rows_to_markdown
from industrial_instruction.extract.text_cleanup import clean_pages
from industrial_instruction.ocr.base import OCRPage
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
        ocr_pending: List[OCRPage] = []
        ocr_text: Dict[int, str] = {}
        ocr_stats = self.new_ocr_stats()

        with pdfplumber.open(str(p)) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                n_images += len(getattr(page, "images", []) or [])
                text = page.extract_text() or ""
                page_texts.append(text)
                if self.ocr.enabled and self.ocr.needs_ocr(text):
                    ocr_pending.append(self._ocr_page(page, number, p, text))
                    if len(ocr_pending) >= self.ocr_batch_size():
                        self.flush_ocr(ocr_pending, ocr_text, ocr_stats)
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
            self.flush_ocr(ocr_pending, ocr_text, ocr_stats)

        for number, markdown in ocr_text.items():
            page_texts[number - 1] = markdown
            page_tables[number - 1] = []  # the OCR markdown carries the tables
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
            meta=self.ocr_meta(ocr_stats),
        )
        return self.validate(doc)

    def _ocr_page(self, page, number: int, path: Path, text_layer: str) -> OCRPage:
        image = page.to_image(resolution=self.config.ocr.dpi).original
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return OCRPage(
            image=buf.getvalue(),
            page_number=number,
            source_path=str(path),
            dpi=self.config.ocr.dpi,
            width=image.width,
            height=image.height,
            text_layer=text_layer,
            options=dict(self.config.ocr.options),
        )
