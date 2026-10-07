"""Default extractor: PyMuPDF text + tables, images discarded.

Why PyMuPDF is the default: pure-CPU, fast on large report sets, and its
``find_tables`` is good enough for the ruled tables typical of industrial
documentation. Heavier layout models (Docling, Marker) remain available as
extras for born-digital-but-messy or scanned corpora.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from industrial_instruction.extract.base import ExtractionError, Extractor
from industrial_instruction.extract.tables import (
    is_degenerate,
    rows_to_markdown,
    table_dimensions,
)
from industrial_instruction.extract.text_cleanup import PAGE_NUM_ONLY, clean_pages
from industrial_instruction.ocr.base import OCRPage
from industrial_instruction.schemas import Document
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


class PyMuPDFExtractor(Extractor):
    name = "pymupdf"

    def _import_pymupdf(self):
        """Import PyMuPDF.

        The package was renamed from ``fitz`` to ``pymupdf`` in 1.24.3; the old
        name still works but prints a deprecation warning on every import, so
        prefer the new one and keep ``fitz`` as a fallback for older installs.
        """
        try:
            import pymupdf

            return pymupdf
        except ImportError:
            pass
        try:
            import fitz  # PyMuPDF < 1.24.3

            return fitz
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ExtractionError(
                "PyMuPDF is required for the 'pymupdf' backend. "
                "Install it with: pip install 'industrial-instruction[pdf]'"
            ) from exc

    def extract(self, path: str | Path) -> Document:
        pymupdf = self._import_pymupdf()
        p = Path(path)
        if not p.exists():
            raise ExtractionError(f"File not found: {p}")

        page_lines: List[List[dict]] = []
        page_tables: List[List[Tuple[tuple, str]]] = []
        n_tables = 0
        n_images = 0
        ocr_pending: List[OCRPage] = []
        ocr_text: Dict[int, str] = {}
        ocr_stats = self.new_ocr_stats()

        with pymupdf.open(p) as doc:
            title = (doc.metadata or {}).get("title") or None
            for page in doc:
                # Images are never rendered or written out; we only count them
                # so the manifest records what was dropped.
                if self.config.drop_images:
                    try:
                        n_images += len(page.get_images(full=True))
                    except Exception:  # pragma: no cover - defensive
                        pass

                if self.ocr.enabled:
                    text_layer = page.get_text("text") or ""
                    if self.ocr.needs_ocr(text_layer):
                        ocr_pending.append(self._ocr_page(page, p, text_layer))
                        if len(ocr_pending) >= self.ocr_batch_size():
                            self.flush_ocr(ocr_pending, ocr_text, ocr_stats)

                tables: List[Tuple[tuple, str]] = []
                if self.config.extract_tables and self.config.table_backend != "none":
                    tables = self._page_tables(page)
                    n_tables += len(tables)
                page_lines.append(_text_lines(page, [bbox for bbox, _ in tables]))
                page_tables.append(tables)
            n_pages = doc.page_count
        self.flush_ocr(ocr_pending, ocr_text, ocr_stats)

        if self.config.strip_headers_footers:
            page_lines = _drop_page_furniture(page_lines)
        scale = (
            _heading_scale([ln for lines in page_lines for ln in lines])
            if self.config.backend_options.get("detect_headings", True)
            else _HeadingScale()
        )
        page_texts = [
            # OCR output is already markdown (headings, tables) and replaces
            # the layout pass for its page.
            ocr_text.get(i + 1) or _render_page(lines, tables, scale)
            for i, (lines, tables) in enumerate(zip(page_lines, page_tables))
        ]
        if not title:
            title = _first_heading(page_texts) or p.stem

        cleaned = clean_pages(
            page_texts,
            # Already done geometrically above; the text-only heuristic in
            # clean_pages can't tell a running header from repeated body text.
            strip_headers_footers=False,
            do_dehyphenate=self.config.dehyphenate,
        )
        markdown = self._assemble(cleaned)

        doc_obj = self._new_document(
            p,
            markdown,
            title=title,
            n_pages=n_pages,
            n_tables=n_tables,
            n_images_dropped=n_images,
            meta=self.ocr_meta(ocr_stats),
        )
        return self.validate(doc_obj)

    def _ocr_page(self, page, path: Path, text_layer: str) -> OCRPage:
        pix = page.get_pixmap(dpi=self.config.ocr.dpi)
        return OCRPage(
            image=pix.tobytes("png"),
            page_number=page.number + 1,
            source_path=str(path),
            dpi=self.config.ocr.dpi,
            width=pix.width,
            height=pix.height,
            text_layer=text_layer,
            options=dict(self.config.ocr.options),
        )

    # ------------------------------------------------------------------

    def _page_tables(self, page) -> List[Tuple[tuple, str]]:
        """Extract tables as ``(bbox, markdown)``. Never raises."""
        out: List[Tuple[tuple, str]] = []
        try:
            finder = page.find_tables()
            tables = getattr(finder, "tables", []) or []
        except Exception as exc:  # pragma: no cover - older PyMuPDF
            logger.debug("table detection unavailable on page: %s", exc)
            return out

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
                out.append((tuple(table.bbox), f"<!-- table {n_rows}x{n_cols} -->\n{md}"))
        return out

    def _assemble(self, pages: List[str]) -> str:
        parts: List[str] = []
        for i, text in enumerate(pages):
            block: List[str] = []
            if self.config.page_markers:
                block.append(f"<!-- page {i + 1} -->")
            if text.strip():
                block.append(text.strip())
            if len(block) > (1 if self.config.page_markers else 0):
                parts.append("\n\n".join(block))
        return "\n\n".join(parts).strip()


# ---------------------------------------------------------------- layout
#
# get_text("text") flattens a page: headings become ordinary lines and table
# cells are emitted a second time next to the markdown table. Working from
# get_text("dict") keeps font sizes (-> markdown headings) and bounding boxes
# (-> skip text inside tables, and place each table where it sits on the page).

_BOLD = 1 << 4
_MAX_HEADING_CHARS = 120
#: Fraction of the page height treated as header/footer margin.
_MARGIN = 0.08
_DIGITS = re.compile(r"\d+")


def _inside(bbox: tuple, tables: List[tuple]) -> bool:
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return any(t[0] <= x <= t[2] and t[1] <= y <= t[3] for t in tables)


def _text_lines(page, table_bboxes: List[tuple]) -> List[dict]:
    """Lines outside tables with their font size, boldness and position."""
    out: List[dict] = []
    try:
        data = page.get_text("dict", sort=True)
    except Exception:  # pragma: no cover - very old PyMuPDF
        return [
            {"text": ln, "size": 0.0, "bold": False, "y": 0.0, "block": 0,
             "solo": False, "margin": False}
            for ln in (page.get_text("text") or "").splitlines()
        ]
    height = float(data.get("height") or page.rect.height or 0.0)
    for b_idx, block in enumerate(data.get("blocks", [])):
        if block.get("type", 0) != 0 or _inside(block["bbox"], table_bboxes):
            continue
        lines = block.get("lines", [])
        for line in lines:
            spans = [sp for sp in line.get("spans", []) if sp.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(sp["text"] for sp in line["spans"]).strip()
            # Dominant span by character count decides size and weight.
            main = max(spans, key=lambda sp: len(sp["text"].strip()))
            out.append(
                {
                    "text": text,
                    "size": round(float(main.get("size", 0.0)) * 2) / 2,
                    "bold": bool(main.get("flags", 0) & _BOLD),
                    "y": float(line["bbox"][1]),
                    "block": b_idx,
                    "solo": len(lines) == 1,
                    "margin": bool(height)
                    and not (height * _MARGIN < line["bbox"][1] < height * (1 - _MARGIN)),
                }
            )
    return out


def _drop_page_furniture(pages: List[List[dict]]) -> List[List[dict]]:
    """Remove running headers/footers and page numbers.

    Only lines in the top/bottom page margin are candidates, which is what
    makes it safe to ignore digits when matching ("Manual - 12" vs "- 13"):
    templated body text such as "rated 10 kW" / "rated 22 kW" is never
    touched.
    """

    def key(line: dict) -> str:
        return _DIGITS.sub("#", line["text"].strip().lower())

    counts: Counter = Counter()
    for lines in pages:
        counts.update({key(ln) for ln in lines if ln["margin"]})
    threshold = max(2, int(len(pages) * 0.5))
    repeated = {k for k, n in counts.items() if n >= threshold} if len(pages) > 1 else set()
    return [
        [
            ln
            for ln in lines
            if not (
                ln["margin"]
                and (key(ln) in repeated or PAGE_NUM_ONLY.match(ln["text"]))
            )
        ]
        for lines in pages
    ]


def _body_size(lines: List[dict]) -> float:
    sizes: Counter = Counter()
    for ln in lines:
        sizes[ln["size"]] += len(ln["text"])
    return sizes.most_common(1)[0][0] if sizes else 0.0


@dataclass
class _HeadingScale:
    """Font sizes that mark headings, relative to the body text size."""

    body: float = 0.0
    by_size: Dict[float, int] = field(default_factory=dict)
    bold_level: int = 1

    def level(self, line: dict) -> Optional[int]:
        text = line["text"]
        if self.body <= 0 or len(text) > _MAX_HEADING_CHARS:
            return None
        if not any(c.isalpha() for c in text):
            return None
        if line["size"] in self.by_size:
            return self.by_size[line["size"]]
        # A bold, standalone, short line at body size is a run-in heading
        # ("Lubrication"), but not a bold sentence or a "Note:" label.
        if (
            line["bold"]
            and line["solo"]
            and line["size"] == self.body
            and len(text) <= 80
            and not text.endswith((".", ",", ";", ":"))
        ):
            return self.bold_level
        return None


def _heading_scale(lines: List[dict], max_levels: int = 3) -> _HeadingScale:
    """Map font sizes clearly larger than body text to heading levels 1..n."""
    body = _body_size(lines)
    if body <= 0:
        return _HeadingScale()
    larger = sorted(
        {ln["size"] for ln in lines if ln["size"] >= body * 1.15}, reverse=True
    )
    return _HeadingScale(
        body=body,
        by_size={size: min(i + 1, max_levels) for i, size in enumerate(larger)},
        bold_level=min(len(larger) + 1, max_levels + 1),
    )


def _render_page(
    lines: List[dict], tables: List[Tuple[tuple, str]], scale: _HeadingScale
) -> str:
    """Rebuild one page as markdown, tables placed in reading order."""
    items: List[Tuple[float, str]] = []
    para: List[str] = []
    para_y = 0.0
    last_block = None
    last_heading: Optional[Tuple[int, float]] = None

    def flush():
        nonlocal para
        if para:
            items.append((para_y, "\n".join(para)))
            para = []

    for ln in lines:
        level = scale.level(ln)
        if level is not None:
            flush()
            # Wrapped headings: merge consecutive lines of the same heading.
            if (
                last_heading
                and last_heading[0] == level
                and ln["block"] == last_block
                and items
            ):
                y, text = items[-1]
                items[-1] = (y, f"{text} {ln['text']}")
            else:
                items.append((ln["y"], "#" * level + " " + ln["text"]))
            last_heading = (level, ln["y"])
            last_block = ln["block"]
            continue
        if ln["block"] != last_block:
            flush()
            para_y = ln["y"]
        para.append(ln["text"])
        last_block = ln["block"]
        last_heading = None
    flush()

    items.extend((bbox[1], md) for bbox, md in tables)
    items.sort(key=lambda item: item[0])
    return "\n\n".join(text for _, text in items)


def _first_heading(pages: List[str]) -> Optional[str]:
    for page in pages:
        for line in page.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    return None
