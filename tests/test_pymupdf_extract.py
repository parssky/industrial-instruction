"""PDF -> markdown structure: headings, tables and page furniture.

Builds a small PDF on the fly, so no binary fixtures are checked in.
"""

import pytest

from industrial_instruction.config import ExtractConfig

pymupdf = pytest.importorskip("pymupdf", reason="PyMuPDF not installed")

from industrial_instruction.extract.pymupdf_extractor import (  # noqa: E402
    PyMuPDFExtractor,
)

BODY = (
    "The pump requires inspection every {hours} operating hours. Check the oil "
    "level weekly through the sight glass on the bearing housing."
)
ROWS = [("Code", "Meaning"), ("E01", "Phase imbalance"), ("E02", "Overtemperature")]


def _draw_table(page, x0, y0, col_w=150, row_h=20):
    for r, row in enumerate(ROWS):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(
                x0 + c * col_w, y0 + r * row_h, x0 + (c + 1) * col_w, y0 + (r + 1) * row_h
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 4, rect.y1 - 6), cell, fontsize=10)


def make_pdf(path):
    doc = pymupdf.open()
    for n in range(4):
        page = doc.new_page()
        page.insert_text((72, 40), "ACME P-100 Service Manual", fontsize=8)
        page.insert_text((290, 810), f"Page {n + 1} of 4", fontsize=8)
        y = 90
        if n == 0:
            page.insert_text((72, y), "Pump Maintenance", fontsize=20)
            y += 40
        page.insert_text((72, y), f"Section {n + 1} Lubrication", fontsize=14)
        y += 10
        page.insert_textbox(
            pymupdf.Rect(72, y, 520, y + 80), BODY.format(hours=500 * (n + 1)), fontsize=10
        )
        if n == 1:
            page.insert_text((72, 260), "Fault codes", fontsize=14)
            _draw_table(page, 72, 280)
            page.insert_textbox(
                pymupdf.Rect(72, 360, 520, 420),
                "Reset the controller after clearing any fault code listed above.",
                fontsize=10,
            )
    doc.set_metadata({})
    doc.save(path)


@pytest.fixture
def markdown(tmp_path):
    pdf = tmp_path / "manual.pdf"
    make_pdf(pdf)
    doc = PyMuPDFExtractor(ExtractConfig(min_chars_per_doc=50)).extract(pdf)
    return doc


def test_font_size_becomes_heading_levels(markdown):
    lines = markdown.markdown.splitlines()
    assert "# Pump Maintenance" in lines
    assert "## Section 1 Lubrication" in lines
    assert "## Fault codes" in lines
    assert markdown.title == "Pump Maintenance"


def test_table_is_markdown_once_and_in_place(markdown):
    md = markdown.markdown
    assert "| E01 | Phase imbalance |" in md
    # cell text must not leak into the prose a second time
    assert md.count("Phase imbalance") == 1
    assert md.index("## Fault codes") < md.index("| E01") < md.index("Reset the controller")


def test_running_header_and_page_numbers_are_stripped(markdown):
    md = markdown.markdown
    assert "ACME P-100 Service Manual" not in md
    assert "Page 2 of 4" not in md
    for hours in (500, 1000, 1500, 2000):
        assert f"every {hours} operating hours" in md
    assert "## Section 2 Lubrication" in md
