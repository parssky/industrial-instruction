"""Post-processing that makes extracted PDF text usable as training context."""

from __future__ import annotations

import re
from collections import Counter
from typing import List

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_BULLET = re.compile(r"^[\u2022\u00b7\u25cf\u25aa\u2023\u2043]\s*", re.MULTILINE)
PAGE_NUM_ONLY = re.compile(
    r"^\s*(page\s*)?\d{1,4}\s*((/|of)\s*\d{1,4})?\s*$", re.IGNORECASE
)


def dehyphenate(text: str) -> str:
    """Join words split across line breaks (``mainte-\\nnance`` -> maintenance)."""
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = _TRAILING_WS.sub("\n", text)
    return _MULTI_BLANK.sub("\n\n", text).strip()


def normalize_bullets(text: str) -> str:
    return _BULLET.sub("- ", text)


def find_repeated_lines(pages: List[str], min_ratio: float = 0.6) -> set:
    """Detect running headers/footers repeated across most pages.

    Only the first and last few lines of each page are considered, so real
    body text that happens to repeat is not removed.
    """
    if len(pages) < 4:
        return set()
    counter: Counter = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        for line in lines[:3] + lines[-3:]:
            # Markdown headings come from large fonts; running headers don't.
            if 3 <= len(line) <= 120 and not line.startswith("#"):
                counter[line] += 1
    threshold = max(2, int(len(pages) * min_ratio))
    return {line for line, count in counter.items() if count >= threshold}


def strip_lines(page: str, banned: set, edge: int = 3) -> str:
    """Drop running headers/footers, and page numbers near the page edges."""
    lines = page.splitlines()
    filled = [i for i, ln in enumerate(lines) if ln.strip()]
    edges = set(filled[:edge] + filled[-edge:])
    kept = [
        ln
        for i, ln in enumerate(lines)
        if (ln.lstrip().startswith("#") or ln.strip() not in banned)
        and not (i in edges and PAGE_NUM_ONLY.match(ln))
    ]
    return "\n".join(kept)


def clean_pages(
    pages: List[str], strip_headers_footers: bool = True, do_dehyphenate: bool = True
) -> List[str]:
    """Clean a list of per-page texts."""
    banned = find_repeated_lines(pages) if strip_headers_footers else set()
    out = []
    for page in pages:
        text = strip_lines(page, banned) if strip_headers_footers else page
        if do_dehyphenate:
            text = dehyphenate(text)
        out.append(normalize_whitespace(normalize_bullets(text)))
    return out
