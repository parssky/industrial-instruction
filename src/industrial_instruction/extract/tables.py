"""Table -> markdown rendering, shared by the PDF backends.

Tables are the reason plain text extraction is not enough for technical
reports: specs, tolerances and fault codes live in them. We render them as
GitHub-style markdown so they survive chunking and stay readable in prompts.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    # Escape pipes so a cell cannot break the markdown table.
    return text.replace("|", "\\|")


def rows_to_markdown(rows: Sequence[Sequence[Any]], header: bool = True) -> str:
    """Render a 2D cell matrix as a markdown table.

    Ragged rows are padded, fully empty rows and columns are dropped.
    Returns an empty string when nothing usable remains.
    """
    cleaned: List[List[str]] = [[_clean_cell(c) for c in row] for row in rows or []]
    cleaned = [r for r in cleaned if any(c for c in r)]
    if not cleaned:
        return ""

    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]

    keep = [i for i in range(width) if any(r[i] for r in cleaned)]
    if not keep:
        return ""
    cleaned = [[r[i] for i in keep] for r in cleaned]
    width = len(keep)

    if header and len(cleaned) > 1:
        head, body = cleaned[0], cleaned[1:]
    else:
        head, body = [f"col{i + 1}" for i in range(width)], cleaned

    head = [h or f"col{i + 1}" for i, h in enumerate(head)]
    lines = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def table_dimensions(rows: Optional[Sequence[Sequence[Any]]]) -> tuple:
    """Return ``(n_rows, n_cols)`` for a cell matrix."""
    if not rows:
        return (0, 0)
    return (len(rows), max((len(r) for r in rows), default=0))


def is_degenerate(rows: Optional[Sequence[Sequence[Any]]], min_cells: int = 4) -> bool:
    """True when a detected 'table' is too small to be worth keeping.

    PDF table detectors frequently fire on layout artifacts such as a single
    boxed line; those add noise to the corpus.
    """
    n_rows, n_cols = table_dimensions(rows)
    if n_rows < 2 or n_cols < 2:
        return True
    filled = sum(1 for row in rows for cell in row if _clean_cell(cell))
    return filled < min_cells
