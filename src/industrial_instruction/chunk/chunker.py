"""Chunking strategies.

The retrieval unit matters more than the embedder for this pipeline: a chunk
that splits a spec table in half produces unanswerable r3/r4 samples. So the
default `HeadingChunker` respects markdown headings and keeps tables
intact.
"""

from __future__ import annotations

import abc
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional

from industrial_instruction.config import ChunkConfig, Config
from industrial_instruction.schemas import Chunk, Document, StageReport
from industrial_instruction.utils.io import stable_id, write_jsonl
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_PAGE_MARKER = re.compile(r"^<!--\s*page\s+\d+\s*-->$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


class BaseChunker(abc.ABC):
    name = "base"

    def __init__(self, config: Optional[ChunkConfig] = None) -> None:
        self.config = config or ChunkConfig()

    @abc.abstractmethod
    def split(self, doc: Document) -> List[Chunk]:
        """Split one document into chunks."""

    def _make_chunk(
        self, doc: Document, text: str, ordinal: int, heading_path: List[str]
    ) -> Chunk:
        text = text.strip()
        return Chunk(
            id=stable_id(doc.id, ordinal, text[:256]),
            doc_id=doc.id,
            text=text,
            ordinal=ordinal,
            heading_path=list(heading_path),
            n_chars=len(text),
            source_path=doc.source_path,
            has_table=bool(_TABLE_ROW.search(text)),
            meta={"doc_title": doc.title, "chunker": self.name},
        )


class FixedChunker(BaseChunker):
    """Character windows with overlap. Simple and layout-agnostic."""

    name = "fixed"

    def split(self, doc: Document) -> List[Chunk]:
        text = _strip_page_markers(doc.markdown)
        size = max(self.config.max_chars, 1)
        step = max(size - max(self.config.overlap, 0), 1)
        chunks: List[Chunk] = []
        ordinal = 0
        start = 0
        while start < len(text):
            end = len(text)
            if start + size < len(text):
                end = _snap(text, start + size, start + max(size // 2, 1))
            window = text[start:end]
            if len(window.strip()) >= self.config.min_chars:
                chunks.append(self._make_chunk(doc, window, ordinal, []))
                ordinal += 1
            if end >= len(text):
                break
            # Next window starts on a word boundary inside the overlap.
            start = max(_snap(text, end - (size - step), start + 1, forward=True), start + 1)
        return chunks


class HeadingChunker(BaseChunker):
    """Split on markdown headings, then window oversized sections.

    Table blocks are kept whole when ``keep_tables_whole`` is set, even if
    that pushes a chunk past ``max_chars``.
    """

    name = "heading"

    def split(self, doc: Document) -> List[Chunk]:
        sections = self._sections(doc.markdown)
        if not any(path for path, _ in sections):
            # No headings at all (common for PDF text): fall back to windows.
            return FixedChunker(self.config).split(doc)

        # Short sections ("Use ISO VG 46 oil.") are real knowledge, so they
        # are carried into the next piece instead of being dropped.
        pieces: List[tuple] = []
        carry: List[str] = []
        carry_path: Optional[List[str]] = None
        for heading_path, body in sections:
            body = body.strip()
            if not body:
                continue
            for piece in self._pack(body):
                text = self._with_heading(piece, heading_path)
                if carry:
                    text = "\n\n".join(carry + [text])
                    heading_path = carry_path or heading_path
                    carry, carry_path = [], None
                if len(text.strip()) < self.config.min_chars:
                    carry, carry_path = [text], heading_path
                    continue
                pieces.append((text, heading_path))
        if carry:
            if pieces:
                text, path = pieces[-1]
                pieces[-1] = (text + "\n\n" + carry[0], path)
            else:
                pieces.append((carry[0], carry_path or []))

        return [
            self._make_chunk(doc, text, ordinal, path)
            for ordinal, (text, path) in enumerate(pieces)
            if text.strip()
        ]

    def _with_heading(self, piece: str, heading_path: List[str]) -> str:
        """Prefix the heading breadcrumb so retrieval and the LLM see it."""
        crumbs = " > ".join(h for h in heading_path if h)
        if not self.config.include_heading_in_text or not crumbs:
            return piece
        return f"## {crumbs}\n\n{piece}"

    # ------------------------------------------------------------------

    def _sections(self, markdown: str):
        """Yield ``(heading_path, body)`` pairs."""
        path: List[str] = []
        buffer: List[str] = []
        out = []
        for line in markdown.splitlines():
            if _PAGE_MARKER.match(line.strip()):
                continue
            m = _HEADING.match(line)
            if m:
                if buffer:
                    out.append((list(path), "\n".join(buffer)))
                    buffer = []
                level = len(m.group(1))
                title = m.group(2).strip()
                path = path[: level - 1]
                while len(path) < level - 1:
                    path.append("")
                path.append(title)
                continue
            buffer.append(line)
        if buffer:
            out.append((list(path), "\n".join(buffer)))
        return out

    def _pack(self, body: str) -> Iterable[str]:
        """Group blocks into <= max_chars pieces without splitting tables."""
        blocks = _split_blocks(body, self.config.keep_tables_whole)
        current: List[str] = []
        size = 0
        for block in blocks:
            blen = len(block)
            if current and size + blen > self.config.max_chars:
                yield "\n\n".join(current)
                if self.config.overlap > 0:
                    tail = "\n\n".join(current)[-self.config.overlap :]
                    current, size = ([tail], len(tail))
                else:
                    current, size = ([], 0)
            if blen > self.config.max_chars and not _is_table_block(block):
                for i in range(0, blen, self.config.max_chars):
                    yield block[i : i + self.config.max_chars]
                current, size = ([], 0)
                continue
            current.append(block)
            size += blen
        if current:
            yield "\n\n".join(current)


def _is_table_block(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return False
    table_lines = sum(1 for ln in lines if _TABLE_ROW.match(ln))
    return table_lines >= max(2, len(lines) // 2)


def _split_blocks(body: str, keep_tables_whole: bool) -> List[str]:
    """Split a section body into paragraph/table blocks."""
    raw = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
    if not keep_tables_whole:
        return raw
    merged: List[str] = []
    for block in raw:
        if merged and _is_table_block(block) and _is_table_block(merged[-1]):
            merged[-1] = merged[-1] + "\n" + block
        else:
            merged.append(block)
    return merged


def _snap(text: str, pos: int, floor: int, forward: bool = False) -> int:
    """Move ``pos`` to the nearest whitespace so windows don't cut words.

    Searches backwards (or forwards) no further than ``floor``/a word's
    length; if there is no whitespace nearby, ``pos`` is returned unchanged.
    """
    pos = max(0, min(pos, len(text)))
    if forward:
        if pos == 0 or text[pos - 1].isspace():
            return pos
        limit = min(len(text), pos + 80)
        for i in range(pos, limit):
            if text[i].isspace():
                return i + 1
        return pos
    for i in range(pos, max(floor, pos - 80), -1):
        if text[i - 1].isspace():
            return i
    return pos


def _strip_page_markers(markdown: str) -> str:
    return "\n".join(
        ln for ln in markdown.splitlines() if not _PAGE_MARKER.match(ln.strip())
    )


def get_chunker(config: ChunkConfig) -> BaseChunker:
    strategy = (config.strategy or "heading").lower()
    if strategy == "heading":
        return HeadingChunker(config)
    if strategy in ("fixed", "window"):
        return FixedChunker(config)
    raise ValueError(f"Unknown chunk strategy {strategy!r}. Use 'heading' or 'fixed'.")


def chunk_documents(
    config: Config,
    documents: Optional[Iterable[Document]] = None,
    output_path: Optional[str] = None,
) -> StageReport:
    """Chunk all documents and write chunks.jsonl."""
    t0 = time.time()
    if documents is None:
        from industrial_instruction.extract.runner import load_documents

        documents = load_documents(config.paths.resolve("documents"))
    docs = list(documents)

    chunker = get_chunker(config.chunk)
    chunks: List[Chunk] = []
    for doc in docs:
        chunks.extend(chunker.split(doc))

    out = Path(output_path).resolve() if output_path else config.paths.resolve("chunks")
    write_jsonl(out, chunks)

    n_chars = sum(c.n_chars for c in chunks)
    report = StageReport(
        stage="chunk",
        inputs=len(docs),
        outputs=len(chunks),
        seconds=round(time.time() - t0, 2),
        details={
            "strategy": chunker.name,
            "output_path": str(out),
            "avg_chars": round(n_chars / len(chunks), 1) if chunks else 0,
            "with_tables": sum(1 for c in chunks if c.has_table),
        },
    )
    logger.info(
        "chunk: %d chunks from %d docs (avg %.0f chars)",
        report.outputs,
        report.inputs,
        report.details["avg_chars"],
    )
    return report


def load_chunks(path: str | Path) -> List[Chunk]:
    from industrial_instruction.utils.io import iter_jsonl

    return [Chunk.model_validate(row) for row in iter_jsonl(path)]
