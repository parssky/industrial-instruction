"""Chunking strategies.

The retrieval unit matters more than the embedder for this pipeline: a chunk
that splits a spec table in half produces unanswerable r3/r4 samples. So the
default :class:`HeadingChunker` respects markdown headings and keeps tables
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
        for start in range(0, len(text), step):
            window = text[start : start + size]
            if len(window.strip()) < self.config.min_chars:
                continue
            chunks.append(self._make_chunk(doc, window, ordinal, []))
            ordinal += 1
            if start + size >= len(text):
                break
        return chunks


class HeadingChunker(BaseChunker):
    """Split on markdown headings, then window oversized sections.

    Table blocks are kept whole when ``keep_tables_whole`` is set, even if
    that pushes a chunk past ``max_chars``.
    """

    name = "heading"

    def split(self, doc: Document) -> List[Chunk]:
        sections = self._sections(doc.markdown)
        chunks: List[Chunk] = []
        ordinal = 0
        for heading_path, body in sections:
            body = body.strip()
            if not body:
                continue
            for piece in self._pack(body):
                if len(piece.strip()) < self.config.min_chars:
                    continue
                chunks.append(self._make_chunk(doc, piece, ordinal, heading_path))
                ordinal += 1
        if not chunks:
            # No headings at all (common for PDF text): fall back to windows.
            return FixedChunker(self.config).split(doc)
        return chunks

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
