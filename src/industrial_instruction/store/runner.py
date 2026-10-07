"""Index-stage orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence

from industrial_instruction.chunk.chunker import chunk_documents, load_chunks
from industrial_instruction.config import Config
from industrial_instruction.schemas import Chunk, StageReport
from industrial_instruction.store.faiss_store import FaissStore
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


def _documents_exist(config: Config) -> bool:
    documents = config.paths.resolve("documents")
    path = documents / "documents.jsonl" if documents.is_dir() else documents
    return path.exists() and path.stat().st_size > 0


def build_index(
    config: Config, chunks: Optional[Sequence[Chunk]] = None
) -> StageReport:
    """Embed chunks and persist a FAISS index."""
    t0 = time.time()
    chunks_path: Path = config.paths.resolve("chunks")
    items = list(chunks) if chunks is not None else load_chunks(chunks_path)

    if not items and chunks is None and _documents_exist(config):
        # Extracted documents are present but never chunked (for example an
        # artifacts dir from before the chunk stage existed). Chunk now rather
        # than making the user re-run an expensive extract.
        logger.warning("%s is missing or empty; chunking documents now", chunks_path)
        chunk_documents(config)
        items = load_chunks(chunks_path)

    if not items:
        raise RuntimeError(
            f"No chunks to index (looked in {chunks_path}). "
            "Run 'ii extract' then 'ii chunk', or just 'ii run'."
        )

    store = FaissStore.from_config(config)
    added = store.add_chunks(items)
    directory = store.save()

    report = StageReport(
        stage="index",
        inputs=len(items),
        outputs=added,
        seconds=round(time.time() - t0, 2),
        details={"index_dir": str(directory), **store.stats()},
    )
    logger.info(
        "index: %d chunks embedded with %s in %.1fs",
        added,
        store.embedder.name,
        report.seconds,
    )
    return report


PASSAGE_SUFFIXES = {".jsonl"}
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def index_corpus(
    config: Config,
    corpus: str | Path,
    text_field: str = "text",
    id_field: Optional[str] = "id",
) -> List[StageReport]:
    """Build the project index from a corpus in one step.

    ``corpus`` may be a directory (or single file) of PDFs - extracted with
    ``extract.backend`` - or of markdown/text files, or a ``.jsonl`` of
    ready-made passages (one ``{"text": ...}`` per line), which are indexed
    as they are. Chunks are written to ``paths.chunks`` too, so generation
    with ``seeds.source: self`` can use the same corpus.
    """
    from industrial_instruction.chunk.chunker import chunk_documents
    from industrial_instruction.extract.runner import extract_documents
    from industrial_instruction.utils.io import iter_jsonl, stable_id, write_jsonl

    path = Path(corpus)
    if not path.is_absolute():
        path = (Path(config.paths.root) / path) if not path.exists() else path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")

    if path.is_file() and path.suffix.lower() in PASSAGE_SUFFIXES:
        chunks: List[Chunk] = []
        for i, row in enumerate(iter_jsonl(path)):
            text = str(row.get(text_field) or "").strip() if isinstance(row, dict) else str(row)
            if not text:
                continue
            key = row.get(id_field) if isinstance(row, dict) and id_field else None
            chunks.append(
                Chunk(
                    id=str(key) if key is not None else stable_id(path.name, i, text[:256]),
                    doc_id=str(row.get("doc_id", path.stem)) if isinstance(row, dict) else path.stem,
                    text=text,
                    ordinal=i,
                    n_chars=len(text),
                    source_path=str(path),
                    meta={k: v for k, v in row.items() if k not in (text_field, id_field)}
                    if isinstance(row, dict) else {},
                )
            )
        write_jsonl(config.paths.resolve("chunks"), chunks)
        logger.info("index: %d passages from %s (indexed as given, not re-chunked)", len(chunks), path)
        return [build_index(config, chunks=chunks)]

    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    has_pdf = any(p.suffix.lower() == ".pdf" for p in files)
    has_text = any(p.suffix.lower() in TEXT_SUFFIXES for p in files)
    if not has_pdf and not has_text:
        raise ValueError(f"{path}: no PDF, markdown or text files found")
    data = config.model_dump(mode="json")
    data["paths"]["pdfs"] = str(path)
    if not has_pdf:
        data["extract"]["backend"] = "markdown"
    run_config = Config.model_validate(data)
    return [
        extract_documents(run_config),
        chunk_documents(run_config),
        build_index(run_config),
    ]


def load_store(config: Config, strict: bool = True) -> FaissStore:
    """Load a previously built index for retrieval."""
    return FaissStore.from_config(config).load(strict=strict)
