"""Index-stage orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

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
        config.embed.model,
        report.seconds,
    )
    return report


def load_store(config: Config, strict: bool = True) -> FaissStore:
    """Load a previously built index for retrieval."""
    return FaissStore.from_config(config).load(strict=strict)
