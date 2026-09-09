"""Batch driver for the index stage."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence

from industrial_instruction.config import Config
from industrial_instruction.schemas import Chunk, StageReport
from industrial_instruction.store.faiss_store import FaissStore
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


def load_store(config: Config) -> FaissStore:
    """Open the configured index for querying."""
    return FaissStore.open(
        config.paths.resolve("index"), config.embed, config.store
    )


def build_index(
    config: Config,
    chunks: Optional[Sequence[Chunk]] = None,
    rebuild: bool = False,
) -> StageReport:
    """Embed chunks and persist the FAISS index.

    Re-running is incremental by default: chunk ids already present are
    skipped, so adding new PDFs does not re-embed the whole corpus.
    """
    t0 = time.time()
    if chunks is None:
        from industrial_instruction.chunk.chunker import load_chunks

        chunks = load_chunks(config.paths.resolve("chunks"))
    chunk_list: List[Chunk] = list(chunks)

    directory = config.paths.resolve("index")
    if rebuild:
        _clear(directory, config)

    store = FaissStore.open(directory, config.embed, config.store)
    added = store.add_chunks(chunk_list, skip_existing=True)
    store.save()

    report = StageReport(
        stage="index",
        inputs=len(chunk_list),
        outputs=added,
        seconds=round(time.time() - t0, 2),
        details={
            "index_dir": str(directory),
            "total_vectors": len(store),
            "skipped_existing": len(chunk_list) - added,
            "embedder": store.embedder.fingerprint(),
        },
    )
    logger.info(
        "index: added %d chunks (%d total vectors) in %.1fs",
        added,
        len(store),
        report.seconds,
    )
    return report


def _clear(directory: Path, config: Config) -> None:
    for name in (
        config.store.index_filename,
        config.store.mapping_filename,
        config.store.meta_filename,
    ):
        target = directory / name
        if target.exists():
            target.unlink()
    logger.info("cleared existing index at %s", directory)
