"""Index-stage orchestration."""

from __future__ import annotations

import time
from typing import Optional, Sequence

from industrial_instruction.chunk.chunker import load_chunks
from industrial_instruction.config import Config
from industrial_instruction.schemas import Chunk, StageReport
from industrial_instruction.store.faiss_store import FaissStore
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


def build_index(
    config: Config, chunks: Optional[Sequence[Chunk]] = None
) -> StageReport:
    """Embed chunks and persist a FAISS index."""
    t0 = time.time()
    items = list(chunks) if chunks is not None else load_chunks(
        config.paths.resolve("chunks")
    )
    if not items:
        raise RuntimeError(
            "No chunks to index. Run the extract stage first (ii extract)."
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
