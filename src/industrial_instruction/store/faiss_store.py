"""FAISS-backed chunk store.

This single class replaces the three drifted copies of
``vector_store_module.py``. Differences that matter in practice:

* documents are embedded in batches, not one call per document;
* the index is created lazily from the embedder's real dimension;
* ``store_meta.json`` records the embedder, model, dimension and index type,
  and loading verifies them, so you cannot silently query an index built
  with a different model;
* the id map stores full :class:`Chunk` records, so retrieval returns
  provenance (source path, headings) and not just text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from industrial_instruction.config import Config, StoreConfig
from industrial_instruction.embed.base import Embedder
from industrial_instruction.embed.registry import get_embedder
from industrial_instruction.schemas import Chunk
from industrial_instruction.utils.io import ensure_dir, read_json, write_json
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "faiss is required for the vector store. Install with: "
            "pip install 'industrial-instruction[faiss]'"
        ) from exc
    return faiss


@dataclass
class SearchHit:
    """One retrieval result."""

    chunk: Chunk
    score: float
    rank: int


class FaissStore:
    """Persistent FAISS index over :class:`Chunk` records."""

    def __init__(
        self,
        embedder: Embedder,
        store_config: Optional[StoreConfig] = None,
        directory: Optional[str | Path] = None,
    ) -> None:
        self.embedder = embedder
        self.store_config = store_config or StoreConfig()
        self.directory = Path(directory).resolve() if directory else None
        self.index = None
        self.chunks: Dict[int, Chunk] = {}
        self._next_id = 0

    # ------------------------------------------------------------- factories

    @classmethod
    def from_config(cls, config: Config) -> "FaissStore":
        return cls(
            embedder=get_embedder(config.embed),
            store_config=config.store,
            directory=config.paths.resolve("index"),
        )

    # ------------------------------------------------------------------ index

    def _create_index(self, dimension: int):
        faiss = _require_faiss()
        kind = (self.store_config.index_type or "flat_ip").lower()
        if kind in ("flat_ip", "ip", "cosine"):
            return faiss.IndexFlatIP(dimension)
        if kind in ("flat_l2", "l2"):
            return faiss.IndexFlatL2(dimension)
        if kind == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, int(self.store_config.hnsw_m))
            index.metric_type = faiss.METRIC_INNER_PRODUCT
            return index
        raise ValueError(
            f"Unknown store.index_type {self.store_config.index_type!r}. "
            "Use 'flat_ip', 'flat_l2' or 'hnsw'."
        )

    def _ensure_index(self):
        if self.index is None:
            self.index = self._create_index(self.embedder.dimension)
        return self.index

    @property
    def size(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------ write

    def add_chunks(self, chunks: Sequence[Chunk], show_progress: bool = True) -> int:
        """Embed and index chunks in batches."""
        items = [c for c in chunks if c.text and c.text.strip()]
        if not items:
            return 0
        index = self._ensure_index()

        iterator: Iterable = self.embedder.batches([c.text for c in items])
        batch_list = list(iterator)
        if show_progress:
            from tqdm import tqdm

            batch_list = tqdm(batch_list, desc="embed", unit="batch")

        cursor = 0
        for batch in batch_list:
            vectors = self.embedder.encode_documents(list(batch))
            index.add(vectors)
            for chunk in items[cursor : cursor + len(batch)]:
                self.chunks[self._next_id] = chunk
                self._next_id += 1
            cursor += len(batch)
        logger.info("indexed %d chunks (total %d)", len(items), self.size)
        return len(items)

    # ------------------------------------------------------------------- read

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        if self.index is None or self.size == 0:
            return []
        vector = self.embedder.encode_queries([query])
        k = max(1, min(int(k), self.size))
        scores, ids = self.index.search(vector, k)
        hits: List[SearchHit] = []
        for rank, (raw_id, score) in enumerate(zip(ids[0], scores[0])):
            chunk = self.chunks.get(int(raw_id))
            if chunk is not None:
                hits.append(SearchHit(chunk=chunk, score=float(score), rank=rank))
        return hits

    def search_many(self, queries: Sequence[str], k: int = 5) -> List[List[SearchHit]]:
        """Batched search; one FAISS call for many queries."""
        if self.index is None or self.size == 0:
            return [[] for _ in queries]
        vectors = self.embedder.encode_queries(list(queries))
        k = max(1, min(int(k), self.size))
        scores, ids = self.index.search(vectors, k)
        results = []
        for row_ids, row_scores in zip(ids, scores):
            row: List[SearchHit] = []
            for rank, (raw_id, score) in enumerate(zip(row_ids, row_scores)):
                chunk = self.chunks.get(int(raw_id))
                if chunk is not None:
                    row.append(SearchHit(chunk=chunk, score=float(score), rank=rank))
            results.append(row)
        return results

    # ------------------------------------------------------------ persistence

    def _paths(self, directory: Optional[str | Path] = None):
        base = Path(directory).resolve() if directory else self.directory
        if base is None:
            raise ValueError("No store directory configured")
        return (
            base,
            base / self.store_config.index_filename,
            base / self.store_config.mapping_filename,
            base / self.store_config.meta_filename,
        )

    def save(self, directory: Optional[str | Path] = None) -> Path:
        faiss = _require_faiss()
        base, index_path, map_path, meta_path = self._paths(directory)
        ensure_dir(base)
        if self.index is None:
            raise RuntimeError("Nothing to save: index is empty")
        faiss.write_index(self.index, str(index_path))
        write_json(
            map_path,
            {str(k): v.model_dump(mode="json") for k, v in self.chunks.items()},
        )
        write_json(
            meta_path,
            {
                "embed_backend": self.embedder.config.backend,
                "embed_model": self.embedder.config.model,
                "dimension": self.embedder.dimension,
                "index_type": self.store_config.index_type,
                "normalize": self.embedder.config.normalize,
                "query_prefix": self.embedder.config.query_prefix,
                "document_prefix": self.embedder.config.document_prefix,
                "count": self.size,
            },
        )
        logger.info("saved index with %d chunks to %s", self.size, base)
        return base

    def load(
        self, directory: Optional[str | Path] = None, strict: bool = True
    ) -> "FaissStore":
        faiss = _require_faiss()
        base, index_path, map_path, meta_path = self._paths(directory)
        if not index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index at {index_path}. Run the index stage first "
                "(ii index)."
            )
        self.index = faiss.read_index(str(index_path))

        raw_map = read_json(map_path, default={}) if map_path.exists() else {}
        self.chunks = {
            int(key): Chunk.model_validate(value) for key, value in raw_map.items()
        }
        self._next_id = (max(self.chunks) + 1) if self.chunks else 0

        if meta_path.exists():
            meta = read_json(meta_path, default={})
            self._check_compatibility(meta, strict=strict)
        logger.info("loaded index with %d chunks from %s", self.size, base)
        return self

    def _check_compatibility(self, meta: dict, strict: bool) -> None:
        """Guard against querying an index built with another model."""
        problems = []
        saved_dim = meta.get("dimension")
        if saved_dim and int(saved_dim) != int(self.embedder.dimension):
            problems.append(
                f"dimension {saved_dim} != current {self.embedder.dimension}"
            )
        saved_model = meta.get("embed_model")
        if saved_model and saved_model != self.embedder.config.model:
            problems.append(
                f"model {saved_model!r} != current {self.embedder.config.model!r}"
            )
        if not problems:
            return
        message = (
            "Index/embedder mismatch: "
            + "; ".join(problems)
            + ". Rebuild the index or point embed.* at the original model."
        )
        if strict and saved_dim and int(saved_dim) != int(self.embedder.dimension):
            raise RuntimeError(message)
        logger.warning(message)

    # ------------------------------------------------------------- utilities

    def all_chunks(self) -> List[Chunk]:
        return [self.chunks[key] for key in sorted(self.chunks)]

    def stats(self) -> dict:
        docs = {c.doc_id for c in self.chunks.values()}
        return {
            "chunks": self.size,
            "documents": len(docs),
            "dimension": self.embedder.dimension if self.index else None,
            "index_type": self.store_config.index_type,
            "embed_model": self.embedder.config.model,
        }
