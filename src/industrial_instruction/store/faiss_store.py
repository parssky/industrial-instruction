"""FAISS-backed chunk store.

This is the single, canonical replacement for the three drifted copies of
``vector_store_module.py``. Improvements over the original ``Retriever``:

* batched adds instead of one ``index.add`` call per document
* stores full :class:`Chunk` records, not bare strings, so provenance
  (document id, page/heading) survives into the dataset
* persists model metadata and refuses to load an index built with a
  different embedder/dimension
* returns similarity scores, and supports duplicate-safe upserts
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from industrial_instruction.config import EmbedConfig, StoreConfig
from industrial_instruction.embed.base import Embedder, normalize
from industrial_instruction.embed.registry import get_embedder
from industrial_instruction.schemas import Chunk
from industrial_instruction.utils.io import ensure_dir, read_json, write_json
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A search hit: the chunk plus its similarity score and rank."""

    chunk: Chunk
    score: float
    rank: int

    @property
    def text(self) -> str:
        return self.chunk.text


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "faiss is required for the vector store. Install with: "
            "pip install 'industrial-instruction[faiss]'"
        ) from exc
    return faiss


class FaissStore:
    """Persistent chunk index.

    ``FaissStore.open(dir, embed_config, store_config)`` loads an existing
    index if present, otherwise creates an empty one.
    """

    def __init__(
        self,
        directory: str | Path,
        embedder: Embedder,
        config: Optional[StoreConfig] = None,
    ) -> None:
        self.dir = Path(directory)
        self.config = config or StoreConfig()
        self.embedder = embedder
        self.index = None
        self.id_to_chunk: Dict[int, Chunk] = {}
        self._chunk_ids: Dict[str, int] = {}
        self.next_id = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def open(
        cls,
        directory: str | Path,
        embed_config: Optional[EmbedConfig] = None,
        store_config: Optional[StoreConfig] = None,
        embedder: Optional[Embedder] = None,
    ) -> "FaissStore":
        embedder = embedder or get_embedder(embed_config or EmbedConfig())
        store = cls(directory, embedder, store_config)
        if store.index_path.exists():
            store.load()
        else:
            store._new_index()
        return store

    @property
    def index_path(self) -> Path:
        return self.dir / self.config.index_filename

    @property
    def mapping_path(self) -> Path:
        return self.dir / self.config.mapping_filename

    @property
    def meta_path(self) -> Path:
        return self.dir / self.config.meta_filename

    def _new_index(self) -> None:
        faiss = _import_faiss()
        dim = self.embedder.dimension
        kind = (self.config.index_type or "flat_ip").lower()
        if kind == "flat_ip":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "flat_l2":
            self.index = faiss.IndexFlatL2(dim)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m)
        else:
            raise ValueError(
                f"Unknown index_type {kind!r}. Use flat_ip, flat_l2 or hnsw."
            )
        logger.info("created %s index (dim=%d)", kind, dim)

    # -- writing -----------------------------------------------------------

    def add_chunks(self, chunks: Sequence[Chunk], skip_existing: bool = True) -> int:
        """Embed and index chunks in batches. Returns the number added."""
        if self.index is None:
            self._new_index()
        pending = [
            c
            for c in chunks
            if not (skip_existing and c.id in self._chunk_ids)
        ]
        if not pending:
            return 0

        batch = max(self.embedder.config.batch_size, 1)
        added = 0
        for start in range(0, len(pending), batch):
            group = pending[start : start + batch]
            texts = [
                c.as_context(include_heading=True) for c in group
            ]
            vectors = self.embedder.embed_documents(texts)
            self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            for chunk in group:
                self.id_to_chunk[self.next_id] = chunk
                self._chunk_ids[chunk.id] = self.next_id
                self.next_id += 1
                added += 1
        return added

    # -- reading -----------------------------------------------------------

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        """Return the top-``k`` chunks for a query, best first."""
        if self.index is None or self.index.ntotal == 0:
            return []
        vector = self.embedder.embed_query(query)
        k = int(min(max(k, 1), self.index.ntotal))
        scores, indices = self.index.search(
            np.ascontiguousarray(vector, dtype=np.float32), k
        )
        hits: List[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            chunk = self.id_to_chunk.get(int(idx))
            if chunk is None:
                continue
            hits.append(RetrievedChunk(chunk=chunk, score=float(score), rank=rank))
        return hits

    def __len__(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def chunks(self) -> List[Chunk]:
        return list(self.id_to_chunk.values())

    # -- persistence -------------------------------------------------------

    def save(self) -> Path:
        faiss = _import_faiss()
        ensure_dir(self.dir)
        faiss.write_index(self.index, str(self.index_path))
        write_json(
            self.mapping_path,
            {str(k): v.model_dump(mode="json") for k, v in self.id_to_chunk.items()},
        )
        write_json(
            self.meta_path,
            {
                "n_vectors": len(self),
                "index_type": self.config.index_type,
                "embedder": self.embedder.fingerprint(),
            },
        )
        logger.info("saved index with %d vectors to %s", len(self), self.dir)
        return self.index_path

    def load(self) -> "FaissStore":
        faiss = _import_faiss()
        self.index = faiss.read_index(str(self.index_path))
        self.id_to_chunk = {}
        self._chunk_ids = {}
        if self.mapping_path.exists():
            raw = read_json(self.mapping_path)
            for key, value in raw.items():
                chunk = _coerce_chunk(value)
                self.id_to_chunk[int(key)] = chunk
                self._chunk_ids[chunk.id] = int(key)
        self.next_id = (max(self.id_to_chunk) + 1) if self.id_to_chunk else 0
        self._check_compatibility()
        logger.info("loaded index with %d vectors from %s", len(self), self.dir)
        return self

    def _check_compatibility(self) -> None:
        """Fail loudly on an embedder/index mismatch instead of silently
        returning garbage neighbours."""
        if not self.meta_path.exists():
            return
        meta = read_json(self.meta_path) or {}
        stored = (meta.get("embedder") or {}).get("dimension")
        current = self.embedder.dimension
        if stored and int(stored) != int(current):
            raise ValueError(
                f"Index at {self.dir} was built with dimension {stored}, but the "
                f"configured embedder produces {current}. Rebuild the index or "
                "restore the original embed.model."
            )
        stored_model = (meta.get("embedder") or {}).get("model")
        if stored_model and stored_model != self.embedder.config.model:
            logger.warning(
                "index was built with model %r but config uses %r; retrieval "
                "quality may degrade",
                stored_model,
                self.embedder.config.model,
            )


def _coerce_chunk(value) -> Chunk:
    """Accept both new Chunk records and the legacy plain-string mapping."""
    if isinstance(value, str):
        from industrial_instruction.utils.io import stable_id

        return Chunk(
            id=stable_id(value[:256]),
            doc_id="legacy",
            text=value,
            n_chars=len(value),
            meta={"legacy": True},
        )
    if isinstance(value, dict) and "text" in value and "id" in value:
        return Chunk.model_validate(value)
    if isinstance(value, dict):
        from industrial_instruction.utils.io import stable_id

        text = value.get("text") or value.get("content") or str(value)
        return Chunk(
            id=str(value.get("id") or stable_id(text[:256])),
            doc_id=str(value.get("doc_id", "legacy")),
            text=text,
            n_chars=len(text),
            meta={"legacy": True},
        )
    raise TypeError(f"Cannot coerce mapping entry of type {type(value)!r} to Chunk")
