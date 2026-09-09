"""Embedder protocol.

Document and query embeddings are separate methods because instruction-tuned
embedders (EmbeddingGemma, E5, BGE, Jina) expect asymmetric prefixes. The
original notebook encoded both sides identically, which silently costs recall.
"""

from __future__ import annotations

import abc
from typing import List, Optional, Sequence

import numpy as np

from industrial_instruction.config import EmbedConfig


class Embedder(abc.ABC):
    name: str = "base"

    def __init__(self, config: Optional[EmbedConfig] = None) -> None:
        self.config = config or EmbedConfig()

    @abc.abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed raw texts. Returns ``(n, dim)`` float32."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Embedding dimensionality."""

    # -- public API --------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        prefix = self.config.document_prefix or ""
        return self._finish(self.embed([prefix + t for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        prefix = self.config.query_prefix or ""
        return self._finish(self.embed([prefix + text]))

    # -- helpers -----------------------------------------------------------

    def _finish(self, vectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self.config.normalize:
            arr = normalize(arr)
        return arr

    def fingerprint(self) -> dict:
        """Recorded in the store metadata to catch index/model mismatches."""
        return {
            "backend": self.name,
            "model": self.config.model,
            "dimension": self.dimension,
            "normalize": self.config.normalize,
            "query_prefix": self.config.query_prefix,
            "document_prefix": self.config.document_prefix,
        }


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product equals cosine similarity."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-9)).astype("float32")


def batched(items: Sequence[str], size: int) -> List[Sequence[str]]:
    size = max(int(size), 1)
    return [items[i : i + size] for i in range(0, len(items), size)]
