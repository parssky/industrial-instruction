"""Embedder interface.

Documents and queries are encoded through separate methods because modern
retrieval models (including EmbeddingGemma, the default) expect different
instruction prefixes for each side. The original code used one ``_encode``
for both, which silently degrades recall.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from industrial_instruction.config import EmbedConfig


class Embedder(ABC):
    """Base class for all embedding backends."""

    def __init__(self, config: EmbedConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector width; required to build the FAISS index."""

    @abstractmethod
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of raw strings into a ``(n, dim)`` float32 array."""

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"{type(self).__name__}({self.config.model})"

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        prefix = self.config.document_prefix or ""
        return self._finalize(self._encode([prefix + t for t in texts]))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefix = self.config.query_prefix or ""
        return self._finalize(self._encode([prefix + t for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]

    # ------------------------------------------------------------------

    def _finalize(self, vectors: np.ndarray) -> np.ndarray:
        """Cast to float32 and L2-normalize so inner product == cosine."""
        array = np.asarray(vectors, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if self.config.normalize:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            array = array / norms
        return array

    def batches(self, texts: Sequence[str]) -> List[Sequence[str]]:
        size = max(int(self.config.batch_size or 32), 1)
        return [texts[i : i + size] for i in range(0, len(texts), size)]
