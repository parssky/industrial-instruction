"""Local embedder via sentence-transformers (default: EmbeddingGemma-300m)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from industrial_instruction.config import EmbedConfig
from industrial_instruction.embed.base import Embedder
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformersEmbedder(Embedder):
    """Batched local encoding. Accepts a hub id or a local model directory."""

    name = "sentence_transformers"

    def __init__(self, config: Optional[EmbedConfig] = None) -> None:
        super().__init__(config)
        self._model = None
        self._dim: Optional[int] = self.config.dimension

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "sentence-transformers is required for this backend. Install with: "
                "pip install 'industrial-instruction[local-embed]'"
            ) from exc
        logger.info("loading embedder %s", self.config.model)
        kwargs = {}
        if self.config.device:
            kwargs["device"] = self.config.device
        self._model = SentenceTransformer(self.config.model, **kwargs)
        self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
        return int(self._dim)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,  # normalization handled centrally
        )
        return np.asarray(vectors, dtype=np.float32)
