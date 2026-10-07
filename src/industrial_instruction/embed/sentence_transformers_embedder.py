"""Local embeddings via sentence-transformers (default backend).

The model is resolved from config, so it can be a Hub id such as
``google/embeddinggemma-300m`` or a local directory. The original module
hardcoded ``../models/embeddinggemma-300m``, which only worked from one
notebook's working directory.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from industrial_instruction.config import EmbedConfig
from industrial_instruction.embed.base import Embedder
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformersEmbedder(Embedder):
    def __init__(self, config: EmbedConfig) -> None:
        super().__init__(config)
        self._model = None
        self._dimension: Optional[int] = config.dimension

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "sentence-transformers is required for the default embedder. "
                "Install with: pip install 'industrial-instruction[local-embed]' "
                "or set embed.backend to 'openai'."
            ) from exc
        logger.info("loading embedding model %s", self.config.model)
        kwargs = {}
        if self.config.device:
            kwargs["device"] = self.config.device
        self._model = SentenceTransformer(self.config.model, **kwargs)
        self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_model()
        return int(self._dimension)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        return model.encode(
            list(texts),
            batch_size=max(int(self.config.batch_size or 32), 1),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
