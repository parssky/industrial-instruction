"""Embeddings from any OpenAI-compatible embeddings endpoint.

Useful when a team cannot run a local GPU model, or already serves an
embedding model behind vLLM / TEI.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from industrial_instruction.config import EmbedConfig
from industrial_instruction.embed.base import Embedder
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)

_KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(Embedder):
    def __init__(self, config: EmbedConfig) -> None:
        super().__init__(config)
        self._client = None
        self._dimension: Optional[int] = config.dimension or _KNOWN_DIMENSIONS.get(
            config.model
        )

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install openai") from exc
        import os

        kwargs = {"api_key": os.environ.get(self.config.api_key_env, "no-key")}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self._encode(["dimension probe"])[0])
        return int(self._dimension)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        client = self._ensure_client()
        vectors = []
        for batch in self.batches(list(texts)):
            last_error = None
            for attempt in range(3):
                try:
                    response = client.embeddings.create(
                        model=self.config.model, input=list(batch)
                    )
                    vectors.extend(item.embedding for item in response.data)
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    time.sleep(2**attempt)
            if last_error is not None:
                raise RuntimeError(f"embedding request failed: {last_error}")
        return np.asarray(vectors, dtype="float32")
