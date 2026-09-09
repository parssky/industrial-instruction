"""Embedder backed by any OpenAI-compatible /embeddings endpoint."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from industrial_instruction.config import EmbedConfig
from industrial_instruction.embed.base import Embedder, batched

_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(Embedder):
    """Useful when no GPU is available, or to reuse a hosted vLLM embedder."""

    name = "openai"

    def __init__(self, config: Optional[EmbedConfig] = None) -> None:
        super().__init__(config)
        self._client = None
        self._dim = self.config.dimension or _KNOWN_DIMS.get(self.config.model)

    def _load(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install openai") from exc
        kwargs = {"api_key": os.environ.get(self.config.api_key_env) or "no-key"}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Probe once; cheaper than requiring users to know the dimension.
            self._dim = int(self.embed(["dimension probe"]).shape[1])
        return int(self._dim)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        client = self._load()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        out = []
        for group in batched(list(texts), self.config.batch_size):
            resp = client.embeddings.create(model=self.config.model, input=list(group))
            out.extend(item.embedding for item in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        self._dim = int(arr.shape[1])
        return arr
