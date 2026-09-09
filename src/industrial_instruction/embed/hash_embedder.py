"""Deterministic hashing embedder.

Not for research use: it exists so tests and smoke runs of the full pipeline
work offline with no model download and no GPU.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, Sequence

import numpy as np

from industrial_instruction.config import EmbedConfig
from industrial_instruction.embed.base import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder(Embedder):
    name = "hash"

    def __init__(self, config: Optional[EmbedConfig] = None) -> None:
        super().__init__(config)
        self._dim = int(self.config.dimension or 256)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _TOKEN.findall((text or "").lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self._dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, idx] += sign
        return out
