"""Deterministic hashing embedder.

Not for production retrieval quality - it exists so the whole pipeline can be
exercised in unit tests and offline smoke runs with no model download, no GPU
and no network. ``embed.dimension`` controls the width.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from industrial_instruction.config import EmbedConfig
from industrial_instruction.embed.base import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder(Embedder):
    def __init__(self, config: EmbedConfig) -> None:
        super().__init__(config)
        self._dimension = int(config.dimension or 256)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dimension), dtype="float32")
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                digest = hashlib.blake2b(
                    token.encode("utf-8"), digest_size=8
                ).digest()
                index = int.from_bytes(digest[:4], "little") % self._dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[row, index] += sign
        return out
