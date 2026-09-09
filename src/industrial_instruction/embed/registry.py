"""Embedder registry keyed by ``embed.backend``."""

from __future__ import annotations

from typing import Callable, Dict

from industrial_instruction.config import EmbedConfig
from industrial_instruction.embed.base import Embedder

_REGISTRY: Dict[str, Callable[[EmbedConfig], Embedder]] = {}


def register_embedder(name: str, factory: Callable[[EmbedConfig], Embedder]) -> None:
    """Plug in a custom embedder, e.g. an in-house domain model."""
    _REGISTRY[name.lower()] = factory


def get_embedder(config: EmbedConfig) -> Embedder:
    key = (config.backend or "sentence_transformers").lower()
    if key in _REGISTRY:
        return _REGISTRY[key](config)
    if key in ("sentence_transformers", "st", "local"):
        from industrial_instruction.embed.sentence_transformers_embedder import (
            SentenceTransformersEmbedder,
        )

        return SentenceTransformersEmbedder(config)
    if key == "openai":
        from industrial_instruction.embed.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder(config)
    if key in ("hash", "test"):
        from industrial_instruction.embed.hash_embedder import HashEmbedder

        return HashEmbedder(config)
    raise ValueError(
        f"Unknown embed backend {key!r}. Built-ins: sentence_transformers, openai, hash."
    )
