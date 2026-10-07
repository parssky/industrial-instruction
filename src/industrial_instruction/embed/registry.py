"""Embedder registry."""

from __future__ import annotations

from typing import Callable, Dict

from industrial_instruction.config import EmbedConfig
from industrial_instruction.embed.base import Embedder

_REGISTRY: Dict[str, Callable[[EmbedConfig], Embedder]] = {}


def register_embedder(name: str, factory: Callable[[EmbedConfig], Embedder]) -> None:
    _REGISTRY[name.lower()] = factory


def _builtin(name: str) -> Callable[[EmbedConfig], Embedder]:
    if name in ("sentence_transformers", "st", "local", "default"):
        from industrial_instruction.embed.sentence_transformers_embedder import (
            SentenceTransformersEmbedder,
        )

        return SentenceTransformersEmbedder
    if name in ("openai", "api", "compatible"):
        from industrial_instruction.embed.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder
    if name in ("hash", "test", "offline"):
        from industrial_instruction.embed.hash_embedder import HashEmbedder

        return HashEmbedder
    raise KeyError(name)


def get_embedder(config: EmbedConfig) -> Embedder:
    """Instantiate the embedder named by ``config.backend``."""
    name = (config.backend or "sentence_transformers").lower()
    if name in _REGISTRY:
        return _REGISTRY[name](config)
    try:
        return _builtin(name)(config)
    except KeyError:
        raise ValueError(
            f"Unknown embed backend {config.backend!r}. Built-ins: "
            "'sentence_transformers', 'openai', 'hash'. "
            f"Registered: {sorted(_REGISTRY)}"
        ) from None
