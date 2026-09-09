"""Backend registry so extractors are selected by name from config."""

from __future__ import annotations

from typing import Callable, Dict, Type

from industrial_instruction.config import ExtractConfig
from industrial_instruction.extract.base import Extractor

_REGISTRY: Dict[str, Callable[[ExtractConfig], Extractor]] = {}


def register_extractor(name: str, factory: Callable[[ExtractConfig], Extractor]) -> None:
    """Register a custom extractor factory under ``name``."""
    _REGISTRY[name.lower()] = factory


def _builtin(name: str) -> Type[Extractor]:
    name = name.lower()
    if name in ("pymupdf", "fitz", "default"):
        from industrial_instruction.extract.pymupdf_extractor import PyMuPDFExtractor

        return PyMuPDFExtractor
    if name == "pdfplumber":
        from industrial_instruction.extract.pdfplumber_extractor import (
            PdfplumberExtractor,
        )

        return PdfplumberExtractor
    if name == "markdown":
        from industrial_instruction.extract.markdown_extractor import MarkdownExtractor

        return MarkdownExtractor
    raise ValueError(
        f"Unknown extract backend {name!r}. Built-ins: pymupdf, pdfplumber, markdown. "
        "Register your own with register_extractor()."
    )


def get_extractor(config: ExtractConfig) -> Extractor:
    """Instantiate the extractor named by ``config.backend``."""
    key = (config.backend or "pymupdf").lower()
    if key in _REGISTRY:
        return _REGISTRY[key](config)
    return _builtin(key)(config)
