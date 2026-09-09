"""PDF -> markdown extraction stage (text + tables, images removed)."""

from industrial_instruction.extract.base import Extractor, ExtractionError
from industrial_instruction.extract.registry import get_extractor, register_extractor
from industrial_instruction.extract.runner import extract_documents

__all__ = [
    "Extractor",
    "ExtractionError",
    "get_extractor",
    "register_extractor",
    "extract_documents",
]
