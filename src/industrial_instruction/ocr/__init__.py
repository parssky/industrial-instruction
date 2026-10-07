"""Pluggable OCR for scanned and image-only PDF pages.

Attach any model with one function::

    from industrial_instruction.ocr import OCRPage, register_ocr

    def my_ocr(page: OCRPage) -> str:
        return call_my_model(page.image)      # PNG bytes in, markdown out

    register_ocr("my-ocr", my_ocr)            # extract.ocr.backend: my-ocr

or skip registration and set ``extract.ocr.backend: my_pkg.ocr:my_ocr``.
See ``examples/custom_ocr.py``.
"""

from industrial_instruction.ocr.base import OCRError, OCRFunction, OCRPage, clean_markdown
from industrial_instruction.ocr.registry import (
    available_ocr_backends,
    get_ocr,
    ocr_backend,
    register_ocr,
)
from industrial_instruction.ocr.runner import OCRResult, PageOCR

__all__ = [
    "OCRError",
    "OCRFunction",
    "OCRPage",
    "OCRResult",
    "PageOCR",
    "available_ocr_backends",
    "clean_markdown",
    "get_ocr",
    "ocr_backend",
    "register_ocr",
]
