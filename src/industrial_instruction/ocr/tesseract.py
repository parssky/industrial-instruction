"""Built-in backend: Tesseract, for offline OCR with no GPU or server.

Also the smallest complete example of a backend - a custom one looks the
same. Needs ``pip install pytesseract pillow`` and the ``tesseract`` binary.
``extract.ocr.options.lang`` selects the language pack (default ``eng``).
Tesseract returns plain text, so headings and tables are not recovered;
use a vision model for those.
"""

from __future__ import annotations

from industrial_instruction.ocr.base import OCRError, OCRPage
from industrial_instruction.ocr.registry import ocr_backend


@ocr_backend("tesseract")
def tesseract(page: OCRPage) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise OCRError(
            "The tesseract OCR backend needs: pip install pytesseract pillow "
            "(and the tesseract binary)"
        ) from exc
    lang = page.options.get("lang", "eng")
    return pytesseract.image_to_string(page.to_pil(), lang=lang)
