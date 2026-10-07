"""The OCR contract: one function, page image in, markdown out.

Any OCR model plugs in by writing a function with this signature::

    def my_ocr(page: OCRPage) -> str:
        ...  # call your model on page.image (PNG bytes)
        return markdown

and attaching it with ``register_ocr("my-ocr", my_ocr)`` or by pointing
``extract.ocr.backend`` at ``my_pkg.module:my_ocr``. Nothing else about the
pipeline needs to know which model produced the text.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class OCRPage:
    """One rendered page handed to an OCR function."""

    image: bytes  # PNG
    page_number: int  # 1-based
    source_path: str
    dpi: int = 200
    width: int = 0  # pixels
    height: int = 0
    text_layer: str = ""  # whatever text the PDF itself had (often empty)
    options: Dict[str, Any] = field(default_factory=dict)  # extract.ocr.options

    mime_type = "image/png"

    @property
    def image_b64(self) -> str:
        return base64.b64encode(self.image).decode("ascii")

    def data_url(self) -> str:
        """``data:image/png;base64,...`` - what OpenAI-style vision APIs take."""
        return f"data:{self.mime_type};base64,{self.image_b64}"

    def to_pil(self):
        """The page as a ``PIL.Image`` (requires Pillow)."""
        from PIL import Image

        return Image.open(io.BytesIO(self.image))


#: What a backend is: a callable from page to markdown. ``None`` or an empty
#: string means "nothing found" and the PDF's own text layer is kept.
OCRFunction = Callable[[OCRPage], Optional[str]]


class OCRError(RuntimeError):
    """Raised when an OCR backend cannot be loaded or configured."""


_FENCE = re.compile(r"^\s*```(?:markdown|md)?\s*\n(.*?)\n?```\s*$", re.S | re.IGNORECASE)


def clean_markdown(text: Optional[str]) -> str:
    """Normalize model output: unwrap a ```markdown fence, trim whitespace.

    Vision models often wrap the whole page in a code fence even when told
    not to; left in place it would turn the page into one code block.
    """
    if not text:
        return ""
    text = text.strip()
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()
    return text
