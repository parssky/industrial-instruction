"""Attach your own OCR model to the extract stage.

Your OCR is one function: it gets an ``OCRPage`` (PNG bytes plus page
number, dpi, the PDF's own text layer and your ``extract.ocr.options``) and
returns the page as markdown. This file shows the three ways to attach one.

    python examples/custom_ocr.py path/to/scanned_pdfs

Assumes an OCR/vision model served by vLLM, e.g.:

    vllm serve nanonets/Nanonets-OCR-s --port 8001
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI

from industrial_instruction import Config, Pipeline
from industrial_instruction.ocr import OCRPage, ocr_backend, register_ocr

VLLM_URL = os.environ.get("OCR_URL", "http://localhost:8001/v1")


# --- 1. A plain function -----------------------------------------------------
#
# Anything goes inside: an HTTP call, a local model, a cloud OCR API.
# Return markdown (headings as #, tables as | a | b |). Return "" or None if
# the page has nothing - the PDF's text layer is kept. Raise on errors - the
# page falls back to its text layer and the failure is counted in the
# manifest; the rest of the document carries on.

client = OpenAI(base_url=VLLM_URL, api_key="not-needed")


def nanonets_ocr(page: OCRPage) -> str:
    response = client.chat.completions.create(
        model="nanonets/Nanonets-OCR-s",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": page.data_url()}},
                    {
                        "type": "text",
                        "text": "Extract the text from the above document as if you were "
                        "reading it naturally. Return the tables in markdown format.",
                    },
                ],
            }
        ],
        max_tokens=page.options.get("max_tokens", 4096),
        temperature=0.0,
    )
    return response.choices[0].message.content


register_ocr("nanonets", nanonets_ocr)


# --- 2. A factory, for backends that need setup once per run ----------------
#
# Called once with the OCRConfig; returns the page function. Use it to load
# weights or open clients a single time.


@ocr_backend("my-local-model", factory=True)
def build_local_model(config):
    weights = config.options.get("weights", "path/to/weights")
    # model = load_my_model(weights)   # once
    def run(page: OCRPage) -> str:
        # return model.transcribe(page.to_pil())
        return f"(transcription of page {page.page_number} using {weights})"

    return run


# --- 3. No Python wiring at all ----------------------------------------------
#
# Put the function in a file next to your config and point the YAML at it:
#
#   extract:
#     ocr:
#       mode: auto
#       backend: ocr/nanonets.py:nanonets_ocr      # or my_pkg.ocr:nanonets_ocr
#
# This works from the CLI too:
#
#   ii extract --set extract.ocr.mode=auto \
#              --set extract.ocr.backend=ocr/nanonets.py:nanonets_ocr
#
# And if your model speaks the OpenAI vision API, no function is needed:
#
#   extract:
#     ocr: { mode: auto, backend: openai, base_url: http://localhost:8001/v1,
#            model: nanonets/Nanonets-OCR-s }


def main(pdf_dir: str) -> int:
    config = Config.from_dict(
        {
            "paths": {"root": ".", "pdfs": pdf_dir},
            "extract": {
                "ocr": {
                    "mode": "auto",  # only pages without a text layer
                    "backend": "nanonets",
                    "max_workers": 8,  # concurrent pages -> full vLLM batches
                }
            },
        }
    )
    report = Pipeline(config).run_extract()
    print(
        f"{report.outputs} documents, {report.details['ocr_pages']} pages OCR'd, "
        f"{report.details['ocr_failed']} failed -> {report.details['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
