# OCR for scanned pages

Pages with no text layer can be sent to an OCR model during
[extraction](extract.md). **Any model plugs in as one function: page image
in, markdown out.**

## Turn it on

```yaml
extract:
  ocr:
    mode: auto          # off | auto | always
    backend: openai     # a registered name, or module:function / file.py:function
```

| `mode` | Pages sent to OCR |
|---|---|
| `off` | none (default) |
| `auto` | pages with fewer than `min_chars_per_page` (50) characters of text: scans and image-only pages |
| `always` | every page; the OCR output replaces the PDF's own text |

## A vision model on vLLM: no code

The built-in `openai` backend sends each page to any OpenAI-compatible
vision endpoint:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001
```

```yaml
extract:
  ocr:
    mode: auto
    backend: openai
    base_url: http://localhost:8001/v1
    model: Qwen/Qwen2.5-VL-7B-Instruct
    max_workers: 8        # pages in flight at once
    dpi: 200
```

The packaged prompt asks for faithful markdown: headings, markdown tables,
and no headers, footers or page numbers. OCR-specific models trained on a
particular instruction (olmOCR, Nanonets-OCR) work best with their own:
set `prompt`. Use `prompt: ""` to send the image alone.

## Your own model: one function

```python
from industrial_instruction.ocr import OCRPage, register_ocr

def my_ocr(page: OCRPage) -> str:
    return call_my_model(page.image)           # return markdown

register_ocr("my-ocr", my_ocr)                 # extract.ocr.backend: my-ocr
```

`OCRPage` gives you:

| Attribute | |
|---|---|
| `image` | the page as PNG bytes |
| `data_url()` | `data:image/png;base64,...`, as OpenAI-style vision APIs take it |
| `to_pil()` | a `PIL.Image` (needs Pillow) |
| `page_number`, `source_path`, `dpi`, `width`, `height` | where the page comes from |
| `text_layer` | whatever text the PDF itself had, often empty |
| `options` | your `extract.ocr.options` dict |

Return `""` or `None` when a page has no text: the PDF's text layer is kept.
Raise an exception on errors: that page keeps its text layer and the failure
is counted, but the document carries on.

### Three ways to attach it

=== "Registered name"

    ```python
    register_ocr("my-ocr", my_ocr)
    ```
    ```yaml
    extract: { ocr: { mode: auto, backend: my-ocr } }
    ```

=== "Import path"

    ```yaml
    extract: { ocr: { mode: auto, backend: my_pkg.ocr:my_ocr } }
    ```

=== "File next to the config"

    ```yaml
    extract: { ocr: { mode: auto, backend: ocr/my_model.py:my_ocr } }
    ```

    This also works from the CLI:

    ```bash
    ii extract --set extract.ocr.mode=auto --set extract.ocr.backend=ocr/my_model.py:my_ocr
    ```

### Setup once per run: factories

For a backend that loads weights or opens a client, register a **factory**.
It is called once with the OCR config and returns the page function:

```python
from industrial_instruction.ocr import ocr_backend

@ocr_backend("my-local-model", factory=True)
def build(config):
    model = load_model(config.options["weights"])
    return lambda page: model.transcribe(page.to_pil())
```

## What the package handles for every backend

| | |
|---|---|
| **Concurrency** | `max_workers` pages are OCR'd in parallel, which keeps a vLLM batch full. |
| **Cache** | Results are stored in `artifacts/ocr_cache/`, keyed by page image and backend settings, so re-running extraction doesn't re-OCR pages. Turn off with `cache: false`. |
| **Memory** | Pages are rendered and OCR'd in batches; a 500-page scan is never held in memory at once. |
| **Failures** | A failing page keeps its text layer; the run manifest counts `ocr_pages`, `ocr_failed`, `ocr_empty` and `ocr_cached`. |
| **Bad config** | A wrong backend name or path stops `ii extract` before the first page. |

## Tesseract (offline)

```bash
pip install -e '.[ocr-tesseract]'      # plus the tesseract binary, e.g. apt install tesseract-ocr
```

```yaml
extract: { ocr: { mode: auto, backend: tesseract, options: { lang: deu } } }
```

Tesseract returns plain text: headings and tables are not recovered. Use a
vision model for those.

A complete example, with a Nanonets-OCR model on vLLM and a factory, is in
[`examples/custom_ocr.py`](https://github.com/parssky/industrial-instruction/blob/main/examples/custom_ocr.py).
