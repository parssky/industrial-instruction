# Extraction <span class="ii-stage">ii extract</span>

Extraction turns every PDF under `paths.pdfs` (default `data/pdfs/`,
including subfolders) into one clean markdown document. Everything else
depends on its quality, so this is the stage worth checking by hand.

```bash
ii extract
ls artifacts/documents/        # <id>.md per document, documents.jsonl, extract_failures.json
```

## What the default extractor does

The `pymupdf` backend reads font sizes and positions, not only the text:

| Feature | How |
|---|---|
| **Headings** | Lines set clearly larger than the body text (≥ 1.15×) become `#`, `##` and `###` by size; short, standalone bold lines at body size become the next level down. |
| **Tables** | Detected tables are written once, as markdown, at their position on the page. The cell text is not repeated as loose text. |
| **Headers and footers** | Lines repeated in the top or bottom 8% of most pages are removed, digits ignored, so "Manual — 12" and "Manual — 13" match. Page numbers in the margins are removed. Body text is never touched by this rule. |
| **Images** | Dropped. They are counted in the manifest. |
| **Hyphenation** | Words broken across lines (`mainte-\nnance`) are joined. |
| **Title** | The PDF's metadata title, or else the first `#` heading. |

A short example of the result:

```markdown
<!-- page 2 -->

## Fault codes

<!-- table 3x2 -->
| Code | Meaning |
| --- | --- |
| E01 | Phase imbalance |
| E02 | Overtemperature |

Reset the controller after clearing any fault code listed above.
```

## Settings

```yaml
extract:
  backend: pymupdf            # pymupdf | pdfplumber | markdown
  extract_tables: true
  table_backend: auto         # auto | pymupdf | pdfplumber | none
  drop_images: true
  page_markers: true          # <!-- page N --> separators
  strip_headers_footers: true
  dehyphenate: true
  min_chars_per_doc: 200      # documents with less text are skipped (see OCR)
  recursive: true             # include subfolders of paths.pdfs
  backend_options:
    detect_headings: true     # pymupdf: font-size headings
```

## Backends

`pymupdf` (default)
:   Fast, CPU-only, layout-aware as described above. Install with `.[pdf]`.

`pdfplumber`
:   Slower; often better on tables without ruling lines. Install with `.[pdfplumber]`.

`markdown`
:   For a corpus that is already converted: reads `.md`, `.markdown` and
    `.txt` files as they are.

## Skipped documents

A document with less than `min_chars_per_doc` characters of text is skipped
and listed in `artifacts/documents/extract_failures.json`. This is almost
always a scanned PDF without a text layer; turn on [OCR](ocr.md) for those.

## Your own extractor

Subclass `Extractor` and register it:

```python
from industrial_instruction.extract import Extractor, register_extractor

class DoclingExtractor(Extractor):
    name = "docling"

    def extract(self, path):
        markdown = convert_with_docling(path)
        return self.validate(self._new_document(path, markdown))

register_extractor("docling", DoclingExtractor)     # extract.backend: docling
```

See [Contributing](../development/contributing.md#83-a-pdf-extractor) for
making it a built-in.
