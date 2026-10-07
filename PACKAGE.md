# industrial-instruction

Build retrieval-grounded instruction datasets from your own PDFs.

This package generalizes the pipeline behind the paper *Industrial
Instruction* so any team can point it at their own manuals, datasheets and
service documents and get a training/evaluation dataset out the other end.
The original notebooks are kept as-is for reproducibility; this package is
the reusable path.

The package does two things: it **builds datasets** from your PDFs, and it
**benchmarks** a model you serve. It does not train models; use the dataset
it writes with any trainer (TRL, Unsloth, axolotl).

## Pipeline

```
PDFs -> extract -> chunk -> index (FAISS) -> generate -> filter -> assemble -> dataset
```

| Stage | What happens |
| --- | --- |
| `extract` | Text + tables out of PDFs, images dropped. Tables become markdown, headers/footers stripped, hyphenation repaired. |
| `chunk` | Heading-aware chunking that keeps tables whole. |
| `index` | Chunks embedded in batches into a FAISS index, with a metadata file that prevents querying it with a different model. |
| `generate` | For each seed instruction, retrieve context and ask an LLM for a QA pair under five retrieval relations (r0-r4). |
| `filter` | Deterministic rules (length, missing answer, meta-references, option count, duplicates) plus an optional LLM judge. |
| `assemble` | Merge relations, seeded and optionally stratified train/test splits, flat + chat formats, optional Hub push. |

## Install

```bash
pip install -e '.[pdf,local-embed,faiss]'    # PyMuPDF + sentence-transformers + FAISS
pip install -e '.[all]'                      # every optional backend
```

Extras: `pdf` (PyMuPDF, the default extractor), `pdfplumber`, `docling`,
`marker`, `local-embed`, `faiss`, `hub` (HuggingFace datasets), `dev`.

## Quick start

```bash
ii init                        # writes industrial_instruction.yaml
cp your-manuals/*.pdf data/pdfs/
export OPENAI_API_KEY=sk-...
ii run                         # extract -> index -> generate -> filter -> assemble
```

Smoke-test on a handful of seeds first:

```bash
ii run --set seeds.limit=20 --set generate.max_workers=4
```

Run one stage at a time, or a subset:

```bash
ii extract
ii index
ii run --stages generate,filter,assemble
ii info                        # resolved config + which artifacts exist
```

Any config value can be overridden without editing the file:

```bash
ii generate --set generate.model=gpt-4.1 \
            --set generate.base_url=http://localhost:8000/v1 \
            --set generate.retrieval_k=5
```

## Python API

```python
from industrial_instruction import Config, Pipeline

pipeline = Pipeline.from_yaml("industrial_instruction.yaml")
pipeline.run_extract()
pipeline.run_index()
report = pipeline.run_generate()
print(report.outputs, "samples")
```

## Bring your own everything

**Extractor.** `extract.backend` selects `pymupdf` (default), `pdfplumber` or
`markdown` (for corpora that are already converted). Register your own with
`register_extractor("name", factory)`.

The `pymupdf` backend reads font sizes and positions, not just text: lines
set clearly larger than the body text become `#`/`##`/`###` headings (bold
standalone lines at body size become the next level down), tables are
written once as markdown at the place they appear on the page, and running
headers/footers and page numbers are removed from the page margins. Headings
are what the default `heading` chunker splits on, and each chunk carries its
`Section > Subsection` breadcrumb so retrieval and the generator see it. Turn
heading detection off with `extract.backend_options: {detect_headings: false}`.

**OCR.** Scanned or image-only pages are sent to an OCR model when
`extract.ocr.mode` is `auto` (pages with no text layer) or `always`. An OCR
backend is just a function, page image in, markdown out:

```python
from industrial_instruction.ocr import OCRPage, register_ocr

def my_ocr(page: OCRPage) -> str:
    # page.image (PNG bytes), page.data_url(), page.to_pil(), page.page_number,
    # page.text_layer, page.options (= extract.ocr.options)
    return call_my_model(page.image)

register_ocr("my-ocr", my_ocr)          # then extract.ocr.backend: my-ocr
```

Or attach it without any Python wiring by an import path, which also works
from the CLI:

```yaml
extract:
  ocr:
    mode: auto
    backend: ocr/my_model.py:my_ocr      # file next to the config, or my_pkg.ocr:my_ocr
    options: { lang: deu }               # anything your function needs
```

Two backends are built in: `openai` sends the page to any OpenAI-compatible
vision endpoint, so a VLM served by vLLM needs only `base_url` and `model`;
`tesseract` runs offline (`pip install '.[ocr-tesseract]'`). For a backend
that loads weights, register a factory with
`@ocr_backend("name", factory=True)`. It is called once with the OCR config
and returns the page function.

The pipeline handles the rest for every backend. Pages are OCR'd
concurrently (`max_workers`) and cached by image hash in
`artifacts/ocr_cache`. A page whose OCR fails keeps its text layer and is
counted in the manifest. See `examples/custom_ocr.py`.

**Embedder.** `embed.backend` selects `sentence_transformers` (default,
any Hub id or local path), `openai` (any OpenAI-compatible embeddings
endpoint) or `hash` (deterministic, offline, for tests). Queries and
documents get separate prefixes, which matters for asymmetric models.

**Seeds.** Three interchangeable sources, set by `seeds.source`:

```yaml
seeds: { source: huggingface, dataset: ibm-research/FailureSensorIQ }
seeds: { source: jsonl, path: data/my_seeds.jsonl, text_field: instruction }
seeds: { source: self, limit: 200 }   # bootstrap from your own PDFs
```

The `self` source needs no external dataset at all: it samples your indexed
chunks and asks the model for a realistic practitioner question per chunk.

**Multiple-choice seeds.** The paper's seeds (FailureSensorIQ) are
multiple-choice prompts with their own options and answer-format template.
With `generate.options_mode: auto` (default), a multiple-choice seed yields a
multiple-choice sample (`q*`, `options*`, `a*` as a label list) in every
relation r0-r4, and a plain seed yields a question/answer pair. Use `always`
or `never` to force one format. Replies are normalized before filtering:

- options are moved out of `q*` if the model embedded them there
- options are labelled `A. ...` to `E. ...`, and P-T labels from perturbed
  seeds are remapped together with the answer
- `{"answer": [...]}`, `"B, D"` and `["B"]` all become `["B", "D"]`

The filter then drops samples with no options, answers that are not option
labels, options copied from the seed (`max_seed_option_overlap`), and
questions that still contain the seed's `{"answer": ...}` template. Retrieval
uses the seed's `question` field (`seeds.query_field`) rather than the full
prompt, whose boilerplate is the same for every seed. In the chat format the
assistant turn is the label list as JSON, e.g. `["B", "D"]`.

**Prompts.** All templates are `.txt` files. Point `generate.prompt_dir` at a
directory containing `useless_doc.txt`, `single_doc_support.txt`,
`multi_doc_support.txt`, `single_doc_answer.txt`, `multi_doc_answer.txt`,
`self_seed.txt` or `judge.txt` to override any of them; the packaged version
is used for the rest. The template hash is recorded on every sample.

**Relations.** The five relations from the paper are the default. Disable or
add relations in `generate.relations`:

```yaml
generate:
  relations:
    - id: r5
      prompt: my_relation      # my_relation.txt in generate.prompt_dir
      doc_mode: multi
      k_docs: 4
      description: comparison across two datasheets
```

## Benchmarking

Serve the model with vLLM (or anything OpenAI-compatible), then point
`ii bench` at it:

```bash
vllm serve your-org/your-model --port 8000

ii bench --base-url http://localhost:8000/v1 --suite ibm
ii bench --suite paper-claude --suite paper-qwen --context none --context gold
ii bench --suite custom --set benchmark.custom.path=my_bench.jsonl --limit 50
```

| Suite | Data | Metrics |
| --- | --- | --- |
| `ibm` | FailureSensorIQ (IBM), original + perturbed form of each question | `acc_original`, `acc_perturb`, `consistency` (both right) |
| `paper-qwen` | held-out test split of the paper's Qwen-generated data | set match, F1, Jaccard |
| `paper-claude` | held-out test split of the Claude-generated data | set match, F1, Jaccard |
| `generated` | the `test.jsonl` this project's `ii assemble` wrote | set match, F1, Jaccard |
| `custom` | any multiple-choice set: jsonl, json, Hub or `save_to_disk` | set match, F1, Jaccard |

`--context` picks what the model sees with each question: `none` (closed
book), `gold` (the item's own documents, the paper's RAG setting) or
`retrieved` (top-k chunks from this project's FAISS index). Documents are
wrapped in the template the paper's models were trained with
(`benchmark.context_template`). Map your own dataset's fields with
`benchmark.custom.*`, or add a suite in Python with
`register_suite("name", loader)`.

Every run writes `artifacts/benchmarks/<model>/<time>/`, containing
`summary.json` and one `<suite>-<context>.samples.jsonl` per run with each
reply, the parsed answer and the gold labels.

**Scoring.** One parser reads every reply format the paper's models
produce: `{"answer": ["B"]}`, `['B']`, malformed `{"answer": ["B"}`,
`<think>` blocks and `Answer: B`. It never matches stray letters in free
text. The original IBM evaluation script compared option ids like `P)` to
answers like `P`, so it could never score a perturbed item as correct. The
`ibm` suite fixes that, and also reports the old rule under `legacy` so new
numbers can be compared with earlier ones.

## Outputs

```
artifacts/
  documents/       <doc_id>.md + documents.jsonl + extract_failures.json
  chunks.jsonl
  faiss/           index.faiss + id_map.json + store_meta.json
  generated/       r0.jsonl ... r4.jsonl + rejects.jsonl
  filtered/        r0.jsonl ... r4.jsonl + rejected.jsonl
  dataset/         train.jsonl, test.jsonl, *.chat.jsonl, dataset_stats.json
  run_manifest.json
```

Nothing is thrown away silently. Failed generations land in `rejects.jsonl`
with the raw model output and the reason, and `run_manifest.json` records the
config fingerprint for every stage, so a dataset can be traced back to the
exact settings that produced it.

## Notes on cost and scale

- `generate` makes one API call per (seed x relation). 500 seeds x 5
  relations = 2500 calls. Start with `seeds.limit`.
- `filter.judge_enabled: true` adds one call per surviving sample.
- A failed JSON reply costs one sample, not the whole seed - each relation is
  isolated and retried independently.
