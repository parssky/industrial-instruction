# industrial-instruction

Build retrieval-grounded instruction datasets from your own PDFs.

This package generalizes the pipeline behind the paper *Industrial
Instruction* so any team can point it at their own manuals, datasheets and
service documents and get a training/evaluation dataset out the other end.
The original notebooks are kept as-is for reproducibility; this package is
the reusable path.

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
