# Industrial-Instruction

[![CI](https://github.com/parssky/industrial-instruction/actions/workflows/ci.yml/badge.svg)](https://github.com/parssky/industrial-instruction/actions/workflows/ci.yml)
[![Docs](https://github.com/parssky/industrial-instruction/actions/workflows/docs.yml/badge.svg)](https://parssky.github.io/industrial-instruction/)
[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue)](https://github.com/parssky/industrial-instruction/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2608.22817-b31b1b)](https://arxiv.org/abs/2608.22817)

Industrial-Instruction turns a folder of technical PDFs (manuals, datasheets,
service reports) into a **retrieval-grounded instruction dataset**, then
**benchmarks** any model you serve against that dataset, IBM's
FailureSensorIQ and the paper's held-out test splits.

**Full documentation: https://parssky.github.io/industrial-instruction/**

It is the reusable package behind the paper:

📄 **[Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports](https://arxiv.org/abs/2608.22817)** (arXiv:2608.22817)

```
PDFs ─► extract ─► chunk ─► index ─► generate (r0–r4) ─► filter ─► assemble ─► dataset
          │ OCR                │                                                  │
          ▼                    ▼                                                  ▼
   scanned pages        your vector store ◄──────── ii bench ◄──── model served on vLLM
```

The package **builds datasets and benchmarks models; it does not train**.
The dataset it writes plugs into any trainer (TRL, Unsloth, axolotl).

| Artifact | Link |
|---|---|
| Paper | https://arxiv.org/abs/2608.22817 |
| Dataset | https://huggingface.co/datasets/Parssky/industrial-instruction-dataset |
| Model (Qwen-generated data) | https://huggingface.co/Parssky/industrial-instruction-qwen4b |
| Model (Claude-generated data) | https://huggingface.co/Parssky/industrial-instruction-qwen4b-claude |
| FAISS retrieval index | https://huggingface.co/datasets/Parssky/industrial-instruction-faiss |
| **Documentation** | https://parssky.github.io/industrial-instruction/ |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## Contents

1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [Configuration](#3-configuration)
4. [Step by step](#4-step-by-step)
   - [4.1 Extract: PDFs to markdown](#41-extract-pdfs-to-markdown)
   - [4.2 OCR for scanned pages](#42-ocr-for-scanned-pages)
   - [4.3 Chunk](#43-chunk)
   - [4.4 Index: the vector store](#44-index-the-vector-store)
   - [4.5 Seeds](#45-seeds)
   - [4.6 Generate: relations r0–r4](#46-generate-relations-r0r4)
   - [4.7 Filter](#47-filter)
   - [4.8 Assemble the dataset](#48-assemble-the-dataset)
5. [Retrieval: your corpus, your index, or your function](#5-retrieval-your-corpus-your-index-or-your-function)
6. [Benchmarking](#6-benchmarking)
7. [Python API](#7-python-api)
8. [Outputs and reproducibility](#8-outputs-and-reproducibility)
9. [Paper results](#9-paper-results)
10. [Repository layout](#10-repository-layout)
11. [Citation](#11-citation)

---

## 1. Install

Python 3.9+.

```bash
git clone https://github.com/parssky/industrial-instruction.git
cd industrial-instruction
pip install -e '.[pdf,local-embed,faiss,bench]'
```

| Extra | Installs | Needed for |
|---|---|---|
| `pdf` | PyMuPDF | the default PDF extractor (and OCR page rendering) |
| `pdfplumber` | pdfplumber | the alternative extractor, often better on unruled tables |
| `local-embed` | sentence-transformers, torch | the default local embedding model |
| `faiss` | faiss-cpu | the vector store |
| `bench` | datasets, huggingface-hub | Hub benchmark suites and `hf:` indexes |
| `hub` | datasets, huggingface-hub | Hub seed datasets, pushing the dataset to the Hub |
| `ocr-tesseract` | pytesseract, Pillow | the offline Tesseract OCR backend |
| `all` | everything above except OCR | |
| `dev` | pytest, ruff | running the tests |

Installing gives you the `ii` command (also available as `industrial-instruction`).

---

## 2. Quick start

```bash
ii init                                   # writes industrial_instruction.yaml
cp ~/manuals/*.pdf data/pdfs/
export OPENAI_API_KEY=sk-...              # or point generate.base_url at vLLM

ii run --set seeds.limit=20               # small first run: extract → … → dataset
```

Check `artifacts/documents/*.md` to see what was extracted, and
`artifacts/dataset/train.jsonl` to see the result. Then run at full size:

```bash
ii run
```

Benchmark a model you serve:

```bash
vllm serve Parssky/industrial-instruction-qwen4b-claude --port 8000
ii bench --base-url http://localhost:8000/v1 --suite ibm --suite generated
```

---

## 3. Configuration

One YAML file drives everything. `ii init` writes a commented starter
([`configs/default.yaml`](src/industrial_instruction/configs/default.yaml));
every key is optional and falls back to the default.

| Section | Controls |
|---|---|
| `paths` | where inputs and artifacts live (relative to the config file) |
| `extract` | PDF backend, tables, header/footer removal, `ocr` |
| `chunk` | chunking strategy and sizes |
| `embed` | embedding model for the vector store |
| `store` | FAISS index type |
| `retrieval` | where generation and benchmarking get documents |
| `seeds` | the "simulated instructions" that steer question style |
| `generate` | the generator LLM, relations r0–r4, multiple-choice mode |
| `filter` | rule checks and the optional LLM judge |
| `assemble` | splits, output formats, Hub upload |
| `benchmark` | the served model, suites, context modes |

Override any value on the command line without editing the file. Values are
parsed as YAML:

```bash
ii generate --set generate.model=gpt-4.1 --set generate.max_workers=16
ii run -c configs/plant_b.yaml --set seeds.limit=50
ii info                                   # resolved config + which artifacts exist
```

Secrets never go in the YAML: `*.api_key_env` names the environment variable
that holds a key (default `OPENAI_API_KEY`). Local vLLM servers accept any key.

---

## 4. Step by step

Each stage reads the previous stage's artifacts, so you can run, inspect and
re-run one stage at a time:

```bash
ii extract && ii chunk && ii index && ii generate && ii filter && ii assemble
ii run --stages generate,filter,assemble  # a subset, always run in pipeline order
```

### 4.1 Extract: PDFs to markdown

```bash
ii extract                                # data/pdfs/**.pdf → artifacts/documents/
```

The default `pymupdf` backend reads font sizes and positions, not just text:

- larger fonts become `#`, `##` and `###` headings; short bold lines become the next level down
- tables are written once, as markdown, at the place they appear on the page
- running headers, footers and page numbers are removed from the page margins
- images are dropped; hyphenated line breaks are repaired

Each document is written as `artifacts/documents/<id>.md`. **Read a few
before generating**, because extraction quality limits everything after it.

```yaml
extract:
  backend: pymupdf           # pymupdf | pdfplumber | markdown (already-converted .md/.txt)
  extract_tables: true
  strip_headers_footers: true
  min_chars_per_doc: 200     # documents with less text are skipped (see OCR)
  backend_options: { detect_headings: true }
```

You can add your own extractor with
`register_extractor("name", factory)` (see `extract/registry.py`).

### 4.2 OCR for scanned pages

A page with no text layer can be sent to an OCR model. **An OCR backend is
one function: page image in, markdown out.**

```python
from industrial_instruction.ocr import OCRPage, register_ocr

def my_ocr(page: OCRPage) -> str:
    # page.image (PNG bytes), page.data_url(), page.to_pil(),
    # page.page_number, page.text_layer, page.options
    return call_my_model(page.image)

register_ocr("my-ocr", my_ocr)            # extract.ocr.backend: my-ocr
```

You can also skip registration and point the config at the function: an
import path (`my_pkg.ocr:my_ocr`) or a file next to the config
(`ocr/my_model.py:my_ocr`). Both also work with `--set`.

**A vision model on vLLM needs no code.** The built-in `openai` backend
works with any OpenAI-compatible vision endpoint:

```yaml
extract:
  ocr:
    mode: auto                 # off | auto (pages without text) | always
    backend: openai            # openai | tesseract | <your name> | module:function
    base_url: http://localhost:8001/v1
    model: Qwen/Qwen2.5-VL-7B-Instruct
    prompt: null               # null = packaged prompt; set one for olmOCR / Nanonets-OCR
    max_workers: 8             # pages in flight at once
    cache: true                # artifacts/ocr_cache: re-runs don't re-OCR pages
    options: {}                # passed to your function as page.options
```

A page whose OCR fails keeps its text layer, and the failure is counted in
the run manifest. A wrong backend path stops the stage at the start. See
[`examples/custom_ocr.py`](examples/custom_ocr.py).

### 4.3 Chunk

```bash
ii chunk                                  # → artifacts/chunks.jsonl
```

The `heading` strategy splits on markdown headings and keeps tables whole.
Each chunk starts with its `## Section > Subsection` breadcrumb, so the
embedder and the generator see the context. Short sections are merged into
the next chunk, never dropped. `fixed` windows break at word boundaries.

```yaml
chunk: { strategy: heading, max_chars: 2000, min_chars: 120, overlap: 200 }
```

### 4.4 Index: the vector store

```bash
ii index                                  # artifacts/chunks.jsonl → artifacts/faiss/
ii index --corpus data/pdfs               # or straight from a corpus: extract + chunk + index
```

```yaml
embed:
  backend: sentence_transformers   # sentence_transformers | openai | hash (offline tests)
  model: google/embeddinggemma-300m
  query_prefix: "task: search result | query: "
  document_prefix: "title: none | text: "
store: { index_type: flat_ip }     # flat_ip (cosine) | flat_l2 | hnsw
```

The index records which embedding model built it. Loading it with a
different model triggers a warning, and an error if the vector width
differs. To use an index you already have, or your own search service, see
[section 5](#5-retrieval-your-corpus-your-index-or-your-function).

### 4.5 Seeds

Seeds are example questions the generator imitates in style and format.
There are three sources:

```yaml
seeds: { source: huggingface, dataset: ibm-research/FailureSensorIQ, splits: [org, pert] }  # the paper
seeds: { source: jsonl, path: data/my_seeds.jsonl, text_field: instruction }
seeds: { source: self, limit: 200 }        # bootstrap from your own chunks, no external data
```

`text_field` is what the generator sees. `query_field` is what retrieval
searches with; for FailureSensorIQ this is the bare `question`, not the full
prompt with its options boilerplate.

### 4.6 Generate: relations r0–r4

```bash
ii generate                               # → artifacts/generated/r0.jsonl … r4.jsonl
```

For each seed, the package retrieves documents and asks the generator LLM
for one sample per **retrieval relation**:

| Relation | Documents | The question is… |
|---|---|---|
| `r0` | 1 | related to the document, but the document cannot answer it (noise robustness) |
| `r1` | 1 | supported by hints in the document, which doesn't state the answer |
| `r2` | 3 | supported by hints spread across several documents |
| `r3` | 1 | fully answerable from the document |
| `r4` | 3 | answerable only by combining documents (multi-hop) |

```yaml
generate:
  base_url: null               # null = OpenAI; http://localhost:8000/v1 for vLLM
  model: gpt-4.1-mini
  temperature: 0.1
  max_workers: 8
  options_mode: auto           # auto: multiple-choice when the seed is | always | never
  n_options: 5
  prompt_dir: null             # a folder of .txt files overriding any packaged prompt
```

With `options_mode: auto`, a multiple-choice seed (as in FailureSensorIQ)
gives multiple-choice samples: question, options A–E and the answer as a
label list (`["B", "D"]`). Replies are normalized before filtering. Options
written inside the question are moved out, P–T labels are remapped to A–E,
and `{"answer": [...]}` or `"B, D"` become `["B", "D"]`.

Each relation is generated and retried on its own. A failed reply costs one
sample, and lands in `rejects.jsonl` with the raw output and the reason.
Prompts are plain `.txt` files in
[`generate/prompts/`](src/industrial_instruction/generate/prompts/). To add a
relation, give it an `id`, a prompt name, `doc_mode` and `k_docs` under
`generate.relations`.

**Cost:** one call per seed × relation; 500 seeds × 5 relations is 2,500
calls. Start with `--set seeds.limit=20`.

### 4.7 Filter

```bash
ii filter                                 # → artifacts/filtered/ + rejected.jsonl
```

Rule checks, each counted in the manifest under `reject_reasons`:

- question length; missing answers
- meta-references such as "according to the documents"; duplicates
- multiple-choice only:
  - missing options or the wrong number of options
  - answers that aren't option labels
  - options copied from the seed
  - the seed's `{"answer": ...}` template left in the question

`filter.judge_enabled: true` adds an LLM judge, which grades faithfulness,
standalone-ness, relation match and usefulness. It costs one call per sample.

### 4.8 Assemble the dataset

```bash
ii assemble                               # → artifacts/dataset/
```

```yaml
assemble:
  splits: { train: 0.9, test: 0.1 }
  stratify_by_relation: true   # every relation appears in every split
  chat_format: true            # also <split>.chat.jsonl (messages) for SFT trainers
  include_documents: true      # put the retrieved documents in the prompt
  push_to_hub: null            # e.g. your-name/your-dataset
```

The `test.jsonl` written here is the `generated` benchmark suite.

---

## 5. Retrieval: your corpus, your index, or your function

Generation and `ii bench --context retrieved` read documents from one
retriever, chosen by `retrieval.source`.

**1. `index` (default): build a vector store from your corpus**, using the
embedding model in `embed`:

```bash
ii index --corpus data/pdfs           # PDFs: extract + chunk + embed
ii index --corpus docs/               # markdown or text files
ii index --corpus passages.jsonl      # one {"text": ..., "id": ...} per line, indexed as given
ii index --corpus kb.jsonl --text-field body --id-field doc_key
```

**2. `faiss`: an index you built yourself.** It can be a directory, an
index file plus id map, or a Hub repo. Both this package's format and the
paper's original format (an `index` file plus an `id_map.json` of
`{"0": "chunk text"}`) are read:

```yaml
retrieval:
  source: faiss
  path: hf:Parssky/industrial-instruction-faiss   # or ./my_index/ or ./my_index/index
  mapping_path: null       # only if id_map.json is not next to the index
  text_field: text         # if id-map entries are dicts
  embed: { backend: sentence_transformers, model: google/embeddinggemma-300m }  # the model it was built with
```

The vector width is checked against the embedding model. Indexes in the
original format were built with plain `encode(text)`, so they are queried
without the `embed` prefixes.

**3. `function`: your own search service** (Elasticsearch, Qdrant, a
reranker…). It takes `(query, k)` and returns texts, dicts (`text`, plus
optional `score`, `id` and metadata) or `Chunk` objects:

```python
# retrievers/es.py
def search(query: str, k: int):
    return [hit["_source"]["text"] for hit in es.search(index="manuals", q=query, size=k)["hits"]["hits"]]
```

```yaml
retrieval: { source: function, backend: retrievers/es.py:search }   # or my_pkg.search:run
```

In Python, use `register_retriever("es", search)`. For a client that should
be created once per run, use `@retriever_backend("es", factory=True)`; the
factory receives `retrieval.options`.

---

## 6. Benchmarking

**1. Serve the model**, on any OpenAI-compatible server. The package never
starts or stops it:

```bash
vllm serve Parssky/industrial-instruction-qwen4b-claude --port 8000
```

**2. Run suites**, optionally in several context modes:

```bash
ii bench --base-url http://localhost:8000/v1 --suite ibm
ii bench --suite paper-claude --suite paper-qwen --context none --context gold
ii bench --suite generated --context retrieved
ii bench --suite custom --set benchmark.custom.path=my_bench.jsonl --limit 50
```

| Suite | Data | Metrics |
|---|---|---|
| `ibm` | [FailureSensorIQ](https://huggingface.co/datasets/ibm-research/FailureSensorIQ): each question in an original and a perturbed form | `acc_original`, `acc_perturb`, `consistency` (both right) |
| `paper-qwen` | held-out test split of the paper's Qwen-generated data (`panasonic_qa_v1_test`) | set match, F1, Jaccard |
| `paper-claude` | held-out test split of the Claude-generated data (`panasonic_qa_claude_v1_test`) | set match, F1, Jaccard |
| `generated` | `artifacts/dataset/test.jsonl` from your own run | set match, F1, Jaccard |
| `custom` | any multiple-choice set: jsonl, json, Hub dataset or `save_to_disk` folder | set match, F1, Jaccard |

| `--context` | The model sees |
|---|---|
| `none` | the question only (closed book) |
| `gold` | the item's own source documents (the paper's RAG setting) |
| `retrieved` | the top `benchmark.retrieval_k` documents from the [retriever](#5-retrieval-your-corpus-your-index-or-your-function) |

```yaml
benchmark:
  endpoint:
    base_url: http://localhost:8000/v1
    model: null                # null = the first model the server lists
    temperature: 0.0
    max_tokens: 1024
    max_workers: 32
    system_prompt: 'You are a helpful assistant. You must output your answer strictly as valid JSON in the format {"answer": ["choice"]}.'
  suites: [ibm]
  contexts: [none]
  retrieval_k: 3
  paper_qwen_split: panasonic_qa_v1_test        # Hub name or a local save_to_disk folder
  paper_claude_split: panasonic_qa_claude_v1_test
  custom:
    source: jsonl              # jsonl | json | huggingface | disk
    path: my_bench.jsonl
    question_field: question
    answer_field: answer       # ["B"], "B", "B, D" or {"answer": [...]}
    options_field: options     # added to the question unless already inside it
    documents_field: documents # used by --context gold
```

The documents are wrapped in the template the paper's models were trained
with (`benchmark.context_template`), so fine-tuned checkpoints see their
training format.

**Scoring.** One parser reads every reply format we have seen:
`{"answer": ["B"]}`, `['B']`, malformed `{"answer": ["B"}`, `<think>`
blocks, and `Answer: B`. It never matches stray letters in free text. A
reply with no answer scores 0, and the unparsed rate is reported. The
original IBM evaluation script compared option ids like `P)` with answers
like `P`, so a perturbed question could never be scored correct. The `ibm`
suite fixes this and reports the old rule under `legacy` for comparison
with earlier numbers.

Add a suite in Python with
`register_suite("name", loader)`, where the loader returns `BenchItem`s.

---

## 7. Python API

```python
from industrial_instruction import Config, Pipeline

config = Config.from_yaml("industrial_instruction.yaml").with_override("seeds.limit", 20)
pipeline = Pipeline(config)
for report in pipeline.run_all():                 # or run_extract(), run_generate(), …
    print(report.stage, report.outputs, report.rejected)

from industrial_instruction.store import get_retriever
hits = get_retriever(config).search("bearing replacement interval", k=3)

from industrial_instruction.benchmark import run_benchmarks, format_table
summary = run_benchmarks(config, suites=["ibm", "generated"], contexts=["none", "gold"])
print(format_table(summary))
```

Plug-in points, all registered by name or given as `module:function` /
`file.py:function` in the config:

| Plug-in | Register with | Config key |
|---|---|---|
| PDF extractor | `register_extractor` | `extract.backend` |
| OCR model | `register_ocr`, `@ocr_backend` | `extract.ocr.backend` |
| Retriever | `register_retriever`, `@retriever_backend` | `retrieval.backend` |
| Benchmark suite | `register_suite` | `benchmark.suites` |
| Prompts | `.txt` files | `generate.prompt_dir` |

---

## 8. Outputs and reproducibility

```
artifacts/
  documents/       <id>.md, documents.jsonl, extract_failures.json
  ocr_cache/       OCR results by page-image hash
  chunks.jsonl
  faiss/           index.faiss, id_map.json, store_meta.json
  generated/       r0.jsonl … r4.jsonl, rejects.jsonl
  filtered/        r0.jsonl … r4.jsonl, rejected.jsonl
  dataset/         train.jsonl, test.jsonl, *.chat.jsonl, dataset_stats.json
  benchmarks/      <model>/<time>/summary.json, <suite>-<context>.samples.jsonl
  run_manifest.json
```

Nothing is dropped silently. Rejected samples keep the reason and the raw
model output. Benchmark samples keep each reply next to its parsed answer.
`run_manifest.json` records every stage's counts and a fingerprint of the
full config, so a dataset can be traced back to the settings that produced
it.

---

## 9. Paper results

Fine-tuning Qwen3-4B-Instruct-2507 on Industrial-Instruction, evaluated on the held-out Panasonic benchmark split:

| Training data | Set-Match Acc. | F1 | Jaccard | MMLU |
|---|---|---|---|---|
| None (base model) | 28.5% | 46.6% | 41.6% | 72.13% |
| `pana_qa_v1` (Qwen-generated) | 42.0% | 63.5% | 58.0% | 70.87% |
| `pana_qa_claude_v1` (Claude-generated) | 56.4% | 72.7% | 68.9% | 72.08% |

> The two fine-tuned models are evaluated on different held-out splits, each generated by its own generator model, so their scores are indicative rather than a controlled head-to-head comparison.

Reproduce a row with `ii bench --suite paper-qwen` or
`ii bench --suite paper-claude`, served with the matching model above.

---

## 10. Repository layout

```
src/industrial_instruction/   the package (this README)
tests/                        offline tests: pytest
examples/                     quickstart.py, custom_ocr.py, seed file example
docs/, mkdocs.yml             the documentation site (mkdocs serve)
scripts/                      docs build helpers
.github/                      CI, docs deployment, releases, PR and issue templates
CONTRIBUTING.md               how to develop, test and submit changes
PACKAGE.md                    short package reference
create_dataset_notebooks/     original research notebooks (paper reproduction)
create_vector_store_notebooks/
training_notebooks/           the paper's fine-tuning scripts (not part of the package)
package_benchmark_*/          original evaluation scripts and saved results
```

The notebook folders are kept as they were run for the paper. New work
should use the package.

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers
setup, code standards, tests, and recipes for adding OCR models,
retrievers, benchmark suites and more. Every pull request runs lint, tests
on Python 3.9–3.13, a package build and a docs build.

---

## 11. Citation

If you use this work, please cite the paper:

```bibtex
@misc{bakhtiari2026industrialinstruction,
      title        = {Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports},
      author       = {Parsa Bakhtiari and Hassan Bashiri and Alireza Khalilipour and Masoud Nasiripour and Moharram Challenger},
      year         = {2026},
      eprint       = {2608.22817},
      archivePrefix= {arXiv},
      primaryClass = {cs.CL},
      url          = {https://arxiv.org/abs/2608.22817}
}
```

To cite the dataset specifically:

```bibtex
@misc{parsa_bakhtiari_2026,
    author       = { Parsa Bakhtiari and Hassan Bashiri and Alireza Khalilipour and Masoud Nasiripour and Moharram Challenger },
    title        = { industrial-instruction-dataset (Revision 7eadea0) },
    year         = 2026,
    url          = { https://huggingface.co/datasets/Parssky/industrial-instruction-dataset },
    doi          = { 10.57967/hf/10098 },
    publisher    = { Hugging Face }
}
```

## License

See [LICENSE](LICENSE).
