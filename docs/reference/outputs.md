# Outputs

All outputs go under `artifacts/` in the project folder. Every path is
configurable under `paths`.

```text
artifacts/
├── documents/              ii extract
│   ├── <id>.md             one markdown file per PDF: read these
│   ├── documents.jsonl     the same documents, with metadata
│   └── extract_failures.json
├── ocr_cache/              OCR results, keyed by page image + backend settings
├── chunks.jsonl            ii chunk
├── faiss/                  ii index
│   ├── index.faiss
│   ├── id_map.json         row id → chunk record
│   └── store_meta.json     embedding model, dimension, prefixes, count
├── generated/              ii generate
│   ├── r0.jsonl … r4.jsonl
│   └── rejects.jsonl       failed or invalid replies, with raw output and reason
├── filtered/               ii filter
│   ├── r0.jsonl … r4.jsonl
│   └── rejected.jsonl      removed samples, with reasons
├── dataset/                ii assemble
│   ├── train.jsonl, test.jsonl
│   ├── train.chat.jsonl, test.chat.jsonl
│   └── dataset_stats.json
├── benchmarks/             ii bench
│   └── <model>/<time>/
│       ├── summary.json
│       └── <suite>-<context>.samples.jsonl
└── run_manifest.json
```

## Records

| Record | Where | Key fields |
|---|---|---|
| `Document` | `documents.jsonl` | `id`, `source_path`, `title`, `markdown`, `n_pages`, `n_tables`, `n_images_dropped`, `meta` (backend, OCR counts) |
| `Chunk` | `chunks.jsonl`, `id_map.json` | `id`, `doc_id`, `text`, `heading_path`, `has_table`, `source_path` |
| `QASample` | `generated/`, `filtered/` | `relation`, `question`, `answer`, `options`, `documents`, `doc_ids`, `seed`, `generator`, `status`, `reject_reason`, `raw_output`, `meta` |

The full schemas are in the [Python API](python-api.md#records) reference.

## The run manifest

`run_manifest.json` is updated after every stage, so it stays accurate even
when a later stage fails:

```json
{
  "run_id": "3f9a1c2b7d10",
  "config_fingerprint": {"hash": "a41c09e2f7b3", "project": "…", "extract": {…}, "generate": {…}},
  "stages": {
    "extract":  {"inputs": 42, "outputs": 41, "errors": 1, "details": {"ocr_pages": 12, …}},
    "generate": {"inputs": 500, "outputs": 2431, "rejected": 69, "details": {"per_relation": {…}}},
    "filter":   {"inputs": 2431, "outputs": 2140, "details": {"reject_reasons": {"duplicate": 120, …}}}
  },
  "runs": [ … the last 20 runs … ]
}
```

`config_fingerprint.hash` changes whenever any setting changes. Two
datasets with the same hash were produced with the same settings.
