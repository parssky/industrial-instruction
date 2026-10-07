# Chunking & indexing <span class="ii-stage">ii chunk</span> <span class="ii-stage">ii index</span>

Chunking splits documents into passages, the unit that is retrieved,
embedded and shown to the generator. Indexing embeds the passages into a
FAISS vector store.

```bash
ii chunk          # artifacts/documents/ → artifacts/chunks.jsonl
ii index          # artifacts/chunks.jsonl → artifacts/faiss/
```

!!! tip "Straight from a corpus"
    `ii index --corpus data/pdfs` runs extract, chunk and index in one step.
    It also accepts a markdown folder or a `.jsonl` of ready-made passages.
    See [Retrieval](retrieval.md#1-index-build-a-vector-store-from-your-corpus).

## Chunking

```yaml
chunk:
  strategy: heading           # heading | fixed
  max_chars: 2000
  min_chars: 120
  overlap: 200
  keep_tables_whole: true
  include_heading_in_text: true
```

`heading` (default)
:   Splits on markdown headings. Sections longer than `max_chars` are packed
    paragraph by paragraph, and a table is never split. That can push a
    chunk past `max_chars`, but a half table produces unanswerable
    questions. Sections shorter than `min_chars` are **merged into the next
    chunk, never dropped**, because one-line specifications ("Use ISO VG 46
    oil.") are often the most valuable content. A document without any
    headings falls back to `fixed`.

`fixed`
:   Character windows of `max_chars` with `overlap`, cut at word boundaries.

With `include_heading_in_text`, every chunk starts with its breadcrumb, so
the embedder and the generator know where a passage comes from:

```markdown
## Pump P-100 Service Manual > 1 Lubrication

Use ISO VG 46 hydraulic oil. Replace the oil every 2000 operating hours.
```

## Embedding model

```yaml
embed:
  backend: sentence_transformers    # sentence_transformers | openai | hash
  model: google/embeddinggemma-300m # Hub id or local folder
  batch_size: 32
  device: null                      # e.g. cuda
  normalize: true                   # inner product = cosine similarity
  query_prefix: "task: search result | query: "
  document_prefix: "title: none | text: "
```

`sentence_transformers`
:   Any model sentence-transformers can load, run locally. Install with `.[local-embed]`.

`openai`
:   Any OpenAI-compatible embeddings endpoint (`base_url`, `api_key_env`), for
    example a vLLM or TEI server.

`hash`
:   Deterministic, offline, meaningless vectors. For tests only (needs `dimension`).

**Prefixes.** Asymmetric retrieval models such as EmbeddingGemma expect
different instructions for queries and documents. The defaults match
EmbeddingGemma; clear both for models that don't use prefixes.

## The FAISS store

```yaml
store:
  index_type: flat_ip     # flat_ip (exact cosine) | flat_l2 | hnsw (approximate, for large corpora)
  hnsw_m: 32
```

The index folder holds `index.faiss`, `id_map.json` (the full chunk records,
with source file and headings) and `store_meta.json` (the embedding model,
dimension and prefixes). When the index is loaded, the stored model is
compared with `embed`. A different model triggers a warning, and a different
vector width is an error, because querying with the wrong model returns
quietly wrong results.

To query an index you built elsewhere, or a search service instead of FAISS,
see [Retrieval](retrieval.md).
