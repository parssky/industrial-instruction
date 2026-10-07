# Retrieval

Generation retrieves documents for every seed, and `ii bench --context
retrieved` retrieves documents for every benchmark question. Both use the
same retriever, chosen by `retrieval.source`:

| `source` | Use it when |
|---|---|
| [`index`](#1-index-build-a-vector-store-from-your-corpus) (default) | you want the package to build the vector store from your corpus |
| [`faiss`](#2-faiss-an-index-you-built-yourself) | you already have a FAISS index, including the paper's published one |
| [`function`](#3-function-your-own-search-service) | your documents live in a search service (Elasticsearch, Qdrant, a reranking pipeline…) |

## 1. `index`: build a vector store from your corpus

```bash
ii index --corpus data/pdfs           # PDFs: extract + chunk + embed
ii index --corpus docs/               # a folder of markdown or text files
ii index --corpus passages.jsonl      # ready-made passages, indexed as given
```

The embedding model is the one in `embed` (see
[Chunking & indexing](chunking-indexing.md#embedding-model)).

**Passages file.** One JSON object per line. The text is required, the id
optional; any other fields are kept as metadata:

```json
{"id": "p-001", "text": "Use ISO VG 46 hydraulic oil...", "doc_id": "pump-manual", "page": 12}
```

```bash
ii index --corpus kb.jsonl --text-field body --id-field doc_key
```

The passages are also written to `artifacts/chunks.jsonl`, so
`seeds.source: self` can bootstrap questions from the same corpus.

If you run `ii generate` or `ii bench --context retrieved` before an index
exists, the command stops before sending any request and tells you which of
these options to use.

## 2. `faiss`: an index you built yourself

```yaml
retrieval:
  source: faiss
  path: my_index/                  # a folder, an index file, or hf:owner/repo
  mapping_path: null               # only if id_map.json isn't next to the index
  text_field: text                 # when id-map entries are dicts
  embed:                           # the model the index was built with
    backend: sentence_transformers
    model: google/embeddinggemma-300m
```

Two formats are read:

| Format | Index file | `id_map.json` |
|---|---|---|
| this package | `index.faiss` | `{"0": {"id": ..., "doc_id": ..., "text": ..., ...}}` |
| the paper's original code, or any simple setup | `index`, `index.faiss`, `faiss.index` or `*.faiss` | `{"0": "chunk text"}` or `{"0": {"text": ..., "page": ...}}` |

In a folder (or a downloaded Hub repo), the index is found anywhere in the
tree, and the id map is looked up next to it.

**The paper's index from the Hub:**

```bash
ii bench --suite paper-claude --context retrieved \
  --set retrieval.source=faiss --set retrieval.path=hf:Parssky/industrial-instruction-faiss
```

!!! info "Use the same embedding model"
    A FAISS index only works with the model that built it. `retrieval.embed`
    names that model (default: the `embed` section). The vector width is
    checked when the index is opened. Indexes in the original format were
    built with plain `encode(text)`, so they are queried without the
    `embed` prefixes, unless you set `retrieval.embed` yourself.

## 3. `function`: your own search service

Write a function that takes the query and `k`, and returns the documents:

```python title="retrievers/es.py"
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def search(query: str, k: int):
    hits = es.search(index="manuals", query={"match": {"text": query}}, size=k)["hits"]["hits"]
    return [{"text": h["_source"]["text"], "score": h["_score"], "id": h["_id"]} for h in hits]
```

```yaml
retrieval:
  source: function
  backend: retrievers/es.py:search     # or my_pkg.search:run, or a registered name
```

The function may return:

- **strings**: the document texts;
- **dicts** with `text` (or `retrieval.text_field`), plus optional `score`,
  `id`, `doc_id` and any metadata;
- **`Chunk` or `SearchHit` objects**, for full control.

### Registering in Python, and factories

```python
from industrial_instruction.store import register_retriever, retriever_backend

register_retriever("es", search)       # retrieval.backend: es

@retriever_backend("qdrant", factory=True)
def build(cfg):                         # called once per run with RetrievalConfig
    client = QdrantClient(cfg.options["url"])
    return lambda query, k: [p.payload["text"] for p in client.query(cfg.options["collection"], query, limit=k)]
```

```yaml
retrieval:
  source: function
  backend: qdrant
  options: { url: http://localhost:6333, collection: manuals }
```

## Which documents each consumer asks for

| Consumer | Query | k |
|---|---|---|
| generation | the seed's `query_field` (FailureSensorIQ: the bare question) | `max(generate.retrieval_k, k_docs of each relation)` |
| `ii bench --context retrieved` | the benchmark question | `benchmark.retrieval_k` (default 3) |
