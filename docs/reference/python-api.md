# Python API

Everything the CLI does is available from Python. Each section below is
generated from the code's docstrings.

```python
from industrial_instruction import Config, Pipeline

config = Config.from_yaml("industrial_instruction.yaml").with_override("seeds.limit", 20)
pipeline = Pipeline(config)
for report in pipeline.run_all():                  # or run_extract(), run_generate(), …
    print(report.stage, report.outputs, report.rejected)

from industrial_instruction.store import get_retriever
hits = get_retriever(config).search("bearing replacement interval", k=3)

from industrial_instruction.benchmark import run_benchmarks, format_table
summary = run_benchmarks(config, suites=["ibm", "generated"], contexts=["none", "gold"])
print(format_table(summary))
```

## Plug-in points

| Plug-in | Register with | Selected by |
|---|---|---|
| OCR model | `register_ocr`, `@ocr_backend` | `extract.ocr.backend` |
| Retriever | `register_retriever`, `@retriever_backend` | `retrieval.backend` |
| PDF extractor | `register_extractor` | `extract.backend` |
| Embedding model | `register_embedder` | `embed.backend` |
| Benchmark suite | `register_suite` | `benchmark.suites`, `--suite` |
| Prompts | `.txt` files | `generate.prompt_dir` |

Where a plain function is enough (OCR, retrievers), the config can also name
it directly as `my_pkg.module:function` or `path/to/file.py:function`,
relative to the config file.

## Pipeline

::: industrial_instruction.pipeline.Pipeline

## Configuration

::: industrial_instruction.config.Config
    options:
      members: [from_yaml, from_dict, with_override, with_overrides, fingerprint]

## OCR

::: industrial_instruction.ocr.OCRPage
::: industrial_instruction.ocr.register_ocr
::: industrial_instruction.ocr.ocr_backend
::: industrial_instruction.ocr.PageOCR

## Retrieval

::: industrial_instruction.store.retrieval.get_retriever
::: industrial_instruction.store.retrieval.register_retriever
::: industrial_instruction.store.retrieval.retriever_backend
::: industrial_instruction.store.faiss_store.FaissStore
::: industrial_instruction.store.faiss_store.SearchHit
::: industrial_instruction.store.runner.index_corpus

## Extraction and embedding

::: industrial_instruction.extract.base.Extractor
::: industrial_instruction.extract.registry.register_extractor
::: industrial_instruction.embed.base.Embedder
::: industrial_instruction.embed.registry.register_embedder

## Generation

::: industrial_instruction.generate.engine.GenerationEngine
::: industrial_instruction.generate.engine.normalize_mcq

## Benchmarking

::: industrial_instruction.benchmark.run_benchmarks
::: industrial_instruction.benchmark.register_suite
::: industrial_instruction.benchmark.BenchItem
::: industrial_instruction.benchmark.Endpoint
::: industrial_instruction.benchmark.parse_answer

## Records

::: industrial_instruction.schemas.Document
::: industrial_instruction.schemas.Chunk
::: industrial_instruction.schemas.QASample
::: industrial_instruction.schemas.RelationSpec
::: industrial_instruction.schemas.StageReport
