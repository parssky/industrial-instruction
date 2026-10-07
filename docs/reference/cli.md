# CLI reference

All commands are subcommands of `ii` (also installed as
`industrial-instruction`).

```text
ii [-v] <command> [-c CONFIG] [--set KEY=VALUE ...] [command options]
```

| Global option | |
|---|---|
| `-v`, `--verbose` | debug logging, and full tracebacks on errors |

| Option on every command except `init` | |
|---|---|
| `-c`, `--config FILE` | the config file (default `./industrial_instruction.yaml`, else built-in defaults) |
| `--set KEY=VALUE` | override one setting; repeatable; values are parsed as YAML. See [Configuration](../guide/configuration.md#overriding-settings-without-editing-the-file). |

## Project

`ii init [-o FILE] [--force]`
:   Write the commented starter config (default `industrial_instruction.yaml`)
    and create `data/pdfs/` and `artifacts/documents/`. Refuses to overwrite
    an existing file without `--force`.

`ii info`
:   Print the resolved configuration summary, its fingerprint, and which
    artifacts exist.

## Building a dataset

| Command | Does | Guide |
|---|---|---|
| `ii extract` | PDFs → markdown documents (with OCR if enabled) | [Extraction](../guide/extract.md) |
| `ii chunk` | documents → `chunks.jsonl` | [Chunking & indexing](../guide/chunking-indexing.md) |
| `ii index` | chunks → FAISS index | [Chunking & indexing](../guide/chunking-indexing.md) |
| `ii generate` | seeds + retrieval + LLM → samples per relation | [Generation](../guide/generation.md) |
| `ii filter` | rule checks (+ judge) → kept samples | [Filtering](../guide/filtering.md) |
| `ii assemble` | splits and formats → dataset | [Assembling](../guide/assemble.md) |
| `ii run [--stages a,b,c]` | all stages, or a subset, always in pipeline order | [User guide](../guide/index.md) |

Each stage prints its report (inputs, outputs, rejected, errors, details) as
JSON and records it in `artifacts/run_manifest.json`.

### `ii index` options

| Option | |
|---|---|
| `--corpus PATH` | build the index in one step from a PDF folder, a markdown/text folder, or a passages `.jsonl` |
| `--text-field NAME` | text key in a passages file (default `text`) |
| `--id-field NAME` | id key in a passages file (default `id`) |

## Benchmarking

`ii bench`
:   Score a served model. See [Benchmarking](../guide/benchmarking.md).

| Option | |
|---|---|
| `--suite NAME` | `ibm`, `paper-qwen`, `paper-claude`, `generated`, `custom` or a registered suite; repeatable (default `benchmark.suites`) |
| `--context MODE` | `none`, `gold` or `retrieved`; repeatable (default `benchmark.contexts`) |
| `--base-url URL` | the model server, e.g. `http://localhost:8000/v1` |
| `--model NAME` | the served model name (default: the first one the server lists) |
| `--limit N` | score only the first N items of each suite |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | an error; the message says what to change (use `-v` for the traceback) |
| `130` | interrupted with Ctrl-C |
