# Configuration

One YAML file drives a whole run. `ii init` writes a commented starter,
`industrial_instruction.yaml`. Every key is optional and falls back to its
default, so the file only needs the settings you change.

```yaml title="industrial_instruction.yaml"
generate:
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen2.5-32B-Instruct
seeds:
  limit: 200
benchmark:
  suites: [ibm, generated]
```

The complete file, with every key and its default, is on the
[configuration reference](../reference/configuration.md) page.

## Sections

| Section | Controls | Guide |
|---|---|---|
| `paths` | where inputs and artifacts live | [Outputs](../reference/outputs.md) |
| `extract` | PDF backend, tables, page cleanup, `ocr` | [Extraction](extract.md), [OCR](ocr.md) |
| `chunk` | chunking strategy and sizes | [Chunking & indexing](chunking-indexing.md) |
| `embed`, `store` | embedding model, FAISS index type | [Chunking & indexing](chunking-indexing.md) |
| `retrieval` | where generation and benchmarking get documents | [Retrieval](retrieval.md) |
| `seeds` | the example questions the generator imitates | [Seeds](seeds.md) |
| `generate` | generator LLM, relations r0–r4, multiple-choice mode | [Generation](generation.md) |
| `filter` | rule checks, LLM judge | [Filtering](filtering.md) |
| `assemble` | splits and output formats | [Assembling the dataset](assemble.md) |
| `benchmark` | served model, suites, contexts | [Benchmarking](benchmarking.md) |

## Paths

All paths are relative to `paths.root`. When the config is loaded from a
file, `paths.root` defaults to that file's folder, so a project is
self-contained and can be run from anywhere:

```bash
ii run -c projects/plant-b/industrial_instruction.yaml
```

## Overriding settings without editing the file

Every command accepts `--set key=value`, as often as needed. Keys are dotted
paths into the config, and values are parsed as YAML, so numbers, booleans,
`null` and lists keep their types:

```bash
ii generate --set generate.model=gpt-4.1 --set generate.max_workers=16
ii run --set seeds.limit=20 --set "seeds.splits=[org]"
ii extract --set extract.ocr.mode=auto --set extract.ocr.backend=ocr/my_model.py:run
```

`-c/--config` picks the file; without it, `./industrial_instruction.yaml` is
used if it exists, and the built-in defaults otherwise.

## Validation

The configuration is validated when it is loaded:

- **unknown keys are errors**, so a typo can't silently fall back to a default;
- **values are type-checked**;
- **choices are checked**, for example `extract.ocr.mode` must be `off`, `auto` or `always`.

```bash
ii info     # print the resolved configuration, its fingerprint, and which artifacts exist
```

!!! warning "`off` and `on` in YAML"
    YAML reads an unquoted `off` as `false`. Settings that accept `off`
    (`extract.ocr.mode`) handle this for you, but quoting is always safe:
    `mode: "off"`.

## Secrets

API keys never go in the YAML file. Settings named `*api_key_env` hold the
**name of an environment variable** (default `OPENAI_API_KEY`):

```yaml
generate:
  api_key_env: MY_PROVIDER_KEY     # the key is read from $MY_PROVIDER_KEY
```

Local vLLM servers accept any key, so nothing needs to be set for them.

## Reproducibility

Every stage writes its counts and a fingerprint of the full configuration to
`artifacts/run_manifest.json`, so any dataset can be traced back to the
settings that produced it. See [Outputs](../reference/outputs.md).
