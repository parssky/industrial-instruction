# Quick start

This walks through one complete run on your own PDFs: from extraction to a
dataset, then a benchmark of a served model.

## 1. Create a project

```bash
mkdir my-dataset && cd my-dataset
ii init                          # writes industrial_instruction.yaml, creates data/pdfs/
cp ~/manuals/*.pdf data/pdfs/
```

All paths in the config are relative to the folder holding the config file.

## 2. Choose the generator

=== "OpenAI API"

    ```bash
    export OPENAI_API_KEY=sk-...
    ```

=== "Local model on vLLM"

    ```bash
    vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000
    ```

    ```yaml title="industrial_instruction.yaml"
    generate:
      base_url: http://localhost:8000/v1
      model: Qwen/Qwen2.5-32B-Instruct
    ```

## 3. Run a small batch first

```bash
ii run --set seeds.limit=20
```

This runs every stage on 20 seeds, which is 100 generator calls across
relations r0–r4. When it finishes:

| Look at | To check |
|---|---|
| `artifacts/documents/*.md` | extraction quality: headings, tables, no headers or footers |
| `artifacts/generated/rejects.jsonl` | replies the generator got wrong, with the raw output |
| `artifacts/filtered/rejected.jsonl` | samples the filter removed, with reasons |
| `artifacts/dataset/train.jsonl` | the result |
| `artifacts/run_manifest.json` | counts per stage and the config fingerprint |

!!! tip "Fix extraction first"
    Everything downstream depends on the markdown. If tables or headings look
    wrong, see [Extraction](../guide/extract.md). If pages are scanned, turn
    on [OCR](../guide/ocr.md).

## 4. Run at full size

```bash
ii run
```

Or one stage at a time, so you can inspect each result:

```bash
ii extract
ii chunk
ii index
ii generate
ii filter
ii assemble
```

## 5. Benchmark a model

Serve the model and point `ii bench` at it:

```bash
vllm serve Parssky/industrial-instruction-qwen4b-claude --port 8000
ii bench --base-url http://localhost:8000/v1 --suite ibm
```

```text
model: Parssky/industrial-instruction-qwen4b-claude
  ibm-none                 acc_org 50.7%  acc_pert 50.8%  consistency 40.5%  (n=2667 pairs)
results: artifacts/benchmarks/Parssky_industrial-instruction-qwen4b-claude/20261007-101500
```

These are this model's FailureSensorIQ scores, re-scored from the paper's
saved replies. Add `--suite generated` to score it on your own test split,
and `--context gold` to show it the source documents.

## Next steps

- [Configuration](../guide/configuration.md): every setting, and how to override it
- [Retrieval](../guide/retrieval.md): use your own index or search service
- [Benchmarking](../guide/benchmarking.md): suites, context modes and scoring
