# Benchmarking <span class="ii-stage">ii bench</span>

`ii bench` scores a model you serve on one or more suites. The package only
calls the model's OpenAI-compatible endpoint; it never starts or stops it.

## 1. Serve the model

```bash
vllm serve Parssky/industrial-instruction-qwen4b-claude --port 8000
# a local fine-tune works the same way:
vllm serve ./checkpoints/my-sft --served-model-name my-sft --port 8000
```

Any OpenAI-compatible server works (vLLM, SGLang, TGI, llama.cpp server, a
hosted API).

## 2. Run

```bash
ii bench --base-url http://localhost:8000/v1 --suite ibm
ii bench --suite paper-claude --suite paper-qwen --context none --context gold
ii bench --suite generated --context retrieved
ii bench --suite custom --set benchmark.custom.path=my_bench.jsonl --limit 50
```

`--suite` and `--context` can be repeated; every suite runs in every
context. `--model` picks the served model name (default: the first one the
server lists). `--limit N` scores the first N items of each suite, for a
quick check.

```text
model: Parssky/industrial-instruction-qwen4b-claude
  ibm-none                 acc_org 50.7%  acc_pert 50.8%  consistency 40.5%  (n=2667 pairs)
results: artifacts/benchmarks/Parssky_industrial-instruction-qwen4b-claude/20261007-101500
```

## Suites

| Suite | Data | Metrics |
|---|---|---|
| `ibm` | [FailureSensorIQ](https://huggingface.co/datasets/ibm-research/FailureSensorIQ): each question in an original (`org`) and a perturbed (`pert`) form; prompts used as published | `acc_original`, `acc_perturb`, `consistency` |
| `paper-qwen` | held-out test split of the paper's Qwen-generated data | set match, F1, Jaccard |
| `paper-claude` | held-out test split of the paper's Claude-generated data | set match, F1, Jaccard |
| `generated` | `artifacts/dataset/test.jsonl` from your own run | set match, F1, Jaccard |
| `custom` | any multiple-choice set | set match, F1, Jaccard |

The paper splits are loaded by name from
[`Parssky/industrial-instruction-dataset`](https://huggingface.co/datasets/Parssky/industrial-instruction-dataset).
The name is matched against the repo's configs and splits; if nothing
matches, the error lists what exists. A local `save_to_disk` folder also
works:

```yaml
benchmark:
  paper_qwen_split: pana_qa_v1_test               # Hub name…
  paper_claude_split: /data/pana_qa_claude_v1_test  # …or a local folder
```

## Context modes

| `--context` | The model sees | Available for |
|---|---|---|
| `none` | the question only (closed book) | every suite |
| `gold` | the item's own source documents: the paper's RAG setting | suites with documents (not `ibm`) |
| `retrieved` | the top `retrieval_k` documents from the [retriever](retrieval.md) | every suite |

Documents are put in front of the question with `benchmark.context_template`.
By default this is the exact template the paper's models were trained
with, so fine-tuned checkpoints see their training format:

```text
        Based on relevat document answer this question.
        relevant document: {documents}
        question: {question}
```

With `retrieved`, the retriever is opened before the first request, so a
missing index fails immediately instead of after the run.

## Your own benchmark

Any multiple-choice set with label answers:

```json title="my_bench.jsonl"
{"id": "q1", "question": "Which fuse rating protects the P-100 motor?", "options": ["5 A", "10 A", "15 A"], "answer": ["B"], "documents": ["10 A fuses protect the motor."]}
```

```yaml
benchmark:
  suites: [custom]
  custom:
    source: jsonl            # jsonl | json | huggingface | disk
    path: my_bench.jsonl     # file, Hub id, or save_to_disk folder
    split: test              # for huggingface / disk / json-with-splits
    question_field: question
    answer_field: answer     # ["B"], "B", "B, D", "['B']" or {"answer": [...]}
    options_field: options   # labelled A., B., … and appended unless already in the question
    documents_field: documents
    id_field: id
```

Items with a free-text answer can't be scored by label matching. They are
skipped, and the number skipped is logged.

To add a suite in Python, write a loader that returns `BenchItem`s:

```python
from industrial_instruction.benchmark import BenchItem, register_suite

register_suite("plant-b", lambda config: [
    BenchItem(id=r["id"], prompt=r["text"], query=r["question"], gold=r["labels"])
    for r in load_rows()
])
```

## Scoring

**Parsing.** One parser reads every reply format the paper's models
produce:

| Reply | Parsed |
|---|---|
| `{"answer": ["B"]}`, `{"answer": "B"}` | `B` |
| `['B', 'D']` (Python list) | `B, D` |
| `{"answer": ["B"}` (malformed) | `B` |
| `<think>…</think>{"answer": ["C"]}` | `C` |
| `{"option_a": "...", "answer": ["T"]}` | `T` |
| `Answer: B`, `The answer is (B) and D` | `B`; `B, D` |
| `B, D` (labels only) | `B, D` |
| `The answer is a pressure valve.` | no answer: never stray letters |

A reply with no answer scores 0. Its share is reported as `parse_failures`
(shown as `unparsed`).

**Set metrics** (paper, generated, custom) compare the set of predicted
labels with the gold set:

- **set match**: exactly equal;
- **F1**: harmonic mean of label precision and recall;
- **Jaccard**: intersection over union.

All three are averaged over items.

**IBM metrics:**

- `acc_original` and `acc_perturb`: exact set match on the two forms;
- `consistency`: the share of question pairs where **both** forms are right.

Perturbed option labels `P)`–`T)` are normalized before matching.

!!! warning "Comparing with the original IBM evaluation script"
    The original script compared option ids such as `P)` with answers such
    as `P`, so a perturbed question could never be scored correct: every
    model got `acc_perturb ≈ 0` and `consistency ≈ 0`. The `ibm` suite fixes
    this, and also reports the old rule under `legacy` in `summary.json`, so
    new results can be compared with numbers produced by that script.
    Replaying the paper's saved replies, the legacy scores match the
    reported ones exactly, and the corrected `acc_perturb` of the base
    Qwen3-4B model is 38–45%.

## Results

Every run writes `artifacts/benchmarks/<model>/<time>/`:

| File | Contents |
|---|---|
| `summary.json` | model, endpoint, settings, retrieval source, and metrics per suite and context |
| `<suite>-<context>.samples.jsonl` | one row per item: gold labels, raw reply, parsed answer, correct, any request error |

A request that fails after retries is scored as wrong, counted under
`errors`, and logged, so a flaky server can't silently inflate or deflate
scores without you seeing it.

## Settings

```yaml
benchmark:
  endpoint:
    base_url: http://localhost:8000/v1
    model: null
    api_key_env: OPENAI_API_KEY
    temperature: 0.0
    max_tokens: 1024
    timeout: 300
    max_retries: 3
    max_workers: 32
    system_prompt: 'You are a helpful assistant. You must output your answer strictly as valid JSON in the format {"answer": ["choice"]}.'
  suites: [ibm]
  contexts: [none]
  retrieval_k: 3
  limit: null
  output_dir: artifacts/benchmarks
```
