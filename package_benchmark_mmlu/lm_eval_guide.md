# `lm-eval` CLI Reference

Quick reference for `lm_eval` command-line switches, based on the `lm-evaluation-harness` package.

---

## Core switches (what you're already using)

| Switch | Description | Example |
|---|---|---|
| `--model` | Backend/framework to load the model with. `hf` = Hugging Face transformers. Other options: `vllm`, `openai-completions`, `anthropic`, `gguf`, etc. | `--model hf` |
| `--model_args` | Comma-separated key=value args passed to the model loader. Most common: `pretrained=<path>`, `dtype=`, `trust_remote_code=`, `peft=`. | `--model_args pretrained=/path/to/model,dtype=bfloat16` |
| `--tasks` | Comma-separated list of benchmark task names to run. Use `lm_eval --tasks list` to see all available tasks. | `--tasks mmlu,arc_challenge,hellaswag` |
| `--num_fewshot` | Number of few-shot examples prepended to each prompt. Applies to all tasks in `--tasks` unless overridden per-task in a YAML config. | `--num_fewshot 5` |
| `--batch_size` | Batch size for inference. `auto` lets the harness find the largest batch size that fits in memory. | `--batch_size auto` or `--batch_size auto:4` |
| `--output_path` | Directory where `results.json` (and sample logs, if enabled) are saved. | `--output_path ./result/my_model` |
| `--log_samples` | Saves every individual prompt/response/label to disk (not just aggregate scores). Needed for error analysis or the `results.json` per-sample breakdown. | `--log_samples` |
| `--device` | Compute device. `cuda:0` for first GPU, `cpu` for CPU-only. | `--device cuda:0` |
| `--limit` | Caps the number of examples evaluated **per subtask**. Accepts an integer (exact count) or a float 0–1 (fraction of the dataset). Use for smoke tests only — not statistically valid for real scores. | `--limit 20` or `--limit 0.01` |

---

## Model-loading args (inside `--model_args`)

These go as comma-separated `key=value` pairs within the single `--model_args` string.

| Arg | Description | Example |
|---|---|---|
| `pretrained` | Path (local) or HF Hub repo ID of the base model. | `pretrained=/home/parsa/.../my-model` |
| `dtype` | Precision to load weights in. `bfloat16`, `float16`, `float32`, or `auto`. | `dtype=bfloat16` |
| `trust_remote_code` | Required for some custom model architectures (e.g. Qwen) that ship custom modeling code. | `trust_remote_code=True` |
| `peft` | Path to a LoRA/PEFT adapter to apply on top of the base model — avoids needing to merge weights first. | `peft=/path/to/lora_adapter` |
| `parallelize` | Splits the model across multiple GPUs (model parallelism) for models too large for one GPU. | `parallelize=True` |
| `attn_implementation` | Attention backend, e.g. `flash_attention_2` for speed if installed. | `attn_implementation=flash_attention_2` |
| `revision` | Specific model revision/commit/branch on HF Hub. | `revision=main` |

---

## Other useful top-level switches

| Switch | Description | Example |
|---|---|---|
| `--tasks list` | Prints every available task/benchmark name instead of running anything. | `lm_eval --tasks list` |
| `--include_path` | Directory containing custom/local task YAML definitions, if you've written your own benchmark task. | `--include_path ./my_tasks/` |
| `--gen_kwargs` | Comma-separated generation parameters for generative tasks (e.g. GSM8K), like `temperature`, `top_p`, `max_gen_toks`. | `--gen_kwargs temperature=0,max_gen_toks=256` |
| `--apply_chat_template` | Formats prompts using the model's chat template (recommended for instruct/chat-tuned models like Qwen3-Instruct). | `--apply_chat_template` |
| `--fewshot_as_multiturn` | When using a chat template, sends few-shot examples as separate conversation turns rather than concatenated into one prompt. | `--fewshot_as_multiturn` |
| `--system_instruction` | Custom system prompt to prepend for chat-template evaluation. | `--system_instruction "You are a helpful assistant."` |
| `--num_fewshot_seeds` | Number of different random seeds to sample few-shot examples from, for variance estimation. | `--num_fewshot_seeds 3` |
| `--seed` | Random seed for reproducibility (few-shot sampling, etc.). | `--seed 1234` |
| `--cache_requests` | Caches tokenized requests to speed up repeated runs on the same task/model. Options: `true`, `refresh`, `delete`. | `--cache_requests true` |
| `--check_integrity` | Runs internal task-definition sanity checks before evaluation. | `--check_integrity` |
| `--write_out` | Writes formatted prompts (before generation) to disk for manual inspection. | `--write_out` |
| `--predict_only` | Only generates model outputs without scoring — useful for later custom evaluation of generative tasks. | `--predict_only` |
| `--use_cache` | Path to a local cache directory to store/reuse computed results. | `--use_cache ./lm_eval_cache` |
| `--verbosity` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. | `--verbosity DEBUG` |
| `--wandb_args` | Logs results directly to Weights & Biases. | `--wandb_args project=my-eval,name=run1` |

---

## Example: your current command, annotated

```bash
lm_eval \
  --model hf \                                    # use HF transformers backend
  --model_args pretrained=/home/parsa/panasonic/gallery/Qwen3-4B-Instruct-2507-ft-panasonic_qa_claude_v1_train-2x32-sft \
  --tasks mmlu \                                  # run MMLU benchmark
  --num_fewshot 5 \                                # standard 5-shot setting for MMLU
  --batch_size auto \                              # auto-detect max batch size
  --output_path ./result/Qwen3-4B-Instruct-2507-ft-panasonic_qa_claude_v1_train-2x32-sft \
  --log_samples \                                  # save per-example outputs
  --device cuda:0 \                                # run on first GPU
  --limit 20                                       # (optional) smoke test on 20 examples/subtask
```

## Recommended addition for Qwen3-Instruct models

Since your model is an **instruct** checkpoint, it's worth adding the chat template flag so prompts are formatted the way the model was fine-tuned to expect:

```bash
--apply_chat_template --fewshot_as_multiturn
```

This can noticeably change scores vs. raw-text prompting, so make sure you use the **same setting** for both the base model and fine-tuned model runs — otherwise the before/after comparison isn't valid.

---

## Comparing base vs. fine-tuned results

After running both, each `--output_path` folder will have a `results.json`. Compare programmatically:

```python
import json

with open("result/base/results.json") as f:
    base = json.load(f)["results"]
with open("result/finetuned/results.json") as f:
    ft = json.load(f)["results"]

for task in base:
    b = base[task].get("acc,none", base[task].get("acc_norm,none"))
    f_ = ft[task].get("acc,none", ft[task].get("acc_norm,none"))
    print(f"{task}: base={b:.4f}  finetuned={f_:.4f}  delta={f_-b:+.4f}")
```
