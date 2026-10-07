# Assembling the dataset <span class="ii-stage">ii assemble</span>

Assembling merges the filtered relations into train/test splits, in two
formats.

```bash
ii assemble         # artifacts/filtered/ → artifacts/dataset/
```

```yaml
assemble:
  splits: { train: 0.9, test: 0.1 }
  split_seed: 42
  stratify_by_relation: true   # every relation appears in every split, in proportion
  chat_format: true            # also write <split>.chat.jsonl
  include_documents: true      # include the documents in the prompt
  push_to_hub: null            # e.g. your-name/your-dataset
  private: true
```

Splits are deterministic: the same samples and `split_seed` always give the
same split.

## Formats

`<split>.jsonl`: one flat record per sample:

```json
{"id": "8f2c…", "relation": "r3", "question": "Which oil grade does the P-100 pump require?",
 "answer": ["A"], "options": ["A. ISO VG 46", "B. ISO VG 100"], "documents": ["…"],
 "seed": "…", "doc_ids": ["c41…"]}
```

`<split>.chat.jsonl`: messages for supervised fine-tuning (TRL, Unsloth,
axolotl):

```json
{"id": "8f2c…", "relation": "r3", "messages": [
  {"role": "user", "content": "<Documents>\n…\n</Documents>\n\nWhich oil grade does the P-100 pump require?\n\nA. ISO VG 46\nB. ISO VG 100"},
  {"role": "assistant", "content": "[\"A\"]"}]}
```

Multiple-choice answers are written as a JSON list of labels. `dataset_stats.json`
records the counts per split and relation.

## Hugging Face Hub

With `push_to_hub: your-name/your-dataset` (and `pip install -e '.[hub]'`,
logged in with `huggingface-cli login`), the splits are uploaded as a
private dataset unless `private: false`.

## Benchmarking your own test split

`artifacts/dataset/test.jsonl` is the `generated` benchmark suite:

```bash
ii bench --suite generated --context none --context gold
```
