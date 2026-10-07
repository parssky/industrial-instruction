# Filtering <span class="ii-stage">ii filter</span>

Filtering removes samples that would teach a model the wrong thing. Every
removed sample is kept, with its reasons, in
`artifacts/filtered/rejected.jsonl`, and the counts per reason are written
to the run manifest.

```bash
ii filter           # artifacts/generated/ → artifacts/filtered/
```

## Rule checks

| Reason | Removes | Setting |
|---|---|---|
| `question_too_short` / `question_too_long` | fragments and runaway generations | `min_question_chars` (20), `max_question_chars` (2000) |
| `missing_answer` / `answer_too_short` | samples without an answer | `require_answer`, `min_answer_chars` |
| `meta_reference` | "according to the documents…", which only makes sense with the context attached | `forbid_meta_references`, `meta_reference_phrases` |
| `no_context` | samples without documents | always on |
| `duplicate` | the same question within a relation (case and punctuation ignored) | `dedupe`, `dedupe_mode: normalized \| exact` |
| `format_instructions_in_question` | a seed's answer template (`{"answer": ...}`) copied into the question | `forbid_format_instructions` |

Multiple-choice samples are also checked for:

| Reason | Removes | Setting |
|---|---|---|
| `missing_options` | a multiple-choice sample without options | `require_options` |
| `wrong_option_count` | not exactly `generate.n_options` options | `enforce_option_count` |
| `answer_not_in_options` | answers that aren't option labels (`"EKMB11 series"`, `"F"`, empty) | `answer_in_options` |
| `copied_seed_options` | options copied from the seed instead of written for the new question | `max_seed_option_overlap` (0.6: more than 60% identical) |

```yaml
filter:
  min_question_chars: 20
  max_question_chars: 2000
  require_answer: true
  forbid_meta_references: true
  enforce_option_count: true
  require_options: true
  answer_in_options: true
  max_seed_option_overlap: 0.6
  forbid_format_instructions: true
  dedupe: true
  dedupe_mode: normalized
```

## LLM judge

```yaml
filter:
  judge_enabled: true
  judge_model: null         # null = generate.model
  judge_min_score: 3.0
  judge_max_workers: 8
```

The judge grades each surviving sample from 1 to 5 on four criteria:

- **faithfulness**: no invented specifications;
- **standalone**: the question makes sense without the documents;
- **relation match**: the documents relate to the question as the relation says;
- **usefulness** to a practitioner.

Samples whose mean score falls below `judge_min_score` are removed. It costs
one LLM call per sample; the prompt is `judge.txt` (see
[Prompts](generation.md#prompts)).

## Reading the results

```bash
jq -r .reject_reason artifacts/filtered/rejected.jsonl | sort | uniq -c | sort -rn
```

A high rate of one reason usually points to a fix upstream rather than a
filter setting:

| Frequent reason | Usually means |
|---|---|
| `copied_seed_options` | the generator imitates seeds too literally; try a stronger model |
| `answer_not_in_options` | the generator returns option text instead of labels |
| `meta_reference` | the prompt's standalone rule is being ignored |
| `duplicate` | too few seeds for the corpus, or a too-small corpus |
