# Generation <span class="ii-stage">ii generate</span>

For every seed, generation retrieves documents and asks the generator LLM for
one question–answer sample per **retrieval relation**: how the documents
relate to the question.

```bash
ii generate         # → artifacts/generated/r0.jsonl … r4.jsonl, rejects.jsonl
```

## The five relations

| Relation | Docs | The question is… | Teaches the model to… |
|---|---|---|---|
| `r0` | 1 | related to the document, which cannot answer it | ignore irrelevant context |
| `r1` | 1 | supported by hints in the document, which doesn't state the answer | combine context with its own knowledge |
| `r2` | 3 | supported by hints spread across several documents | combine partial evidence |
| `r3` | 1 | fully answerable from the document | read the answer from context |
| `r4` | 3 | answerable only by combining documents (multi-hop) | reason across documents |

All relations of one seed share a single retrieval call. Single-document
relations use the top hit; multi-document relations use the top `k_docs`.

## The generator

```yaml
generate:
  base_url: null            # null = OpenAI API; http://localhost:8000/v1 for vLLM
  model: gpt-4.1-mini
  api_key_env: OPENAI_API_KEY
  temperature: 0.1
  max_tokens: null
  max_workers: 8            # seeds processed in parallel
  max_retries: 3
  request_json_object: true # JSON mode; dropped automatically if the server rejects it
  retrieval_k: 3
```

Any OpenAI-compatible server works. Each relation is generated and retried
on its own: a reply that isn't valid JSON costs one sample, not the whole
seed. Code fences, prose before the JSON and trailing commas are repaired
before parsing.

## Multiple-choice samples

```yaml
generate:
  options_mode: auto        # auto | always | never
  n_options: 5
```

| `options_mode` | Output |
|---|---|
| `auto` | multiple-choice when the seed is multiple-choice, as in the paper's seeds; question–answer otherwise |
| `always` | every sample is multiple-choice |
| `never` | question–answer only |

A multiple-choice sample has a question, options `A.`–`E.`, and the answer
as a list of labels. Model replies vary, so they are normalized before
filtering:

| Reply | Stored as |
|---|---|
| options inside the question text | moved to `options` |
| `["resistance", "torque", ...]` (no labels) | `["A. resistance", "B. torque", ...]` |
| labels `P`–`T` copied from a perturbed seed | remapped to `A`–`E`, answer included |
| `{"answer": ["Q"]}`, `"B, D"`, `["b"]` | `["B"]`, `["B", "D"]`, `["B"]` |

## Prompts

Every prompt is a plain text file:

| File | Used for |
|---|---|
| `useless_doc.txt` | r0 |
| `single_doc_support.txt` | r1 |
| `multi_doc_support.txt` | r2 |
| `single_doc_answer.txt` | r3 |
| `multi_doc_answer.txt` | r4 |
| `self_seed.txt` | `seeds.source: self` |
| `judge.txt` | the [LLM judge](filtering.md#llm-judge) |

To change one, copy it into a folder, edit it, and point the config there.
Any file you don't override uses the packaged version:

```yaml
generate:
  prompt_dir: prompts/
```

Templates use `{docs}`, `{simulated_instruction}`, `{options_rule}` and
`{output_rule}` placeholders. Literal JSON braces are fine. Every sample
records its template name and hash (`meta.prompt_template`,
`meta.prompt_sha`).

## Adding a relation

Relations are configuration:

```yaml
generate:
  relations:
    - { id: r0, prompt: useless_doc }
    - { id: r3, prompt: single_doc_answer }
    - id: r5
      prompt: comparison            # prompts/comparison.txt
      doc_mode: multi
      k_docs: 4
      description: compare two datasheets
      requires_options: false
```

Setting `relations` replaces the default list, so include the defaults you
want to keep, or disable one with `enabled: false`.

## Cost

One generator call per seed × relation: 500 seeds × 5 relations is 2,500
calls, plus one per seed with `seeds.source: self`. Start with
`--set seeds.limit=20`.

## Output

`artifacts/generated/<relation>.jsonl` holds one sample per line:

```json
{"id": "8f2c…", "relation": "r3", "question": "Which oil grade does the P-100 pump require?",
 "answer": ["A"], "options": ["A. ISO VG 46", "B. ISO VG 100", "…"],
 "documents": ["## Pump P-100 > Lubrication\n\nUse ISO VG 46 …"], "doc_ids": ["c41…"],
 "seed": "Please select the correct option(s)…", "generator": "gpt-4.1-mini",
 "meta": {"mcq": true, "retrieval_scores": [0.83], "prompt_sha": "1a2b…"}}
```

Failed or invalid replies go to `rejects.jsonl`, with the reason and the raw
model output.
