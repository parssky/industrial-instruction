# Seeds

Seeds are example questions ("simulated instructions"). For every seed, the
generator writes new questions in the same style and format, grounded in
documents retrieved for that seed. Seeds steer **how** questions are asked;
your documents decide **what** they are about.

## Sources

=== "Hugging Face dataset (paper)"

    ```yaml
    seeds:
      source: huggingface
      dataset: ibm-research/FailureSensorIQ   # or a local save_to_disk folder
      splits: [org, pert]
      text_field: prompt          # what the generator imitates
      query_field: question       # what retrieval searches with
    ```

=== "Your own file"

    ```yaml
    seeds:
      source: jsonl               # jsonl or json
      path: data/my_seeds.jsonl
      text_field: instruction
      id_field: id
    ```

    ```json
    {"id": "s1", "instruction": "Which inspection interval applies to the drive-end bearing?"}
    ```

    A JSON file may also be a plain list of strings.

=== "Bootstrapped from your corpus"

    ```yaml
    seeds:
      source: self
      limit: 200
    ```

    Samples your chunks and asks the generator for one realistic
    practitioner question per chunk, so a new corpus needs no seed data at
    all. Uses the `self_seed` prompt.

## Common settings

```yaml
seeds:
  shuffle: true
  seed: 42          # shuffling is reproducible
  limit: null       # cap the number of seeds; try 20 for a first run
```

## `text_field` and `query_field`

They do different jobs:

- `text_field` is shown to the generator as `<Simulated Instruction>`, so it
  should be the full example, options and wording included.
- `query_field` is what retrieval searches with. FailureSensorIQ's `prompt`
  wraps every question in the same option and answer-format boilerplate,
  which dilutes the search. Its `question` field holds only the question,
  and retrieves much better.

If the row has no `query_field`, the `text_field` is used for both.

## Multiple-choice seeds

When a seed contains options (an `options` list in the row, or an
`Options:` block in its text), it is a multiple-choice seed. With
`generate.options_mode: auto`, it produces multiple-choice samples. See
[Generation](generation.md#multiple-choice-samples).
