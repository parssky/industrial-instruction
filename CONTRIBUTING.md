# Contributing to Industrial-Instruction

Thank you for helping. This guide covers everything from a first local
setup to getting a pull request merged: how the code is organized, the
rules the code follows, and step-by-step recipes for the most common
contributions (a new OCR model, retriever, extractor, embedder, benchmark
suite, generation relation or prompt).

If anything here is unclear or out of date, that is a bug too. Please
open an issue or fix it in your PR.

---

## Contents

1. [Ground rules](#1-ground-rules)
2. [Development setup](#2-development-setup)
3. [Project layout](#3-project-layout)
4. [How the pipeline fits together](#4-how-the-pipeline-fits-together)
5. [Development workflow](#5-development-workflow)
6. [Code standards](#6-code-standards)
7. [Testing](#7-testing)
8. [Recipes: common contributions](#8-recipes-common-contributions)
9. [Documentation](#9-documentation)
10. [Pull requests and review](#10-pull-requests-and-review)
11. [Continuous integration](#11-continuous-integration)
12. [Releases](#12-releases)
13. [Reporting bugs and proposing features](#13-reporting-bugs-and-proposing-features)

---

## 1. Ground rules

- **Be kind and constructive.** Review the code, not the person.
- **Scope.** The package does two things: it **builds datasets** from
  documents, and it **benchmarks** a served model. Model training is out of
  scope; the dataset feeds external trainers.
- **The research folders are frozen.** `create_dataset_notebooks/`,
  `create_vector_store_notebooks/`, `training_notebooks/` and
  `package_benchmark_*/` are kept exactly as they were run for the paper.
  Don't edit them. New behavior goes into `src/industrial_instruction/`.
- **Nothing is dropped silently.** If a stage discards data (a page, a
  sample, a reply), it records what and why, in a rejects file or in the
  stage report.
- **Reproducibility over cleverness.** A change to a prompt, a filter rule
  or a metric changes the dataset or the scores. Say so in the PR and in
  the docs.

---

## 2. Development setup

Requirements: Python 3.9 or newer, and git.

```bash
git clone https://github.com/parssky/industrial-instruction.git
cd industrial-instruction
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e ".[dev,pdf,pdfplumber,faiss,bench,docs]"
```

That is everything the tests and docs need. The full local embedding model
(`.[local-embed]`, which pulls in torch) is **not** needed for the tests,
which use the offline `hash` embedder.

Check the setup:

```bash
pytest -q                 # all tests, about 10 s, no network, no GPU
ruff check .
mkdocs serve              # docs at http://127.0.0.1:8000
```

With [uv](https://docs.astral.sh/uv/) the setup is
`uv venv && uv pip install -e ".[dev,pdf,pdfplumber,faiss,bench,docs]"`.

---

## 3. Project layout

```
src/industrial_instruction/
  cli.py              the `ii` command: thin wrapper over Pipeline and benchmark
  config.py           every setting, as pydantic models (one class per YAML section)
  configs/default.yaml  the starter config `ii init` writes
  schemas.py          data records: Document, Chunk, Seed, QASample, StageReport, RelationSpec
  pipeline.py         Pipeline: runs stages in order, writes run_manifest.json

  extract/            PDF → markdown
    base.py             Extractor base class (+ shared OCR helpers)
    pymupdf_extractor.py  default: font-size headings, in-place tables, margin cleanup
    pdfplumber_extractor.py, markdown_extractor.py
    registry.py         register_extractor / get_extractor
    runner.py           extract stage: walks paths.pdfs, writes documents
    tables.py, text_cleanup.py

  ocr/                pluggable OCR for scanned pages
    base.py             OCRPage, the page → markdown contract
    registry.py         register_ocr / @ocr_backend / import-path loading
    runner.py           PageOCR: concurrency, disk cache, per-page isolation
    openai_vision.py, tesseract.py   built-in backends

  chunk/chunker.py    heading-aware and fixed-window chunking
  embed/              embedding backends (sentence_transformers, openai, hash) + registry
  store/
    faiss_store.py      FaissStore: batched embedding, persistence, model checks
    retrieval.py        get_retriever: project index | user FAISS | user function
    runner.py           index stage, `ii index --corpus`

  generate/
    engine.py           GenerationEngine: seed × relation → QASample
    prompts/*.txt       the generation, self-seed and judge prompts
    prompt_loader.py    template rendering and user prompt overrides
    seeds.py            Hub / file / self-bootstrapped seeds
    mcq.py              multiple-choice detection and normalization
    client.py           OpenAI-compatible chat client with retries + JSON repair

  filter/             rule checks (rules.py), optional LLM judge (judge.py), stage (runner.py)
  assemble/           splits, flat and chat formats, Hub upload

  benchmark/
    suites.py           ibm, paper-qwen, paper-claude, generated, custom + register_suite
    runner.py           Endpoint (served model), ContextBuilder, run_benchmarks
    parsing.py          reply → answer labels (one parser for every suite)
    metrics.py          set match / F1 / Jaccard; IBM accuracy and consistency

  utils/              io (jsonl, hashing), logging, plugins (import-path loader)

tests/                one test module per area; see section 7
examples/             runnable examples (quickstart, custom OCR)
docs/                 the documentation site (MkDocs Material)
.github/              CI, docs deploy, release workflows; PR and issue templates
```

---

## 4. How the pipeline fits together

```
paths.pdfs ─► extract ─► documents/ ─► chunk ─► chunks.jsonl ─► index ─► faiss/
                │                                                           │
              OCR                                       retrieval (index | faiss | function)
                                                                            │
seeds ─────────────────────────────────────────► generate ◄─────────────────┤
                                                   │                        │
                                     generated/ ─► filter ─► filtered/ ─► assemble ─► dataset/
                                                                                        │
                                         ii bench (served model) ◄── suites ◄───────────┘
```

Principles to keep when changing it:

- **Each stage reads files and writes files.** Stages talk to each other
  only through `artifacts/` (paths come from `config.paths`), so any stage
  can be re-run alone, and users can read every intermediate result.
- **Each stage returns a `StageReport`** (`inputs`, `outputs`, `rejected`,
  `errors`, `details`). `Pipeline` writes it to `run_manifest.json` with the
  config fingerprint.
- **All settings live in `config.py`.** Nothing is hard-coded: no endpoints,
  model paths or prompt text in code. A new setting goes in a config class
  with a default and a comment, in `configs/default.yaml` if users should
  see it, and in the docs.
- **Plug-in points take a name or an import path.** OCR backends,
  retrievers, extractors, embedders and benchmark suites are registered by
  name. Where a function is enough, users can also give `module:function`
  or `file.py:function` (`utils/plugins.load_callable`).
- **Optional dependencies are imported lazily**, inside the function that
  needs them. Missing ones raise an error that names the extra to install.
  `import industrial_instruction` must work with only the core dependencies.

---

## 5. Development workflow

1. **Find or open an issue** for anything bigger than a small fix, so the
   approach can be agreed before you invest time.
2. **Branch from `main`:**

   | Prefix | For |
   |---|---|
   | `feat/<topic>` | new functionality |
   | `fix/<topic>` | bug fixes |
   | `docs/<topic>` | documentation only |
   | `refactor/<topic>`, `test/<topic>`, `ci/<topic>`, `chore/<topic>` | the rest |

3. **Make the change with its tests and docs** (sections 6–9).
4. **Run the checks locally**: `ruff check . && pytest -q && mkdocs build --strict`.
5. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):

   ```
   fix(benchmark): normalize "P)" option ids before scoring

   The IBM scorer compared "P)" ids to "P" answers, so perturbed items
   could never be correct. Labels are now normalized on both sides.
   ```

   Format: `type(scope): summary` in the imperative, at most about 72
   characters, then a body explaining **why**. Common scopes are `extract`,
   `ocr`, `chunk`, `embed`, `store`, `retrieval`, `generate`, `filter`,
   `assemble`, `benchmark`, `cli`, `config`, `docs` and `ci`. Mark breaking
   changes with `!` (`feat(config)!: …`) and a `BREAKING CHANGE:` footer.

6. **Open a pull request** against `main` and fill in the template.

Keep PRs focused: one logical change per PR is easier to review, revert and
describe in release notes.

---

## 6. Code standards

**Style and linting.** `ruff check .` must pass. The rule
set is in `pyproject.toml`: errors, likely bugs (`B`) and import order
(`I`). `ruff check --fix` sorts imports. Keep lines within about 100
characters.

**Python version.** Code must run on Python **3.9**, which CI tests. In
practice:

- start modules with `from __future__ import annotations`
- inside pydantic models, use `Optional[X]` and `List[X]`, not `X | None`
  or `list[X]`
- no `match` statements and no `tomllib` in package code

**Typing.** Annotate public functions and methods. Data that crosses stage
boundaries is a pydantic model in `schemas.py`; settings are pydantic
models in `config.py` with `extra="forbid"`, so a typo in YAML is an error,
not a silent default.

**Errors.**

- Configuration and setup problems fail early, with a message that says
  what to change. For example, a wrong OCR backend path stops the extract
  stage before the first page.
- Per-item failures (one page, one LLM reply, one benchmark request) are
  isolated. Record them, and the stage goes on. Never let one bad item
  cost a whole run.
- Don't catch broad `Exception` unless you are at such an isolation
  boundary, and then record the error.

**Logging.** Get loggers with `utils.logging.get_logger(__name__)`. Use no
`print` in library code; the CLI prints final results.

**Docstrings.** Every module starts with a docstring saying what it is for.
Public classes and functions get a docstring that explains *why* and any
non-obvious behavior. The API reference pages are generated from them.
Comments explain intent and constraints, not what the next line does.

**Backward compatibility.** Config keys are a public interface. When you
rename or replace one, keep the old key working (see `ocr_fallback` and
`require_options` in `config.py`), note it in the docs, and mark it as
deprecated in a comment.

**Data and secrets.** Never commit API keys, PDFs, datasets, model weights
or FAISS indexes. Secrets are read from environment variables named by
`*_api_key_env` settings.

---

## 7. Testing

```bash
pytest -q                                   # everything
pytest tests/test_benchmark.py -q           # one area
pytest -k "ocr and cache" -q                # by name
pytest -q -rs                               # show why tests were skipped
```

**Rules:**

- **No network, no GPU, no API keys.** Tests use:
  - the `hash` embedder (`EmbedConfig(backend="hash", dimension=64)`);
  - fake LLM clients (see `FakeClient` in `test_generation_payload.py` and
    `FakeOpenAI` in `test_benchmark.py`);
  - monkeypatched Hub loaders (see `fake_hub` and `StubDatasets` in
    `test_benchmark.py`).
- **Build inputs in the test.** PDFs are generated with PyMuPDF
  (`test_pymupdf_extract.py`, `test_ocr.py`), so no binary fixtures are
  checked in.
- **Optional dependencies skip, never fail.** Use
  `pytest.importorskip("faiss", reason="faiss-cpu not installed")`. CI runs
  the suite once with only the core install to enforce this.
- **Use `tmp_path`** for every file a test writes, and set
  `paths.root` to it.
- **Every bug fix comes with a test that fails without the fix.** Name it
  after the behavior, not the ticket: `test_perturbed_labels_are_normalized`.
- **Test the wiring, not just the unit.** If a component is reachable only
  through a stage or the CLI, test it from there too (see
  `test_pipeline_stages.py`).
- When you change scoring or parsing, add real reply strings to the
  parametrized cases in `test_benchmark.py`.

**Where tests live:**

| Area | Test module |
|---|---|
| config, `--set` overrides | `test_config.py` |
| extraction and page cleanup | `test_pymupdf_extract.py` |
| OCR | `test_ocr.py` |
| chunking, FAISS store | `test_chunk_and_store.py` |
| retrieval sources, `ii index --corpus` | `test_retrieval.py` |
| prompts, MCQ normalization, generation payload | `test_generation_payload.py`, `test_json_parsing.py` |
| filter rules, assemble | `test_filter_and_assemble.py` |
| benchmark suites, parsing, metrics, runner | `test_benchmark.py` |
| pipeline wiring, manifest, logging | `test_pipeline_stages.py`, `test_io_and_manifest.py`, `test_logging_and_imports.py` |

---

## 8. Recipes: common contributions

Each recipe lists the code, the test and the docs to touch. Following the
existing implementation named in each recipe is the fastest way to get it
right.

### 8.1 A built-in OCR backend

A backend is a function `OCRPage -> markdown`. See `ocr/tesseract.py` (a
function) and `ocr/openai_vision.py` (a factory that holds a client).

1. Create `ocr/<name>.py`. Import the model's library **inside** the
   function and raise `OCRError` naming the install command if it's
   missing:

   ```python
   from industrial_instruction.ocr.base import OCRError, OCRPage
   from industrial_instruction.ocr.registry import ocr_backend

   @ocr_backend("my-ocr", factory=True)          # factory: setup once per run
   def build(config):
       model = load_model(config.options.get("weights"))
       def run(page: OCRPage) -> str:
           return model.transcribe(page.to_pil())
       return run
   ```

2. Import the module in `ocr/registry._load_builtins()`.
3. Add an extra to `pyproject.toml` if it needs new dependencies.
4. Test it with a stubbed model (see `test_openai_backend_sends_image_and_prompt`).
5. Document it in `docs/guide/ocr.md`.

Concurrency, caching and per-page failure handling come from `PageOCR`;
don't re-implement them.

### 8.2 A built-in retriever

User retrievers need no code changes: they are functions
`(query, k) -> texts | dicts | Chunks`. To ship one with the package
(for example, an Elasticsearch adapter), add a module under `store/` that
registers itself with `@retriever_backend("name", factory=True)`. The
factory receives `RetrievalConfig`, whose `options` carry hosts and index
names. Then import it lazily in `store/retrieval.py`, test it with a fake
client, and document it in `docs/guide/retrieval.md`.

To read another **index file format**, extend `find_index_files` and
`_chunk_from_entry` in `store/` and add a fixture writing that format to
`test_retrieval.py` (see `write_paper_format`).

### 8.3 A PDF extractor

1. Subclass `extract.base.Extractor`, implement `extract(path) -> Document`
   and end with `return self.validate(doc)`.
2. Use `clean_pages` from `extract/text_cleanup.py`. Support OCR with
   `self.ocr.enabled`, `self.ocr.needs_ocr(text)` and `self.flush_ocr(...)`,
   as `pdfplumber_extractor.py` does.
3. Add it to `_builtin()` in `extract/registry.py` and to
   `SUPPORTED_SUFFIXES` in `extract/runner.py`.
4. Test it on a PDF generated in the test, and document it in
   `docs/guide/extract.md`.

### 8.4 An embedding backend

Subclass `embed.base.Embedder` and implement `dimension` and `_encode`.
Normalization and the query/document prefixes are handled by the base
class. Register it in `embed/registry._builtin()`, or users can call
`register_embedder("name", factory)`.

### 8.5 A benchmark suite

A suite is a loader `config -> list[BenchItem]`:

```python
from industrial_instruction.benchmark import BenchItem, register_suite

def my_suite(config):
    return [BenchItem(id=row["id"], prompt=row["question_with_options"],
                      query=row["question"], gold=["B"], labels=list("ABCDE"),
                      documents=row.get("docs", []))
            for row in load_rows()]

register_suite("my-suite", my_suite)
```

For a built-in suite, add it to `_SUITES` in `benchmark/suites.py`, add its
settings to `BenchmarkConfig`, and keep these properties:

- gold labels are normalized with `normalize_label`;
- Hub data goes through `load_split`, so a wrong split name lists the
  available ones;
- items without a scoreable label answer are skipped with a warning,
  never scored.

If it needs metrics other than set match, F1 and Jaccard, add them to
`benchmark/metrics.py` and branch in `runner.summarize`. Add the suite to
the table in `docs/guide/benchmarking.md`.

### 8.6 A generation relation or prompt

Relations are data, not code:

1. Write `generate/prompts/<name>.txt`. Use `{docs}`,
   `{simulated_instruction}`, `{options_rule}` and `{output_rule}`. Literal
   JSON braces are safe, because templates are rendered by placeholder
   replacement, not `str.format`.
2. Add a `RelationSpec` to `DEFAULT_RELATIONS` in `schemas.py` **only** if
   it should run for everyone. Otherwise document how to add it under
   `generate.relations`.
3. Add a case to `test_generation_payload.py` asserting the rendered
   prompt.

**Changing an existing prompt changes every dataset generated afterwards.**
The prompt hash is recorded on each sample (`meta.prompt_sha`). Explain the
change and its expected effect in the PR.

### 8.7 A filter rule

Add the check to `RuleFilter.check` (or `_check_mcq`) in `filter/rules.py`:

- record it with `self._note("reason_name")`;
- put its switch or threshold on `FilterConfig` (and in `default.yaml`);
- add both a rejecting and a passing case to `test_filter_and_assemble.py`
  or `test_generation_payload.py`.

The reason name appears in users' manifests, so make it descriptive.

### 8.8 A config setting or CLI flag

- Add the field with a default and a comment to the right class in
  `config.py`. If you replace an old field, keep it as a deprecated alias.
- Add it to `configs/default.yaml` if users should see it. CI loads that
  file through the schema.
- CLI flags are shortcuts for config keys. Implement them as `with_override`
  calls in `cli.py`, so `--set` keeps working the same way.
- Document it in the matching `docs/` page.

---

## 9. Documentation

The docs site lives in `docs/` and is built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/):

```bash
mkdocs serve             # live preview at http://127.0.0.1:8000
mkdocs build --strict    # what CI runs: broken links and references fail
```

| Path | Content |
|---|---|
| `docs/index.md` | landing page |
| `docs/getting-started/` | installation, quick start |
| `docs/guide/` | one page per stage and feature |
| `docs/reference/` | CLI, configuration, outputs, Python API |
| `docs/development/` | contributing (this file, included), architecture |
| `mkdocs.yml` | navigation and theme; add new pages to `nav` |

Two reference pages are generated from the code:

- The **Python API** pages come from docstrings, via `mkdocstrings`
  (`::: industrial_instruction.module` blocks), so a good docstring is
  documentation.
- The **settings tables** on the configuration reference come from
  `config.py`, via `scripts/docs_config_reference.py`. A field's
  description is the comment on its line, or the comment lines just above
  it. Give every new setting one.

A user-visible change is not done until it is documented: a new option, a
changed default or a new plug-in point. The README is the short overview;
put details in `docs/`.

---

## 10. Pull requests and review

Before you request review:

- [ ] `ruff check .` passes
- [ ] `pytest -q` passes; new behavior and bug fixes have tests
- [ ] `mkdocs build --strict` passes; user-visible changes are documented
- [ ] new settings have defaults, comments and an entry in `default.yaml` if user-facing
- [ ] changes to prompts, filter rules, metrics or parsing are called out in the PR description
- [ ] no secrets, datasets, PDFs or model files are committed

**Review.** At least one maintainer approval and a green **CI passed**
check are required to merge. Reviewers look for:

1. correctness;
2. tests that would catch a regression;
3. clear errors for users;
4. consistency with the principles in section 4;
5. docs.

Reply to every review comment, either with a change or with your
reasoning, and push follow-up commits instead of force-pushing during
review, so reviewers can see what changed. PRs are squash-merged, so the
PR title becomes the commit message: make it a Conventional Commit.

---

## 11. Continuous integration

`.github/workflows/ci.yml` runs on every pull request and every push to
`main`:

| Job | What it checks |
|---|---|
| Lint | `ruff check .` |
| Tests (Python 3.9–3.13) | the full suite with the test extras installed |
| Tests (core install only) | optional-dependency tests skip instead of failing |
| Build package | wheel and sdist build; prompts and default config are packaged; the installed `ii` runs |
| Docs build | `mkdocs build --strict` |
| **CI passed** | succeeds only if all of the above did; use it as the required check |

To reproduce a failing job locally, run its commands from `ci.yml` in a
fresh virtual environment with the same Python version.
`uv venv -p 3.9 && uv pip install -e ".[dev,pdf,pdfplumber,faiss,bench]"`
is the quickest way to test an older Python.

`.github/workflows/docs.yml` publishes the site to GitHub Pages on every
push to `main`.

---

## 12. Releases

Maintainers release from `main`:

1. Update `version` in `pyproject.toml` and `__version__` in
   `src/industrial_instruction/__init__.py`, following
   [semantic versioning](https://semver.org/). While the version is 0.x,
   breaking changes bump the minor number.
2. Merge that change with a `chore(release): vX.Y.Z` PR.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.

`.github/workflows/release.yml` checks that the tag matches the package
version, builds the wheel and sdist, and creates a GitHub release with
generated notes. Upload to PyPI is enabled once a maintainer configures
PyPI trusted publishing and sets the repository variable
`PUBLISH_TO_PYPI=true`.

---

## 13. Reporting bugs and proposing features

Use the issue templates:

- **Bug report:** what you ran, the config (with secrets removed), what
  happened and what you expected, the error or log, and versions (`pip show
  industrial-instruction`, Python, OS). For extraction bugs, attach a
  one-page PDF that reproduces the problem, if you are allowed to share it.
- **Feature request:** the problem you want to solve and who has it,
  before the solution you have in mind.

For security issues (for example, a way to make the package execute
unintended code), don't open a public issue. Contact the maintainers
privately through the repository's security advisories page.
