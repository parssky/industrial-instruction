## What and why

<!-- What does this change, and why is it needed? Link the issue: Fixes #123 -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change (config keys, CLI, output formats or the Python API change)
- [ ] Documentation
- [ ] Refactor / tests / CI

## Effect on datasets or scores

<!-- Does this change generated datasets (prompts, filter rules, chunking,
extraction) or benchmark scores (parsing, metrics)? Describe the expected
effect, or write "none". -->

## How it was tested

<!-- New or updated tests, and any manual run (command + result). -->

## Checklist

- [ ] `ruff check .` passes
- [ ] `pytest -q` passes, and new behavior / bug fixes have tests
- [ ] `mkdocs build --strict` passes, and user-visible changes are documented in `docs/`
- [ ] New settings have defaults, comments, and an entry in `configs/default.yaml` if user-facing
- [ ] No secrets, datasets, PDFs or model files are committed
