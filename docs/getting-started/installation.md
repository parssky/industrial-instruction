# Installation

Industrial-Instruction needs **Python 3.9 or newer**.

```bash
git clone https://github.com/parssky/industrial-instruction.git
cd industrial-instruction
pip install -e '.[pdf,local-embed,faiss,bench]'
```

That covers the default path: PyMuPDF extraction, a local embedding model,
the FAISS vector store and the Hub-hosted benchmark suites. Installing adds
the `ii` command (also available as `industrial-instruction`).

## Optional extras

Install only what you use. Every optional dependency is imported when a
feature first needs it, and a missing one is reported with the extra to
install.

| Extra | Installs | Needed for |
|---|---|---|
| `pdf` | PyMuPDF | the default PDF extractor, and rendering pages for OCR |
| `pdfplumber` | pdfplumber | the alternative extractor, often better on unruled tables |
| `local-embed` | sentence-transformers, torch | the default local embedding model |
| `faiss` | faiss-cpu | the vector store |
| `bench` | datasets, huggingface-hub | Hub benchmark suites and `hf:` indexes |
| `hub` | datasets, huggingface-hub | Hub seed datasets, pushing the dataset to the Hub |
| `ocr-tesseract` | pytesseract, Pillow | the offline Tesseract OCR backend (also needs the `tesseract` binary) |
| `all` | everything above except OCR | |
| `dev` | pytest, ruff, build | contributing |
| `docs` | mkdocs-material, mkdocstrings | building this site |

!!! tip "No GPU needed for the package itself"
    Generation and benchmarking call models over HTTP. Embeddings run on CPU
    by default (`embed.device` picks a GPU), or use `embed.backend: openai`
    for an embeddings API.

## Check the install

```bash
ii --help
ii info        # resolved configuration and which artifacts exist
```

## Models you will talk to

| Role | Default | Change with |
|---|---|---|
| Generator LLM | `gpt-4.1-mini` on the OpenAI API | `generate.base_url`, `generate.model`: any OpenAI-compatible server, such as vLLM |
| Embedding model | `google/embeddinggemma-300m` (local) | `embed.backend`, `embed.model` |
| OCR model (optional) | an OpenAI-compatible vision model | `extract.ocr.*`, or your own function |
| Model under test | whatever you serve | `ii bench --base-url ...` |

API keys are read from environment variables named in the config
(`OPENAI_API_KEY` by default), never from the YAML file. Local vLLM servers
accept any key.
