"""Typed configuration for the whole pipeline.

One YAML file drives an entire run. Every stage receives its own sub-config,
so nothing is hardcoded the way the original notebooks were (endpoints, model
paths, index locations, prompt text).

Secrets are never stored in YAML: ``generate.api_key_env`` names the
environment variable that holds the key.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from industrial_instruction.schemas import DEFAULT_RELATIONS, RelationSpec

STAGE_KEYS = (
    "extract",
    "chunk",
    "embed",
    "store",
    "retrieval",
    "seeds",
    "generate",
    "filter",
    "assemble",
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Base):
    """Where each artifact lands. All relative to ``root``."""

    root: str = "."
    pdfs: str = "data/pdfs"
    documents: str = "artifacts/documents"
    chunks: str = "artifacts/chunks.jsonl"
    index: str = "artifacts/faiss"
    generated: str = "artifacts/generated"
    filtered: str = "artifacts/filtered"
    dataset: str = "artifacts/dataset"
    manifest: str = "artifacts/run_manifest.json"

    def resolve(self, key: str) -> Path:
        """Absolute path for a named artifact."""
        value = getattr(self, key)
        p = Path(value)
        return p if p.is_absolute() else (Path(self.root) / p).resolve()


class OCRConfig(_Base):
    """Page OCR for scanned or image-only PDF pages.

    ``backend`` is either a registered name (``openai``, ``tesseract``, or
    anything added with ``register_ocr``) or an import path to your own
    function: ``my_pkg.ocr:run`` or ``ocr/my_model.py:run``. The function
    receives an :class:`~industrial_instruction.ocr.OCRPage` and returns the
    page as markdown.
    """

    mode: str = "off"  # off | auto (pages without a text layer) | always
    backend: str = "openai"
    min_chars_per_page: int = 50  # auto: OCR pages with less text than this
    dpi: int = 200
    max_workers: int = 4  # pages OCR'd concurrently (servers batch these)
    cache: bool = True  # reuse results for identical page images
    cache_dir: str = "artifacts/ocr_cache"  # relative to paths.root
    # Built-in ``openai`` backend: any OpenAI-compatible vision endpoint,
    # e.g. a vLLM server running Qwen2.5-VL, olmOCR or Nanonets-OCR.
    base_url: Optional[str] = None
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    prompt: Optional[str] = None  # null = packaged prompt; "" = image only
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: float = 180.0
    max_retries: int = 3
    # Free-form settings handed to custom functions as ``page.options``.
    options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("mode", mode="before")
    @classmethod
    def _yaml_bools(cls, value: Any) -> Any:
        # YAML 1.1 reads an unquoted `off` as False (and `on` as True).
        if isinstance(value, bool):
            return "auto" if value else "off"
        return value.lower() if isinstance(value, str) else value

    def resolve_api_key(self) -> str:
        return os.environ.get(self.api_key_env) or "no-key"


class ExtractConfig(_Base):
    """PDF -> image-free markdown (text + tables)."""

    backend: str = "pymupdf"  # pymupdf | pdfplumber | docling | marker
    extract_tables: bool = True
    table_backend: str = "auto"  # auto | pymupdf | pdfplumber | none
    drop_images: bool = True
    page_markers: bool = True  # emit <!-- page N --> separators
    min_chars_per_doc: int = 200  # skip scanned/empty PDFs
    strip_headers_footers: bool = True
    dehyphenate: bool = True
    recursive: bool = True  # walk subdirectories of paths.pdfs
    ocr_fallback: bool = False  # deprecated: same as ocr.mode = auto
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    backend_options: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _legacy_ocr_flag(self) -> "ExtractConfig":
        if self.ocr_fallback and self.ocr.mode == "off":
            self.ocr.mode = "auto"
        if self.ocr.mode not in ("off", "auto", "always"):
            raise ValueError(
                f"extract.ocr.mode must be off, auto or always, got {self.ocr.mode!r}"
            )
        return self


class ChunkConfig(_Base):
    """Markdown -> retrievable chunks."""

    strategy: str = "heading"  # heading | fixed
    max_chars: int = 2000
    min_chars: int = 120
    overlap: int = 200
    keep_tables_whole: bool = True
    include_heading_in_text: bool = True


class EmbedConfig(_Base):
    """Embedding model used to build and query the vector store."""

    backend: str = "sentence_transformers"  # sentence_transformers | openai | hash
    model: str = "google/embeddinggemma-300m"
    batch_size: int = 32
    device: Optional[str] = None
    normalize: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    dimension: Optional[int] = None  # required for the 'hash' test backend
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None


class StoreConfig(_Base):
    """FAISS index layout."""

    index_type: str = "flat_ip"  # flat_ip | flat_l2 | hnsw
    index_filename: str = "index.faiss"
    mapping_filename: str = "id_map.json"
    meta_filename: str = "store_meta.json"
    hnsw_m: int = 32


class RetrievalConfig(_Base):
    """Where ``generate`` and ``ii bench --context retrieved`` get documents.

    ``source``:
      * ``index``    - this project's FAISS index (``paths.index``), built from
                      your corpus by ``ii index`` or ``ii index --corpus``
      * ``faiss``    - a FAISS index you built yourself: a directory or index
                      file plus ``id_map.json``, in this package's format or
                      the original paper format (``{"0": "chunk text"}``).
                      ``hf:owner/repo`` downloads it from the Hub.
      * ``function`` - your own retriever, ``fn(query, k) -> [text | dict]``,
                      registered by name or given as ``my_pkg.mod:fn`` /
                      ``retrievers/mine.py:fn``
    """

    source: str = "index"  # index | faiss | function
    path: Optional[str] = None  # faiss: directory, index file or hf:owner/repo
    mapping_path: Optional[str] = None  # faiss: id_map.json if not next to the index
    text_field: str = "text"  # faiss/function: text key when entries are dicts
    # Embedding model for querying a `faiss` index. null = the top-level
    # `embed` section; for the original paper format the prefixes are then
    # cleared, because that index was built with plain encode(text).
    embed: Optional[EmbedConfig] = None
    backend: Optional[str] = None  # function: registered name or import path
    options: Dict[str, Any] = Field(default_factory=dict)  # passed to factories

    @model_validator(mode="after")
    def _check(self) -> "RetrievalConfig":
        if self.source not in ("index", "faiss", "function"):
            raise ValueError("retrieval.source must be index, faiss or function")
        return self


class SeedsConfig(_Base):
    """Seed ('simulated instruction') source that steers question style.

    Three interchangeable sources are supported:
      * ``huggingface`` - any HF dataset, e.g. FailureSensorIQ
      * ``jsonl`` / ``json`` - a local file of your own instructions
      * ``self`` - derive seeds from your own indexed chunks (no external data)
    """

    source: str = "huggingface"
    dataset: str = "ibm-research/FailureSensorIQ"
    splits: List[str] = Field(default_factory=lambda: ["org", "pert"])
    text_field: str = "prompt"  # shown to the generator as <Simulated Instruction>
    # Field used as the retrieval query. FailureSensorIQ's `prompt` wraps the
    # question in identical option/format boilerplate; `question` is just the
    # question, which retrieves far better. Falls back to text_field.
    query_field: Optional[str] = "question"
    id_field: Optional[str] = None
    path: Optional[str] = None  # for jsonl/json sources
    revision: Optional[str] = None
    shuffle: bool = True
    seed: int = 42
    limit: Optional[int] = None
    self_seed_prompt: str = "self_seed"  # template used when source == 'self'
    self_seed_per_chunk: int = 1


class GenerateConfig(_Base):
    """LLM call settings and which relations to produce."""

    base_url: Optional[str] = None  # OpenAI-compatible endpoint (vLLM, etc.)
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    system_prompt: str = (
        "You are a helpful assistant. Follow exactly the Instructions and reply "
        "with valid JSON only."
    )
    request_json_object: bool = True  # use response_format when supported
    max_workers: int = 8
    max_retries: int = 3
    retry_backoff: float = 2.0
    timeout: float = 120.0
    retrieval_k: int = 3  # candidate pool; per-relation k_docs slices it
    limit: Optional[int] = None  # cap seeds processed (useful for smoke runs)
    # Multiple-choice output (q*, a*, options*):
    #   auto   - when the seed is multiple-choice (as in the paper's seeds)
    #   always - every sample;  never - plain question/answer only
    options_mode: str = "auto"
    require_options: bool = False  # deprecated: same as options_mode = always
    n_options: int = 5
    prompt_dir: Optional[str] = None  # user-supplied prompt templates override
    relations: List[RelationSpec] = Field(
        default_factory=lambda: [r.model_copy() for r in DEFAULT_RELATIONS]
    )

    @model_validator(mode="after")
    def _options_mode(self) -> "GenerateConfig":
        if self.require_options:
            self.options_mode = "always"
        if self.options_mode not in ("auto", "always", "never"):
            raise ValueError(
                "generate.options_mode must be auto, always or never, "
                f"got {self.options_mode!r}"
            )
        return self

    def enabled_relations(self) -> List[RelationSpec]:
        return [r for r in self.relations if r.enabled]

    def resolve_api_key(self) -> str:
        # Local vLLM servers accept any non-empty key.
        return os.environ.get(self.api_key_env) or "no-key"


class FilterConfig(_Base):
    """Rule-based validation plus optional LLM-as-judge."""

    min_question_chars: int = 20
    max_question_chars: int = 2000
    min_answer_chars: int = 1
    require_answer: bool = True
    forbid_meta_references: bool = True
    meta_reference_phrases: List[str] = Field(
        default_factory=lambda: [
            "based on the provided context",
            "according to the document",
            "according to the documents",
            "in the text",
            "the passage above",
            "as shown in the excerpt",
        ]
    )
    enforce_option_count: bool = True
    # Multiple-choice checks (samples generated as MCQ):
    require_options: bool = True  # an MCQ sample must have options
    answer_in_options: bool = True  # every answer label must be an option label
    # Reject when more than this share of the options are copied verbatim
    # from the seed's options (the model emulated the seed too literally).
    max_seed_option_overlap: float = 0.6
    forbid_format_instructions: bool = True  # e.g. '{"answer": ...}' left in q*
    dedupe: bool = True
    dedupe_mode: str = "normalized"  # exact | normalized
    judge_enabled: bool = False
    judge_model: Optional[str] = None
    judge_min_score: float = 3.0
    judge_max_workers: int = 8


class AssembleConfig(_Base):
    """Merge, split and export the final dataset."""

    splits: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.9, "test": 0.1}
    )
    split_seed: int = 42
    stratify_by_relation: bool = True
    formats: List[str] = Field(default_factory=lambda: ["jsonl"])  # jsonl | chat
    chat_format: bool = True  # also emit messages-style SFT rows
    include_documents: bool = True
    push_to_hub: Optional[str] = None  # e.g. "you/your-dataset"
    private: bool = True

    def resolved_formats(self) -> List[str]:
        """``formats`` plus the implicit 'chat' output when enabled."""
        formats = list(self.formats)
        if self.chat_format and "chat" not in formats:
            formats.append("chat")
        return formats


class EndpointConfig(_Base):
    """The model under test: any OpenAI-compatible server, e.g. vLLM.

    The package never starts or stops the model; serve it yourself
    (``vllm serve <model> --served-model-name my-model``) and point here.
    """

    base_url: str = "http://localhost:8000/v1"
    model: Optional[str] = None  # null = the first model the server lists
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 300.0
    max_retries: int = 3
    max_workers: int = 32
    # The paper's evaluation system prompt.
    system_prompt: Optional[str] = (
        'You are a helpful assistant. You must output your answer strictly as valid '
        'JSON in the format {"answer": ["choice"]}.'
    )

    def resolve_api_key(self) -> str:
        return os.environ.get(self.api_key_env) or "no-key"


class CustomBenchmarkConfig(_Base):
    """A user benchmark: multiple-choice items with label answers."""

    source: str = "jsonl"  # jsonl | json | huggingface | disk
    path: Optional[str] = None  # file, Hub id or save_to_disk directory
    config_name: Optional[str] = None  # Hub dataset config
    split: Optional[str] = "test"
    question_field: str = "question"
    answer_field: str = "answer"  # ["B"], "B", "B, D" or {"answer": [...]}
    options_field: Optional[str] = "options"  # appended unless already in the question
    documents_field: Optional[str] = "documents"  # used by context: gold
    id_field: Optional[str] = "id"


class BenchmarkConfig(_Base):
    """``ii bench``: score a served model on one or more suites."""

    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    # ibm | paper-qwen | paper-claude | generated | custom
    suites: List[str] = Field(default_factory=lambda: ["ibm"])
    # none: question only | gold: the item's own documents (the paper's RAG
    # setting) | retrieved: top-k chunks from this project's FAISS index
    contexts: List[str] = Field(default_factory=lambda: ["none"])
    retrieval_k: int = 3
    # How documents are put in front of the question. This is the template
    # the paper's models were trained with, typo included, so fine-tuned
    # checkpoints see exactly their training format.
    context_template: str = (
        "\n        Based on relevat document answer this question.\n"
        "        relevant document: {documents}\n"
        "        question: {question}\n    "
    )
    limit: Optional[int] = None  # items per suite, for smoke runs
    output_dir: str = "artifacts/benchmarks"  # relative to paths.root
    # Hub sources. The paper splits are matched by name inside the dataset
    # repo, so a renamed config/split is found or reported, never guessed.
    ibm_dataset: str = "ibm-research/FailureSensorIQ"
    ibm_splits: List[str] = Field(default_factory=lambda: ["org", "pert"])
    paper_dataset: str = "Parssky/industrial-instruction-dataset"
    paper_qwen_split: str = "panasonic_qa_v1_test"
    paper_claude_split: str = "panasonic_qa_claude_v1_test"
    custom: CustomBenchmarkConfig = Field(default_factory=CustomBenchmarkConfig)

    @model_validator(mode="after")
    def _check(self) -> "BenchmarkConfig":
        bad = [c for c in self.contexts if c not in ("none", "gold", "retrieved")]
        if bad:
            raise ValueError(f"benchmark.contexts: unknown {bad}; use none, gold, retrieved")
        return self


class Config(_Base):
    """Root configuration object."""

    project: str = "industrial-instruction"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    assemble: AssembleConfig = Field(default_factory=AssembleConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from YAML, defaulting ``paths.root`` to the file's dir."""
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        data.setdefault("paths", {})
        if not data["paths"].get("root"):
            data["paths"]["root"] = str(p.parent.resolve())
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls.model_validate(data or {})

    def to_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(
                self.model_dump(mode="json"), sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )
        return p

    # ------------------------------------------------------------ overrides

    def with_override(self, key: str, value: Any) -> "Config":
        """Return a copy with one dotted ``key`` set to ``value``.

        Backs the CLI's ``--set generate.model=gpt-4.1``. Strings are parsed
        as YAML so ``true``, ``8``, ``0.2``, ``null`` and ``[a, b]`` arrive as
        the right Python types, and validation still runs on the result.
        """
        parts = [p for p in key.split(".") if p]
        if not parts:
            raise ValueError("override key must not be empty")

        if isinstance(value, str):
            try:
                parsed = yaml.safe_load(value)
            except yaml.YAMLError:
                parsed = value
            if parsed is None and value.strip() not in ("null", "~", ""):
                parsed = value
        else:
            parsed = value

        data = self.model_dump(mode="json")
        cursor: Any = data
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(
                    f"Unknown config key {key!r}: no section {part!r}. "
                    "Run 'ii info' to see the resolved config."
                )
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise ValueError(
                f"Unknown config key {key!r}. Run 'ii info' to see valid keys."
            )
        cursor[leaf] = parsed
        return type(self).model_validate(data)

    def with_overrides(self, overrides: Dict[str, Any]) -> "Config":
        config = self
        for key, value in (overrides or {}).items():
            config = config.with_override(key, value)
        return config

    # ---------------------------------------------------------- fingerprint

    def fingerprint(self, *stages: str) -> Dict[str, Any]:
        """Config subset used for stage cache keys / the run manifest.

        With no arguments this covers every stage, so the manifest records the
        full settings that produced a dataset. ``hash`` is a short digest for
        quick equality checks between runs.
        """
        dumped = self.model_dump(mode="json")
        keys = stages or STAGE_KEYS
        subset = {s: dumped.get(s) for s in keys}
        digest = hashlib.sha256(
            json.dumps(subset, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]
        return {"hash": digest, "project": self.project, **subset}
