"""Typed configuration for the whole pipeline.

One YAML file drives an entire run. Every stage receives its own sub-config,
so nothing is hardcoded the way the original notebooks were (endpoints, model
paths, index locations, prompt text).

Secrets are never stored in YAML: ``generate.api_key_env`` names the
environment variable that holds the key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from industrial_instruction.schemas import DEFAULT_RELATIONS, RelationSpec


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
    ocr_fallback: bool = False
    backend_options: Dict[str, Any] = Field(default_factory=dict)


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
    text_field: str = "prompt"
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
    require_options: bool = False  # force multiple-choice A-E for all relations
    n_options: int = 5
    prompt_dir: Optional[str] = None  # user-supplied prompt templates override
    relations: List[RelationSpec] = Field(
        default_factory=lambda: [r.model_copy() for r in DEFAULT_RELATIONS]
    )

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
    formats: List[str] = Field(default_factory=lambda: ["jsonl"])  # jsonl | hf
    chat_format: bool = True  # also emit messages-style SFT rows
    include_documents: bool = True
    push_to_hub: Optional[str] = None  # e.g. "you/your-dataset"
    private: bool = True


class Config(_Base):
    """Root configuration object."""

    project: str = "industrial-instruction"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    assemble: AssembleConfig = Field(default_factory=AssembleConfig)

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

    def fingerprint(self, *stages: str) -> Dict[str, Any]:
        """Config subset used for stage cache keys / the run manifest."""
        dumped = self.model_dump(mode="json")
        return {s: dumped.get(s) for s in stages}
