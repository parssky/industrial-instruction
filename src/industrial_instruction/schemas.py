"""Data contracts shared by every pipeline stage.

All artifacts on disk are JSONL files whose lines validate against these
models. Keeping the contracts here means a user can swap any stage
implementation (extractor, embedder, generator) without touching the rest.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocMode(str, Enum):
    """How many retrieved documents a relation prompt consumes."""

    SINGLE = "single"
    MULTI = "multi"


class SampleStatus(str, Enum):
    """Outcome of generating or validating one sample."""

    OK = "ok"
    INVALID = "invalid"  # produced, but failed schema/rule validation
    ERROR = "error"  # generation itself failed (API/parse error)
    REJECTED = "rejected"  # dropped by the filter stage


class Table(BaseModel):
    """A table lifted out of a PDF page, rendered as markdown."""

    page: int
    markdown: str
    n_rows: int = 0
    n_cols: int = 0


class Document(BaseModel):
    """One source PDF converted to image-free markdown (text + tables)."""

    model_config = ConfigDict(extra="allow")

    id: str
    source_path: str
    title: str = ""
    markdown: str = ""
    n_pages: int = 0
    n_tables: int = 0
    n_images_dropped: int = 0
    content_sha256: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable passage. This is the unit stored in the vector store."""

    model_config = ConfigDict(extra="allow")

    id: str
    doc_id: str
    text: str
    ordinal: int = 0
    heading_path: List[str] = Field(default_factory=list)
    n_chars: int = 0
    source_path: str = ""
    has_table: bool = False
    meta: Dict[str, Any] = Field(default_factory=dict)

    def as_context(self, include_heading: bool = True) -> str:
        """Render the chunk the way it is injected into a prompt."""
        if include_heading and self.heading_path:
            return " > ".join(self.heading_path) + "\n" + self.text
        return self.text


class RelationSpec(BaseModel):
    """Declarative definition of one retrieval relation (r0..r4, or custom).

    This is what replaces the copy-pasted per-relation blocks: adding a new
    relation means adding one spec plus a prompt template.
    """

    id: str
    prompt: str  # prompt template name, resolved by generate.prompts
    doc_mode: DocMode = DocMode.SINGLE
    k_docs: int = 1  # how many retrieved chunks to place in the prompt
    description: str = ""
    requires_options: bool = False  # multiple-choice (A-E) output expected
    enabled: bool = True


class QASample(BaseModel):
    """One generated dataset row."""

    model_config = ConfigDict(extra="allow")

    id: str
    relation: str
    question: str
    answer: Any = None
    options: Optional[List[str]] = None
    documents: List[str] = Field(default_factory=list)
    doc_ids: List[str] = Field(default_factory=list)
    seed: str = ""
    seed_id: str = ""
    generator: str = ""
    status: SampleStatus = SampleStatus.OK
    reject_reason: str = ""
    raw_output: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_llm_dict(cls, payload: Dict[str, Any], **kwargs: Any) -> "QASample":
        """Build a sample from the generator's ``q*``/``a*``/``options*`` JSON.

        Accepts both the starred keys used by the paper prompts and plain
        ``question``/``answer``/``options`` keys, so custom prompts work too.
        """
        question = payload.get("q*", payload.get("question", ""))
        answer = payload.get("a*", payload.get("answer"))
        options = payload.get("options*", payload.get("options"))
        if isinstance(options, dict):
            options = [f"{k}. {v}" for k, v in options.items()]
        return cls(
            question=(question or "").strip(),
            answer=answer,
            options=options,
            **kwargs,
        )


class Seed(BaseModel):
    """A 'simulated instruction' used to steer question style and format."""

    model_config = ConfigDict(extra="allow")

    id: str
    text: str
    source: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)


class StageReport(BaseModel):
    """Per-stage counters written into the run manifest."""

    stage: str
    inputs: int = 0
    outputs: int = 0
    rejected: int = 0
    errors: int = 0
    seconds: float = 0.0
    details: Dict[str, Any] = Field(default_factory=dict)


# The five relations from the paper, as the shipped default.
DEFAULT_RELATIONS: List[RelationSpec] = [
    RelationSpec(
        id="r0",
        prompt="useless_doc",
        doc_mode=DocMode.SINGLE,
        k_docs=1,
        description="Retrieved document is related but unhelpful (noise robustness).",
    ),
    RelationSpec(
        id="r1",
        prompt="single_doc_support",
        doc_mode=DocMode.SINGLE,
        k_docs=1,
        description="One document supports the answer without stating it.",
    ),
    RelationSpec(
        id="r2",
        prompt="multi_doc_support",
        doc_mode=DocMode.MULTI,
        k_docs=3,
        description="Several documents support the answer without stating it.",
    ),
    RelationSpec(
        id="r3",
        prompt="single_doc_answer",
        doc_mode=DocMode.SINGLE,
        k_docs=1,
        description="Answer fully derivable from one document.",
    ),
    RelationSpec(
        id="r4",
        prompt="multi_doc_answer",
        doc_mode=DocMode.MULTI,
        k_docs=3,
        description="Answer requires multi-hop reasoning across documents.",
    ),
]
