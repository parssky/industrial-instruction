"""One retrieval interface for generation and benchmarking.

``get_retriever(config)`` returns an object with ``search(query, k)`` that
yields :class:`SearchHit` rows, whichever ``retrieval.source`` is set:

``index``     this project's FAISS index (``ii index`` / ``ii index --corpus``)
``faiss``     a FAISS index you built yourself, in this package's format or the
              original paper format, from disk or ``hf:owner/repo``
``function``  your own retriever (Elasticsearch, Qdrant, a reranker, ...):

    from industrial_instruction.store import register_retriever

    def my_search(query: str, k: int) -> list[str]:   # or list of dicts
        return [hit["text"] for hit in es.search(query, size=k)]

    register_retriever("es", my_search)    # retrieval: {source: function, backend: es}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from industrial_instruction.config import Config, RetrievalConfig
from industrial_instruction.schemas import Chunk
from industrial_instruction.store.faiss_store import FaissStore, SearchHit, _chunk_from_entry
from industrial_instruction.utils.io import stable_id
from industrial_instruction.utils.logging import get_logger
from industrial_instruction.utils.plugins import load_callable

logger = get_logger(__name__)

_INDEX_NAMES = ("index.faiss", "index", "faiss.index")
_MAP_NAMES = ("id_map.json", "mapping.json")


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> List[SearchHit]: ...


class RetrievalError(RuntimeError):
    pass


# ------------------------------------------------------------- functions

_FUNCTIONS: Dict[str, Tuple[Callable, bool]] = {}
_FACTORY_ATTR = "_ii_retriever_factory"


def register_retriever(name: str, fn: Callable, factory: bool = False) -> None:
    """``fn(query, k) -> [text | dict]``; with ``factory=True``,
    ``fn(retrieval_config) -> search_function`` is called once per run."""
    if not callable(fn):
        raise TypeError(f"retriever {name!r} must be callable")
    _FUNCTIONS[name.lower()] = (fn, factory)


def retriever_backend(name: str, factory: bool = False):
    """Decorator form of :func:`register_retriever`."""

    def wrap(fn):
        register_retriever(name, fn, factory=factory)
        try:
            setattr(fn, _FACTORY_ATTR, factory)
        except AttributeError:  # pragma: no cover
            pass
        return fn

    return wrap


class FunctionRetriever:
    """Adapts a user function's results to :class:`SearchHit` rows."""

    def __init__(self, fn: Callable, text_field: str = "text", name: str = "function") -> None:
        self.fn = fn
        self.text_field = text_field
        self.name = name

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        results = self.fn(query, k) or []
        hits: List[SearchHit] = []
        for rank, item in enumerate(list(results)[:k]):
            if isinstance(item, SearchHit):
                hits.append(item)
                continue
            if isinstance(item, Chunk):
                chunk, score = item, 0.0
            else:
                score = float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0
                text = item.get(self.text_field, "") if isinstance(item, dict) else str(item)
                key = item.get("id") if isinstance(item, dict) else None
                chunk = _chunk_from_entry(key or stable_id(text), item, self.text_field)
                chunk.doc_id = chunk.doc_id if chunk.doc_id != "external" else self.name
            hits.append(SearchHit(chunk=chunk, score=score, rank=rank))
        return hits


# ----------------------------------------------------------------- faiss


def _download(spec: str) -> Path:
    repo = spec[3:].strip("/")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RetrievalError("hf: paths need: pip install huggingface-hub") from exc
    logger.info("downloading index %s from the Hugging Face Hub", repo)
    return Path(snapshot_download(repo_id=repo, repo_type="dataset"))


def find_index_files(path: Path, mapping: Optional[Path] = None) -> Tuple[Path, Path]:
    """Locate the FAISS index and its id map in a file or directory tree."""
    if path.is_file():
        index = path
        candidates = [mapping] if mapping else [path.parent / n for n in _MAP_NAMES]
    else:
        index = next(
            (p for name in _INDEX_NAMES for p in sorted(path.rglob(name)) if p.is_file()),
            None,
        ) or next(iter(sorted(path.rglob("*.faiss"))), None)
        if index is None:
            raise RetrievalError(f"no FAISS index file ({', '.join(_INDEX_NAMES)}) under {path}")
        candidates = [mapping] if mapping else [index.parent / n for n in _MAP_NAMES]
    found = next((c for c in candidates if c and c.exists()), None)
    if found is None:
        raise RetrievalError(f"no id map ({', '.join(_MAP_NAMES)}) next to {index}")
    return index, found


def _is_package_format(mapping: Path) -> bool:
    import json

    with open(mapping, encoding="utf-8") as f:
        data = json.load(f)
    first = next(iter(data.values()), None) if isinstance(data, dict) else None
    return isinstance(first, dict) and {"id", "doc_id", "text"} <= set(first)


def open_faiss(cfg: RetrievalConfig, config: Config) -> FaissStore:
    from industrial_instruction.embed.registry import get_embedder

    if not cfg.path:
        raise RetrievalError("retrieval.path must point at your FAISS index (or hf:owner/repo)")
    root = Path(config.paths.root)
    path = _download(cfg.path) if cfg.path.startswith("hf:") else Path(cfg.path)
    if not path.is_absolute() and not cfg.path.startswith("hf:"):
        path = root / path
    if not path.exists():
        raise RetrievalError(f"retrieval.path {path} does not exist")
    mapping = Path(cfg.mapping_path) if cfg.mapping_path else None
    if mapping and not mapping.is_absolute():
        mapping = root / mapping
    index_file, map_file = find_index_files(path, mapping)

    embed = cfg.embed
    if embed is None:
        embed = config.embed
        if not _is_package_format(map_file) and (embed.query_prefix or embed.document_prefix):
            # The original Retriever embedded with plain encode(text).
            logger.info(
                "retrieval: %s is in the original format; querying without "
                "embed prefixes (set retrieval.embed to override)", map_file,
            )
            embed = embed.model_copy(update={"query_prefix": "", "document_prefix": ""})
    store = FaissStore(embedder=get_embedder(embed), store_config=config.store)
    return store.load_files(index_file, map_file, text_field=cfg.text_field)


# ------------------------------------------------------------------ entry


def get_retriever(config: Config) -> Retriever:
    cfg = config.retrieval
    if cfg.source == "index":
        from industrial_instruction.store.runner import load_store

        try:
            return load_store(config)
        except FileNotFoundError as exc:
            raise RetrievalError(
                f"{exc}\nBuild it from your corpus with: ii index --corpus <pdf-dir | "
                "md-dir | passages.jsonl>\nor use an index you built yourself: "
                "retrieval.source=faiss, retrieval.path=<dir or hf:owner/repo>"
            ) from exc
    if cfg.source == "faiss":
        return open_faiss(cfg, config)

    spec = (cfg.backend or "").strip()
    if not spec:
        raise RetrievalError("retrieval.backend must name your retriever function")
    if spec.lower() in _FUNCTIONS:
        fn, is_factory = _FUNCTIONS[spec.lower()]
    elif ":" in spec:
        fn = load_callable(spec, config.paths.root, error=RetrievalError, what="retriever")
        is_factory = getattr(fn, _FACTORY_ATTR, False)
    else:
        raise RetrievalError(
            f"unknown retriever {spec!r}; registered: {sorted(_FUNCTIONS)}. "
            "Use register_retriever() or an import path like 'my_pkg.search:run'."
        )
    return FunctionRetriever(fn(cfg) if is_factory else fn, cfg.text_field, name=spec)


def describe(config: Config) -> Dict[str, Any]:
    cfg = config.retrieval
    if cfg.source == "index":
        return {"source": "index", "path": str(config.paths.resolve("index"))}
    if cfg.source == "faiss":
        return {"source": "faiss", "path": cfg.path}
    return {"source": "function", "backend": cfg.backend}
