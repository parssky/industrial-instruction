"""Small IO helpers. Every stage reads and writes JSONL through here."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union

PathLike = Union[str, Path]


class _Missing:
    """Sentinel so that ``default=None`` is distinguishable from no default."""


_MISSING = _Missing()


def ensure_dir(path: PathLike) -> Path:
    """Create ``path`` (as a directory) if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_parent(path: PathLike) -> Path:
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """Stream a JSONL file, skipping blank lines."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: PathLike, rows: Iterable[Any]) -> int:
    """Write rows (dicts or pydantic models) as JSONL. Returns row count."""
    p = _ensure_parent(path)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: PathLike, rows: Iterable[Any]) -> int:
    p = _ensure_parent(path)
    n = 0
    with p.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_json(path: PathLike, default: Any = _MISSING) -> Any:
    """Read a JSON file.

    If ``default`` is given it is returned when the file is missing, empty or
    contains invalid JSON. A half-written manifest from an interrupted run
    should not be able to break the next run, so a corrupt file is treated the
    same as a missing one. Without ``default`` the underlying error is raised.
    """
    p = Path(path)
    has_default = not isinstance(default, _Missing)
    if not p.exists():
        if has_default:
            return default
        raise FileNotFoundError(f"No such file: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        if has_default:
            return default
        raise


def write_json(path: PathLike, obj: Any, indent: int = 2) -> None:
    p = _ensure_parent(path)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(obj), fh, ensure_ascii=False, indent=indent)


def _to_jsonable(obj: Any) -> Any:
    """Convert pydantic models / sets / Paths into JSON-serializable data."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except TypeError:
            pass
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: PathLike, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_id(*parts: Any, length: int = 16) -> str:
    """Deterministic short id, so re-runs produce identical row ids."""
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 64) -> str:
    slug = _SLUG_RE.sub("-", str(text).lower()).strip("-")
    return slug[:max_len] or "untitled"
