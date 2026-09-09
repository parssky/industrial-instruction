"""Seed ('simulated instruction') providers.

Seeds steer the style and format of generated questions. Three sources are
supported so a team is never forced to depend on an external dataset:

``huggingface``
    Any HF dataset, e.g. the FailureSensorIQ set used in the paper.
``jsonl`` / ``json``
    Your own instructions from a local file.
``self``
    Seeds derived from your own indexed chunks with one cheap LLM call each,
    so a brand-new corpus can bootstrap with no seed data at all.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from industrial_instruction.config import Config, SeedsConfig
from industrial_instruction.schemas import Seed
from industrial_instruction.utils.io import iter_jsonl, read_json, stable_id
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


def load_seeds(config: Config, limit: Optional[int] = None) -> List[Seed]:
    """Load seeds according to ``config.seeds``."""
    cfg = config.seeds
    source = (cfg.source or "huggingface").lower()

    if source in ("huggingface", "hf", "datasets"):
        seeds = _from_huggingface(cfg)
    elif source in ("jsonl", "json", "file", "local"):
        seeds = _from_file(cfg)
    elif source in ("self", "self_seed", "bootstrap"):
        seeds = _from_self(config)
    elif source in ("none", "empty"):
        seeds = []
    else:
        raise ValueError(
            f"Unknown seeds.source {cfg.source!r}. "
            "Use 'huggingface', 'jsonl', 'self' or 'none'."
        )

    if cfg.shuffle:
        random.Random(cfg.seed).shuffle(seeds)
    cap = limit if limit is not None else cfg.limit
    if cap:
        seeds = seeds[: int(cap)]
    logger.info("loaded %d seeds from %s", len(seeds), source)
    return seeds


# ----------------------------------------------------------------------


def _from_huggingface(cfg: SeedsConfig) -> List[Seed]:
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "The 'datasets' library is required for huggingface seeds. Install with: "
            "pip install 'industrial-instruction[hub]'"
        ) from exc

    target = cfg.dataset
    if not target:
        raise ValueError("seeds.dataset must be set for the huggingface source")

    local = Path(target)
    if local.exists():
        ds = load_from_disk(str(local))
    else:
        kwargs = {"revision": cfg.revision} if cfg.revision else {}
        ds = load_dataset(target, **kwargs)

    seeds: List[Seed] = []
    available = list(getattr(ds, "keys", lambda: [])())
    splits = cfg.splits or available or ["train"]
    for split in splits:
        if available and split not in available:
            logger.warning(
                "split %r not in dataset (available: %s); skipping", split, available
            )
            continue
        subset = ds[split] if available else ds
        for row in subset:
            seed = _row_to_seed(row, cfg, source=f"{target}:{split}")
            if seed:
                seeds.append(seed)
    return seeds


def _from_file(cfg: SeedsConfig) -> List[Seed]:
    if not cfg.path:
        raise ValueError("seeds.path must be set for the jsonl/json source")
    path = Path(cfg.path)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: List[Any] = list(iter_jsonl(path))
    else:
        loaded = read_json(path)
        rows = loaded if isinstance(loaded, list) else [loaded]

    seeds: List[Seed] = []
    for row in rows:
        if isinstance(row, str):
            text = row.strip()
            if text:
                seeds.append(
                    Seed(id=stable_id(text), text=text, source=str(path.name))
                )
            continue
        seed = _row_to_seed(row, cfg, source=str(path.name))
        if seed:
            seeds.append(seed)
    return seeds


def _from_self(config: Config) -> List[Seed]:
    """Bootstrap seeds from the user's own corpus via the generator model."""
    from industrial_instruction.chunk.chunker import load_chunks
    from industrial_instruction.generate.client import LLMClient
    from industrial_instruction.generate.prompt_loader import (
        PromptLibrary,
        format_documents,
    )

    cfg = config.seeds
    chunks = load_chunks(config.paths.resolve("chunks"))
    if not chunks:
        raise RuntimeError(
            "No chunks available for self-seeding. Run the extract and chunk "
            "stages first."
        )
    rng = random.Random(cfg.seed)
    rng.shuffle(chunks)
    target = int(cfg.limit or min(len(chunks), 200))
    selected = chunks[:target]

    library = PromptLibrary(config.generate.prompt_dir)
    client = LLMClient(config.generate)
    seeds: List[Seed] = []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    def one(chunk):
        prompt = library.render(
            cfg.self_seed_prompt,
            docs=format_documents([chunk.as_context()]),
        )
        payload, _ = client.complete_json(prompt)
        text = (payload.get("q*") or payload.get("question") or "").strip()
        if not text:
            return None
        return Seed(
            id=stable_id(chunk.id, text),
            text=text,
            source="self",
            meta={"chunk_id": chunk.id, "doc_id": chunk.doc_id},
        )

    workers = max(config.generate.max_workers, 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, c) for c in selected]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="self-seed", unit="seed"
        ):
            try:
                seed = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("self-seed failed: %s", exc)
                continue
            if seed:
                seeds.append(seed)
    return seeds


def _row_to_seed(
    row: Dict[str, Any], cfg: SeedsConfig, source: str
) -> Optional[Seed]:
    if not isinstance(row, dict):
        return None
    text = row.get(cfg.text_field)
    if text is None:
        for fallback in ("prompt", "instruction", "question", "text", "input"):
            if row.get(fallback):
                text = row[fallback]
                break
    if not text:
        return None
    text = str(text).strip()
    if not text:
        return None
    raw_id = row.get(cfg.id_field) if cfg.id_field else None
    meta = {
        k: v
        for k, v in row.items()
        if k != cfg.text_field and isinstance(v, (str, int, float, bool))
    }
    return Seed(
        id=str(raw_id) if raw_id else stable_id(source, text),
        text=text,
        source=source,
        meta=meta,
    )
