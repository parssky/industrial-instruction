"""Merge filtered relations into a splittable, publishable dataset.

This replaces ``merge_all_dataset.ipynb`` (an 18 MB notebook with committed
outputs) with a deterministic, seeded function. Splits are reproducible and
optionally stratified by relation, so r0-r4 stay proportionally represented
in train and test.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from industrial_instruction.config import Config
from industrial_instruction.filter.runner import load_generated
from industrial_instruction.schemas import QASample, StageReport
from industrial_instruction.utils.io import append_jsonl, ensure_dir, write_json
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


def _split_indices(
    n: int, ratios: Dict[str, float], rng: random.Random
) -> Dict[str, List[int]]:
    order = list(range(n))
    rng.shuffle(order)
    total = sum(ratios.values()) or 1.0
    result: Dict[str, List[int]] = {}
    cursor = 0
    names = list(ratios)
    for i, name in enumerate(names):
        if i == len(names) - 1:
            result[name] = order[cursor:]
        else:
            take = int(round(n * ratios[name] / total))
            result[name] = order[cursor : cursor + take]
            cursor += take
    return result


def to_chat_record(sample: QASample, include_documents: bool) -> dict:
    """Chat-style record for SFT trainers (TRL, Unsloth, axolotl)."""
    question = sample.question
    if sample.options:
        rendered = "\n".join(str(o) for o in sample.options)
        question = f"{question}\n\n{rendered}"
    if include_documents and sample.documents:
        context = "\n\n".join(sample.documents)
        user = f"<Documents>\n{context}\n</Documents>\n\n{question}"
    else:
        user = question
    return {
        "id": sample.id,
        "relation": sample.relation,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": sample.answer or ""},
        ],
    }


def to_flat_record(sample: QASample, include_documents: bool) -> dict:
    record = {
        "id": sample.id,
        "relation": sample.relation,
        "question": sample.question,
        "answer": sample.answer,
        "options": sample.options,
        "seed": sample.seed,
        "doc_ids": sample.doc_ids,
    }
    if include_documents:
        record["documents"] = sample.documents
    return record


def assemble_dataset(
    config: Config,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    samples: Optional[Sequence[QASample]] = None,
) -> StageReport:
    """Write train/test splits in the configured formats."""
    t0 = time.time()
    cfg = config.assemble
    out = Path(output_dir).resolve() if output_dir else config.paths.resolve("dataset")
    ensure_dir(out)

    if samples is None:
        src = (
            Path(input_dir).resolve()
            if input_dir
            else config.paths.resolve("filtered")
        )
        grouped = load_generated(src)
        grouped.pop("rejected", None)
        if not grouped:
            raise RuntimeError(
                f"No filtered samples found in {src}. Run the filter stage first."
            )
        pool = [s for rows in grouped.values() for s in rows]
    else:
        pool = list(samples)

    rng = random.Random(cfg.split_seed)
    ratios = dict(cfg.splits) or {"train": 1.0}

    assignments: Dict[str, List[QASample]] = defaultdict(list)
    if cfg.stratify_by_relation:
        by_relation: Dict[str, List[QASample]] = defaultdict(list)
        for sample in pool:
            by_relation[sample.relation].append(sample)
        for relation in sorted(by_relation):
            rows = by_relation[relation]
            for split, indices in _split_indices(len(rows), ratios, rng).items():
                assignments[split].extend(rows[i] for i in indices)
    else:
        for split, indices in _split_indices(len(pool), ratios, rng).items():
            assignments[split].extend(pool[i] for i in indices)

    written: Dict[str, Dict[str, int]] = {}
    for split, rows in assignments.items():
        rng.shuffle(rows)
        counts: Dict[str, int] = {}
        for fmt in cfg.formats:
            if fmt == "jsonl":
                path = out / f"{split}.jsonl"
                if path.exists():
                    path.unlink()
                append_jsonl(
                    path, [to_flat_record(s, cfg.include_documents) for s in rows]
                )
                counts["jsonl"] = len(rows)
            elif fmt == "chat":
                path = out / f"{split}.chat.jsonl"
                if path.exists():
                    path.unlink()
                append_jsonl(
                    path, [to_chat_record(s, cfg.include_documents) for s in rows]
                )
                counts["chat"] = len(rows)
            else:
                raise ValueError(
                    f"Unknown assemble.formats entry {fmt!r}. Use 'jsonl' or 'chat'."
                )
        written[split] = counts

    per_relation: Dict[str, Dict[str, int]] = {}
    for split, rows in assignments.items():
        breakdown: Dict[str, int] = {}
        for sample in rows:
            breakdown[sample.relation] = breakdown.get(sample.relation, 0) + 1
        per_relation[split] = breakdown

    stats = {
        "total": len(pool),
        "splits": {k: len(v) for k, v in assignments.items()},
        "per_relation": per_relation,
        "formats": list(cfg.formats),
        "output_dir": str(out),
    }
    write_json(out / "dataset_stats.json", stats)

    if cfg.push_to_hub:
        _push_to_hub(cfg.push_to_hub, out, assignments, cfg)
        stats["pushed_to"] = cfg.push_to_hub

    report = StageReport(
        stage="assemble",
        inputs=len(pool),
        outputs=sum(len(v) for v in assignments.values()),
        seconds=round(time.time() - t0, 2),
        details={**stats, "written": written},
    )
    logger.info(
        "assemble: %d samples -> %s in %.1fs",
        len(pool),
        stats["splits"],
        report.seconds,
    )
    return report


def _push_to_hub(repo_id: str, out: Path, assignments, cfg) -> None:
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Publishing needs the 'datasets' library: "
            "pip install 'industrial-instruction[hub]'"
        ) from exc
    bundle = DatasetDict(
        {
            split: Dataset.from_list(
                [to_flat_record(s, cfg.include_documents) for s in rows]
            )
            for split, rows in assignments.items()
            if rows
        }
    )
    logger.info("pushing dataset to %s", repo_id)
    bundle.push_to_hub(repo_id, private=cfg.private)
