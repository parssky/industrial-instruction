"""Filter-stage orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from industrial_instruction.config import Config
from industrial_instruction.filter.rules import RuleFilter
from industrial_instruction.schemas import QASample, SampleStatus, StageReport
from industrial_instruction.utils.io import append_jsonl, ensure_dir, iter_jsonl
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


def load_generated(directory: Path) -> Dict[str, List[QASample]]:
    """Load ``r*.jsonl`` files produced by the generate stage."""
    grouped: Dict[str, List[QASample]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        if path.name == "rejects.jsonl":
            continue
        rows = []
        for row in iter_jsonl(path):
            try:
                rows.append(QASample.model_validate(row))
            except Exception as exc:  # noqa: BLE001
                logger.warning("skipping malformed row in %s: %s", path.name, exc)
        if rows:
            grouped[path.stem] = rows
    return grouped


def filter_samples(
    config: Config,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> StageReport:
    """Apply rule filters (and optionally the judge) to generated samples."""
    t0 = time.time()
    src = (
        Path(input_dir).resolve() if input_dir else config.paths.resolve("generated")
    )
    out = (
        Path(output_dir).resolve() if output_dir else config.paths.resolve("filtered")
    )
    grouped = load_generated(src)
    if not grouped:
        raise RuntimeError(
            f"No generated samples found in {src}. Run the generate stage first."
        )

    ensure_dir(out)
    for path in out.glob("*.jsonl"):
        path.unlink()

    rules = RuleFilter(config.filter)
    kept_counts: Dict[str, int] = {}
    total_in = 0
    rejected: List[QASample] = []
    survivors: Dict[str, List[QASample]] = {}

    for relation, samples in grouped.items():
        total_in += len(samples)
        keep: List[QASample] = []
        for sample in samples:
            result = rules.check(sample)
            if result.passed:
                keep.append(sample)
            else:
                sample.status = SampleStatus.REJECTED
                sample.reject_reason = result.reason
                rejected.append(sample)
        survivors[relation] = keep

    if config.filter.judge_enabled:
        for relation, samples in list(survivors.items()):
            from industrial_instruction.filter.judge import judge_samples

            kept, dropped = judge_samples(config, samples)
            for sample in dropped:
                sample.status = SampleStatus.REJECTED
            rejected.extend(dropped)
            survivors[relation] = kept

    for relation, samples in survivors.items():
        if samples:
            append_jsonl(out / f"{relation}.jsonl", samples)
        kept_counts[relation] = len(samples)
    if rejected:
        append_jsonl(out / "rejected.jsonl", rejected)

    total_out = sum(kept_counts.values())
    report = StageReport(
        stage="filter",
        inputs=total_in,
        outputs=total_out,
        rejected=len(rejected),
        seconds=round(time.time() - t0, 2),
        details={
            "output_dir": str(out),
            "per_relation": kept_counts,
            "reject_reasons": rules.counts,
            "judge_enabled": config.filter.judge_enabled,
            "pass_rate": round(total_out / total_in, 4) if total_in else 0.0,
        },
    )
    logger.info(
        "filter: kept %d/%d samples (%.1f%%) in %.1fs",
        total_out,
        total_in,
        100 * total_out / total_in if total_in else 0.0,
        report.seconds,
    )
    return report
