"""Optional LLM-as-judge scoring, applied after the rule filters.

Disabled by default: it costs one extra API call per surviving sample. Enable
with ``filter.judge_enabled: true`` when you want a quality bar beyond the
deterministic rules.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm

from industrial_instruction.config import Config
from industrial_instruction.generate.client import LLMClient
from industrial_instruction.generate.prompt_loader import (
    PromptLibrary,
    format_documents,
)
from industrial_instruction.schemas import QASample
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)

_CRITERIA = ("faithful", "standalone", "relation_match", "usefulness")


class Judge:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.library = PromptLibrary(config.generate.prompt_dir)
        self.client = LLMClient(config.generate)
        self.relation_descriptions = {
            r.id: (r.description or r.prompt)
            for r in config.generate.enabled_relations()
        }

    def score(self, sample: QASample) -> Tuple[float, Dict[str, object]]:
        prompt = self.library.render(
            "judge",
            docs=format_documents(sample.documents),
            question=sample.question,
            answer=sample.answer or "",
            relation_description=self.relation_descriptions.get(
                sample.relation, sample.relation
            ),
        )
        payload, _ = self.client.complete_json(
            prompt, model=self.config.filter.judge_model or None
        )
        scores = {}
        for key in _CRITERIA:
            try:
                scores[key] = float(payload.get(key, 0) or 0)
            except (TypeError, ValueError):
                scores[key] = 0.0
        mean = sum(scores.values()) / len(_CRITERIA)
        details: Dict[str, object] = dict(scores)
        details["reason"] = str(payload.get("reason", ""))[:500]
        details["mean"] = round(mean, 3)
        return mean, details


def judge_samples(
    config: Config, samples: Sequence[QASample]
) -> Tuple[List[QASample], List[QASample]]:
    """Split samples into ``(kept, rejected)`` using judge scores."""
    judge = Judge(config)
    threshold = config.filter.judge_min_score
    workers = max(config.filter.judge_max_workers or config.generate.max_workers, 1)
    kept: List[QASample] = []
    rejected: List[QASample] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge.score, s): s for s in samples}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="judge", unit="sample"
        ):
            sample = futures[future]
            try:
                mean, details = future.result()
            except Exception as exc:  # noqa: BLE001 - never fail the stage
                logger.warning("judge failed for %s: %s", sample.id, exc)
                sample.meta["judge_error"] = repr(exc)[:200]
                kept.append(sample)
                continue
            sample.meta["judge"] = details
            if mean >= threshold:
                kept.append(sample)
            else:
                sample.reject_reason = f"judge_score {mean:.2f} < {threshold}"
                rejected.append(sample)
    return kept, rejected
