"""Deterministic rule-based filters.

These encode the checks that were previously done by hand across ten
near-identical ``filter_rX_samples.ipynb`` notebooks: length bounds, missing
answers, meta-references such as "according to the provided context", wrong
option counts, and duplicates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from industrial_instruction.config import FilterConfig
from industrial_instruction.generate.mcq import (
    has_format_instructions,
    option_labels,
    option_texts,
)
from industrial_instruction.schemas import QASample
from industrial_instruction.utils.io import stable_id

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


@dataclass
class RuleResult:
    """Outcome of running the rule filters on one sample."""

    passed: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def reason(self) -> Optional[str]:
        return "; ".join(self.reasons) if self.reasons else None


def normalize(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


class RuleFilter:
    """Stateful filter: tracks seen questions so it can drop duplicates."""

    def __init__(self, config: FilterConfig, n_options: int = 5) -> None:
        self.config = config
        # Expected option count; comes from generate.n_options.
        self.n_options = n_options
        self._seen: Set[str] = set()
        self.counts: Dict[str, int] = {}

    def _note(self, reason: str) -> str:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        return reason

    def check(self, sample: QASample) -> RuleResult:
        cfg = self.config
        reasons: List[str] = []
        question = (sample.question or "").strip()
        answer = _answer_text(sample.answer)
        mcq = bool(sample.meta.get("mcq")) or bool(sample.options)

        if len(question) < cfg.min_question_chars:
            reasons.append(self._note("question_too_short"))
        if cfg.max_question_chars and len(question) > cfg.max_question_chars:
            reasons.append(self._note("question_too_long"))

        if cfg.require_answer and not answer and not sample.options:
            reasons.append(self._note("missing_answer"))
        elif answer and len(answer) < cfg.min_answer_chars:
            reasons.append(self._note("answer_too_short"))

        if cfg.forbid_meta_references:
            lowered = question.lower()
            for phrase in cfg.meta_reference_phrases:
                if phrase.lower() in lowered:
                    reasons.append(self._note("meta_reference"))
                    break

        if cfg.enforce_option_count and sample.options:
            if len(sample.options) != self.n_options:
                reasons.append(self._note("wrong_option_count"))

        if mcq:
            reasons.extend(self._check_mcq(sample))

        if cfg.forbid_format_instructions and has_format_instructions(question):
            reasons.append(self._note("format_instructions_in_question"))

        if not sample.documents:
            reasons.append(self._note("no_context"))

        if cfg.dedupe:
            key = (
                normalize(question)
                if cfg.dedupe_mode == "normalized"
                else question
            )
            fingerprint = stable_id(sample.relation, key)
            if fingerprint in self._seen:
                reasons.append(self._note("duplicate"))
            else:
                self._seen.add(fingerprint)

        return RuleResult(passed=not reasons, reasons=reasons)

    def _check_mcq(self, sample: QASample) -> List[str]:
        cfg = self.config
        reasons: List[str] = []
        options = sample.options or []
        if cfg.require_options and not options:
            return [self._note("missing_options")]
        labels = set(option_labels(options))
        answers = sample.answer if isinstance(sample.answer, list) else [sample.answer]
        if cfg.answer_in_options and options:
            if not answers or any(str(a) not in labels for a in answers):
                reasons.append(self._note("answer_not_in_options"))
        seed_opts = sample.meta.get("seed_options") or []
        if seed_opts and options and cfg.max_seed_option_overlap < 1:
            seen = set(option_texts(seed_opts))
            copied = sum(1 for t in option_texts(options) if t in seen)
            if copied / len(options) > cfg.max_seed_option_overlap:
                reasons.append(self._note("copied_seed_options"))
        return reasons


def _answer_text(answer) -> str:
    """Answers are strings for QA samples and label lists for MCQ samples."""
    if answer is None:
        return ""
    if isinstance(answer, (list, tuple)):
        return " ".join(str(a) for a in answer).strip()
    if isinstance(answer, dict):
        return " ".join(str(v) for v in answer.values()).strip()
    return str(answer).strip()
