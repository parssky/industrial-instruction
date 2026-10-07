"""Benchmark suites: load each source into one item format.

``ibm``           FailureSensorIQ (IBM): every question in an original and a
                  perturbed form (``org``/``pert``), prompts used verbatim.
``paper-qwen``    held-out test split of the paper's Qwen-generated data.
``paper-claude``  held-out test split of the Claude-generated data.
``generated``     the test split this package wrote (``ii assemble``).
``custom``        any multiple-choice set, mapped by ``benchmark.custom``.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from industrial_instruction.benchmark.parsing import normalize_label
from industrial_instruction.config import BenchmarkConfig, Config, CustomBenchmarkConfig
from industrial_instruction.generate.mcq import find_options, normalize_answer, option_labels
from industrial_instruction.utils.io import iter_jsonl, read_json
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_LABELS = ["A", "B", "C", "D", "E"]


@dataclass
class BenchItem:
    """One question, ready to be prompted and scored."""

    id: str
    prompt: str  # the text shown to the model (before any context)
    query: str  # used for retrieval
    gold: List[str]  # correct labels, normalized ("P)" -> "P")
    labels: List[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    documents: List[str] = field(default_factory=list)  # for context: gold
    group: str = ""  # ibm: org / pert
    pair: Tuple = ()  # ibm: (subject, id) shared by both variants
    extra: Dict[str, Any] = field(default_factory=dict)


class SuiteError(RuntimeError):
    pass


# ------------------------------------------------------------- loading


def load_split(dataset: str, split: Optional[str], config_name: Optional[str] = None):
    """A dataset split from a ``save_to_disk`` directory or the Hub.

    On the Hub the split is looked up by name across the repo's configs and
    splits, so ``panasonic_qa_v1_test`` is found whether it is published as
    a config, a split, or a data directory. If nothing matches, the error
    lists what exists instead of loading the wrong thing.
    """
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SuiteError(
            "Hub/disk benchmarks need: pip install 'industrial-instruction[hub]'"
        ) from exc

    local = Path(dataset)
    if local.exists():
        ds = datasets.load_from_disk(str(local))
        if isinstance(ds, datasets.DatasetDict):
            if split not in ds:
                raise SuiteError(f"{local}: no split {split!r}; has {list(ds)}")
            return ds[split]
        return ds

    if config_name:
        return datasets.load_dataset(dataset, config_name, split=split)
    try:
        configs = datasets.get_dataset_config_names(dataset)
    except Exception as exc:  # noqa: BLE001
        raise SuiteError(f"cannot reach dataset {dataset!r}: {exc}") from exc
    if split in configs:  # published as a config
        splits = datasets.get_dataset_split_names(dataset, split)
        return datasets.load_dataset(dataset, split, split="test" if "test" in splits else splits[0])
    found = []
    for cfg in configs:
        names = datasets.get_dataset_split_names(dataset, cfg)
        found.extend(f"{cfg}/{n}" for n in names)
        if split in names:
            return datasets.load_dataset(dataset, cfg, split=split)
    raise SuiteError(
        f"{dataset!r} has no config or split named {split!r}. Available: {found}. "
        "Set the right name in benchmark.* (or pass a local save_to_disk path)."
    )


def _as_list(value: Any) -> List[Any]:
    """Lists stored as strings ("['B']") in some exports become lists again."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            for loader in (json.loads, ast.literal_eval):
                try:
                    out = loader(text)
                    return list(out) if isinstance(out, (list, tuple)) else [out]
                except Exception:  # noqa: BLE001
                    continue
        return [value]
    return list(value)


def _render_question(question: str, options: List[str]) -> Tuple[str, List[str]]:
    """Question text with its options, and the option labels."""
    stem, embedded = find_options(question)
    if embedded:
        return question, option_labels(embedded) or DEFAULT_LABELS
    if options:
        labelled = [
            o if option_labels([o]) else f"{chr(65 + i)}. {o}" for i, o in enumerate(options)
        ]
        return question + "\n" + "\n".join(labelled), option_labels(labelled)
    return question, list(DEFAULT_LABELS)


def _gold(answer: Any) -> List[str]:
    return [normalize_label(a) for a in normalize_answer(_as_list(answer) or answer)]


# -------------------------------------------------------------- suites


def ibm_suite(cfg: BenchmarkConfig) -> List[BenchItem]:
    items: List[BenchItem] = []
    for split in cfg.ibm_splits:
        for row in load_split(cfg.ibm_dataset, split):
            option_ids = [str(o) for o in _as_list(row["option_ids"])]
            correct = [bool(c) if not isinstance(c, str) else c == "True" for c in _as_list(row["correct"])]
            labels = [normalize_label(o) for o in option_ids]
            items.append(
                BenchItem(
                    id=f"{split}:{row['subject']}:{row['id']}",
                    prompt=row["prompt"],
                    query=row.get("question") or row["prompt"],
                    gold=[lab for lab, ok in zip(labels, correct) if ok],
                    labels=labels,
                    group=split,
                    pair=(row["subject"], str(row["id"])),
                    extra={"raw_option_ids": option_ids, "correct": correct},
                )
            )
    return items


def _qa_items(rows, *, name: str, question_field="question", answer_field="answer",
              options_field: Optional[str] = "options", documents_field: Optional[str] = "documents",
              id_field: Optional[str] = "id") -> List[BenchItem]:
    items: List[BenchItem] = []
    skipped = 0
    for i, row in enumerate(rows):
        question = str(row.get(question_field) or "").strip()
        options = [str(o) for o in _as_list(row.get(options_field))] if options_field else []
        prompt, labels = _render_question(question, options)
        gold = _gold(row.get(answer_field))
        if not question or not gold or not set(gold) <= set(labels):
            skipped += 1  # free-text answers can't be scored by label matching
            continue
        docs = _as_list(row.get(documents_field)) if documents_field else []
        items.append(
            BenchItem(
                id=str(row.get(id_field) if id_field and row.get(id_field) is not None else f"{name}:{i}"),
                prompt=prompt,
                query=question,
                gold=gold,
                labels=labels,
                documents=[str(d) for d in docs],
            )
        )
    if skipped:
        logger.warning(
            "%s: skipped %d item(s) without a question or a label answer "
            "(free-text answers are not scored)", name, skipped,
        )
    return items


def paper_suite(cfg: BenchmarkConfig, which: str) -> List[BenchItem]:
    split = cfg.paper_qwen_split if which == "qwen" else cfg.paper_claude_split
    # The split name may also be a local save_to_disk directory.
    source = split if Path(split).exists() else cfg.paper_dataset
    rows = load_split(source, None if Path(split).exists() else split)
    return _qa_items(rows, name=f"paper-{which}", id_field=None)


def generated_suite(config: Config) -> List[BenchItem]:
    path = config.paths.resolve("dataset") / "test.jsonl"
    if not path.exists():
        raise SuiteError(f"{path} not found; run 'ii assemble' first")
    return _qa_items(iter_jsonl(path), name="generated")


def custom_suite(cfg: CustomBenchmarkConfig, root: Path) -> List[BenchItem]:
    if not cfg.path:
        raise SuiteError("benchmark.custom.path is not set")
    source = cfg.source.lower()
    path = Path(cfg.path)
    if not path.is_absolute() and (root / path).exists():
        path = root / path
    if source == "jsonl":
        rows = list(iter_jsonl(path))
    elif source == "json":
        loaded = read_json(path)
        rows = loaded if isinstance(loaded, list) else loaded.get(cfg.split or "test", [])
    elif source in ("huggingface", "hf", "disk"):
        rows = load_split(str(path) if source == "disk" else cfg.path, cfg.split, cfg.config_name)
    else:
        raise SuiteError(f"benchmark.custom.source {cfg.source!r}: use jsonl, json, huggingface or disk")
    return _qa_items(
        rows,
        name="custom",
        question_field=cfg.question_field,
        answer_field=cfg.answer_field,
        options_field=cfg.options_field,
        documents_field=cfg.documents_field,
        id_field=cfg.id_field,
    )


_SUITES: Dict[str, Callable[[Config], List[BenchItem]]] = {
    "ibm": lambda c: ibm_suite(c.benchmark),
    "paper-qwen": lambda c: paper_suite(c.benchmark, "qwen"),
    "paper-claude": lambda c: paper_suite(c.benchmark, "claude"),
    "generated": generated_suite,
    "custom": lambda c: custom_suite(c.benchmark.custom, Path(c.paths.root)),
}


def register_suite(name: str, loader: Callable[[Config], List[BenchItem]]) -> None:
    """Add a suite: ``loader(config) -> [BenchItem, ...]``."""
    _SUITES[name.lower()] = loader


def available_suites() -> List[str]:
    return sorted(_SUITES)


def load_suite(name: str, config: Config) -> List[BenchItem]:
    key = name.lower()
    if key not in _SUITES:
        raise SuiteError(f"unknown suite {name!r}; available: {available_suites()}")
    items = _SUITES[key](config)
    if config.benchmark.limit:
        items = _limit(items, int(config.benchmark.limit))
    return items


def _limit(items: List[BenchItem], n: int) -> List[BenchItem]:
    """First ``n`` items; for paired suites, the first ``n`` complete pairs."""
    if not any(i.pair for i in items):
        return items[:n]
    keep: set = set()
    for item in items:
        if item.pair not in keep and len(keep) < n:
            keep.add(item.pair)
    return [i for i in items if i.pair in keep]
