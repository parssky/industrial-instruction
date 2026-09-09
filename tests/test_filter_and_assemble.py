"""Offline tests for the filter and assemble stages (no API, no model)."""

from industrial_instruction.assemble.assemble import (
    assemble_dataset,
    to_chat_record,
)
from industrial_instruction.config import Config, FilterConfig
from industrial_instruction.filter.rules import RuleFilter
from industrial_instruction.schemas import QASample


def make_sample(**kwargs) -> QASample:
    defaults = dict(
        id="s1",
        relation="r1",
        question="What is the recommended torque for the main bearing bolts?",
        answer="Around 120 Nm, tightened in three stages.",
        documents=["doc text"],
    )
    defaults.update(kwargs)
    return QASample(**defaults)


def test_accepts_a_good_sample():
    assert RuleFilter(FilterConfig()).check(make_sample()).passed


def test_rejects_short_question_and_missing_answer():
    rules = RuleFilter(FilterConfig())
    assert not rules.check(make_sample(question="why?")).passed
    assert not rules.check(make_sample(answer="")).passed


def test_rejects_meta_reference():
    result = RuleFilter(FilterConfig()).check(
        make_sample(
            question="According to the documents, what is the bearing torque value?"
        )
    )
    assert not result.passed
    assert "meta_reference" in result.reasons


def test_rejects_missing_context():
    assert not RuleFilter(FilterConfig()).check(make_sample(documents=[])).passed


def test_dedupes_normalized_duplicates():
    rules = RuleFilter(FilterConfig())
    assert rules.check(make_sample(id="a")).passed
    second = rules.check(
        make_sample(
            id="b",
            question="What is the recommended TORQUE for the main bearing bolts??",
        )
    )
    assert not second.passed
    assert "duplicate" in second.reasons


def test_wrong_option_count_is_rejected():
    rules = RuleFilter(FilterConfig(enforce_option_count=5))
    assert not rules.check(make_sample(options=["A. x", "B. y"])).passed


def test_assemble_splits_are_deterministic_and_stratified(tmp_path):
    samples = [
        make_sample(
            id=f"{relation}-{i}",
            relation=relation,
            question=f"Question number {i} about maintenance intervals?",
        )
        for relation in ("r0", "r1", "r2")
        for i in range(10)
    ]
    config = Config.from_dict(
        {
            "paths": {"root": str(tmp_path)},
            "assemble": {"splits": {"train": 0.8, "test": 0.2}},
        }
    )

    first = assemble_dataset(config, samples=samples)
    assert first.inputs == 30
    assert first.details["splits"] == {"train": 24, "test": 6}
    # every relation appears in test, i.e. the split is stratified
    assert set(first.details["per_relation"]["test"]) == {"r0", "r1", "r2"}

    dataset_dir = config.paths.resolve("dataset")
    assert (dataset_dir / "train.jsonl").exists()
    assert (dataset_dir / "train.chat.jsonl").exists()
    assert (dataset_dir / "dataset_stats.json").exists()

    train_before = (dataset_dir / "train.jsonl").read_text(encoding="utf-8")
    assemble_dataset(config, samples=samples)
    assert (dataset_dir / "train.jsonl").read_text(encoding="utf-8") == train_before


def test_chat_record_shape():
    record = to_chat_record(make_sample(options=["A. x", "B. y"]), True)
    assert [m["role"] for m in record["messages"]] == ["user", "assistant"]
    assert "<Documents>" in record["messages"][0]["content"]
    assert "A. x" in record["messages"][0]["content"]
