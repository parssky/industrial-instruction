"""Benchmark suites, answer parsing, metrics and the runner (no GPU, no network)."""

import glob
import json
from pathlib import Path

import pytest

from industrial_instruction.benchmark import (
    BenchItem,
    ContextBuilder,
    Endpoint,
    format_table,
    load_suite,
    parse_answer,
    register_suite,
    run_benchmarks,
)
from industrial_instruction.benchmark import suites as suites_mod
from industrial_instruction.benchmark.metrics import ibm_metrics, set_scores
from industrial_instruction.benchmark.parsing import normalize_label
from industrial_instruction.benchmark.runner import _legacy_ibm_correct, summarize
from industrial_instruction.config import Config, EndpointConfig
from industrial_instruction.schemas import Chunk
from industrial_instruction.store.faiss_store import SearchHit

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"answer": ["B"]}', ["B"]),
        ("['E']", ["E"]),  # what the Claude-trained model learned to output
        ('{"answer": ["B"}', ["B"]),  # malformed JSON seen in the paper runs
        ('```json\n{"answer": ["A", "C"]}\n```', ["A", "C"]),
        ('{"option_a": "...", "answer": ["T"]}', ["T"]),  # IBM perturbed format
        ('{"answer": "P)"}', ["P"]),
        ("<think>maybe A</think>{\"answer\": [\"D\"]}", ["D"]),
        ("Answer: B", ["B"]),
        ("The answer is (B) and D.", ["B", "D"]),
        ("B, D", ["B", "D"]),
        ("[]", []),
        ("The answer is a pressure valve.", None),  # no stray-letter matching
        ("", None),
    ],
)
def test_parse_answer(reply, expected):
    assert parse_answer(reply, valid="ABCDE") == expected


def test_normalize_label():
    assert [normalize_label(x) for x in ["P)", "(b)", " C. ", "EKM-B"]] == ["P", "B", "C", "EKM-B"]


# ---------------------------------------------------------------- metrics


def test_set_scores():
    assert set_scores(["A"], ["A"]) == (1.0, 1.0, 1.0)
    em, jac, f1 = set_scores(["A", "B"], ["A"])
    assert em == 0 and jac == 0.5 and round(f1, 3) == 0.667
    assert set_scores(None, ["A"]) == (0.0, 0.0, 0.0)


def test_ibm_consistency_needs_both_variants_right():
    rows = [
        {"pair": ("s", "1"), "group": "org", "pred": ["A"], "gold": ["A"]},
        {"pair": ("s", "1"), "group": "pert", "pred": ["P"], "gold": ["P"]},
        {"pair": ("s", "2"), "group": "org", "pred": ["B"], "gold": ["B"]},
        {"pair": ("s", "2"), "group": "pert", "pred": ["Q"], "gold": ["R"]},
    ]
    m = ibm_metrics(rows)
    assert m["acc_original"] == 1.0 and m["acc_perturb"] == 0.5 and m["consistency"] == 0.5


SAVED = sorted(glob.glob(str(ROOT / "package_benchmark_ibm/results_with_table/Qwen3-4B-Instruct-2507-2026*samples-RAG.json")))


@pytest.mark.skipif(not SAVED, reason="saved IBM results not in checkout")
def test_saved_ibm_run_legacy_matches_report_and_fix_scores_perturbed():
    """Replays the paper's saved base-model replies through both scorers."""
    data = json.load(open(SAVED[0]))
    reported = json.load(open(SAVED[0].replace("samples", "evaluation")))
    rows = []
    for split in ("org", "pert"):
        for r in data[split]:
            labels = [normalize_label(o) for o in r["option_ids"]]
            item = BenchItem(
                id=str(r["id"]), prompt="", query="",
                gold=[lab for lab, ok in zip(labels, r["correct"]) if ok], labels=labels,
                group=split, pair=(r["subject"], str(r["id"])),
                extra={"raw_option_ids": r["option_ids"], "correct": r["correct"]},
            )
            row = {"group": split, "pair": list(item.pair), "gold": item.gold,
                   "reply": r["model_original_output"]}
            row["pred"] = parse_answer(row["reply"], valid=labels)
            row["legacy_correct"] = _legacy_ibm_correct(row, item)
            rows.append(row)
    s = summarize("ibm", rows)
    assert s["legacy"]["acc_original"] == pytest.approx(reported["acc_original"])
    assert s["legacy"]["acc_perturb"] == pytest.approx(reported["acc_perturb"])
    # "P)" option ids made every perturbed item unscoreable before:
    assert reported["acc_perturb"] < 0.01 and s["acc_perturb"] > 0.3


# ----------------------------------------------------------------- suites

IBM_ROWS = {
    "org": [{"subject": "fm", "id": 1, "prompt": "Q1?\nOptions:\nA x\nB y", "question": "Q1?",
             "option_ids": ["A", "B"], "correct": [False, True]}],
    "pert": [{"subject": "fm", "id": 1, "prompt": "Q1 again?\nOptions:\nP) y\nQ) x", "question": "Q1 again?",
              "option_ids": ["P)", "Q)"], "correct": [True, False]}],
}


@pytest.fixture
def fake_hub(monkeypatch):
    def load_split(dataset, split, config_name=None):
        if dataset == "ibm-research/FailureSensorIQ":
            return IBM_ROWS[split]
        if split == "pana_qa_claude_v1_test":
            return [{"question": "Which relay? Options: A. EKM B. AMN C. ERJ", "answer": ["B"],
                     "documents": ["AMN relays are used for..."]}]
        raise suites_mod.SuiteError(f"no {split}")

    monkeypatch.setattr(suites_mod, "load_split", load_split)


def test_ibm_suite_normalizes_perturbed_labels(fake_hub):
    items = load_suite("ibm", Config())
    pert = next(i for i in items if i.group == "pert")
    assert pert.labels == ["P", "Q"] and pert.gold == ["P"]
    assert pert.prompt.startswith("Q1 again?") and pert.query == "Q1 again?"
    assert {i.pair for i in items} == {("fm", "1")}


def test_paper_suite_keeps_embedded_options_and_documents(fake_hub):
    (item,) = load_suite("paper-claude", Config())
    assert item.gold == ["B"] and item.labels == ["A", "B", "C"]
    assert item.prompt.count("AMN") == 1 and item.documents


def test_custom_jsonl_with_separate_options_and_string_answers(tmp_path):
    path = tmp_path / "bench.jsonl"
    rows = [
        {"q": "Which fuse rating?", "opts": ["5 A", "10 A", "15 A"], "ans": "B"},
        {"q": "Which two apply?", "opts": ["x", "y", "z"], "ans": "['A', 'C']"},
        {"q": "Explain the wiring.", "opts": [], "ans": "Connect L to terminal 1."},  # not scoreable
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    config = Config.from_dict({
        "paths": {"root": str(tmp_path)},
        "benchmark": {"custom": {"path": "bench.jsonl", "question_field": "q",
                                 "answer_field": "ans", "options_field": "opts", "id_field": None}},
    })
    items = load_suite("custom", config)
    assert [i.gold for i in items] == [["B"], ["A", "C"]]
    assert "B. 10 A" in items[0].prompt


def test_generated_suite_reads_assemble_output(tmp_path):
    config = Config.from_dict({"paths": {"root": str(tmp_path)}})
    out = config.paths.resolve("dataset")
    out.mkdir(parents=True)
    (out / "test.jsonl").write_text(json.dumps({
        "id": "g1", "question": "Which oil grade?", "answer": ["A"],
        "options": ["A. ISO VG 46", "B. ISO VG 100"], "documents": ["Use ISO VG 46."]}))
    (item,) = load_suite("generated", config)
    assert item.id == "g1" and item.gold == ["A"] and "B. ISO VG 100" in item.prompt


def test_limit_keeps_whole_ibm_pairs(fake_hub):
    config = Config.from_dict({"benchmark": {"limit": 1}})
    assert len(load_suite("ibm", config)) == 2


def test_register_custom_suite():
    register_suite("tiny", lambda c: [BenchItem(id="t", prompt="Q?", query="Q?", gold=["A"])])
    assert load_suite("tiny", Config())[0].id == "t"


# ----------------------------------------------------------------- runner


class FakeOpenAI:
    """Answers from a lookup on the prompt; records what it was sent."""

    def __init__(self, answer_for):
        self.answer_for = answer_for
        self.sent = []
        self.chat = self
        self.completions = self
        self.models = self

    def list(self):
        return type("L", (), {"data": [type("M", (), {"id": "org/my-model"})]})

    def create(self, **kw):
        self.sent.append(kw)
        content = self.answer_for(kw["messages"][-1]["content"])
        if isinstance(content, Exception):
            raise content
        msg = type("Msg", (), {"content": content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


class FakeStore:
    def search(self, query, k):
        chunk = Chunk(id="c", doc_id="d", text="Retrieved: AMN relays.")
        return [SearchHit(chunk=chunk, score=1.0, rank=0)] * k


def test_run_benchmarks_end_to_end(tmp_path, fake_hub):
    def answer(prompt):
        if "Q1 again" in prompt:
            return '{"option_a": "...", "answer": ["P"]}'
        if "Q1?" in prompt:
            return "['B']"
        return '{"answer": ["B"]}' if "AMN relays are used" in prompt else '{"answer": ["A"]}'

    fake = FakeOpenAI(answer)
    config = Config.from_dict({
        "paths": {"root": str(tmp_path)},
        "benchmark": {"endpoint": {"base_url": "http://x/v1", "max_workers": 2}},
    })
    summary = run_benchmarks(
        config,
        suites=["ibm", "paper-claude"],
        contexts=["none", "gold", "retrieved"],
        endpoint=Endpoint(config.benchmark.endpoint, client=fake),
        store=FakeStore(),
    )
    r = summary["results"]
    assert summary["model"] == "org/my-model"
    assert r["ibm-none"]["acc_original"] == 1.0 and r["ibm-none"]["acc_perturb"] == 1.0
    assert r["ibm-none"]["legacy"]["acc_perturb"] == 0.0  # the old rule
    assert "ibm-gold" not in r  # IBM has no gold documents
    assert r["paper-claude-gold"]["set_match"] == 1.0  # gold docs reached the model
    assert r["paper-claude-none"]["set_match"] == 0.0
    assert r["paper-claude-retrieved"]["n"] == 1

    gold_prompt = next(s for s in fake.sent if "AMN relays are used" in s["messages"][-1]["content"])
    user = gold_prompt["messages"][-1]["content"]
    assert "Based on relevat document answer this question." in user
    assert gold_prompt["messages"][0]["content"].startswith("You are a helpful assistant")
    assert gold_prompt["temperature"] == 0.0

    out = Path(summary["output_dir"])
    assert (out / "summary.json").exists() and (out / "ibm-none.samples.jsonl").exists()
    assert "acc_pert" in format_table(summary) and "set-match" in format_table(summary)


def test_request_errors_are_counted_not_fatal(tmp_path):
    register_suite("two", lambda c: [BenchItem(id=str(i), prompt=f"Q{i}?", query="", gold=["A"]) for i in range(2)])
    fake = FakeOpenAI(lambda p: RuntimeError("503") if "Q1" in p else '{"answer": ["A"]}')
    config = Config.from_dict({"paths": {"root": str(tmp_path)},
                               "benchmark": {"endpoint": {"max_retries": 1}}})
    summary = run_benchmarks(config, suites=["two"], contexts=["none"],
                             endpoint=Endpoint(config.benchmark.endpoint, client=fake))
    res = summary["results"]["two-none"]
    assert res["errors"] == 1 and res["set_match"] == 0.5


def test_context_template_keeps_braces_in_documents():
    item = BenchItem(id="x", prompt="Q?", query="Q?", gold=["A"], documents=['{"spec": 1}'])
    text = ContextBuilder(Config()).prompt(item, "gold")
    assert '{"spec": 1}' in text and "question: Q?" in text


def test_unreachable_server_has_a_clear_error():
    class Down:
        models = property(lambda self: (_ for _ in ()).throw(ConnectionError("refused")))

    with pytest.raises(RuntimeError, match="vllm serve"):
        Endpoint(EndpointConfig(), client=Down())


def test_cli_bench_parser():
    from industrial_instruction.cli import build_parser

    args = build_parser().parse_args(
        ["bench", "--suite", "ibm", "--suite", "custom", "--context", "gold", "--base-url", "http://h/v1"]
    )
    assert args.suite == ["ibm", "custom"] and args.context == ["gold"]


# ------------------------------------------------------------- load_split


class StubDatasets:
    """Hub stand-in: repo with configs -> split names."""

    DatasetDict = dict

    def __init__(self, layout):
        self.layout = layout
        self.loaded = None

    def get_dataset_config_names(self, repo):
        return list(self.layout)

    def get_dataset_split_names(self, repo, config):
        return self.layout[config]

    def load_dataset(self, repo, config, split):
        self.loaded = (config, split)
        return [{"config": config, "split": split}]

    def load_from_disk(self, path):  # pragma: no cover - not used here
        raise AssertionError


@pytest.mark.parametrize(
    "layout, expected",
    [
        ({"default": ["pana_qa_v1_test", "train"]}, ("default", "pana_qa_v1_test")),
        ({"pana_qa_v1_test": ["test"], "other": ["test"]}, ("pana_qa_v1_test", "test")),
    ],
)
def test_load_split_finds_a_split_or_a_config(monkeypatch, layout, expected):
    import sys

    stub = StubDatasets(layout)
    monkeypatch.setitem(sys.modules, "datasets", stub)
    suites_mod.load_split("Parssky/industrial-instruction-dataset", "pana_qa_v1_test")
    assert stub.loaded == expected


def test_load_split_lists_what_exists_when_nothing_matches(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "datasets", StubDatasets({"default": ["r0", "r1"]}))
    with pytest.raises(suites_mod.SuiteError, match=r"default/r0"):
        suites_mod.load_split("Parssky/industrial-instruction-dataset", "pana_qa_v1_test")


def test_paper_split_can_be_a_local_save_to_disk_dir(tmp_path):
    datasets = pytest.importorskip("datasets")
    path = tmp_path / "pana_qa_claude_v1_test"
    datasets.Dataset.from_list(
        [{"question": "Which? Options: A. x B. y", "answer": ["A"], "documents": ["d"]}]
    ).save_to_disk(str(path))
    config = Config.from_dict({"benchmark": {"paper_claude_split": str(path)}})
    (item,) = load_suite("paper-claude", config)
    assert item.gold == ["A"]
