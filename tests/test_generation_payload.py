"""What actually goes to the generator, and what comes back, for r0-r4.

Uses real FailureSensorIQ rows (the paper's seeds): multiple-choice prompts
that carry their own options and answer-format template. The shapes of the
fake replies below are the ones the original filter notebooks had to clean
up by hand.
"""

import json

import pytest

from industrial_instruction.assemble.assemble import to_chat_record
from industrial_instruction.config import Config, SeedsConfig
from industrial_instruction.filter.rules import RuleFilter
from industrial_instruction.generate.engine import GenerationEngine
from industrial_instruction.generate.mcq import (
    find_options,
    normalize_answer,
    normalize_options,
)
from industrial_instruction.generate.seeds import _row_to_seed
from industrial_instruction.schemas import Chunk, QASample, SampleStatus
from industrial_instruction.store.faiss_store import SearchHit

ORG = {
    "prompt": (
        "Please select the correct option(s) from the following options given the "
        "question:\nQuestion: Which sensor out of the choices can indicate the presence "
        "of rotor windings fault in asset electric motor?\nOptions:\nA resistance\n"
        "B partial discharge\nC torque\nD coast down time\nE voltage\nYour output must "
        'strictly follow this format:\n{"answer": <the list of selected options, e.g., '
        '["A", "B", "C", "D", "E"]>}\n\nYour output in a single line:'
    ),
    "question": "Which sensor out of the choices can indicate the presence of rotor "
    "windings fault in asset electric motor?",
    "options": ["resistance", "partial discharge", "torque", "coast down time", "voltage"],
}
PERT = {
    "prompt": (
        "Please select the correct option(s) from the following options given the "
        "question. To solve the problem, follow the Let's think Step by Step reasoning "
        "strategy.\nQuestion: In electric motor, when rotor windings fault occurs, which "
        "sensor from the choices is most critical in detecting the occurrence of the "
        "failure event?\nOptions:\nP) oil debris\nQ) coast down time\nR) voltage\n"
        'S) resistance\nT) power\n{"option_a": "<Reasoning for option a>", ..., '
        '"answer": <the list of selected option>}\nYour output in a single line:'
    ),
    "question": "In electric motor, when rotor windings fault occurs, which sensor "
    "from the choices is most critical?",
    "options": ["oil debris", "coast down time", "voltage", "resistance", "power"],
}
CHUNK = Chunk(
    id="c1",
    doc_id="d1",
    text="## Motor > Rotor\n\nRotor winding faults show as rising partial discharge.",
    heading_path=["Motor", "Rotor"],
)


def seed(row=ORG):
    return _row_to_seed(row, SeedsConfig(), source="FailureSensorIQ:org")


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete_json(self, prompt):
        self.prompts.append(prompt)
        return self.reply, json.dumps(self.reply)


class FakeStore:
    def __init__(self):
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        return [SearchHit(chunk=CHUNK, score=0.9, rank=0)] * k


def engine(reply=None, **generate):
    config = Config.from_dict({"generate": generate})
    return GenerationEngine(config, store=FakeStore(), client=FakeClient(reply or {}))


def relation(eng, rid):
    return next(r for r in eng.config.generate.relations if r.id == rid)


# ------------------------------------------------------------------ seeds


def test_seed_keeps_options_and_queries_with_the_bare_question():
    s = seed()
    assert s.meta["options"] == ORG["options"]
    assert s.search_text == ORG["question"]
    eng = engine({"q*": "x"})
    eng.generate_for_seed(s, eng.config.generate.relations)
    assert eng.store.queries == [ORG["question"]]  # not the boilerplate prompt


# ---------------------------------------------------------------- payload


@pytest.mark.parametrize("rid", ["r0", "r1", "r2", "r3", "r4"])
def test_mcq_seed_asks_for_options_in_every_relation(rid):
    eng = engine()
    rel = relation(eng, rid)
    prompt = eng.build_prompt(
        rel, [CHUNK.as_context()], seed().text, wants_options=eng.wants_options(rel, seed())
    )
    assert "<Simulated Instruction>\n" + ORG["prompt"] in prompt
    assert "exactly 5 options labeled A-E" in prompt
    assert '"options*"' in prompt
    assert "do not reuse the options of the <Simulated Instruction>" in prompt
    assert prompt.count("Motor > Rotor") == 1  # heading not duplicated


def test_plain_seed_gets_question_answer_format():
    eng = engine()
    plain = _row_to_seed({"prompt": "How often should pump oil be changed?"}, SeedsConfig(), "x")
    rel = relation(eng, "r3")
    assert not eng.wants_options(rel, plain)
    prompt = eng.build_prompt(rel, ["doc"], plain.text, wants_options=False)
    assert '"options*"' not in prompt and '{"q*": ..., "a*": ...}' in prompt


@pytest.mark.parametrize(
    "mode, expected", [("auto", True), ("always", True), ("never", False)]
)
def test_options_mode(mode, expected):
    eng = engine(options_mode=mode)
    assert eng.wants_options(relation(eng, "r1"), seed()) is expected


def test_legacy_require_options_means_always():
    assert Config.from_dict({"generate": {"require_options": True}}).generate.options_mode == "always"


def test_multi_doc_relations_number_their_documents():
    eng = engine({"q*": "q", "a*": ["A"], "options*": ["a", "b", "c", "d", "e"]})
    eng.generate_one(relation(eng, "r4"), seed(), [SearchHit(chunk=CHUNK, score=1.0, rank=0)] * 3)
    prompt = eng.client.prompts[0]
    assert "[1] " in prompt and "[3] " in prompt


# ---------------------------------------------------------------- replies

GOOD_OPTIONS = [
    "A. partial discharge",
    "B. winding insulation resistance",
    "C. stator current harmonics",
    "D. rotor bar temperature",
    "E. acoustic emission",
]


@pytest.mark.parametrize(
    "reply, answer",
    [
        ({"q*": "Which signal reveals a rotor winding fault?", "a*": ["A"], "options*": GOOD_OPTIONS}, ["A"]),
        ({"q*": "Which signal reveals a rotor winding fault?", "a*": {"answer": ["A", "C"]}, "options*": GOOD_OPTIONS}, ["A", "C"]),
        ({"q*": "Which signal reveals a rotor winding fault?", "a*": "B, D", "options*": GOOD_OPTIONS}, ["B", "D"]),
        # perturbed seeds: P-T labels are mapped to A-E, answer included
        (
            {
                "q*": "Which signal reveals a rotor winding fault?",
                "a*": ["Q"],
                "options*": [o.replace(o[0], chr(ord("P") + i), 1) for i, o in enumerate(GOOD_OPTIONS)],
            },
            ["B"],
        ),
        # options only inside q*: moved out into options
        (
            {"q*": "Which signal reveals a rotor winding fault?\nOptions:\n" + "\n".join(GOOD_OPTIONS), "a*": ["E"]},
            ["E"],
        ),
    ],
)
def test_reply_shapes_normalize_to_one_form(reply, answer):
    eng = engine(reply)
    sample = eng.generate_one(relation(eng, "r1"), seed(), [SearchHit(chunk=CHUNK, score=1.0, rank=0)])
    assert sample.status == SampleStatus.OK
    assert sample.question == "Which signal reveals a rotor winding fault?"
    assert sample.options == GOOD_OPTIONS
    assert sample.answer == answer
    assert sample.meta["mcq"] is True
    assert RuleFilter(eng.config.filter).check(sample).passed


def test_unlabeled_options_get_labels():
    options, mapping = normalize_options(["resistance", "torque", "voltage"])
    assert options == ["A. resistance", "B. torque", "C. voltage"] and mapping == {}


def test_find_options_inline_and_absent():
    stem, opts = find_options("Which? Options: A. x B. y C. z")
    assert stem == "Which?" and opts == ["A. x", "B. y", "C. z"]
    assert find_options("Tighten to 25 Nm.\nA torque wrench is required.")[1] is None


def test_non_label_answer_is_kept_for_the_filter():
    assert normalize_answer("EKMB11 series") == ["EKMB11 series"]


# ----------------------------------------------------------------- filter


def mcq_sample(**kw):
    base = dict(
        id="s",
        relation="r1",
        question="Which signal best indicates a rotor winding fault in an induction motor?",
        answer=["A"],
        options=GOOD_OPTIONS,
        documents=["d"],
        meta={"mcq": True, "seed_options": ORG["options"]},
    )
    base.update(kw)
    return QASample(**base)


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"answer": ["EKMB11 series"]}, "answer_not_in_options"),
        ({"answer": ["F"]}, "answer_not_in_options"),
        ({"answer": []}, "answer_not_in_options"),
        ({"options": None}, "missing_options"),
        (
            {"options": ["A. resistance", "B. partial discharge", "C. torque", "D. coast down time", "E. power"]},
            "copied_seed_options",
        ),
        (
            {"question": 'Which signal indicates a rotor fault? Your output must strictly follow this format: {"answer": [...]}'},
            "format_instructions_in_question",
        ),
    ],
)
def test_mcq_rules(change, reason):
    result = RuleFilter(Config().filter).check(mcq_sample(**change))
    assert not result.passed and reason in result.reasons


def test_some_overlap_with_seed_options_is_fine():
    options = GOOD_OPTIONS[:3] + ["D. resistance", "E. voltage"]  # 2/5 shared
    assert RuleFilter(Config().filter).check(mcq_sample(options=options)).passed


def test_chat_record_renders_options_once_and_answer_as_json():
    record = to_chat_record(mcq_sample(answer=["A", "C"]), include_documents=True)
    user, assistant = (m["content"] for m in record["messages"])
    assert user.count("A. partial discharge") == 1
    assert assistant == '["A", "C"]'
