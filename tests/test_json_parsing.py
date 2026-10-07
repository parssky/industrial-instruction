import pytest

from industrial_instruction.generate.client import parse_json_object
from industrial_instruction.schemas import QASample


def test_plain_object():
    assert parse_json_object('{"q*": "why?", "a*": "because"}')["q*"] == "why?"


def test_fenced_block():
    raw = 'Sure!\n```json\n{"q*": "why?", "a*": "because"}\n```'
    assert parse_json_object(raw)["a*"] == "because"


def test_leading_prose_and_trailing_comma():
    raw = 'Here you go: {"q*": "why?", "a*": "because",}'
    assert parse_json_object(raw)["q*"] == "why?"


def test_rejects_garbage():
    with pytest.raises(ValueError):
        parse_json_object("no json at all")
    with pytest.raises(ValueError):
        parse_json_object("")


def test_sample_accepts_both_key_styles():
    starred = QASample.from_llm_dict(
        {"q*": "q", "a*": "a"}, id="1", relation="r1", documents=["d"]
    )
    plain = QASample.from_llm_dict(
        {"question": "q", "answer": "a"}, id="2", relation="r1", documents=["d"]
    )
    assert starred.question == plain.question == "q"
    assert starred.answer == plain.answer == "a"
