from industrial_instruction.config import Config


def test_defaults_have_five_relations():
    config = Config()
    assert [r.id for r in config.generate.enabled_relations()] == [
        "r0",
        "r1",
        "r2",
        "r3",
        "r4",
    ]


def test_with_override_parses_types():
    config = Config()
    assert config.with_override("generate.model", "gpt-4.1").generate.model == "gpt-4.1"
    assert config.with_override("generate.max_workers", "16").generate.max_workers == 16
    assert config.with_override("filter.judge_enabled", "true").filter.judge_enabled
    assert config.with_override("generate.temperature", "0.7").generate.temperature == 0.7


def test_with_override_keeps_original_immutable():
    config = Config()
    config.with_override("embed.model", "other/model")
    assert config.embed.model == "google/embeddinggemma-300m"


def test_with_override_rejects_unknown_key():
    import pytest

    with pytest.raises(ValueError):
        Config().with_override("generate.nope", "1")
    with pytest.raises(ValueError):
        Config().with_override("nosection.key", "1")


def test_fingerprint_covers_all_stages_and_is_stable():
    a = Config().fingerprint()
    b = Config().fingerprint()
    assert a["hash"] == b["hash"]
    assert "generate" in a and "extract" in a
    changed = Config().with_override("generate.model", "gpt-4.1").fingerprint()
    assert changed["hash"] != a["hash"]


def test_roundtrip_yaml(tmp_path):
    path = Config().to_yaml(tmp_path / "c.yaml")
    assert Config.from_yaml(path).generate.model == Config().generate.model


def test_resolved_formats_adds_chat():
    config = Config()
    assert config.assemble.resolved_formats() == ["jsonl", "chat"]
    config = config.with_override("assemble.chat_format", "false")
    assert config.assemble.resolved_formats() == ["jsonl"]
