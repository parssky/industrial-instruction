"""Regression tests for run_manifest.json handling.

The bug these cover: ``ii extract`` succeeded on a clean checkout and then
failed on the *second* invocation, because write_manifest only read the
existing manifest once the file was there, and passed a ``default`` keyword
that read_json did not accept.
"""

import json

import pytest

from industrial_instruction.config import Config
from industrial_instruction.pipeline import MAX_RUN_HISTORY, Pipeline
from industrial_instruction.schemas import StageReport
from industrial_instruction.utils.io import read_json, write_json


def _config(tmp_path) -> Config:
    return Config.from_dict({"paths": {"root": str(tmp_path)}})


# ---------------------------------------------------------------- read_json


def test_read_json_missing_file_returns_default(tmp_path):
    assert read_json(tmp_path / "nope.json", default={}) == {}
    assert read_json(tmp_path / "nope.json", default=None) is None


def test_read_json_missing_file_raises_without_default(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "nope.json")


def test_read_json_corrupt_file_returns_default(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert read_json(bad, default={"ok": True}) == {"ok": True}
    with pytest.raises(json.JSONDecodeError):
        read_json(bad)


def test_read_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "x.json"
    write_json(path, {"a": [1, 2]})
    assert read_json(path) == {"a": [1, 2]}


# ----------------------------------------------------------- write_manifest


def test_write_manifest_is_repeatable(tmp_path):
    """The original failure: second write blew up once the file existed."""
    pipeline = Pipeline(_config(tmp_path), configure_logging=False)
    path = pipeline.write_manifest()
    assert path.exists()
    pipeline.write_manifest()  # must not raise
    assert read_json(path)["runs"]


def test_manifest_records_stage_reports(tmp_path):
    pipeline = Pipeline(_config(tmp_path), configure_logging=False)
    pipeline._record(StageReport(stage="extract", inputs=1, outputs=1))
    data = read_json(pipeline.config.paths.resolve("manifest"))
    assert data["stages"]["extract"]["outputs"] == 1
    assert data["config_fingerprint"]["hash"]


def test_one_history_entry_per_run_not_per_stage(tmp_path):
    config = _config(tmp_path)
    pipeline = Pipeline(config, configure_logging=False)
    pipeline._record(StageReport(stage="extract"))
    pipeline._record(StageReport(stage="index"))
    runs = read_json(config.paths.resolve("manifest"))["runs"]
    assert len(runs) == 1
    assert set(runs[0]["stages"]) == {"extract", "index"}


def test_separate_runs_accumulate_history(tmp_path):
    config = _config(tmp_path)
    for _ in range(3):
        Pipeline(config, configure_logging=False)._record(StageReport(stage="extract"))
    runs = read_json(config.paths.resolve("manifest"))["runs"]
    assert len(runs) == 3
    assert len({r["run_id"] for r in runs}) == 3


def test_history_is_capped(tmp_path):
    config = _config(tmp_path)
    for _ in range(MAX_RUN_HISTORY + 5):
        Pipeline(config, configure_logging=False).write_manifest()
    runs = read_json(config.paths.resolve("manifest"))["runs"]
    assert len(runs) == MAX_RUN_HISTORY


def test_corrupt_manifest_does_not_block_next_run(tmp_path):
    config = _config(tmp_path)
    manifest = config.paths.resolve("manifest")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"runs": [truncated', encoding="utf-8")
    Pipeline(config, configure_logging=False).write_manifest()
    assert len(read_json(manifest)["runs"]) == 1
