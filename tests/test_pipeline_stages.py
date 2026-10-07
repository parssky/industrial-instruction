"""Stage-wiring tests.

The bug these cover: chunk_documents() was implemented but unreachable, so
'ii run' always died at the index stage with an empty chunks.jsonl. A stage
that exists but is never called is invisible to unit tests of that stage,
so these assert the wiring itself.
"""

import pytest

from industrial_instruction.cli import build_parser
from industrial_instruction.config import Config
from industrial_instruction.pipeline import STAGE_ORDER, Pipeline

CLI_STAGE_COMMANDS = [s for s in STAGE_ORDER]


def test_chunk_is_between_extract_and_index():
    assert STAGE_ORDER.index("extract") < STAGE_ORDER.index("chunk")
    assert STAGE_ORDER.index("chunk") < STAGE_ORDER.index("index")


@pytest.mark.parametrize("stage", CLI_STAGE_COMMANDS)
def test_every_stage_has_a_runner(stage):
    assert hasattr(Pipeline(Config(), configure_logging=False), f"run_{stage}")


@pytest.mark.parametrize("stage", CLI_STAGE_COMMANDS)
def test_every_stage_has_a_cli_command(stage):
    args = build_parser().parse_args([stage])
    assert callable(args.func)


def test_run_all_rejects_unknown_stage(tmp_path):
    pipeline = Pipeline(
        Config.from_dict({"paths": {"root": str(tmp_path)}}), configure_logging=False
    )
    with pytest.raises(ValueError, match="Unknown stage"):
        pipeline.run_all(stages=["extract", "nope"])


def test_run_all_normalizes_stage_order(tmp_path, monkeypatch):
    """Stages run in dependency order even if requested out of order."""
    pipeline = Pipeline(
        Config.from_dict({"paths": {"root": str(tmp_path)}}), configure_logging=False
    )
    called = []
    for stage in STAGE_ORDER:
        monkeypatch.setattr(
            pipeline,
            f"run_{stage}",
            lambda s=stage: called.append(s) or __import__(
                "industrial_instruction.schemas", fromlist=["StageReport"]
            ).StageReport(stage=s),
        )
    pipeline.run_all(stages=["index", "chunk", "extract"])
    assert called == ["extract", "chunk", "index"]
