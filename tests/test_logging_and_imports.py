"""Guards against the import-time breakage that shipped in the first pass.

Every module below was importable in isolation but referenced a logging
helper that did not exist, so the CLI and Pipeline both failed at import.
These tests are cheap and catch that class of bug immediately.
"""

import importlib
import logging

import pytest

from industrial_instruction.config import Config
from industrial_instruction.utils import logging as ii_logging

MODULES = [
    "industrial_instruction",
    "industrial_instruction.cli",
    "industrial_instruction.pipeline",
    "industrial_instruction.config",
    "industrial_instruction.schemas",
    "industrial_instruction.utils.io",
    "industrial_instruction.utils.logging",
    "industrial_instruction.chunk.chunker",
    "industrial_instruction.extract.registry",
    "industrial_instruction.extract.runner",
    "industrial_instruction.embed.registry",
    "industrial_instruction.embed.hash_embedder",
    "industrial_instruction.generate.client",
    "industrial_instruction.generate.engine",
    "industrial_instruction.generate.prompt_loader",
    "industrial_instruction.generate.seeds",
    "industrial_instruction.filter.rules",
    "industrial_instruction.filter.runner",
    "industrial_instruction.filter.judge",
    "industrial_instruction.assemble.assemble",
    "industrial_instruction.store.faiss_store",
    "industrial_instruction.store.runner",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_logging_exports_configure_logging():
    assert callable(ii_logging.configure_logging)
    assert not hasattr(ii_logging, "setup_logging")


def test_configure_logging_is_idempotent():
    ii_logging.configure_logging()
    ii_logging.configure_logging("DEBUG")
    root = logging.getLogger("industrial_instruction")
    assert len(root.handlers) == 1
    assert root.level == logging.DEBUG
    ii_logging.configure_logging("INFO")


def test_get_logger_is_namespaced():
    assert ii_logging.get_logger("chunk").name == "industrial_instruction.chunk"
    assert (
        ii_logging.get_logger("industrial_instruction.x").name
        == "industrial_instruction.x"
    )


def test_pipeline_constructs_with_logging_both_ways():
    """The keyword argument shadows the function name; both paths must work."""
    from industrial_instruction.pipeline import Pipeline

    assert Pipeline(Config(), configure_logging=True).config is not None
    assert Pipeline(Config(), configure_logging=False).config is not None


def test_cli_parser_builds_and_info_runs(capsys):
    from industrial_instruction.cli import main

    assert main(["info"]) == 0
    assert "fingerprint" in capsys.readouterr().out
