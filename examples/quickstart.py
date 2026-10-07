"""Minimal end-to-end run using the Python API.

    python examples/quickstart.py path/to/pdf_dir

Uses a small seed limit so the first run is cheap. Set OPENAI_API_KEY (or
point generate.base_url at a local vLLM server) before running.
"""

from __future__ import annotations

import sys

from industrial_instruction import Config, Pipeline


def main(pdf_dir: str) -> int:
    config = Config.from_dict(
        {
            "paths": {"root": ".", "pdfs": pdf_dir},
            "seeds": {"source": "self", "limit": 10},
            "generate": {"model": "gpt-4.1-mini", "max_workers": 4},
            "assemble": {"splits": {"train": 0.8, "test": 0.2}},
        }
    )
    pipeline = Pipeline(config)

    for report in pipeline.run_all():
        print(
            f"{report.stage:>9}: {report.outputs} out / {report.inputs} in "
            f"({report.rejected} rejected, {report.seconds}s)"
        )
    print("dataset written to", config.paths.resolve("dataset"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
