"""``ii`` command line interface.

    ii init                 write a starter config
    ii extract              PDFs -> markdown documents + chunks
    ii index                chunks -> FAISS index
    ii generate             seeds + retrieval + LLM -> raw samples
    ii filter               rules (and optional judge) -> kept samples
    ii assemble             merge + split -> dataset
    ii run                  all of the above
    ii info                 show resolved config and artifact status

Every command takes ``-c/--config`` and repeatable ``--set key=value``
overrides, so nothing needs to be edited in source to change a model,
endpoint or path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from industrial_instruction.config import Config
from industrial_instruction.utils.logging import configure_logging

DEFAULT_CONFIG_NAME = "industrial_instruction.yaml"
_PACKAGED_DEFAULT = Path(__file__).parent / "configs" / "default.yaml"


def _load_config(args) -> Config:
    path = Path(args.config) if args.config else Path(DEFAULT_CONFIG_NAME)
    if path.exists():
        config = Config.from_yaml(path)
    elif args.config:
        raise SystemExit(
            f"Config file not found: {path}\nCreate one with: ii init"
        )
    else:
        config = Config()
    for override in args.set or []:
        if "=" not in override:
            raise SystemExit(f"--set expects key=value, got {override!r}")
        key, value = override.split("=", 1)
        config = config.with_override(key.strip(), value.strip())
    return config


def _pipeline(args):
    from industrial_instruction.pipeline import Pipeline

    # Logging is already configured in main(); don't reset the level here.
    return Pipeline(_load_config(args), configure_logging=False)


def _print_report(report) -> None:
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


# --------------------------------------------------------------- commands


def cmd_init(args) -> int:
    target = Path(args.output or DEFAULT_CONFIG_NAME)
    if target.exists() and not args.force:
        print(f"{target} already exists; pass --force to overwrite.")
        return 1
    if _PACKAGED_DEFAULT.exists():
        shutil.copyfile(_PACKAGED_DEFAULT, target)
    else:  # pragma: no cover - defensive
        Config().to_yaml(target)
    root = Config().paths
    for sub in (root.pdfs, root.documents):
        Path(sub).mkdir(parents=True, exist_ok=True)
    print(f"Wrote {target}")
    print(f"Next: put your PDFs in {root.pdfs}/ then run: ii run")
    return 0


def cmd_extract(args) -> int:
    _print_report(_pipeline(args).run_extract())
    return 0


def cmd_index(args) -> int:
    _print_report(_pipeline(args).run_index())
    return 0


def cmd_generate(args) -> int:
    _print_report(_pipeline(args).run_generate())
    return 0


def cmd_filter(args) -> int:
    _print_report(_pipeline(args).run_filter())
    return 0


def cmd_assemble(args) -> int:
    _print_report(_pipeline(args).run_assemble())
    return 0


def cmd_run(args) -> int:
    pipeline = _pipeline(args)
    stages = args.stages.split(",") if args.stages else None
    pipeline.run_all(stages=stages)
    print(json.dumps(pipeline.summary(), indent=2, ensure_ascii=False))
    return 0


def cmd_info(args) -> int:
    config = _load_config(args)
    paths = config.paths
    status = {}
    for key in (
        "pdfs",
        "documents",
        "chunks",
        "index",
        "generated",
        "filtered",
        "dataset",
    ):
        resolved = paths.resolve(key)
        status[key] = {"path": str(resolved), "exists": resolved.exists()}
    print(
        json.dumps(
            {
                "fingerprint": config.fingerprint(),
                "extract_backend": config.extract.backend,
                "embed": {
                    "backend": config.embed.backend,
                    "model": config.embed.model,
                },
                "generate": {
                    "model": config.generate.model,
                    "base_url": config.generate.base_url or "default",
                    "relations": [
                        r.id for r in config.generate.enabled_relations()
                    ],
                },
                "seeds": {
                    "source": config.seeds.source,
                    "dataset": config.seeds.dataset,
                    "path": config.seeds.path,
                },
                "artifacts": status,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


# ----------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ii",
        description=(
            "Build retrieval-grounded instruction datasets from your own PDFs."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("-c", "--config", help=f"defaults to ./{DEFAULT_CONFIG_NAME}")
        sub.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="override a config value, e.g. --set generate.model=gpt-4.1",
        )
        return sub

    init = subparsers.add_parser("init", help="write a starter config file")
    init.add_argument("-o", "--output")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    for name, help_text, func in (
        ("extract", "extract text and tables from PDFs", cmd_extract),
        ("index", "embed chunks into a FAISS index", cmd_index),
        ("generate", "generate QA samples", cmd_generate),
        ("filter", "filter generated samples", cmd_filter),
        ("assemble", "merge and split into a dataset", cmd_assemble),
        ("info", "show resolved config and artifact status", cmd_info),
    ):
        sub = add_common(subparsers.add_parser(name, help=help_text))
        sub.set_defaults(func=func)

    run = add_common(subparsers.add_parser("run", help="run the full pipeline"))
    run.add_argument(
        "--stages",
        help="comma-separated subset, e.g. --stages generate,filter,assemble",
    )
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level="DEBUG" if args.verbose else "INFO")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
