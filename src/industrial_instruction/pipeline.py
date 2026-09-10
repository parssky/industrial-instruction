"""End-to-end pipeline orchestration.

``Pipeline`` is the programmatic entry point; the CLI is a thin wrapper over
it. Every stage returns a :class:`StageReport`, and each run appends to
``run_manifest.json`` together with the config fingerprint, so a dataset can
always be traced back to the settings that produced it.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from industrial_instruction.config import Config
from industrial_instruction.schemas import StageReport
from industrial_instruction.utils.io import ensure_dir, read_json, write_json

# Aliased on import: the ``Pipeline.__init__`` keyword argument is also named
# ``configure_logging``, and would otherwise shadow this function.
from industrial_instruction.utils.logging import (
    configure_logging as _configure_logging,
)
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)

#: Stages in dependency order. ``chunk`` sits between extract and index:
#: extract writes documents.jsonl, chunk writes chunks.jsonl, index embeds it.
STAGE_ORDER = ("extract", "chunk", "index", "generate", "filter", "assemble")

#: How many past runs to keep in run_manifest.json.
MAX_RUN_HISTORY = 20


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Pipeline:
    """Runs the dataset-construction stages against one :class:`Config`."""

    def __init__(self, config: Config, configure_logging: bool = True) -> None:
        self.config = config
        self.reports: List[StageReport] = []
        self.run_id = uuid.uuid4().hex[:12]
        self.started_at = _utcnow()
        if configure_logging:
            _configure_logging()

    # ------------------------------------------------------------ factories

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Pipeline":
        return cls(Config.from_yaml(path))

    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline":
        return cls(Config.from_dict(data))

    # --------------------------------------------------------------- stages

    def run_extract(self, **kwargs) -> StageReport:
        from industrial_instruction.extract.runner import extract_documents

        return self._record(extract_documents(self.config, **kwargs))

    def run_chunk(self, **kwargs) -> StageReport:
        from industrial_instruction.chunk.chunker import chunk_documents

        return self._record(chunk_documents(self.config, **kwargs))

    def run_index(self, **kwargs) -> StageReport:
        from industrial_instruction.store.runner import build_index

        return self._record(build_index(self.config, **kwargs))

    def run_generate(self, **kwargs) -> StageReport:
        from industrial_instruction.generate.engine import generate_samples

        return self._record(generate_samples(self.config, **kwargs))

    def run_filter(self, **kwargs) -> StageReport:
        from industrial_instruction.filter.runner import filter_samples

        return self._record(filter_samples(self.config, **kwargs))

    def run_assemble(self, **kwargs) -> StageReport:
        from industrial_instruction.assemble.assemble import assemble_dataset

        return self._record(assemble_dataset(self.config, **kwargs))

    # ------------------------------------------------------------------ all

    def run_all(
        self,
        stages: Optional[List[str]] = None,
        stop_on_error: bool = True,
    ) -> List[StageReport]:
        """Run the requested stages in dependency order."""
        requested = [s.strip() for s in (stages or [])if s.strip()]
        unknown = [s for s in requested if s not in STAGE_ORDER]
        if unknown:
            raise ValueError(
                f"Unknown stage(s): {sorted(unknown)}. Valid: {list(STAGE_ORDER)}"
            )
        # Always run in dependency order, whatever order they were listed in.
        selected = [s for s in STAGE_ORDER if not requested or s in requested]

        started = time.time()
        runners = {
            "extract": self.run_extract,
            "chunk": self.run_chunk,
            "index": self.run_index,
            "generate": self.run_generate,
            "filter": self.run_filter,
            "assemble": self.run_assemble,
        }
        out: List[StageReport] = []
        for stage in selected:
            logger.info("=== stage: %s ===", stage)
            try:
                out.append(runners[stage]())
            except Exception as exc:  # noqa: BLE001
                logger.error("stage %s failed: %s", stage, exc)
                self._record(
                    StageReport(
                        stage=stage,
                        errors=1,
                        details={"error": repr(exc)[:1000]},
                    )
                )
                if stop_on_error:
                    raise
        logger.info(
            "pipeline finished %d stage(s) in %.1fs", len(out), time.time() - started
        )
        return out

    # ------------------------------------------------------------- manifest

    def _record(self, report: StageReport) -> StageReport:
        self.reports.append(report)
        self.write_manifest()
        return report

    def summary(self) -> Dict[str, dict]:
        return {r.stage: r.model_dump(mode="json") for r in self.reports}

    def write_manifest(self) -> Path:
        """Write the manifest, upserting this run into the rolling history.

        Called after every stage, so the manifest stays truthful even if a
        later stage crashes.
        """
        path = self.config.paths.resolve("manifest")
        ensure_dir(path.parent)

        existing = read_json(path, default={})
        if not isinstance(existing, dict):
            existing = {}
        history = existing.get("runs")
        if not isinstance(history, list):
            history = []

        now = _utcnow()
        summary = self.summary()
        entry = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "updated_at": now,
            "config_fingerprint": self.config.fingerprint(),
            "stages": summary,
        }
        # Replace this run's entry rather than appending one per stage.
        history = [
            r
            for r in history
            if not (isinstance(r, dict) and r.get("run_id") == self.run_id)
        ]
        history.append(entry)
        history = history[-MAX_RUN_HISTORY:]

        write_json(
            path,
            {
                "updated_at": now,
                "run_id": self.run_id,
                "config_fingerprint": self.config.fingerprint(),
                "config": self.config.model_dump(mode="json"),
                "stages": summary,
                "runs": history,
            },
        )
        return path
