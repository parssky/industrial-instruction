"""Batch driver for the extract stage."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm

from industrial_instruction.config import Config
from industrial_instruction.extract.base import ExtractionError
from industrial_instruction.extract.registry import get_extractor
from industrial_instruction.schemas import Document, StageReport
from industrial_instruction.utils.io import ensure_dir, write_json, write_jsonl
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {
    "pymupdf": {".pdf"},
    "pdfplumber": {".pdf"},
    "markdown": {".md", ".markdown", ".txt"},
}


def discover_inputs(
    root: Path, backend: str, recursive: bool = True
) -> List[Path]:
    """Find candidate input files for a backend."""
    suffixes = SUPPORTED_SUFFIXES.get(backend.lower(), {".pdf"})
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith(".")
    )


def extract_documents(
    config: Config,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> StageReport:
    """Convert every input file into markdown and write documents.jsonl.

    Also writes one ``.md`` per document so users can eyeball extraction
    quality before spending money on generation.
    """
    t0 = time.time()
    src = Path(input_dir).resolve() if input_dir else config.paths.resolve("pdfs")
    out = Path(output_dir).resolve() if output_dir else config.paths.resolve("documents")
    ensure_dir(out)

    files = discover_inputs(src, config.extract.backend, config.extract.recursive)
    if not files:
        logger.warning("No input files found under %s", src)

    extractor = get_extractor(config.extract)
    docs: List[Document] = []
    failures: List[dict] = []

    for path in tqdm(files, desc=f"extract[{extractor.name}]", unit="doc"):
        try:
            doc = extractor.extract(path)
        except ExtractionError as exc:
            logger.warning("skipped %s: %s", path.name, exc)
            failures.append({"path": str(path), "error": str(exc)})
            continue
        except Exception as exc:  # pragma: no cover - backend specific
            logger.exception("failed %s", path.name)
            failures.append({"path": str(path), "error": repr(exc)})
            continue
        docs.append(doc)
        (out / f"{doc.id}.md").write_text(doc.markdown, encoding="utf-8")

    write_jsonl(out / "documents.jsonl", docs)
    if failures:
        write_json(out / "extract_failures.json", failures)

    report = StageReport(
        stage="extract",
        inputs=len(files),
        outputs=len(docs),
        errors=len(failures),
        seconds=round(time.time() - t0, 2),
        details={
            "backend": extractor.name,
            "input_dir": str(src),
            "output_dir": str(out),
            "tables": sum(d.n_tables for d in docs),
            "images_dropped": sum(d.n_images_dropped for d in docs),
            "pages": sum(d.n_pages for d in docs),
        },
    )
    logger.info(
        "extract: %d/%d docs, %d tables, %d images dropped (%.1fs)",
        report.outputs,
        report.inputs,
        report.details["tables"],
        report.details["images_dropped"],
        report.seconds,
    )
    return report


def load_documents(path: str | Path) -> Iterable[Document]:
    """Read a documents.jsonl written by :func:`extract_documents`."""
    from industrial_instruction.utils.io import iter_jsonl

    p = Path(path)
    if p.is_dir():
        p = p / "documents.jsonl"
    for row in iter_jsonl(p):
        yield Document.model_validate(row)
