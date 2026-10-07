"""Industrial-Instruction: build instruction/benchmark datasets from technical PDFs.

Typical library usage::

    from industrial_instruction import Config, Pipeline

    cfg = Config.from_yaml("config.yaml")
    Pipeline(cfg).run_all()

Each stage is also usable on its own; see ``industrial_instruction.pipeline``.
"""

from industrial_instruction.config import Config
from industrial_instruction.schemas import (
    Chunk,
    Document,
    QASample,
    RelationSpec,
    SampleStatus,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Chunk",
    "Document",
    "QASample",
    "RelationSpec",
    "SampleStatus",
    "Pipeline",
    "__version__",
]


def __getattr__(name):
    # Lazy import so `import industrial_instruction` stays light and does not
    # require optional deps (faiss, sentence-transformers, PyMuPDF).
    if name == "Pipeline":
        from industrial_instruction.pipeline import Pipeline

        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
