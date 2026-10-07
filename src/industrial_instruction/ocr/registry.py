"""Find the OCR function named by ``extract.ocr.backend``.

Three ways to attach a model, from most to least code:

1. Python: ``register_ocr("my-vllm", fn)`` (or ``@ocr_backend("my-vllm")``),
   then ``extract.ocr.backend: my-vllm``.
2. Import path, no registration: ``extract.ocr.backend: my_pkg.ocr:fn``.
3. A file next to your config: ``extract.ocr.backend: ocr/my_model.py:fn``
   (relative paths resolve against ``paths.root``).

A registered object may also be a *factory*: a class or function decorated
with ``@ocr_backend(..., factory=True)`` that takes the :class:`OCRConfig`
and returns the page function. Use that when the backend needs setup
(loading weights, opening a client) once per run instead of once per page.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from industrial_instruction.config import OCRConfig
from industrial_instruction.ocr.base import OCRError, OCRFunction

_Factory = Callable[[OCRConfig], OCRFunction]
_REGISTRY: Dict[str, Tuple[Callable, bool]] = {}
#: Set by ``@ocr_backend(factory=True)`` so import-path backends keep the flag.
_FACTORY_ATTR = "_ii_ocr_factory"


def register_ocr(
    name: str, fn: Union[OCRFunction, _Factory], factory: bool = False
) -> None:
    """Make ``fn`` available as ``extract.ocr.backend: <name>``.

    ``fn(page) -> markdown`` by default; with ``factory=True``,
    ``fn(ocr_config) -> page_function`` and is called once per run.
    """
    if not callable(fn):
        raise TypeError(f"OCR backend {name!r} must be callable, got {type(fn).__name__}")
    _REGISTRY[name.lower()] = (fn, factory)


def ocr_backend(name: str, factory: bool = False):
    """Decorator form of :func:`register_ocr`."""

    def wrap(fn):
        register_ocr(name, fn, factory=factory)
        try:
            setattr(fn, _FACTORY_ATTR, factory)
        except AttributeError:  # pragma: no cover - builtins, bound methods
            pass
        return fn

    return wrap


def available_ocr_backends() -> List[str]:
    _load_builtins()
    return sorted(_REGISTRY)


def get_ocr(config: OCRConfig, root: Optional[Union[str, Path]] = None) -> OCRFunction:
    """Resolve ``config.backend`` to a ready-to-call page function."""
    _load_builtins()
    spec = (config.backend or "").strip()
    if not spec:
        raise OCRError("extract.ocr.backend is empty")

    if spec.lower() in _REGISTRY:
        fn, is_factory = _REGISTRY[spec.lower()]
    elif ":" in spec:
        fn = _import(spec, root)
        is_factory = getattr(fn, _FACTORY_ATTR, False)
    else:
        raise OCRError(
            f"Unknown OCR backend {spec!r}. Registered: {available_ocr_backends()}. "
            "Use register_ocr(), or an import path like 'my_pkg.ocr:run' or "
            "'ocr/my_model.py:run'."
        )
    return fn(config) if is_factory else fn


def _import(spec: str, root: Optional[Union[str, Path]]) -> Callable:
    module_part, _, attr = spec.rpartition(":")
    if not module_part or not attr:
        raise OCRError(f"OCR backend {spec!r} must look like 'module:function'")

    if module_part.endswith(".py"):
        path = Path(module_part)
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        if not path.exists():
            raise OCRError(f"OCR backend file not found: {path}")
        mod_name = f"_ii_ocr_{path.stem}"
        module_spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[mod_name] = module
        module_spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_part)
        except ImportError as exc:
            raise OCRError(f"cannot import OCR backend module {module_part!r}: {exc}") from exc

    fn = getattr(module, attr, None)
    if not callable(fn):
        raise OCRError(f"{spec!r}: {attr!r} is not a callable in {module_part}")
    return fn


_BUILTINS_LOADED = False


def _load_builtins() -> None:
    """Register the packaged backends without importing their heavy deps."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from industrial_instruction.ocr import openai_vision, tesseract  # noqa: F401
