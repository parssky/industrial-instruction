"""Load a user function from an import path or a file.

Shared by every pluggable part of the package (OCR backends, retrievers):

    my_pkg.module:function        an importable module
    plugins/my_model.py:function  a file, relative to ``root`` (paths.root)
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Optional, Type, Union


def load_callable(
    spec: str,
    root: Optional[Union[str, Path]] = None,
    error: Type[Exception] = ValueError,
    what: str = "plugin",
) -> Callable:
    module_part, _, attr = spec.rpartition(":")
    if not module_part or not attr:
        raise error(f"{what} {spec!r} must look like 'module:function' or 'file.py:function'")

    if module_part.endswith(".py"):
        path = Path(module_part)
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        if not path.exists():
            raise error(f"{what} file not found: {path}")
        mod_name = f"_ii_plugin_{path.stem}_{abs(hash(str(path.resolve())))}"
        module_spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[mod_name] = module
        module_spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_part)
        except ImportError as exc:
            raise error(f"cannot import {what} module {module_part!r}: {exc}") from exc

    fn = getattr(module, attr, None)
    if not callable(fn):
        raise error(f"{spec!r}: {attr!r} is not a callable in {module_part}")
    return fn
