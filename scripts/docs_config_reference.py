"""MkDocs hook: render the configuration reference from config.py.

Replaces ``<!-- config-reference -->`` in a page with one table per YAML
section (key, type, default, description). Descriptions come from the
comments next to each field in ``config.py``, so the reference is
regenerated from the source on every build and cannot drift from it.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MARKER = "<!-- config-reference -->"
SOURCE = Path(__file__).resolve().parents[1] / "src" / "industrial_instruction" / "config.py"

# YAML path -> class, in the order sections appear in the starter file.
SECTIONS: List[Tuple[str, str]] = [
    ("project", "Config"),
    ("paths", "PathsConfig"),
    ("extract", "ExtractConfig"),
    ("extract.ocr", "OCRConfig"),
    ("chunk", "ChunkConfig"),
    ("embed", "EmbedConfig"),
    ("store", "StoreConfig"),
    ("retrieval", "RetrievalConfig"),
    ("seeds", "SeedsConfig"),
    ("generate", "GenerateConfig"),
    ("filter", "FilterConfig"),
    ("assemble", "AssembleConfig"),
    ("benchmark", "BenchmarkConfig"),
    ("benchmark.endpoint", "EndpointConfig"),
    ("benchmark.custom", "CustomBenchmarkConfig"),
]
NESTED = {cls for _, cls in SECTIONS}


def _comments(source: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """``(inline, standalone)`` comments by line number."""
    inline: Dict[int, str] = {}
    standalone: Dict[int, str] = {}
    lines = source.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            text = tok.string.lstrip("#").strip()
            line = tok.start[0]
            if lines[line - 1].lstrip().startswith("#"):
                standalone[line] = text
            else:
                inline[line] = text
    return inline, standalone


def _default(node: Optional[ast.expr], source: str) -> str:
    if node is None:
        return "*required*"
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Field":
        for kw in node.keywords:
            if kw.arg == "default_factory":
                value = kw.value
                if isinstance(value, ast.Lambda):
                    text = ast.get_source_segment(source, value.body) or ""
                    if "DEFAULT_RELATIONS" in text:
                        return "r0 – r4 (see Generation)"
                    return text
                name = getattr(value, "id", "")
                if name in NESTED:
                    return "see below"
                return {"dict": "{}", "list": "[]"}.get(name, f"{name}()")
            if kw.arg == "default":
                return ast.get_source_segment(source, kw.value) or ""
    text = ast.get_source_segment(source, node) or ""
    if isinstance(node, ast.Constant) and not isinstance(node.value, str):
        return {True: "true", False: "false", None: "null"}.get(node.value, text) \
            if node.value is None or isinstance(node.value, bool) else text
    if isinstance(node, (ast.Constant, ast.UnaryOp)):
        return text
    # Implicitly concatenated strings: show the evaluated value.
    try:
        return repr(ast.literal_eval(node))
    except Exception:  # noqa: BLE001
        return re.sub(r"\s+", " ", text)


def _type(node: ast.expr, source: str) -> str:
    text = ast.get_source_segment(source, node) or ""
    match = re.fullmatch(r"Optional\[(.*)\]", text)
    return f"{match.group(1)} | null" if match else text


def _escape(text: str) -> str:
    """Make text safe inside a markdown table cell."""
    text = text.replace("``", "`").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("|", "\\|").replace("\n", " ")


def _fields(cls: ast.ClassDef, source: str, comments):
    rows = []
    for stmt in cls.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        name = stmt.target.id
        if name.startswith("_"):
            continue
        inline, standalone = comments
        notes = [inline[n] for n in range(stmt.lineno, stmt.end_lineno + 1) if n in inline]
        line = stmt.lineno - 1
        above = []
        while line in standalone:
            above.insert(0, standalone[line])
            line -= 1
        description = " ".join(above + notes)
        rows.append((name, _type(stmt.annotation, source), _default(stmt.value, source), description))
    return rows


def render() -> str:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    comments = _comments(source)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    parts = []
    for path, name in SECTIONS:
        cls = classes[name]
        doc = (ast.get_docstring(cls) or "").split("\n\n")[0].replace("\n", " ")
        rows = _fields(cls, source, comments)
        if name == "Config":
            rows = [r for r in rows if r[0] == "project"]
            parts.append("### Top level\n")
        else:
            parts.append(f"### `{path}`\n\n{_escape(doc)}\n")
        parts.append("| Key | Type | Default | Description |\n|---|---|---|---|")
        for key, typ, default, desc in rows:
            types = " \\| ".join(f"`{t}`" for t in typ.split(" | "))
            parts.append(f"| `{key}` | {types} | `{_escape(default)}` | {_escape(desc)} |")
        parts.append("")
    return "\n".join(parts)


def on_page_markdown(markdown: str, page, config, files) -> str:
    if MARKER in markdown:
        markdown = markdown.replace(MARKER, render())
    return markdown
