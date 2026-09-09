"""Prompt template loading and rendering.

Templates are plain text with ``{placeholder}`` tokens. Rendering uses
explicit token substitution rather than :meth:`str.format`, because the
templates contain literal JSON braces that ``format`` would choke on. That
was the source of the doubled ``{{...}}`` escaping in the original scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from industrial_instruction.utils.io import sha256_text
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)

_BUILTIN_DIR = Path(__file__).parent / "prompts"

OUTPUT_RULE_QA = (
    'Please directly generate the question-answer pair (q*, a*) following all the rules '
    'above in the format of {"q*": ..., "a*": ...}. Return valid JSON only, with no '
    'markdown fences and no commentary. Ensure the quality of the generated (q*, a*).'
)

OUTPUT_RULE_MCQ = (
    'Please directly generate the question-answer-options (q*, a*, options*) following all '
    'the rules above in the format of {"q*": ..., "a*": ..., "options*": [...]}. For a*, give '
    'the correct option label(s) as a JSON list. Return valid JSON only, with no markdown '
    'fences and no commentary. Ensure the quality of the generated (q*, a*, options*).'
)


def options_rule(n_options: int = 5) -> str:
    last = chr(ord("A") + max(n_options, 1) - 1)
    return (
        f"IMPORTANT: your generated q* MUST include exactly {n_options} options labeled "
        f"A-{last}, and a* MUST contain the correct option label(s) as a JSON list. Always "
        f"generate {n_options} options, even if the source documents contain no option-like "
        "content."
    )


class PromptLibrary:
    """Resolves template names to text, with a user override directory.

    Lookup order: ``user_dir/<name>.txt`` then the packaged template. This
    lets a team adapt wording to their domain while keeping the defaults
    that reproduce the paper.
    """

    def __init__(self, user_dir: Optional[str | Path] = None) -> None:
        self.user_dir = Path(user_dir) if user_dir else None
        self._cache: Dict[str, str] = {}

    def path_for(self, name: str) -> Path:
        filename = name if name.endswith(".txt") else f"{name}.txt"
        if self.user_dir:
            candidate = self.user_dir / filename
            if candidate.exists():
                return candidate
        builtin = _BUILTIN_DIR / filename
        if not builtin.exists():
            available = sorted(p.stem for p in _BUILTIN_DIR.glob("*.txt"))
            raise FileNotFoundError(
                f"Prompt template {name!r} not found. Built-in templates: {available}. "
                "Add your own by setting generate.prompt_dir."
            )
        return builtin

    def get(self, name: str) -> str:
        if name not in self._cache:
            self._cache[name] = self.path_for(name).read_text(encoding="utf-8")
        return self._cache[name]

    def render(self, name: str, **values: str) -> str:
        """Substitute ``{token}`` placeholders; unknown tokens become ''."""
        text = self.get(name)
        for key in _tokens(text):
            text = text.replace("{" + key + "}", str(values.get(key, "")))
        return text.strip()

    def fingerprint(self, name: str) -> str:
        """Template hash, recorded in the manifest for reproducibility."""
        return sha256_text(self.get(name))[:12]


def _tokens(text: str):
    """Find single-word ``{token}`` placeholders, ignoring JSON braces."""
    import re

    return set(re.findall(r"\{([a-z_][a-z0-9_]*)\}", text))


def format_documents(texts, numbered: bool = True) -> str:
    """Render retrieved chunks as the ``<Documents>`` block.

    The original code interpolated a raw Python list, so prompts contained
    Python repr artifacts. Numbered blocks are cleaner and let the model
    reference documents unambiguously for multi-hop relations.
    """
    items = list(texts)
    if not numbered:
        return "\n\n".join(items)
    return "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(items))
