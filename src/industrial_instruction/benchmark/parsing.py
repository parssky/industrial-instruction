"""Turn a model reply into a set of option labels.

One parser for every suite. The original evaluation scripts each had their
own, and their differences changed scores: a JSON-only parser scored the
Claude-trained model's ``['E']`` replies as failures, and the IBM scorer
compared ``"P)"`` option ids to ``"P"`` answers, so no perturbed item could
ever be correct.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, List, Optional, Sequence

_BLOCK = re.compile(r"\{.*?\}|\[.*?\]", re.S)
_ANSWER_KEY = re.compile(r'["\']answer["\']\s*:\s*(\[[^\]}]*\]?|"[^"]*"|\'[^\']*\')', re.S)
# "Answer: B", "the answer is (B) and D"
_ANSWER_PROSE = re.compile(
    r"(?i:\banswers?)\s*(?i:is|are|:)\s*((?:\(?[A-Z]\)?(?:\s*(?:,|and|&)\s*)?)+)(?![A-Za-z])"
)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.IGNORECASE)


def normalize_label(label: Any) -> str:
    """``"P)"``, ``"(b)"``, ``" B. "`` -> ``"P"``, ``"B"``, ``"B"``."""
    text = str(label).strip().strip("()[]{}.:,;'\" ").strip()
    if len(text) == 1:
        return text.upper()
    m = re.match(r"^\(?([A-Za-z])[).:]?$", text)
    return m.group(1).upper() if m else text


def _labels(value: Any) -> Optional[List[str]]:
    if isinstance(value, dict):
        for key in ("answer", "answers", "choice", "choices"):
            if key in value:
                return _labels(value[key])
        return None
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (list, tuple)):
        return [normalize_label(v) for v in value if str(v).strip()]
    return None


def _literal(text: str) -> Any:
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:  # noqa: BLE001
            continue
    return None


def parse_answer(reply: Optional[str], valid: Optional[Sequence[str]] = None) -> Optional[List[str]]:
    """Labels chosen in ``reply``, or ``None`` if no answer can be found.

    Tries, in order: a JSON/Python object or list (``{"answer": ["B"]}``,
    ``['B']``), the ``"answer": [...]`` fragment of a malformed object
    (``{"answer": ["B"}``), then - only if ``valid`` labels are given - a
    reply that is nothing but labels (``B``, ``B, D``). Free text is never
    scanned for stray capital letters.
    """
    if not reply or not reply.strip():
        return None
    text = _THINK.sub("", reply).strip()

    for block in _BLOCK.findall(text):
        labels = _labels(_literal(block))
        if labels is not None:
            return labels

    m = _ANSWER_KEY.search(text)
    if m:
        frag = m.group(1)
        if frag.startswith("[") and not frag.endswith("]"):
            frag += "]"
        labels = _labels(_literal(frag))
        if labels is not None:
            return labels

    m = _ANSWER_PROSE.search(text)
    if m:
        letters = re.findall(r"(?<![A-Za-z])[A-Z](?![A-Za-z])", m.group(1))
        if letters:
            return [normalize_label(x) for x in letters]

    if valid:
        allowed = {normalize_label(v) for v in valid}
        tokens = [t for t in re.split(r"[\s,;/&]+|\band\b", text) if t]
        labels = [normalize_label(t) for t in tokens]
        if labels and all(lab in allowed for lab in labels):
            return labels
    return None
