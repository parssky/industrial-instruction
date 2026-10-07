"""Multiple-choice handling: detect, parse and normalize options and answers.

The paper's seeds (FailureSensorIQ) are multiple-choice prompts, so most
generated samples are too. Model replies vary a lot in shape, and the
original filter notebooks cleaned them up by hand:

- ``a*`` as ``["B"]``, ``"B"``, ``"B, D"`` or ``{"answer": ["B"]}``
- options as ``["A. x", ...]``, ``["A x", ...]``, ``["x", ...]`` or a dict
- options written inside ``q*`` instead of (or as well as) ``options*``
- labels ``P``-``T`` copied from perturbed seeds instead of ``A``-``E``

Everything here turns those into one canonical form: a question stem,
options ``["A. text", ...]`` and an answer list of labels ``["B", "D"]``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# "A. text", "A) text", "(A) text", "A: text", "A - text", "A text"
_LABELED = re.compile(r"^\s*\(?([A-Z])\s*(?:[.):\-]|\s)\s*(.+?)\s*$")
_OPTIONS_MARKER = re.compile(r"(?im)^\s*options?\s*:\s*$|\boptions?\s*:")
_INLINE_SPLIT = re.compile(r"(?:(?<=\s)|^)\(?([A-Z])[.)]\s+")
_FORMAT_HINTS = (
    "your output must",
    "your output in a single line",
    '{"answer"',
    '"option_a"',
    "<the list of selected option",
)


def parse_labeled(option: str) -> Tuple[Optional[str], str]:
    """``"B) partial discharge"`` -> ``("B", "partial discharge")``."""
    m = _LABELED.match(option or "")
    if m and len(m.group(2)) > 0:
        return m.group(1), m.group(2)
    return None, (option or "").strip()


def _consecutive(labels: Sequence[str]) -> bool:
    return all(ord(b) - ord(a) == 1 for a, b in zip(labels, labels[1:]))


def find_options(text: str) -> Tuple[str, Optional[List[str]]]:
    """Split ``text`` into ``(stem, options)`` if it embeds an option list.

    Recognizes the block after an ``Options:`` marker (one option per line,
    or inline ``A. x B. y``), and trailing runs of labeled lines. Returns
    ``(text, None)`` when there are fewer than two consecutive labels.
    """
    if not text:
        return text, None
    lines = text.splitlines()

    # One option per line: the longest run of consecutive labeled lines.
    best: Tuple[int, int] = (0, 0)
    i = 0
    while i < len(lines):
        label, _ = parse_labeled(lines[i])
        if label and len(lines[i].strip()) <= 300:
            j, labels = i, []
            while j < len(lines):
                lab, _ = parse_labeled(lines[j])
                if not lab or (labels and ord(lab) - ord(labels[-1]) != 1):
                    break
                labels.append(lab)
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = max(j, i + 1)
        else:
            i += 1
    if best[1] - best[0] >= 2:
        start, end = best
        stem_lines = lines[:start]
        if stem_lines and _OPTIONS_MARKER.fullmatch(stem_lines[-1].strip() or "x"):
            stem_lines = stem_lines[:-1]
        options = [lines[k].strip() for k in range(start, end)]
        stem = "\n".join(stem_lines + lines[end:]).strip()
        return stem, options

    # Inline: "... Options: A. x B. y C. z"
    marker = list(_OPTIONS_MARKER.finditer(text))
    if marker:
        head, tail = text[: marker[-1].start()], text[marker[-1].end() :]
        parts = _INLINE_SPLIT.split(tail)
        # parts = [prefix, label, text, label, text, ...]
        labels, texts = parts[1::2], [p.strip() for p in parts[2::2]]
        if len(labels) >= 2 and _consecutive(labels) and all(texts):
            return head.strip(), [f"{lab}. {t}" for lab, t in zip(labels, texts)]
    return text, None


def normalize_options(options: Any) -> Tuple[List[str], Dict[str, str]]:
    """Canonical ``["A. text", ...]`` plus a map from original to new labels."""
    if isinstance(options, dict):
        options = [f"{k}. {v}" for k, v in options.items()]
    if not isinstance(options, (list, tuple)):
        return [], {}
    parsed = [parse_labeled(str(o)) for o in options if str(o).strip()]
    labels = [lab for lab, _ in parsed]
    keep_labels = all(labels) and _consecutive(labels)  # type: ignore[arg-type]
    out: List[str] = []
    mapping: Dict[str, str] = {}
    for i, (label, text) in enumerate(parsed):
        new = chr(ord("A") + i)
        if keep_labels and label:
            mapping[label] = new
        else:
            # Unlabeled (or a broken sequence): label by position, keep text.
            text = str(options[i]).strip() if not keep_labels else text
        out.append(f"{new}. {text}")
    return out, mapping


_LABEL_LIST = re.compile(r"^\s*[A-Za-z](?:\s*(?:,|;|/|&|\band\b|\s)\s*[A-Za-z])*\s*$")


def _one_label(item: str) -> str:
    item = item.strip()
    bare = item.strip("()[]. ")
    if len(bare) == 1 and bare.isalpha():
        return bare.upper()
    label, _ = parse_labeled(item)  # "B. partial discharge" -> "B"
    return label or item


def normalize_answer(answer: Any, mapping: Optional[Dict[str, str]] = None) -> List[str]:
    """``{"answer": ["P"]}`` / ``"B, D"`` / ``["B"]`` -> ``["B", "D"]`` (remapped).

    Anything that isn't a label (``"EKMB11 series"``) is kept as-is so the
    filter can reject it as ``answer_not_in_options``.
    """
    if isinstance(answer, dict):
        answer = answer.get("answer", answer.get("a*", answer.get("answers")))
    if answer is None:
        return []
    if isinstance(answer, str):
        text = answer.strip()
        if _LABEL_LIST.match(text):
            items = re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", text.replace(" and ", " "))
        else:
            items = [text] if text else []
    elif isinstance(answer, (list, tuple)):
        items = [str(a) for a in answer]
    else:
        items = [str(answer)]
    out: List[str] = []
    for item in items:
        label = _one_label(item)
        if mapping:
            label = mapping.get(label, label)
        if label and label not in out:
            out.append(label)
    return out


def option_labels(options: Sequence[str]) -> List[str]:
    return [lab for lab, _ in (parse_labeled(o) for o in options) if lab]


def option_texts(options: Sequence[str]) -> List[str]:
    return [re.sub(r"\W+", " ", t).strip().lower() for _, t in (parse_labeled(o) for o in options)]


def has_format_instructions(text: str) -> bool:
    """True if a benchmark's answer-format scaffolding leaked into ``text``."""
    lowered = (text or "").lower()
    return any(h in lowered for h in _FORMAT_HINTS)


def seed_options(seed_text: str, meta: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
    """The seed's own options, from dataset fields or parsed from its text."""
    meta = meta or {}
    raw = meta.get("options")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return [str(o) for o in raw]
    _, found = find_options(seed_text)
    return found
