"""Benchmark metrics.

``set_metrics`` are the paper's Panasonic metrics (exact set match,
Jaccard, F1 over label sets, unparseable replies scoring 0).
``ibm_metrics`` are FailureSensorIQ's: accuracy on the original and the
perturbed version of each question, and consistency (both right).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def set_scores(pred: Optional[Iterable[str]], gold: Iterable[str]) -> Tuple[float, float, float]:
    """``(exact_match, jaccard, f1)`` for one item; no prediction scores 0."""
    if pred is None:
        return 0.0, 0.0, 0.0
    p, g = set(pred), set(gold)
    inter = len(p & g)
    union = len(p | g)
    jaccard = inter / union if union else 0.0
    precision = inter / len(p) if p else 0.0
    recall = inter / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(p == g), jaccard, f1


def set_metrics(rows: Sequence[dict]) -> Dict[str, float]:
    n = len(rows)
    if not n:
        return {"n": 0}
    scores = [set_scores(r["pred"], r["gold"]) for r in rows]
    return {
        "n": n,
        "set_match": sum(s[0] for s in scores) / n,
        "jaccard": sum(s[1] for s in scores) / n,
        "f1": sum(s[2] for s in scores) / n,
        "parse_failures": sum(1 for r in rows if r["pred"] is None) / n,
    }


def ibm_metrics(rows: Sequence[dict]) -> Dict[str, object]:
    """Accuracy per variant and consistency over (subject, id) pairs."""
    by_key: Dict[tuple, Dict[str, bool]] = {}
    for r in rows:
        correct = r["pred"] is not None and set(r["pred"]) == set(r["gold"])
        by_key.setdefault(r["pair"], {})[r["group"]] = correct
    pairs = [v for v in by_key.values() if "org" in v and "pert" in v]
    n = len(pairs)
    out: Dict[str, object] = {"n_pairs": n}
    if n:
        orig = [v["org"] for v in pairs]
        pert = [v["pert"] for v in pairs]
        out.update(
            acc_original=sum(orig) / n,
            acc_perturb=sum(pert) / n,
            consistency=sum(a and b for a, b in zip(orig, pert)) / n,
            # The original script's "f1": agreement between the two variants'
            # correctness (original as y_true). Kept for comparison only.
            agreement_f1_macro=_macro_f1(orig, pert),
            agreement_f1_micro=sum(a == b for a, b in zip(orig, pert)) / n,
        )
    out["set"] = set_metrics(rows)
    return out


def _macro_f1(y_true: List[bool], y_pred: List[bool]) -> float:
    scores = []
    for cls in (True, False):
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / len(scores)
