"""Lightweight transition metrics for pre-loaded log dictionaries.

Usage:
    from simple_transition import evaluate_transition
    metrics = evaluate_transition(original_logs, perturbed_logs)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

try:
    from sklearn.metrics import f1_score
except ImportError:  
    f1_score = None


def _is_correct(entry: dict) -> int:
    """Return 1 if model_output matches true_answer exactly, else 0."""
    
    result = [i in entry["model_output"] for i in entry['option_ids']]
    return 1 if  result == entry["correct"] else 0


def _filter_subjects(
    records: Iterable[dict], subjects: set[str] | None
) -> Iterable[dict]:
    if not subjects:
        return records
    return (record for record in records if record["subject"] in subjects)


def evaluate_transition(
    original_records: Sequence[dict],
    perturbed_records: Sequence[dict],
    subjects: Sequence[str] | None = None,
) -> dict:
    """Compute the same metrics as transition_analysis without file IO.

    Args:
        original_records: iterable of dicts each containing at least
            subject, id, true_answer, model_output.
        perturbed_records: iterable with the same schema.
        subjects: optional iterable of subjects to include; defaults to all.

    Returns:
        dict with acc_original, acc_perturb, consistency, count.
    """
    subjects_filter = set(subjects) if subjects else None

    original_map = {
        (entry["subject"], entry["id"]): entry
        for entry in _filter_subjects(original_records, subjects_filter)
    }
    perturbed_map = {
        (entry["subject"], entry["id"]): entry
        for entry in _filter_subjects(perturbed_records, subjects_filter)
    }

    shared_keys = sorted(set(original_map) & set(perturbed_map))
    if not shared_keys:
        raise ValueError("No overlapping (subject, id) pairs between logs.")

    orig_scores: List[int] = []
    pert_scores: List[int] = []
    consist_scores: List[int] = []

    for key in shared_keys:
        orig_entry = original_map[key]
        pert_entry = perturbed_map[key]
        orig_correct = _is_correct(orig_entry)
        pert_correct = _is_correct(pert_entry)
        orig_scores.append(orig_correct)
        pert_scores.append(pert_correct)
        consist_scores.append(1 if (orig_correct and pert_correct) else 0)

    n = len(shared_keys)
    acc_original = sum(orig_scores) / n
    acc_perturb = sum(pert_scores) / n
    consistency = sum(consist_scores) / n

    if f1_score:
        macro_f1 = f1_score(
            orig_scores, pert_scores, average="macro", zero_division=0
        )
        micro_f1 = f1_score(
            orig_scores, pert_scores, average="micro", zero_division=0
        )
    else:  # fallback if sklearn is unavailable
        macro_f1 = None
        micro_f1 = None

    return {
        "acc_original": acc_original,
        "acc_perturb": acc_perturb,
        "consistency": consistency,
        "f1_macro": macro_f1,
        "f1_micro": micro_f1,
        "n_samples": n,
    }


__all__ = ["evaluate_transition"]

