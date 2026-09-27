"""Holm step-down adjustment over a metric's comparator family.

Ref: Table 1 caption (the exact sign-flip p value Holm-adjusted within each metric
across the eleven comparators of Tables 1 and 2).
"""

from __future__ import annotations

import numpy as np

COMPARATOR_COUNT = 11


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Adjusted p values: the running maximum of (remaining tests) x (ordered p)."""

    raw = np.asarray(p_values, dtype=np.float64).reshape(-1)
    count = raw.shape[0]
    if count == 0:
        empty: np.ndarray = raw.copy()
        return empty
    order = np.argsort(raw, kind="mergesort")
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        scaled = float(count - position) * float(raw[index])
        running = max(running, min(1.0, scaled))
        adjusted[index] = running
    ordered: np.ndarray = adjusted
    return ordered


def adjust_within_metric(p_values: np.ndarray) -> np.ndarray:
    return holm_bonferroni(p_values)


def family_width(comparators: int = COMPARATOR_COUNT) -> int:
    return int(comparators)


def significant(p_values: np.ndarray, adjusted: np.ndarray, level: float = 0.05) -> np.ndarray:
    raw = np.asarray(p_values, dtype=np.float64).reshape(-1)
    corrected = np.asarray(adjusted, dtype=np.float64).reshape(-1)
    flags: np.ndarray = np.asarray((raw <= float(level)) & (corrected <= float(level)), dtype=bool)
    return flags
