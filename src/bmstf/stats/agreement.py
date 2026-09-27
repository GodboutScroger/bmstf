"""Agreement statistics for the reader-adjudication cells.

Ref: Table 4 (outcome adjudication agreement, two readers, Cohen's kappa with its
95 % interval; inter-reader maximum-diameter difference, median absolute).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class KappaResult:
    kappa: float
    observed: float
    expected: float
    counts: int

    def as_mapping(self) -> dict[str, float]:
        return {
            "kappa": float(self.kappa),
            "observed_agreement": float(self.observed),
            "expected_agreement": float(self.expected),
            "counts": float(self.counts),
        }


def cohen_kappa(first: np.ndarray, second: np.ndarray) -> KappaResult:
    left = np.asarray(first).reshape(-1)
    right = np.asarray(second).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("the two rating vectors must share a length")
    labels = np.unique(np.concatenate([left, right]))
    total = left.shape[0]
    if total == 0:
        return KappaResult(kappa=0.0, observed=0.0, expected=0.0, counts=0)
    observed = float(np.count_nonzero(left == right)) / float(total)
    expected = 0.0
    for label in labels:
        expected += (float(np.count_nonzero(left == label)) / total) * (
            float(np.count_nonzero(right == label)) / total
        )
    denominator = 1.0 - expected
    kappa = (observed - expected) / denominator if abs(denominator) > 1.0e-12 else 0.0
    return KappaResult(
        kappa=float(kappa), observed=observed, expected=float(expected), counts=int(total)
    )


def mean_squares(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    subjects, raters = values.shape
    grand = float(values.mean())
    rows = values.mean(axis=1)
    columns = values.mean(axis=0)
    between_rows = raters * float(np.sum((rows - grand) ** 2)) / (subjects - 1)
    between_columns = subjects * float(np.sum((columns - grand) ** 2)) / (raters - 1)
    residual = float(np.sum((values - rows[:, None] - columns[None, :] + grand) ** 2)) / (
        (subjects - 1) * (raters - 1)
    )
    return {
        "subjects": float(subjects),
        "raters": float(raters),
        "between_rows": between_rows,
        "between_columns": between_columns,
        "residual": residual,
    }


def icc_two_one(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    assembly = mean_squares(values)
    subjects = int(assembly["subjects"])
    raters = int(assembly["raters"])
    between_rows = assembly["between_rows"]
    between_columns = assembly["between_columns"]
    residual = assembly["residual"]
    denominator = (
        between_rows + (raters - 1) * residual + raters * (between_columns - residual) / subjects
    )
    if abs(denominator) < 1.0e-15:
        return 0.0
    return float((between_rows - residual) / denominator)


def median_absolute_difference(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    return float(np.median(np.abs(left - right)))


def median_absolute_deviation(values: np.ndarray) -> float:
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size == 0:
        return 0.0
    return float(np.median(np.abs(sample - np.median(sample))))
