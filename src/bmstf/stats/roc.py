"""DeLong variance of the area under the ROC curve and its paired contrast.

Ref: Table 5 panel A (the intervals quoted beside every AUROC), Table 1 panel A
(the seed-paired increment in AUROC with its own interval).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DelongResult:
    area: float
    variance: float

    def standard_error(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))

    def as_mapping(self) -> dict[str, float]:
        return {"area": float(self.area), "variance": float(self.variance)}


@dataclass(frozen=True, slots=True)
class PairedContrast:
    difference: float
    standard_error: float
    z_statistic: float
    p_value: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "difference": float(self.difference),
            "standard_error": float(self.standard_error),
            "z_statistic": float(self.z_statistic),
            "p_value": float(self.p_value),
        }


def placement_matrix(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive = np.asarray(scores, dtype=np.float64).reshape(-1)[np.asarray(labels) == 1.0]
    negative = np.asarray(scores, dtype=np.float64).reshape(-1)[np.asarray(labels) == 0.0]
    matrix = np.zeros((positive.shape[0], negative.shape[0]), dtype=np.float64)
    for row, value in enumerate(positive):
        matrix[row] = np.where(value > negative, 1.0, np.where(value == negative, 0.5, 0.0))
    return matrix


def delong(scores: np.ndarray, labels: np.ndarray) -> DelongResult:
    matrix = placement_matrix(scores, labels)
    if matrix.size == 0:
        return DelongResult(area=0.5, variance=0.0)
    v10 = matrix.mean(axis=1)
    v01 = matrix.mean(axis=0)
    variance = float(np.var(v10, ddof=1) / matrix.shape[1] + np.var(v01, ddof=1) / matrix.shape[0])
    return DelongResult(area=float(matrix.mean()), variance=variance)


def paired_contrast(first: np.ndarray, second: np.ndarray, labels: np.ndarray) -> PairedContrast:
    outcomes = np.asarray(labels, dtype=np.float64).reshape(-1)
    matrix_first = placement_matrix(first, outcomes)
    matrix_second = placement_matrix(second, outcomes)
    if matrix_first.size == 0 or matrix_second.size == 0:
        return PairedContrast(0.0, 0.0, 0.0, 1.0)
    first_positive = matrix_first.mean(axis=1)
    second_positive = matrix_second.mean(axis=1)
    first_negative = matrix_first.mean(axis=0)
    second_negative = matrix_second.mean(axis=0)
    difference = float(matrix_first.mean() - matrix_second.mean())
    covariance = float(
        np.cov(first_positive, second_positive, ddof=1)[0, 1] / matrix_first.shape[1]
        + np.cov(first_negative, second_negative, ddof=1)[0, 1] / matrix_first.shape[0]
    )
    variance = (
        delong(first, outcomes).variance + delong(second, outcomes).variance - 2.0 * covariance
    )
    standard_error = float(np.sqrt(max(variance, 1.0e-15)))
    z_statistic = difference / standard_error
    p_value = float(2.0 * (1.0 - _normal_cdf(abs(z_statistic))))
    return PairedContrast(difference, standard_error, z_statistic, p_value)


def _normal_cdf(value: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(float(value)))
