"""Calibration read-outs: expected calibration error, reliability and coverage.

Ref: Table 1 panel A (expected calibration error of the 12-month crossing
probability; empirical 90 % coverage of the 12-month forecast of the true
diameter and of the true low-shear burden).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BINS = 10
NOMINAL_COVERAGE = 0.90


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    lower_edges: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    counts: np.ndarray

    def expected_calibration_error(self) -> float:
        total = float(np.sum(self.counts))
        if total <= 0.0:
            return 0.0
        gaps = np.abs(self.predicted - self.observed)
        return float(np.sum(self.counts * gaps) / total)

    def as_mapping(self) -> dict[str, float]:
        return {
            "ece": float(self.expected_calibration_error()),
            "bins": float(self.counts.shape[0]),
            "nonempty_bins": float(np.count_nonzero(self.counts)),
        }


def reliability_curve(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = DEFAULT_BINS
) -> ReliabilityCurve:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("probabilities and outcomes must share a shape")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    index = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, int(bins) - 1)
    predicted = np.zeros(int(bins), dtype=np.float64)
    observed = np.zeros(int(bins), dtype=np.float64)
    counts = np.zeros(int(bins), dtype=np.float64)
    for position in range(int(bins)):
        selected = index == position
        counts[position] = float(np.count_nonzero(selected))
        if counts[position] > 0:
            predicted[position] = float(np.mean(scores[selected]))
            observed[position] = float(np.mean(labels[selected]))
    return ReliabilityCurve(edges[:-1], predicted, observed, counts)


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = DEFAULT_BINS
) -> float:
    return reliability_curve(probabilities, outcomes, bins).expected_calibration_error()


def coverage(indicators: np.ndarray) -> float:
    values = np.asarray(indicators, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def coverage_gap(indicators: np.ndarray, nominal: float = NOMINAL_COVERAGE) -> float:
    return float(coverage(indicators) - float(nominal))
