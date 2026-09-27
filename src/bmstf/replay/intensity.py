"""Visit-intensity model for the inverse-intensity-weighted replay.

Ref: Algorithm 3 line 2: fit i_hat_i(t | H_{i,k-1}); w_ik = 1 / i_hat_i(t_ik |
H_{i,k-1}) for every visit, stabilised by the unconditional intensity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MONTH_CEILING = 60
SMOOTHING = 1.0
PRIOR_STRENGTH = 4.0


@dataclass(frozen=True, slots=True)
class VisitIntensity:
    hazard: np.ndarray
    unconditional: float

    def at(self, month: float) -> float:
        index = int(np.clip(round(float(month)), 0, self.hazard.shape[0] - 1))
        return float(max(self.hazard[index], 1.0e-6))

    def stabilised_weight(self, month: float) -> float:
        """The unconditional intensity over the conditional one, the usual stabiliser."""

        return float(self.unconditional / self.at(month))

    def as_mapping(self) -> dict[str, float]:
        return {
            "unconditional": float(self.unconditional),
            "months": float(self.hazard.shape[0]),
            "mean_hazard": float(np.mean(self.hazard)),
        }


def fit_intensity(intervals: np.ndarray, ceiling: int = MONTH_CEILING) -> VisitIntensity:
    """Discrete-time per-month visit hazard with a Gamma prior and light smoothing."""

    values = np.asarray(intervals, dtype=np.float64).reshape(-1)
    counts = np.zeros(int(ceiling) + 1, dtype=np.float64)
    exposure = np.zeros(int(ceiling) + 1, dtype=np.float64)
    for value in values:
        index = int(np.clip(round(float(value)), 1, int(ceiling)))
        counts[index] += 1.0
        exposure[index:] += 1.0
    exposure = np.maximum(exposure, 1.0)
    raw = (counts + PRIOR_STRENGTH * float(np.mean(counts) / max(np.mean(exposure), 1e-9))) / (
        exposure + PRIOR_STRENGTH
    )
    smoothed = _smooth(raw, SMOOTHING)
    unconditional = float(np.mean(counts) / max(np.mean(exposure), 1.0e-9))
    return VisitIntensity(hazard=np.clip(smoothed, 1.0e-6, 1.0), unconditional=unconditional)


def _smooth(values: np.ndarray, width: float) -> np.ndarray:
    if width <= 0.0 or values.shape[0] < 3:
        return values
    kernel = np.exp(-0.5 * (np.arange(-3, 4, dtype=np.float64) / float(width)) ** 2)
    kernel = kernel / float(np.sum(kernel))
    padded = np.pad(values, (3, 3), mode="edge")
    return np.asarray(np.convolve(padded, kernel, mode="valid"), dtype=np.float64)


def intervals_from_months(months: np.ndarray) -> np.ndarray:
    grid = np.asarray(months, dtype=np.float64).reshape(-1)
    if grid.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    return np.diff(grid)


def weights_for_visits(
    model: VisitIntensity, months: np.ndarray, since_previous: np.ndarray
) -> np.ndarray:
    """w_ik for each visit, using the gap since the previous visit as the conditioning state."""

    grid = np.asarray(months, dtype=np.float64).reshape(-1)
    gaps = np.asarray(since_previous, dtype=np.float64).reshape(-1)
    if grid.shape[0] != gaps.shape[0]:
        raise ValueError("months and gaps must share a length")
    weights = np.empty(grid.shape[0], dtype=np.float64)
    for index in range(grid.shape[0]):
        gap = float(gaps[index]) if index > 0 else float(grid[index]) + 1.0
        weights[index] = model.stabilised_weight(gap)
    return weights


def truncated_deferral(values: np.ndarray, cap: float) -> float:
    """min(sum of weighted defers / sum of weights, rho_max) as Algorithm 3 line 15 states."""

    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size == 0:
        return 0.0
    estimate = float(np.mean(sample))
    return min(estimate, float(cap))
