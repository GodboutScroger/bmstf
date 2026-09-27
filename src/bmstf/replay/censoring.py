"""Competing-risk censoring and the inverse-probability-of-censoring correction.

Ref: Algorithm 3 lines 3-12: elective repair as a competing risk that
cause-specifically censors inside the recommendation window, and the
inverse-probability-of-censoring correction of v^1_ik, v^2_ik and d^c_ik.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CENSORING_FLOOR = 1.0e-3


@dataclass(frozen=True, slots=True)
class CensoringModel:
    survival: np.ndarray

    def probability(self, month: float) -> float:
        index = int(np.clip(round(float(month)), 0, self.survival.shape[0] - 1))
        return float(max(self.survival[index], CENSORING_FLOOR))

    def weight(self, month: float) -> float:
        return float(1.0 / self.probability(month))

    def as_mapping(self) -> dict[str, float]:
        return {
            "months": float(self.survival.shape[0]),
            "final_survival": float(self.survival[-1]),
        }


def fit_censoring(censoring_months: np.ndarray, ceiling: int = 60) -> CensoringModel:
    """A discrete-time censoring survival curve from the observed censoring times."""

    observed = np.asarray(censoring_months, dtype=np.float64).reshape(-1)
    at_risk = np.zeros(int(ceiling) + 1, dtype=np.float64)
    events = np.zeros(int(ceiling) + 1, dtype=np.float64)
    count = observed.shape[0]
    for month in range(int(ceiling) + 1):
        at_risk[month] = float(count)
        events[month] = float(np.count_nonzero(np.round(observed) == month))
    survival = np.ones(int(ceiling) + 1, dtype=np.float64)
    running = 1.0
    for month in range(1, int(ceiling) + 1):
        denominator = max(at_risk[month], 1.0)
        running *= max(1.0 - events[month] / denominator, CENSORING_FLOOR)
        survival[month] = running
    return CensoringModel(survival=np.clip(survival, CENSORING_FLOOR, 1.0))


def censored_inside_window(
    repair_month: float | None,
    crossing_month: float | None,
    event_month: float | None,
    start: float,
    horizon: float,
) -> bool:
    """True when elective repair falls inside the window before either barrier event."""

    if repair_month is None:
        return False
    repair = float(repair_month)
    if not (start < repair <= start + horizon):
        return False
    crossing_before = crossing_month is not None and float(crossing_month) <= repair
    event_before = event_month is not None and float(event_month) <= repair
    return not (crossing_before or event_before)


def apply_correction(
    indicators: np.ndarray,
    weights: np.ndarray,
    censor_weight: float,
) -> np.ndarray:
    """Scale the indicators of the censored records by the inverse censoring probability."""

    values = np.asarray(indicators, dtype=np.float64).reshape(-1)
    factor = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != factor.shape:
        raise ValueError("indicators and weights must share a length")
    adjusted = values.copy()
    mask = adjusted < 0.0
    adjusted[mask] = 0.0
    return np.asarray(adjusted * factor, dtype=np.float64)
