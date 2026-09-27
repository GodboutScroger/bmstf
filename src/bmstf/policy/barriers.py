"""Two-barrier read-out and the interval policy.

Ref: Eq. (1) tau*(H_K) = min{max{h in H : F^1_K(h) <= alpha_1}, h_gate(H_K)} with
h_gate(H_K) = max{h in H : Lambda_strat(D_K, sex)(h) <= alpha_2}; Algorithm 2
line 21, where the crossing barrier is checked on the Wilson upper bound and the
event barrier on F^2(h) + z_W se(F^2(h)); an empty admissible set returns the
escalate state, scored at the 3-month floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.policy.wilson import BOUND_LEVEL, standard_error, wilson_upper

ALPHA1 = 0.10
ALPHA2 = 0.01
ACTION_GRID: tuple[float, ...] = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 24.0, 36.0)
MINIMUM_INTERVAL_MONTHS = 3.0


@dataclass(frozen=True, slots=True)
class IntervalDecision:
    horizon_months: float
    escalate: bool
    crossing_probability: float
    event_probability: float
    gate_month: float
    admissible: bool
    feasible_crossing_upper: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "horizon_months": float(self.horizon_months),
            "escalate": float(self.escalate),
            "crossing_probability": float(self.crossing_probability),
            "event_probability": float(self.event_probability),
            "gate_month": float(self.gate_month),
            "admissible": float(self.admissible),
            "feasible_crossing_upper": float(self.feasible_crossing_upper),
        }


def action_grid(configuration_grid: tuple[float, ...] | None = None) -> np.ndarray:
    grid = ACTION_GRID if configuration_grid is None else configuration_grid
    return np.asarray(grid, dtype=np.float64)


def admissible_horizons(
    crossing_curve: np.ndarray,
    event_curve: np.ndarray,
    grid: np.ndarray,
    paths: int,
    alpha1: float = ALPHA1,
    alpha2: float = ALPHA2,
    bound_level: float = BOUND_LEVEL,
    event_barrier: bool = True,
) -> list[float]:
    admissible: list[float] = []
    for horizon in np.asarray(grid, dtype=np.float64):
        index = int(np.argmin(np.abs(np.asarray(grid, dtype=np.float64) - float(horizon))))
        crossing = float(crossing_curve[index])
        event = float(event_curve[index])
        crossing_ok = wilson_upper(crossing, paths, bound_level) <= float(alpha1)
        event_ok = (
            event + bound_level * standard_error(event, paths) <= float(alpha2)
            if event_barrier
            else True
        )
        if crossing_ok and event_ok:
            admissible.append(float(horizon))
    return admissible


def read_out(
    crossing_curve: np.ndarray,
    event_curve: np.ndarray,
    grid: np.ndarray,
    paths: int,
    gate_month: float,
    alpha1: float = ALPHA1,
    alpha2: float = ALPHA2,
    bound_level: float = BOUND_LEVEL,
    event_barrier: bool = True,
) -> IntervalDecision:
    values = np.asarray(grid, dtype=np.float64)
    admissible = admissible_horizons(
        crossing_curve, event_curve, values, paths, alpha1, alpha2, bound_level, event_barrier
    )
    if not admissible:
        floor = float(values[0])
        index = int(np.argmin(np.abs(values - floor)))
        return IntervalDecision(
            horizon_months=floor,
            escalate=True,
            crossing_probability=float(crossing_curve[index]),
            event_probability=float(event_curve[index]),
            gate_month=float(gate_month),
            admissible=False,
            feasible_crossing_upper=float(wilson_upper(crossing_curve[index], paths, bound_level)),
        )
    chosen = float(min(max(admissible), float(gate_month)))
    index = int(np.argmin(np.abs(values - chosen)))
    return IntervalDecision(
        horizon_months=chosen,
        escalate=False,
        crossing_probability=float(crossing_curve[index]),
        event_probability=float(event_curve[index]),
        gate_month=float(gate_month),
        admissible=True,
        feasible_crossing_upper=float(wilson_upper(crossing_curve[index], paths, bound_level)),
    )


def unconditional_read_out(
    crossing_curve: np.ndarray,
    event_curve: np.ndarray,
    grid: np.ndarray,
    paths: int,
    gate_month: float,
    alpha1: float = ALPHA1,
) -> IntervalDecision:
    """The single-barrier variant: the event barrier is dropped (Table 2)."""

    return read_out(
        crossing_curve,
        event_curve,
        grid,
        paths,
        gate_month,
        alpha1=alpha1,
        event_barrier=False,
    )
