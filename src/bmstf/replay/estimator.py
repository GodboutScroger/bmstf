"""Inverse-intensity-weighted retrospective policy replay.

Ref: Algorithm 3 lines 15-17: V_hat^a = sum w_ik v^a_ik / sum w_ik for a in {1, 2};
the deferral estimator truncated at rho_max; the imaging burden as the
weighted mean of tau*; the escalate fraction overall and by 5-mm band and sex;
B record-level bootstrap resamples with percentile intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bmstf.replay.censoring import CensoringModel, censored_inside_window
from bmstf.replay.intensity import VisitIntensity, intervals_from_months, weights_for_visits
from bmstf.replay.landmarks import landmark_indices
from bmstf.stats.bootstrap import Interval, cluster_bootstrap


@dataclass(frozen=True, slots=True)
class ReplayVisit:
    record: str
    index: int
    month: float
    horizon: float
    escalate: bool
    crossing_hit: bool
    event_hit: bool
    comparator_months: float
    band: str
    sex: str
    repair_month: float | None


@dataclass(frozen=True, slots=True)
class ReplayEstimates:
    crossing_rate: Interval
    event_rate: Interval
    deferral: dict[str, float]
    imaging_burden: float
    escalate_fraction: float
    by_band: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, object]:
        return {
            "crossing_rate": self.crossing_rate.as_mapping(),
            "event_rate": self.event_rate.as_mapping(),
            "deferral": dict(self.deferral),
            "imaging_burden": float(self.imaging_burden),
            "escalate_fraction": float(self.escalate_fraction),
            "by_band": {key: dict(value) for key, value in self.by_band.items()},
        }


def build_visits(records: list[ReplayVisit]) -> list[ReplayVisit]:
    return list(records)


def estimate(
    visits: list[ReplayVisit],
    intensity: VisitIntensity,
    censoring: CensoringModel,
    deferral_cap: float,
    bandwidth_mm: float = 5.0,
    resamples: int = 2000,
    rng: np.random.Generator | None = None,
) -> ReplayEstimates:
    """Weighted rates, deferral and burden with record-level bootstrap intervals."""

    if not visits:
        empty = Interval(0.0, 0.0, 0.0, 0.95, 0)
        return ReplayEstimates(empty, empty, {}, 0.0, 0.0)
    weights = np.asarray(
        [
            intensity.stabilised_weight(visit.month) * censoring.weight(visit.month + visit.horizon)
            for visit in visits
        ],
        dtype=np.float64,
    )
    crossing = np.asarray([1.0 if visit.crossing_hit else 0.0 for visit in visits])
    event = np.asarray([1.0 if visit.event_hit else 0.0 for visit in visits])
    escalate = np.asarray([1.0 if visit.escalate else 0.0 for visit in visits])
    horizon = np.asarray([float(visit.horizon) for visit in visits])
    deferred = np.asarray(
        [
            (
                1.0
                if (
                    (not visit.escalate)
                    and float(visit.horizon) > float(visit.comparator_months)
                    and (not visit.crossing_hit)
                    and (not visit.event_hit)
                )
                else 0.0
            )
            for visit in visits
        ]
    )
    censored = np.asarray(
        [
            (
                1.0
                if censored_inside_window(
                    visit.repair_month,
                    float(visit.month + visit.horizon) if visit.crossing_hit else None,
                    float(visit.month + visit.horizon) if visit.event_hit else None,
                    visit.month,
                    visit.horizon,
                )
                else 0.0
            )
            for visit in visits
        ]
    )
    total = float(np.sum(weights))
    crossing_rate = float(np.sum(weights * crossing) / total) if total > 0 else 0.0
    event_rate = float(np.sum(weights * event) / total) if total > 0 else 0.0
    deferral_point = (
        float(min(np.sum(weights * deferred) / total, deferral_cap)) if total > 0 else 0.0
    )
    burden = float(np.sum(weights * horizon) / total) if total > 0 else 0.0
    escalate_point = float(np.sum(weights * escalate) / total) if total > 0 else 0.0
    labels = np.asarray([visit.record for visit in visits])
    generator = rng if rng is not None else np.random.default_rng(0)
    crossing_interval = cluster_bootstrap(crossing, labels, resamples=resamples, rng=generator)
    event_interval = cluster_bootstrap(event, labels, resamples=resamples, rng=generator)
    by_band: dict[str, dict[str, float]] = {}
    for band in sorted({visit.band for visit in visits}):
        selected = np.asarray([visit.band == band for visit in visits])
        if not bool(np.any(selected)):
            continue
        by_band[band] = {
            "escalate": float(np.mean(escalate[selected])),
            "crossing": float(np.mean(crossing[selected])),
            "event": float(np.mean(event[selected])),
            "interval": float(np.mean(horizon[selected])),
            "count": float(np.count_nonzero(selected)),
        }
    return ReplayEstimates(
        crossing_rate=Interval(
            crossing_rate, crossing_interval.lower, crossing_interval.upper, 0.95, resamples
        ),
        event_rate=Interval(
            event_rate, event_interval.lower, event_interval.upper, 0.95, resamples
        ),
        deferral={
            "point": deferral_point,
            "censored_share": float(np.mean(censored)),
            "cap": float(deferral_cap),
        },
        imaging_burden=burden,
        escalate_fraction=escalate_point,
        by_band=by_band,
    )


def replay_indices(channels: tuple[str, ...], start: int = 2) -> tuple[int, ...]:
    return landmark_indices(channels, start)


def intensity_from_records(months: list[np.ndarray]) -> VisitIntensity:
    intervals = np.concatenate([intervals_from_months(grid) for grid in months if grid.size > 1])
    from bmstf.replay.intensity import fit_intensity

    return fit_intensity(intervals)


def visit_weights(months: np.ndarray, intensity: VisitIntensity) -> np.ndarray:
    grid = np.asarray(months, dtype=np.float64)
    gaps = np.concatenate([[grid[0] + 1.0], np.diff(grid)]) if grid.size else np.zeros(0)
    return weights_for_visits(intensity, grid, gaps)
