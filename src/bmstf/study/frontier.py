"""Tolerance sweep and fixed-shift diagnostic.

Ref: Table 3 panel A (the crossing tolerance alpha_1 swept over 0.05, 0.10, 0.15
and 0.20 with alpha_2 = 0.01 fixed, the calibrated bands re-calibrated at each
tolerance on a 6,000-history cohort) and panel B (every feasible recommended
interval lengthened by one, two and three months).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from bmstf.inference.forecast import ForecastSpec
from bmstf.policy.bands import BandPolicy
from bmstf.process.cohort import HistorySample
from bmstf.process.hazards import StratumRisk
from bmstf.process.regimes import ShearSystem
from bmstf.study.engines import ArmResult, DecisionRecord, build_arm, run_arm

SWEEP_TOLERANCES: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
SHIFTS: tuple[int, ...] = (0, 1, 2, 3)


@dataclass(frozen=True, slots=True)
class SweepRow:
    alpha1: float
    arm: str
    interval_pooled: float
    interval_feasible: float
    escalate_fraction: float
    crossing_feasible: float
    crossing_pooled: float
    event_pooled: float

    def as_mapping(self) -> dict[str, object]:
        return {
            "alpha1": float(self.alpha1),
            "arm": self.arm,
            "interval_pooled": float(self.interval_pooled),
            "interval_feasible": float(self.interval_feasible),
            "escalate_fraction": float(self.escalate_fraction),
            "crossing_feasible": float(self.crossing_feasible),
            "crossing_pooled": float(self.crossing_pooled),
            "event_pooled": float(self.event_pooled),
        }


@dataclass(frozen=True, slots=True)
class ShiftRow:
    forecaster: str
    readout: str
    coverage_burden: float
    ece: float
    at_interval: dict[str, float]

    def as_mapping(self) -> dict[str, object]:
        return {
            "forecaster": self.forecaster,
            "readout": self.readout,
            "coverage_burden": float(self.coverage_burden),
            "ece": float(self.ece),
            "at_interval": {key: float(value) for key, value in self.at_interval.items()},
        }


def _summarise(result: ArmResult) -> dict[str, float]:
    records = list(result.records)
    feasible = [record for record in records if not record.escalate]
    return {
        "interval_pooled": (
            float(np.mean([record.interval_months for record in records]))
            if records
            else float("nan")
        ),
        "interval_feasible": (
            float(np.mean([record.interval_months for record in feasible]))
            if feasible
            else float("nan")
        ),
        "escalate_fraction": (
            float(np.mean([record.escalate for record in records])) if records else float("nan")
        ),
        "crossing_feasible": (
            float(np.mean([record.crossing_hit for record in feasible]))
            if feasible
            else float("nan")
        ),
        "crossing_pooled": (
            float(np.mean([record.crossing_hit for record in records])) if records else float("nan")
        ),
        "event_pooled": (
            float(np.mean([record.event_hit for record in records])) if records else float("nan")
        ),
    }


def sweep(
    histories: list[HistorySample],
    system: ShearSystem,
    forecast_spec: ForecastSpec,
    strata: StratumRisk,
    seed: int,
    calibrate: Callable[[float], BandPolicy],
    tolerances: tuple[float, ...] = SWEEP_TOLERANCES,
    arms: tuple[str, ...] = ("BM-STF", "single_barrier", "calibrated_bands"),
) -> list[SweepRow]:
    """One block of rows per tolerance, each block holding every arm.

    The band comparator is re-calibrated at each tolerance by the caller, which owns
    the calibration cohort; the sweep only varies alpha_1 and the arm. Every block
    starts from the same seed, so a change between two tolerances is the
    tolerance and not a different draw of the forecast paths.
    """

    rows: list[SweepRow] = []
    for alpha1 in tolerances:
        policy = calibrate(float(alpha1))
        rng = np.random.default_rng(int(seed))
        for name in arms:
            arm = build_arm(name, system)
            result = run_arm(
                histories,
                system,
                arm,
                forecast_spec,
                strata,
                rng,
                band_policy=policy if name == "calibrated_bands" else None,
                alpha1=float(alpha1),
            )
            summary = _summarise(result)
            rows.append(SweepRow(alpha1=float(alpha1), arm=name, **summary))
    return rows


def fixed_shift_rows(results: dict[str, ArmResult]) -> list[ShiftRow]:
    """Aggregate the per-record shift read-outs into Table 3 panel B rows."""

    rows: list[ShiftRow] = []
    for name, result in results.items():
        feasible = [record for record in result.records if not record.escalate]
        coverage = [
            record.coverage_burden.get("12")
            for record in feasible
            if "12" in record.coverage_burden
        ]
        finite = [value for value in coverage if value is not None]
        crossing: dict[str, float] = {}
        event: dict[str, float] = {}
        for shift in SHIFTS:
            key = str(shift)
            crossing_values = [
                record.shift_crossing[key] for record in feasible if key in record.shift_crossing
            ]
            event_values = [
                record.shift_event[key] for record in feasible if key in record.shift_event
            ]
            crossing[str(shift)] = (
                float(np.mean(crossing_values)) if crossing_values else float("nan")
            )
            event[str(shift)] = float(np.mean(event_values)) if event_values else float("nan")
        rows.append(
            ShiftRow(
                forecaster=name,
                readout="crossing",
                coverage_burden=float(np.mean(finite)) if finite else float("nan"),
                ece=_ece(feasible),
                at_interval=crossing,
            )
        )
        rows.append(
            ShiftRow(
                forecaster=name,
                readout="event",
                coverage_burden=float(np.mean(finite)) if finite else float("nan"),
                ece=_ece(feasible),
                at_interval=event,
            )
        )
    return rows


def _ece(records: list[DecisionRecord]) -> float:
    from bmstf.scoring.calibration import expected_calibration_error

    if not records:
        return float("nan")
    probabilities = np.asarray([record.crossing_12 for record in records], dtype=np.float64)
    labels = np.asarray(
        [1.0 if record.crossing_12_hit else 0.0 for record in records], dtype=np.float64
    )
    return float(expected_calibration_error(probabilities, labels))


def monotone_crossing(rows: list[SweepRow]) -> bool:
    values = [row.crossing_feasible for row in rows if row.arm == "BM-STF"]
    return all(later >= earlier - 1.0e-12 for earlier, later in pairwise(values))


def shift_increases(row: ShiftRow) -> bool:
    keys = [str(shift) for shift in SHIFTS if str(shift) in row.at_interval]
    values = [row.at_interval[key] for key in keys if np.isfinite(row.at_interval[key])]
    return all(later >= earlier - 1.0e-12 for earlier, later in pairwise(values))
