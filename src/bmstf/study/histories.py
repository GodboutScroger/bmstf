"""Seed-level orchestration of the in-silico study.

Ref: Table 1 caption (every value is the mean over 20 seeds of 300 in-silico
histories with its 95 % percentile-bootstrap interval over seeds); Table 2 (the
same construction for the ablations); Table 3 (a tolerance sweep and a
6,000-history calibration cohort for the band comparator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bmstf.inference.forecast import ForecastSpec
from bmstf.policy.bands import (
    BAND_ORDER,
    BandPolicy,
    band_from_diameter,
    calibrate_band_policy,
    literal_band_policy,
)
from bmstf.policy.barriers import ALPHA1, ALPHA2, action_grid
from bmstf.process.cohort import (
    INITIAL_BURDEN_BAND,
    INITIAL_DIAMETER_BAND_MM,
    INTERVAL_LOG_SIGMA,
    INTERVAL_MAX_MONTHS,
    INTERVAL_MEDIAN_MONTHS,
    INTERVAL_MIN_MONTHS,
    US_PROBABILITY,
    US_STANDARD_DEVIATION_MM,
    HistorySample,
    HistorySpec,
    SensorContext,
    patient_batch,
)
from bmstf.process.hazards import StratumRisk, repair_threshold_mm
from bmstf.process.regimes import ShearSystem
from bmstf.runtime.seeds import seed_sequence, set_seed
from bmstf.study.engines import ARM_NAMES, ArmResult, build_arm, run_arm, tune_scalar_noise

SEED_COUNT = 20
PATIENTS_PER_SEED = 300
CALIBRATION_PATIENTS = 6000
BASE_SEED = 20260925


@dataclass(frozen=True, slots=True)
class StudySpec:
    seeds: int = SEED_COUNT
    patients: int = PATIENTS_PER_SEED
    calibration_patients: int = CALIBRATION_PATIENTS
    horizon_months: int = 36
    forecast_paths: int = 256
    women_share: float = 0.19
    arms: tuple[str, ...] = ARM_NAMES
    base_seed: int = BASE_SEED
    calibration_grid: tuple[float, ...] = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 24.0, 36.0)
    decision_row_limit: int = 2000
    interval_median_months: float = INTERVAL_MEDIAN_MONTHS
    interval_log_sigma: float = INTERVAL_LOG_SIGMA
    interval_min_months: int = INTERVAL_MIN_MONTHS
    interval_max_months: int = INTERVAL_MAX_MONTHS
    us_probability: float = US_PROBABILITY
    us_standard_deviation_mm: float = US_STANDARD_DEVIATION_MM
    initial_diameter_band_mm: tuple[float, float] = INITIAL_DIAMETER_BAND_MM
    initial_burden_band: tuple[float, float] = INITIAL_BURDEN_BAND
    alpha1: float = ALPHA1
    alpha2: float = ALPHA2

    def as_mapping(self) -> dict[str, Any]:
        return {
            "seeds": int(self.seeds),
            "patients": int(self.patients),
            "calibration_patients": int(self.calibration_patients),
            "horizon_months": int(self.horizon_months),
            "forecast_paths": int(self.forecast_paths),
            "women_share": float(self.women_share),
            "arms": list(self.arms),
            "base_seed": int(self.base_seed),
            "decision_row_limit": int(self.decision_row_limit),
            "alpha1": float(self.alpha1),
            "alpha2": float(self.alpha2),
            "cohort": self.history_spec().as_mapping(),
        }

    def history_spec(self) -> HistorySpec:
        """The cohort-process parameters this study runs on."""

        return HistorySpec(
            horizon_months=int(self.horizon_months),
            interval_median_months=float(self.interval_median_months),
            interval_log_sigma=float(self.interval_log_sigma),
            interval_min_months=int(self.interval_min_months),
            interval_max_months=int(self.interval_max_months),
            us_probability=float(self.us_probability),
            us_standard_deviation_mm=float(self.us_standard_deviation_mm),
            initial_diameter_band_mm=(
                float(self.initial_diameter_band_mm[0]),
                float(self.initial_diameter_band_mm[1]),
            ),
            initial_burden_band=(
                float(self.initial_burden_band[0]),
                float(self.initial_burden_band[1]),
            ),
        )

    def forecast_spec(self, horizon: int | None = None) -> ForecastSpec:
        return ForecastSpec(
            paths=int(self.forecast_paths),
            horizon_months=int(self.horizon_months if horizon is None else horizon),
        )


@dataclass(frozen=True, slots=True)
class SeedBlock:
    seed: int
    metrics: dict[str, dict[str, float]]
    records: dict[str, int]
    calibration: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StudyResult:
    spec: StudySpec
    seeds: tuple[SeedBlock, ...]
    band_policies: dict[str, BandPolicy]
    scalar_noise: np.ndarray
    aggregate: dict[str, dict[str, float]]
    decision_rows: tuple[dict[str, object], ...]
    decision_count: int

    def as_mapping(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_mapping(),
            "scalar_noise": [float(value) for value in np.asarray(self.scalar_noise).reshape(-1)],
            "aggregate": self.aggregate,
            "decision_count": int(self.decision_count),
            "decision_row_limit": int(self.spec.decision_row_limit),
            "band_policies": {
                name: policy.as_mapping() for name, policy in self.band_policies.items()
            },
        }


def metric_names() -> tuple[str, ...]:
    return (
        "coverage_diameter",
        "coverage_burden",
        "coverage_burden_high_noise",
        "expected_calibration_error",
        "auroc_crossing",
        "auroc_crossing_24",
        "escalate_fraction",
        "crossing_feasible",
        "crossing_pooled",
        "event_feasible",
        "event_pooled",
        "mean_interval_feasible",
        "mean_interval_pooled",
        "deferral_fraction",
        "burden_change_rmse",
        "crps_diameter",
    )


def calibrate_bands(
    sensor: SensorContext,
    system: ShearSystem,
    spec: StudySpec,
    alpha1: float = ALPHA1,
) -> BandPolicy:
    """Calibrate the band comparator on a separate cohort, as Table 3 specifies."""

    streams = set_seed(spec.base_seed + 991)
    histories = patient_batch(
        spec.calibration_patients,
        sensor,
        spec.history_spec(),
        streams.generator,
        prefix="CAL",
        women_share=spec.women_share,
    )
    grid = np.asarray(spec.calibration_grid, dtype=np.float64)
    labels: list[str] = []
    within: list[list[bool]] = []
    for history in histories:
        diameter = float(history.monthly_diameter[int(history.visit_months[1])])
        labels.append(band_from_diameter(diameter))
        crossing = history.outcome.crossing_month
        decision = float(history.visit_months[1])
        within.append(
            [
                bool(
                    crossing is not None and decision < float(crossing) <= decision + float(horizon)
                )
                for horizon in grid
            ]
        )
    return calibrate_band_policy(
        np.asarray(labels), np.asarray(within, dtype=bool), grid, alpha1=float(alpha1)
    )


def _metrics_for_arm(
    result: ArmResult,
    history_by_patient: dict[str, HistorySample],
    horizon: int,
) -> dict[str, float]:
    records = list(result.records)
    if not records:
        return {name: float("nan") for name in metric_names()}
    run_average = _run_average_boundary_trace(records, history_by_patient)
    coverage_d = [
        record.coverage_diameter.get("12", float("nan"))
        for record in records
        if "12" in record.coverage_diameter and record.under_observation
    ]
    coverage_b = [
        record.coverage_burden.get("12", float("nan"))
        for record in records
        if "12" in record.coverage_burden and record.under_observation
    ]
    high_noise = [
        record.coverage_burden.get("12", float("nan"))
        for record in records
        if "12" in record.coverage_burden
        and record.under_observation
        and _is_high_noise(record, history_by_patient, run_average)
    ]
    probabilities = np.asarray([record.crossing_12 for record in records], dtype=np.float64)
    labels = np.asarray(
        [1.0 if record.crossing_12_hit else 0.0 for record in records], dtype=np.float64
    )
    labels_24 = np.asarray(
        [
            1.0 if (record.crossing_hit and record.interval_months >= 24.0) else 0.0
            for record in records
        ],
        dtype=np.float64,
    )
    from bmstf.scoring.calibration import expected_calibration_error
    from bmstf.scoring.discrimination import roc_auc

    escalated = np.asarray([record.escalate for record in records], dtype=bool)
    crossing_feasible = [record for record in records if not record.escalate]
    crps_values = [
        record.crps_diameter_12 for record in records if np.isfinite(record.crps_diameter_12)
    ]
    rmse = (
        float(np.sqrt(np.mean(result.burden_change_errors**2)))
        if result.burden_change_errors.size
        else float("nan")
    )
    return {
        "coverage_diameter": float(np.mean(coverage_d)) if coverage_d else float("nan"),
        "coverage_burden": float(np.mean(coverage_b)) if coverage_b else float("nan"),
        "coverage_burden_high_noise": float(np.mean(high_noise)) if high_noise else float("nan"),
        "expected_calibration_error": float(expected_calibration_error(probabilities, labels)),
        "auroc_crossing": float(roc_auc(probabilities, labels)),
        "auroc_crossing_24": float(
            roc_auc(np.asarray([r.crossing_12 for r in records]), labels_24)
        ),
        "escalate_fraction": float(np.mean(escalated)) if records else float("nan"),
        "crossing_feasible": (
            float(np.mean([r.crossing_hit for r in crossing_feasible]))
            if crossing_feasible
            else float("nan")
        ),
        "crossing_pooled": float(np.mean([r.crossing_hit for r in records])),
        "event_feasible": (
            float(np.mean([r.event_hit for r in crossing_feasible]))
            if crossing_feasible
            else float("nan")
        ),
        "event_pooled": float(np.mean([r.event_hit for r in records])),
        "mean_interval_feasible": (
            float(np.mean([r.interval_months for r in crossing_feasible]))
            if crossing_feasible
            else float("nan")
        ),
        "mean_interval_pooled": float(np.mean([r.interval_months for r in records])),
        "deferral_fraction": float(np.mean([r.deferred for r in records])),
        "burden_change_rmse": rmse,
        "crps_diameter": float(np.mean(crps_values)) if crps_values else float("nan"),
    }


def _run_average_boundary_trace(
    records: list[Any], history_by_patient: dict[str, HistorySample]
) -> float:
    """The run's mean per-scan boundary-condition variance, the high-noise denominator."""

    values: list[float] = []
    for record in records:
        history = history_by_patient.get(str(record.patient))
        if history is None or history.boundary_trace.size == 0:
            continue
        index = int(np.clip(int(record.landmark_index), 0, history.boundary_trace.shape[0] - 1))
        values.append(float(history.boundary_trace[index]))
    return float(np.mean(values)) if values else 0.0


def _is_high_noise(
    record: Any, history_by_patient: dict[str, HistorySample], run_average: float
) -> bool:
    """Table 2 caption: scans whose boundary-condition variance exceeds 1.5x the run's mean.

    The denominator is the run's own average, not the record's own history: normalising
    each patient by its own scans would drop exactly the patients whose every scan carries
    a wide boundary-condition budget.
    """

    history = history_by_patient.get(str(record.patient))
    if history is None or history.boundary_trace.size == 0 or run_average <= 0.0:
        return False
    index = int(np.clip(int(record.landmark_index), 0, history.boundary_trace.shape[0] - 1))
    return bool(float(history.boundary_trace[index]) > 1.5 * float(run_average))


def run_study(
    sensor: SensorContext,
    system: ShearSystem,
    spec: StudySpec,
    strata: StratumRisk | None = None,
    alpha1: float = ALPHA1,
    alpha2: float = ALPHA2,
) -> StudyResult:
    table = strata if strata is not None else StratumRisk()
    seeds = seed_sequence(spec.base_seed, spec.seeds)
    blocks: list[SeedBlock] = []
    decision_rows: list[dict[str, object]] = []
    decision_count = 0
    scalar = np.full(4, 0.05, dtype=np.float64)
    for position, seed in enumerate(seeds):
        streams = set_seed(int(seed))
        histories = patient_batch(
            spec.patients,
            sensor,
            spec.history_spec(),
            streams.generator,
            prefix=f"S{position:02d}",
            women_share=spec.women_share,
        )
        if position == 0:
            scalar = tune_scalar_noise(histories)
        policies = {
            "literal_bands": literal_band_policy(),
            "calibrated_bands": calibrate_bands(sensor, system, spec, alpha1),
        }
        metrics: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        forecast_spec = spec.forecast_spec()
        by_patient = {history.patient: history for history in histories}
        for arm_name in spec.arms:
            arm = build_arm(arm_name, system, scalar_noise=scalar)
            policy = policies.get(arm_name)
            result = run_arm(
                histories,
                system,
                arm,
                forecast_spec,
                table,
                streams.forecast,
                band_policy=policy,
                alpha1=alpha1,
                alpha2=alpha2,
            )
            metrics[arm_name] = _metrics_for_arm(result, by_patient, spec.horizon_months)
            counts[arm_name] = len(result.records)
            decision_count += len(result.records)
            if len(decision_rows) < spec.decision_row_limit:
                remaining = spec.decision_row_limit - len(decision_rows)
                decision_rows.extend(record.as_mapping() for record in result.records[:remaining])
        blocks.append(
            SeedBlock(
                seed=int(seed),
                metrics=metrics,
                records=counts,
                calibration={
                    name: float(len(policy.schedule)) for name, policy in policies.items()
                },
            )
        )
    aggregate = _aggregate(blocks, spec.arms)
    return StudyResult(
        spec=spec,
        seeds=tuple(blocks),
        band_policies={
            "literal_bands": literal_band_policy(),
        },
        scalar_noise=np.asarray(scalar, dtype=np.float64),
        aggregate=aggregate,
        decision_rows=tuple(decision_rows),
        decision_count=int(decision_count),
    )


def _aggregate(blocks: list[SeedBlock], arms: tuple[str, ...]) -> dict[str, dict[str, float]]:
    from bmstf.stats.bootstrap import percentile_interval

    aggregate: dict[str, dict[str, float]] = {}
    for arm in arms:
        entry: dict[str, float] = {}
        for metric in metric_names():
            values = np.asarray([block.metrics[arm][metric] for block in blocks], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                entry[metric] = float("nan")
                entry[f"{metric}_lower"] = float("nan")
                entry[f"{metric}_upper"] = float("nan")
                continue
            summary = percentile_interval(finite, resamples=2000)
            entry[metric] = float(summary.estimate)
            entry[f"{metric}_lower"] = float(summary.lower)
            entry[f"{metric}_upper"] = float(summary.upper)
        aggregate[arm] = entry
    return aggregate


def contrast(
    aggregate: dict[str, dict[str, float]], reference: str, comparator: str, metric: str
) -> float:
    left = aggregate.get(reference, {}).get(metric, float("nan"))
    right = aggregate.get(comparator, {}).get(metric, float("nan"))
    return float(left - right)


def band_schedule_summary(policy: BandPolicy) -> dict[str, float]:
    summary: dict[str, float] = {}
    for band in BAND_ORDER:
        summary[band] = float(policy.schedule.get(band, -1.0))
    return summary


def grid_of(spec: StudySpec) -> np.ndarray:
    return action_grid(spec.calibration_grid)


def reproducibility_digest(spec: StudySpec) -> str:
    from bmstf.runtime.store import sha256_mapping

    return sha256_mapping({key: str(value) for key, value in spec.as_mapping().items()})


def threshold_for(sex: str) -> float:
    return repair_threshold_mm(sex)
