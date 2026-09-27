"""Decision engines: filtering, snapshot and band comparators on one record.

Ref: Algorithm 2 (inference at one surveillance CT angiogram); Algorithm 3 lines
3-14 (the landmark at the second observation and the per-visit read-out);
Table 1 (the snapshot comparator propagates process noise from a point state);
Table 3 (the calibrated band policy re-calibrated at each tolerance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bmstf.inference.forecast import Forecast, ForecastSpec, monte_carlo_forecast
from bmstf.inference.imm import (
    FilterConfiguration,
    FilterTrace,
    IMMState,
    augmented_modes,
    run_filter_trace,
    single_regime_modes,
    two_regime_modes,
)
from bmstf.policy.bands import (
    BandPolicy,
    band_from_diameter,
    literal_band_policy,
    point_crossing_probability,
)
from bmstf.policy.barriers import (
    ALPHA1,
    ALPHA2,
    IntervalDecision,
    action_grid,
    read_out,
    unconditional_read_out,
)
from bmstf.process.cohort import CHANNEL_CTA, US_SITE_OFFSET_MM, HistorySample, Visit
from bmstf.process.hazards import StratumRisk, repair_threshold_mm
from bmstf.process.regimes import AREA_UNIT_MM2, ShearSystem
from bmstf.scoring.crps import crps_sample

LANDMARK_INDEX = 1
PRIOR_COVARIANCE = {
    "diameter": 25.0,
    "burden": 0.04,
    "area": 0.04,
    "osi": 0.04,
    "mode": 0.01,
    "offset": 0.25,
}
COVERAGE_HORIZON = 12.0
SECONDARY_HORIZONS: tuple[float, ...] = (6.0, 12.0, 24.0)
STANDARDISED_COVERAGE_LEVEL = 0.10
SNAPSHOT_POINT_VARIANCE = 1.0e-3


def corrected_visits(history: HistorySample) -> tuple[Visit, ...]:
    """Remove the known site offset from the ultrasound diameter channel.

    Algorithm 2 line 10 reads the ultrasound diameter with the offset delta_US,site
    in the measurement model; with oracle parameters that offset is known, so the
    filter consumes the offset-corrected diameter.
    """

    bias = float(US_SITE_OFFSET_MM[int(history.site) % len(US_SITE_OFFSET_MM)])
    corrected: list[Visit] = []
    for visit in history.visits:
        if visit.channel == CHANNEL_CTA:
            corrected.append(visit)
            continue
        corrected.append(
            Visit(
                month=visit.month,
                channel=visit.channel,
                measurement=np.asarray(visit.measurement, dtype=np.float64) - bias,
                covariance=visit.covariance,
                flag=visit.flag,
            )
        )
    return tuple(corrected)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    filter: FilterConfiguration
    policy: str = "two_barrier"
    comparator: str = "literal_band"
    is_band_policy: bool = False
    snapshot: bool = False

    def as_mapping(self) -> dict[str, Any]:
        return {
            "arm": self.name,
            "policy": self.policy,
            "modes": self.filter.modes.name,
            "use_ultrasound": self.filter.use_ultrasound,
            "hemodynamic_channels": self.filter.hemodynamic_channels,
            "bias_augmented": self.filter.bias_augmented,
            "spread_factor": self.filter.spread_factor,
            "scalar_noise": self.filter.scalar_noise,
            "snapshot": self.snapshot,
        }


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    arm: str
    patient: str
    site: int
    sex: str
    decision_month: float
    landmark_index: int
    diameter_mm: float
    band: str
    gate_month: float
    interval_months: float
    escalate: bool
    comparator_months: float
    crossing_probability: float
    event_probability: float
    crossing_12: float
    event_12: float
    crossing_hit: bool
    event_hit: bool
    crossing_12_hit: bool
    event_12_hit: bool
    deferred: bool
    under_observation: bool
    coverage_diameter: dict[str, float]
    coverage_burden: dict[str, float]
    interval_width: tuple[float, float]
    crps_diameter_12: float
    burden_error: float
    active_probability: float
    shift_crossing: dict[str, float] = field(default_factory=dict)
    shift_event: dict[str, float] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "patient": self.patient,
            "site": self.site,
            "sex": self.sex,
            "decision_month": float(self.decision_month),
            "diameter_mm": float(self.diameter_mm),
            "band": self.band,
            "gate_month": float(self.gate_month),
            "interval_months": float(self.interval_months),
            "escalate": bool(self.escalate),
            "comparator_months": float(self.comparator_months),
            "crossing_probability": float(self.crossing_probability),
            "event_probability": float(self.event_probability),
            "crossing_12": float(self.crossing_12),
            "event_12": float(self.event_12),
            "crossing_hit": bool(self.crossing_hit),
            "event_hit": bool(self.event_hit),
            "crossing_12_hit": bool(self.crossing_12_hit),
            "event_12_hit": bool(self.event_12_hit),
            "deferred": bool(self.deferred),
            "under_observation": bool(self.under_observation),
            "coverage_diameter": dict(self.coverage_diameter),
            "coverage_burden": dict(self.coverage_burden),
            "interval_width_diameter": float(self.interval_width[0]),
            "interval_width_burden": float(self.interval_width[1]),
            "crps_diameter_12": float(self.crps_diameter_12),
            "burden_error": float(self.burden_error),
            "active_probability": float(self.active_probability),
            "shift_crossing": dict(self.shift_crossing),
            "shift_event": dict(self.shift_event),
        }


@dataclass(frozen=True, slots=True)
class ArmResult:
    arm: str
    records: tuple[DecisionRecord, ...]
    burden_change_errors: np.ndarray
    coverage_by_horizon: dict[str, dict[str, float]]
    extra: dict[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "count": len(self.records),
            "burden_change_rmse": (
                float(np.sqrt(np.mean(self.burden_change_errors**2)))
                if self.burden_change_errors.size
                else float("nan")
            ),
            "coverage": self.coverage_by_horizon,
        }


def prior_moments(
    system: ShearSystem, configuration: FilterConfiguration, history: HistorySample
) -> tuple[np.ndarray, np.ndarray]:
    """The initial augmented prior: a diffuse state plus the residual covariance on the bias."""

    dimension = system.dimension + (
        system.observation_dimension if configuration.bias_augmented else 0
    )
    mean = np.zeros(dimension, dtype=np.float64)
    first = history.visits[0]
    mean[0] = float(first.measurement[0])
    mean[1] = 0.35
    mean[2] = 0.35
    mean[3] = 0.40
    covariance = np.diag(
        [
            PRIOR_COVARIANCE["diameter"],
            PRIOR_COVARIANCE["burden"],
            PRIOR_COVARIANCE["area"],
            PRIOR_COVARIANCE["osi"],
        ]
        + [PRIOR_COVARIANCE["mode"]] * (system.dimension - 5)
        + [PRIOR_COVARIANCE["offset"]]
    )
    if configuration.bias_augmented:
        bias = np.asarray(history.surrogate_covariance, dtype=np.float64)
        covariance = np.block(
            [
                [covariance, np.zeros((covariance.shape[0], bias.shape[0]))],
                [np.zeros((bias.shape[0], covariance.shape[1])), bias],
            ]
        )
    return mean, covariance


def base_filter(name: str, system: ShearSystem, **overrides: Any) -> FilterConfiguration:
    modes = two_regime_modes(system)
    if overrides.get("single_regime") == "population":
        modes = single_regime_modes(system, "population")
    elif overrides.get("single_regime") == "quiescent":
        modes = single_regime_modes(system, "quiescent", noise_factor=2.0)
    bias = bool(overrides.get("bias_augmented", True))
    if bias:
        modes = augmented_modes(modes, system)
    return FilterConfiguration(
        name=name,
        modes=modes,
        use_ultrasound=bool(overrides.get("use_ultrasound", True)),
        bias_augmented=bias,
        scalar_noise=overrides.get("scalar_noise"),
        spread_factor=float(overrides.get("spread_factor", 1.0)),
        use_event_barrier=bool(overrides.get("use_event_barrier", True)),
        hemodynamic_channels=bool(overrides.get("hemodynamic_channels", True)),
    )


ARM_NAMES: tuple[str, ...] = (
    "BM-STF",
    "snapshot",
    "literal_bands",
    "calibrated_bands",
    "scalar_noise",
    "no_ultrasound",
    "single_regime_mean_drift",
    "single_regime_quiescent",
    "single_barrier",
    "diameter_only",
    "spread_half",
    "spread_double",
)


def build_arm(name: str, system: ShearSystem, scalar_noise: np.ndarray | None = None) -> ArmSpec:
    if name == "BM-STF":
        return ArmSpec(name=name, filter=base_filter("bmstf", system))
    if name == "snapshot":
        modes = single_regime_modes(system, "population")
        return ArmSpec(
            name=name,
            filter=FilterConfiguration(
                name="snapshot",
                modes=modes,
                bias_augmented=False,
                use_ultrasound=True,
                hemodynamic_channels=True,
            ),
            snapshot=True,
        )
    if name == "literal_bands":
        return ArmSpec(
            name=name,
            filter=base_filter("bmstf", system),
            policy="literal_band",
            is_band_policy=True,
        )
    if name == "calibrated_bands":
        return ArmSpec(
            name=name,
            filter=base_filter("bmstf", system),
            policy="calibrated_band",
            is_band_policy=True,
        )
    if name == "scalar_noise":
        return ArmSpec(
            name=name,
            filter=base_filter(
                "scalar_noise",
                system,
                bias_augmented=False,
                scalar_noise=(
                    scalar_noise if scalar_noise is not None else np.full(4, 0.05, dtype=np.float64)
                ),
            ),
        )
    if name == "no_ultrasound":
        return ArmSpec(name=name, filter=base_filter("no_ultrasound", system, use_ultrasound=False))
    if name == "single_regime_mean_drift":
        return ArmSpec(
            name=name, filter=base_filter("single_regime_mean", system, single_regime="population")
        )
    if name == "single_regime_quiescent":
        return ArmSpec(
            name=name,
            filter=base_filter("single_regime_quiescent", system, single_regime="quiescent"),
        )
    if name == "single_barrier":
        return ArmSpec(
            name=name,
            filter=base_filter("single_barrier", system, use_event_barrier=False),
            policy="single_barrier",
        )
    if name == "diameter_only":
        return ArmSpec(
            name=name, filter=base_filter("diameter_only", system, hemodynamic_channels=False)
        )
    if name == "spread_half":
        return ArmSpec(name=name, filter=base_filter("spread_half", system, spread_factor=0.5))
    if name == "spread_double":
        return ArmSpec(name=name, filter=base_filter("spread_double", system, spread_factor=2.0))
    raise ValueError(f"unknown arm {name!r}")


def snapshot_state(history: HistorySample, index: int, system: ShearSystem) -> IMMState:
    """A point state carrying the scan's own observation, with no filtering."""

    visit = history.visits[index]
    dimension = system.dimension
    mean = np.zeros(dimension, dtype=np.float64)
    covariance = np.diag(
        [SNAPSHOT_POINT_VARIANCE]
        + [PRIOR_COVARIANCE["burden"], PRIOR_COVARIANCE["area"], PRIOR_COVARIANCE["osi"]]
        + [PRIOR_COVARIANCE["mode"]] * (dimension - 5)
        + [PRIOR_COVARIANCE["offset"]]
    )
    if visit.channel == CHANNEL_CTA:
        mean[0] = float(visit.measurement[0])
        mean[1] = float(visit.measurement[2])
        mean[2] = float(visit.measurement[1]) / AREA_UNIT_MM2
        mean[3] = float(visit.measurement[3])
    else:
        mean[0] = float(visit.measurement[0])
    means = mean[None, :]
    covariances = covariance[None, :, :]
    return IMMState(
        means=means,
        covariances=covariances,
        weights=np.ones(1, dtype=np.float64),
        steps=index,
        log_likelihood=0.0,
        mode_switches=0,
    )


def observed_diameters(history: HistorySample, upto: int) -> tuple[np.ndarray, np.ndarray]:
    months = np.asarray(history.visit_months[: upto + 1], dtype=np.float64)
    values = np.asarray([float(visit.measurement[0]) for visit in history.visits[: upto + 1]])
    return months, values


def band_decision(
    forecast: Forecast,
    policy: BandPolicy,
    diameter_mm: float,
    grid: np.ndarray,
    aggregate: bool,
) -> IntervalDecision:
    interval = policy.interval(diameter_mm)
    escalated = interval is None
    chosen = float(grid[0]) if interval is None else float(interval)
    index = int(np.argmin(np.abs(np.asarray(grid, dtype=np.float64) - chosen)))
    crossing = float(forecast.crossing_curve[index])
    event = float(forecast.event_curve[index])
    if aggregate:
        crossing = float(np.mean(forecast.crossing_curve))
        event = float(np.mean(forecast.event_curve))
    return IntervalDecision(
        horizon_months=chosen,
        escalate=escalated,
        crossing_probability=crossing,
        event_probability=event,
        gate_month=chosen,
        admissible=not escalated,
        feasible_crossing_upper=crossing,
    )


def evaluate_history(
    history: HistorySample,
    system: ShearSystem,
    arm: ArmSpec,
    forecast_spec: ForecastSpec,
    band_policy: BandPolicy,
    strata: StratumRisk,
    rng: np.random.Generator,
    alpha1: float = ALPHA1,
    alpha2: float = ALPHA2,
) -> list[DecisionRecord]:
    grid = action_grid()
    threshold = repair_threshold_mm(history.sex)
    prior_mean, prior_covariance = prior_moments(system, arm.filter, history)
    trace: FilterTrace | None = None
    if not arm.snapshot:
        trace = run_filter_trace(
            system,
            arm.filter,
            prior_mean,
            prior_covariance,
            history.visit_months,
            history.inputs,
            corrected_visits(history),
        )
    comparator_policy = literal_band_policy()
    records: list[DecisionRecord] = []
    horizon = int(forecast_spec.horizon_months)
    crossing_month = history.outcome.crossing_month
    event_month = history.outcome.event_month
    for index in range(LANDMARK_INDEX, len(history.visits)):
        if history.visits[index].channel != CHANNEL_CTA:
            continue
        decision_month = float(history.visit_months[index])
        if arm.snapshot:
            state = snapshot_state(history, index, system)
        else:
            assert trace is not None
            state = trace.state_at(index)
        forecast = monte_carlo_forecast(
            state, system, arm.filter, forecast_spec, history.inputs[index], history.sex, rng
        )
        diameter_hat = float(state.collapsed()[0][0])
        gate = strata.gate_month(diameter_hat, history.sex, grid, alpha2)
        if arm.policy == "two_barrier":
            decision = read_out(
                forecast.crossing_curve,
                forecast.event_curve,
                grid,
                forecast.paths,
                gate,
                alpha1=alpha1,
                alpha2=alpha2,
                event_barrier=arm.filter.use_event_barrier,
            )
        elif arm.policy == "single_barrier":
            decision = unconditional_read_out(
                forecast.crossing_curve,
                forecast.event_curve,
                grid,
                forecast.paths,
                gate,
                alpha1=alpha1,
            )
        elif arm.policy in ("literal_band", "calibrated_band"):
            decision = band_decision(forecast, band_policy, diameter_hat, grid, aggregate=False)
        elif arm.policy == "band_extrapolation":
            observed_months, observed = observed_diameters(history, index)
            probability = point_crossing_probability(observed_months, observed, 12.0, threshold)
            decision = IntervalDecision(
                horizon_months=float(grid[np.argmin(np.abs(grid - 12.0))]),
                escalate=False,
                crossing_probability=probability,
                event_probability=probability,
                gate_month=12.0,
                admissible=True,
                feasible_crossing_upper=probability,
            )
        else:
            raise ValueError(f"unknown policy {arm.policy!r}")
        spread = decision.horizon_months
        crossing_hit = _hit(crossing_month, decision_month, spread)
        event_hit = _hit(event_month, decision_month, spread)
        crossing_12_hit = _hit(crossing_month, decision_month, 12.0)
        event_12_hit = _hit(event_month, decision_month, 12.0)
        coverage_diameter, coverage_burden, widths = _coverage(
            forecast, history, decision_month, horizon
        )
        crps_value = _crps_at(forecast, history, decision_month, 12.0, horizon)
        comparator_months = float(
            comparator_policy.schedule.get(band_from_diameter(diameter_hat), float(grid[-1]))
        )
        deferred = bool(
            (not decision.escalate)
            and decision.horizon_months > comparator_months
            and (not crossing_hit)
            and (not event_hit)
        )
        records.append(
            DecisionRecord(
                arm=arm.name,
                patient=history.patient,
                site=history.site,
                sex=history.sex,
                decision_month=decision_month,
                landmark_index=index,
                diameter_mm=diameter_hat,
                band=band_from_diameter(diameter_hat),
                gate_month=float(gate),
                interval_months=float(decision.horizon_months),
                escalate=bool(decision.escalate),
                comparator_months=comparator_months,
                crossing_probability=float(decision.crossing_probability),
                event_probability=float(decision.event_probability),
                crossing_12=float(forecast.crossing_at(12.0)),
                event_12=float(forecast.event_at(12.0)),
                crossing_hit=bool(crossing_hit),
                event_hit=bool(event_hit),
                crossing_12_hit=bool(crossing_12_hit),
                event_12_hit=bool(event_12_hit),
                deferred=deferred,
                under_observation=bool(crossing_month is None or crossing_month > decision_month),
                coverage_diameter=coverage_diameter,
                coverage_burden=coverage_burden,
                interval_width=widths,
                crps_diameter_12=float(crps_value),
                burden_error=_burden_error(trace, history, index),
                active_probability=float(_active(state)),
                shift_crossing=_shift_readout(
                    forecast, decision, decision_month, horizon, "crossing"
                ),
                shift_event=_shift_readout(forecast, decision, decision_month, horizon, "event"),
            )
        )
    return records


def _shift_readout(
    forecast: Forecast,
    decision: IntervalDecision,
    decision_month: float,
    horizon: int,
    kind: str,
) -> dict[str, float]:
    """Crossing or event probability at tau*, tau*+1, tau*+2 and tau*+3 (Table 3 panel B)."""

    if decision.escalate:
        return {}
    readout: dict[str, float] = {}
    for shift in range(4):
        month = float(decision.horizon_months) + float(shift)
        if month > float(forecast.horizon_months) or decision_month + month > float(horizon):
            break
        readout[str(shift)] = float(
            forecast.crossing_at(month) if kind == "crossing" else forecast.event_at(month)
        )
    return readout


def _active(state: IMMState) -> float:
    if state.weights.shape[0] < 2:
        return 1.0
    return float(state.weights[1])


def _hit(month: float | None, start: float, span: float) -> bool:
    if month is None:
        return False
    return bool(start < float(month) <= start + float(span))


def _coverage(
    forecast: Forecast, history: HistorySample, decision_month: float, horizon: int
) -> tuple[dict[str, float], dict[str, float], tuple[float, float]]:
    diameter: dict[str, float] = {}
    burden: dict[str, float] = {}
    widths = forecast.width(float(min(COVERAGE_HORIZON, horizon - decision_month)))
    for level in SECONDARY_HORIZONS:
        target = decision_month + level
        if target > horizon:
            continue
        index = round(target)
        diameter[str(int(level))] = forecast.covers_diameter(
            level, float(history.monthly_diameter[index])
        )
        burden[str(int(level))] = forecast.covers_burden(
            level, float(history.monthly_burden[index])
        )
    return diameter, burden, widths


def _crps_at(
    forecast: Forecast, history: HistorySample, decision_month: float, level: float, horizon: int
) -> float:
    target = decision_month + level
    if target > horizon:
        return float("nan")
    index = round(target)
    band_index = forecast.band_index(level)
    return crps_sample(forecast.diameters[:, band_index], float(history.monthly_diameter[index]))


def _burden_error(trace: FilterTrace | None, history: HistorySample, index: int) -> float:
    if trace is None or index == 0:
        return 0.0
    predicted_change = float(trace.predicted_means[index][1] - trace.updated_means[index - 1][1])
    true_change = float(
        history.monthly_burden[int(history.visit_months[index])]
        - history.monthly_burden[int(history.visit_months[index - 1])]
    )
    return float(predicted_change - true_change)


def run_arm(
    histories: list[HistorySample],
    system: ShearSystem,
    arm: ArmSpec,
    forecast_spec: ForecastSpec,
    strata: StratumRisk,
    rng: np.random.Generator,
    band_policy: BandPolicy | None = None,
    alpha1: float = ALPHA1,
    alpha2: float = ALPHA2,
) -> ArmResult:
    policy = band_policy if band_policy is not None else literal_band_policy()
    records: list[DecisionRecord] = []
    errors: list[float] = []
    for history in histories:
        history_records = evaluate_history(
            history, system, arm, forecast_spec, policy, strata, rng, alpha1, alpha2
        )
        records.extend(history_records)
        if not arm.snapshot:
            prior_mean, prior_covariance = prior_moments(system, arm.filter, history)
            trace = run_filter_trace(
                system,
                arm.filter,
                prior_mean,
                prior_covariance,
                history.visit_months,
                history.inputs,
                corrected_visits(history),
            )
            errors.extend(
                _burden_error(trace, history, index) for index in range(1, len(history.visits))
            )
    coverage: dict[str, dict[str, float]] = {}
    for horizon in SECONDARY_HORIZONS:
        key = str(int(horizon))
        diameters = [
            record.coverage_diameter[key]
            for record in records
            if key in record.coverage_diameter and record.under_observation
        ]
        burdens = [
            record.coverage_burden[key]
            for record in records
            if key in record.coverage_burden and record.under_observation
        ]
        coverage[key] = {
            "diameter": float(np.mean(diameters)) if diameters else 0.0,
            "burden": float(np.mean(burdens)) if burdens else 0.0,
            "decisions": float(len(diameters)),
        }
    return ArmResult(
        arm=arm.name,
        records=tuple(records),
        burden_change_errors=np.asarray(errors, dtype=np.float64),
        coverage_by_horizon=coverage,
    )


def tune_scalar_noise(histories: list[HistorySample], channels: int = 4) -> np.ndarray:
    """The tuned scalar noise of the ablation: the pooled per-channel observation spread.

    The decomposition is replaced by this diagonal, so the ablation keeps a
    channel-wise scale but loses the cross-channel and bias structure.
    """

    pooled = np.zeros(channels, dtype=np.float64)
    count = 0
    for history in histories:
        for observation in history.observations:
            pooled += np.diag(np.asarray(observation.covariance, dtype=np.float64))
            count += 1
    if count == 0:
        return np.full(channels, 0.05, dtype=np.float64)
    return np.asarray(np.sqrt(pooled / float(count)), dtype=np.float64)
