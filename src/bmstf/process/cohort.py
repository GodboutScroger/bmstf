"""In-silico histories from the two-regime latent shear-state process.

Ref: Methods, the in-silico trajectory mechanism (300 in-silico histories
per seed); Eq. (2)-(3); Algorithm 1 (the observation structure the process emits);
Table 4 (the retained inter-observation interval, median 8.6 months, IQR 5.8-12.1).

The process and the filter share one parameterisation: the filter is run with the
process's own matrices, so the observation mean is the latent state and the
boundary-condition and reconstruction machinery fixes the noise covariance. That is
the oracle-parameter regime the in-silico study reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bmstf.geometry.perturbations import boundary_condition_scale, topology_class
from bmstf.geometry.shapes import ShapeSpec, geometry_from_state, target_ilt_fraction
from bmstf.geometry.summaries import summarise
from bmstf.geometry.template import ScanGeometry
from bmstf.process.hazards import CompositeOutcome, composite_outcome
from bmstf.process.regimes import (
    AREA_UNIT_MM2,
    GROWTH_RATE_MM_PER_MONTH,
    GROWTH_SPREAD_MM_PER_MONTH,
    ShearSystem,
)
from bmstf.process.transition import covariance_sqrt, discretise_regime, regime_transition
from bmstf.sensor.boundary import BoundaryPrior
from bmstf.sensor.covariance import ResidualLookup
from bmstf.sensor.marginalise import MarginalisationSpec, Observation, marginalise
from bmstf.sensor.surrogate import PhysicsAttentionSurrogate

CHANNEL_CTA = "CTA"
CHANNEL_US = "US"
CHANNELS: tuple[str, ...] = (CHANNEL_CTA, CHANNEL_US)

INTERVAL_MEDIAN_MONTHS = 8.6
INTERVAL_LOG_SIGMA = 0.42
INTERVAL_MIN_MONTHS = 3
INTERVAL_MAX_MONTHS = 24

US_SITE_OFFSET_MM: tuple[float, ...] = (-0.65, 0.45, -0.25)
US_STANDARD_DEVIATION_MM = 1.10
US_PROBABILITY = 0.35

INITIAL_DIAMETER_BAND_MM: tuple[float, float] = (30.0, 54.0)
FEMALE_DIAMETER_CEILING_MM = 48.0
INITIAL_BURDEN_BAND = (0.18, 0.55)
MIN_DIAMETER_MM = 12.0
MAX_BURDEN = 0.98


@dataclass(frozen=True, slots=True)
class HistorySpec:
    horizon_months: int = 36
    min_observations: int = 3
    max_observations: int = 6
    interval_median_months: float = INTERVAL_MEDIAN_MONTHS
    interval_log_sigma: float = INTERVAL_LOG_SIGMA
    interval_min_months: int = INTERVAL_MIN_MONTHS
    interval_max_months: int = INTERVAL_MAX_MONTHS
    us_probability: float = US_PROBABILITY
    us_standard_deviation_mm: float = US_STANDARD_DEVIATION_MM
    initial_diameter_band_mm: tuple[float, float] = INITIAL_DIAMETER_BAND_MM
    initial_burden_band: tuple[float, float] = INITIAL_BURDEN_BAND
    observation_seed_stride: int = 7919

    def as_mapping(self) -> dict[str, Any]:
        return {
            "horizon_months": int(self.horizon_months),
            "min_observations": int(self.min_observations),
            "max_observations": int(self.max_observations),
            "interval_median_months": float(self.interval_median_months),
            "interval_log_sigma": float(self.interval_log_sigma),
            "interval_min_months": int(self.interval_min_months),
            "interval_max_months": int(self.interval_max_months),
            "us_probability": float(self.us_probability),
            "us_standard_deviation_mm": float(self.us_standard_deviation_mm),
            "initial_diameter_band_mm": [float(value) for value in self.initial_diameter_band_mm],
            "initial_burden_band": [float(value) for value in self.initial_burden_band],
        }


@dataclass(frozen=True, slots=True)
class Visit:
    month: float
    channel: str
    measurement: np.ndarray
    covariance: np.ndarray
    flag: int

    @property
    def dimension(self) -> int:
        return int(self.measurement.shape[0])


@dataclass(frozen=True, slots=True)
class HistorySample:
    patient: str
    sex: str
    site: int
    visits: tuple[Visit, ...]
    inputs: np.ndarray
    visit_states: np.ndarray
    visit_months: np.ndarray
    monthly_months: np.ndarray
    monthly_diameter: np.ndarray
    monthly_burden: np.ndarray
    monthly_regime: np.ndarray
    growth_offset: float
    outcome: CompositeOutcome
    surrogate_covariance: np.ndarray
    boundary_trace: np.ndarray
    observations: tuple[Observation, ...]

    @property
    def observations_count(self) -> int:
        return len(self.visits)

    def input_at(self, index: int) -> np.ndarray:
        return np.asarray(self.inputs[index], dtype=np.float64)

    def strategy_label(self) -> str:
        return f"{self.patient}:{self.observations_count}"


@dataclass(frozen=True, slots=True)
class SensorContext:
    """Everything the marginalisation needs, bundled for one generator."""

    system: ShearSystem
    lookup: ResidualLookup
    spec: MarginalisationSpec
    surrogate: PhysicsAttentionSurrogate
    prior: BoundaryPrior

    def observe(
        self,
        geometry: ScanGeometry,
        site_index: int,
        seed: int,
        rng: np.random.Generator,
    ) -> Observation:
        summary = summarise(geometry)
        return marginalise(
            geometry,
            summary,
            site_index,
            self.surrogate,
            self.prior,
            self.lookup,
            self.spec,
            seed,
            rng,
        )


def draw_visit_months(spec: HistorySpec, rng: np.random.Generator) -> np.ndarray:
    """Integer monthly intervals whose median and spread follow the cohort's own."""

    months = [0]
    while len(months) < spec.max_observations:
        raw = rng.lognormal(np.log(spec.interval_median_months), spec.interval_log_sigma)
        step = int(np.clip(round(float(raw)), spec.interval_min_months, spec.interval_max_months))
        candidate = months[-1] + step
        if candidate > spec.horizon_months:
            break
        months.append(candidate)
    if len(months) < spec.min_observations:
        months = [
            round(float(value))
            for value in np.linspace(0, spec.horizon_months, spec.min_observations)
        ]
    return np.asarray(months, dtype=np.int64)


def regime_path(
    system: ShearSystem, start: int, horizon: int, rng: np.random.Generator
) -> np.ndarray:
    path = np.empty(horizon + 1, dtype=np.int64)
    path[0] = int(start)
    current = int(start)
    transition = regime_transition(system.chain, 1.0)
    for month in range(1, horizon + 1):
        weights = transition[current] / float(np.sum(transition[current]))
        current = int(rng.choice(2, p=weights))
        path[month] = current
    return path


def _entry_band(spec: HistorySpec, sex: str) -> tuple[float, float]:
    """Records at or above the sex-specific repair threshold never enter the replay."""

    lower, upper = spec.initial_diameter_band_mm
    if sex == "female":
        upper = min(upper, FEMALE_DIAMETER_CEILING_MM)
    return (float(lower), float(upper))


def initial_state(diameter_mm: float, burden: float, system: ShearSystem) -> np.ndarray:
    state = np.zeros(system.dimension, dtype=np.float64)
    state[0] = float(diameter_mm)
    state[1] = float(burden)
    state[2] = float(burden)
    state[3] = float(np.clip(0.25 + 0.4 * burden, 0.0, 1.0))
    return state


def build_history(
    patient: str,
    sex: str,
    site_index: int,
    sensor: SensorContext,
    spec: HistorySpec,
    rng: np.random.Generator,
    uniform: float,
) -> HistorySample:
    system = sensor.system
    horizon = int(spec.horizon_months)
    visit_months = draw_visit_months(spec, rng)
    offset = float(rng.normal(0.0, 1.0))
    growth = offset * GROWTH_SPREAD_MM_PER_MONTH + GROWTH_RATE_MM_PER_MONTH

    start = int(rng.choice(2, p=system.chain.stationary()))
    regimes = regime_path(system, start, horizon, rng)
    initial_diameter = float(rng.uniform(*_entry_band(spec, sex)))
    initial_burden = float(rng.uniform(*spec.initial_burden_band))

    monthly = np.empty((horizon + 1, system.dimension), dtype=np.float64)
    monthly[0] = initial_state(initial_diameter, initial_burden, system)
    monthly[0, system.offset_index] = growth

    visit_flags = np.zeros(visit_months.shape[0], dtype=int)
    inputs = np.zeros((visit_months.shape[0], system.input_dimension), dtype=np.float64)
    observations: list[Observation] = []
    visit_cursor = 0

    active_input = _input_from_state(monthly[0])
    for month in range(1, horizon + 1):
        if (
            visit_cursor + 1 < visit_months.shape[0]
            and int(visit_months[visit_cursor + 1]) == month
        ):
            visit_cursor += 1
            active_input = _input_from_state(monthly[int(visit_months[visit_cursor - 1])])
        step = discretise_regime(system, int(regimes[month - 1]), 1.0)
        mean = step.transition @ monthly[month - 1] + step.input_gain @ active_input
        if step.covariance.any():
            noise = covariance_sqrt(step.covariance) @ rng.standard_normal(system.dimension)
        else:
            noise = np.zeros(system.dimension, dtype=np.float64)
        state = mean + noise
        state[0] = max(float(state[0]), MIN_DIAMETER_MM)
        state[1] = float(np.clip(state[1], 0.0, MAX_BURDEN))
        state[2] = max(float(state[2]), 0.0)
        state[3] = float(np.clip(state[3], 0.0, 1.0))
        state[system.offset_index] = growth
        monthly[month] = state

    for position, month in enumerate(visit_months):
        state = monthly[int(month)].copy()
        geometry = geometry_from_state(
            f"{patient}-{position:02d}", float(state[0]), float(state[1])
        )
        inputs[position] = _input_from_state(state)
        observation = sensor.observe(
            geometry,
            site_index,
            seed=position * spec.observation_seed_stride + abs(int(rng.integers(0, 10**6))),
            rng=rng,
        )
        observations.append(observation)
        visit_flags[position] = observation.flag
    if visit_months.shape[0] > 1:
        inputs[1:] = np.stack(
            [
                _input_from_state(monthly[int(visit_months[position])])
                for position in range(visit_months.shape[0] - 1)
            ]
        )

    base_surrogate = observations[0].sigma_surrogate
    offset_draw = covariance_sqrt(base_surrogate) @ rng.standard_normal(base_surrogate.shape[0])

    visits = tuple(
        _build_visit(
            float(visit_months[position]),
            monthly[int(visit_months[position])],
            observations[position],
            offset_draw,
            spec,
            site_index,
            rng,
            int(visit_flags[position]),
        )
        for position in range(visit_months.shape[0])
    )

    outcome = composite_outcome(
        np.arange(horizon + 1, dtype=np.float64),
        monthly[:, 0],
        monthly[:, 1],
        sex,
        uniform,
    )
    return HistorySample(
        patient=patient,
        sex=sex,
        site=int(site_index),
        visits=visits,
        inputs=inputs,
        visit_states=np.stack([monthly[int(month)] for month in visit_months]),
        visit_months=np.asarray(visit_months, dtype=np.float64),
        monthly_months=np.arange(horizon + 1, dtype=np.float64),
        monthly_diameter=np.asarray(monthly[:, 0], dtype=np.float64),
        monthly_burden=np.asarray(monthly[:, 1], dtype=np.float64),
        monthly_regime=np.asarray(regimes, dtype=np.int64),
        growth_offset=growth,
        outcome=outcome,
        surrogate_covariance=base_surrogate,
        boundary_trace=np.asarray([float(np.trace(o.sigma_boundary)) for o in observations]),
        observations=tuple(observations),
    )


def _build_visit(
    month: float,
    state: np.ndarray,
    observation: Observation,
    offset_draw: np.ndarray,
    spec: HistorySpec,
    site_index: int,
    rng: np.random.Generator,
    flag: int,
) -> Visit:
    signal = np.asarray([state[0], state[2] * AREA_UNIT_MM2, state[1], state[3]], dtype=np.float64)
    if rng.random() < spec.us_probability:
        bias = float(US_SITE_OFFSET_MM[int(site_index) % len(US_SITE_OFFSET_MM)])
        measurement = np.asarray(
            [state[0] + bias + rng.normal(0.0, spec.us_standard_deviation_mm)], dtype=np.float64
        )
        covariance = np.asarray([[spec.us_standard_deviation_mm**2]], dtype=np.float64)
        return Visit(
            month=month,
            channel=CHANNEL_US,
            measurement=measurement,
            covariance=covariance,
            flag=flag,
        )
    noise = covariance_sqrt(observation.covariance) @ rng.standard_normal(
        observation.covariance.shape[0]
    )
    measurement = signal + offset_draw + noise
    return Visit(
        month=month,
        channel=CHANNEL_CTA,
        measurement=measurement,
        covariance=observation.covariance,
        flag=flag,
    )


def _stabilise(covariance: np.ndarray) -> np.ndarray:
    symmetry = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetry)
    floor = float(np.min(eigenvalues))
    if floor <= 1.0e-12:
        symmetry = symmetry + (1.0e-12 - floor) * np.eye(symmetry.shape[0])
    stabilised: np.ndarray = symmetry
    return stabilised


def _input_from_state(state: np.ndarray) -> np.ndarray:
    """The scan-level geometry summaries u(g_k) that drive the exogenous drift."""

    diameter = float(state[0])
    burden = float(state[1])
    area = float(state[2])
    osi = float(state[3])
    inputs: np.ndarray = np.asarray(
        [
            (diameter - 40.0) / 20.0,
            target_ilt_fraction(burden, ShapeSpec()),
            area,
            float(np.clip(osi, 0.0, 1.0)),
        ],
        dtype=np.float64,
    )
    return inputs


def geometry_metadata(geometry: ScanGeometry, lookup: ResidualLookup) -> dict[str, float]:
    return {
        "topology_class": float(topology_class(geometry)),
        "topology_scale": float(boundary_condition_scale(geometry)),
        "residual_trace": float(
            np.trace(lookup(topology_class(geometry), boundary_condition_scale(geometry)))
        ),
    }


def patient_batch(
    count: int,
    sensor: SensorContext,
    spec: HistorySpec,
    rng: np.random.Generator,
    prefix: str = "SIM",
    women_share: float = 0.19,
) -> list[HistorySample]:
    histories: list[HistorySample] = []
    for index in range(int(count)):
        sex = "female" if rng.random() < women_share else "male"
        site = int(rng.integers(0, 3))
        uniform = float(rng.random())
        histories.append(
            build_history(
                f"{prefix}-{index:05d}",
                sex,
                site,
                sensor,
                spec,
                rng,
                uniform,
            )
        )
    return histories
