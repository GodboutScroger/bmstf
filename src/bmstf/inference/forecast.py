"""Monte-Carlo horizon forecast with a Rao-Blackwellised event law.

Ref: Eq. (4) (the two-component predictive mixture) and Algorithm 2 lines 16-21:
draw a regime path from Pi and a state path from N(z~_K, P_K) under Eq. (3);
c1_n(h) = 1{D crosses D_thr by h}; c2_n(h) = 1 - prod_{s<=h}(1 - chi(D_s, B_s))
along the path, frozen once the path has crossed; F^1 and F^2 are the path means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.inference.imm import FilterConfiguration, IMMState
from bmstf.process.hazards import repair_threshold_mm, step_hazard_array
from bmstf.process.regimes import ShearSystem
from bmstf.process.transition import covariance_sqrt, discretise

FORECAST_PATHS = 256
FORECAST_HORIZON_MONTHS = 36
QUANTILES: tuple[float, ...] = (0.05, 0.50, 0.95)
QUANTILE_HORIZONS: tuple[int, ...] = (6, 12, 24)


@dataclass(frozen=True, slots=True)
class ForecastSpec:
    paths: int = FORECAST_PATHS
    horizon_months: int = FORECAST_HORIZON_MONTHS
    quantiles: tuple[float, ...] = QUANTILES
    quantile_horizons: tuple[int, ...] = QUANTILE_HORIZONS


@dataclass(frozen=True, slots=True)
class Forecast:
    crossing_curve: np.ndarray
    event_curve: np.ndarray
    diameter_bands: np.ndarray
    burden_bands: np.ndarray
    diameters: np.ndarray
    burdens: np.ndarray
    regime_weights: np.ndarray
    paths: int
    spread_factor: float
    horizon_months: int
    quantiles: tuple[float, ...] = QUANTILES

    def at_month(self, month: float) -> int:
        return int(np.clip(round(float(month)), 1, self.horizon_months)) - 1

    def band_index(self, month: float) -> int:
        return int(np.clip(round(float(month)), 0, self.horizon_months))

    def crossing_at(self, month: float) -> float:
        return float(self.crossing_curve[self.at_month(month)])

    def event_at(self, month: float) -> float:
        return float(self.event_curve[self.at_month(month)])

    def diameter_band(self, month: float) -> np.ndarray:
        return np.asarray(self.diameter_bands[self.band_index(month)], dtype=np.float64)

    def burden_band(self, month: float) -> np.ndarray:
        return np.asarray(self.burden_bands[self.band_index(month)], dtype=np.float64)

    def covers_diameter(self, month: float, truth: float, level_index: int = 0) -> float:
        band = self.diameter_band(month)
        return 1.0 if band[level_index] <= truth <= band[-1 - level_index] else 0.0

    def covers_burden(self, month: float, truth: float, level_index: int = 0) -> float:
        band = self.burden_band(month)
        return 1.0 if band[level_index] <= truth <= band[-1 - level_index] else 0.0

    def width(self, month: float) -> tuple[float, float]:
        return (
            float(self.diameter_band(month)[-1] - self.diameter_band(month)[0]),
            float(self.burden_band(month)[-1] - self.burden_band(month)[0]),
        )

    def as_mapping(self) -> dict[str, float]:
        return {
            "crossing_12": self.crossing_at(12.0),
            "event_12": self.event_at(12.0),
            "crossing_24": self.crossing_at(24.0),
            "event_24": self.event_at(24.0),
            "paths": float(self.paths),
            "active_weight": (
                float(self.regime_weights[-1]) if self.regime_weights.size > 1 else 1.0
            ),
        }


def _mode_operators(
    system: ShearSystem, configuration: FilterConfiguration
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    operators = []
    for index in range(configuration.modes.modes):
        step = discretise(
            configuration.modes.drifts[index],
            configuration.modes.inputs[index],
            configuration.modes.noises[index],
            1.0,
        )
        root = covariance_sqrt(step.covariance)
        operators.append((step.transition, step.input_gain, root))
    return operators


def monte_carlo_forecast(
    state: IMMState,
    system: ShearSystem,
    configuration: FilterConfiguration,
    spec: ForecastSpec,
    exogenous: np.ndarray,
    sex: str,
    rng: np.random.Generator,
) -> Forecast:
    mean, covariance = state.collapsed()
    root = covariance_sqrt(covariance)
    factor = float(configuration.spread_factor)
    deviations = (root @ rng.standard_normal((mean.shape[0], spec.paths))).T
    paths = mean[None, :] + factor * deviations
    modes = np.asarray(
        rng.choice(
            configuration.modes.modes,
            size=spec.paths,
            p=np.asarray(state.weights, dtype=np.float64) / float(np.sum(state.weights)),
        ),
        dtype=np.int64,
    )
    operators = _mode_operators(system, configuration)
    transitions = configuration.modes.transition(1.0)

    horizon = int(spec.horizon_months)
    diameter = np.empty((spec.paths, horizon + 1), dtype=np.float64)
    burden = np.empty((spec.paths, horizon + 1), dtype=np.float64)
    diameter[:, 0] = paths[:, 0]
    burden[:, 0] = paths[:, 1]
    for month in range(1, horizon + 1):
        if configuration.modes.modes > 1:
            weights = transitions[modes]
            modes = np.asarray(
                [int(rng.choice(2, p=row / float(np.sum(row)))) for row in weights], dtype=np.int64
            )
        for index in range(configuration.modes.modes):
            selected = modes == index
            if not bool(np.any(selected)):
                continue
            transition, gain, noise_root = operators[index]
            propagated = paths[selected] @ transition.T + gain @ exogenous
            propagated = (
                propagated
                + rng.standard_normal((int(np.count_nonzero(selected)), mean.shape[0]))
                @ noise_root.T
            )
            paths[selected] = propagated
        diameter[:, month] = paths[:, 0]
        burden[:, month] = paths[:, 1]

    threshold = repair_threshold_mm(sex)
    crossing = np.maximum.accumulate(diameter >= threshold, axis=1).astype(np.float64)
    hazards = step_hazard_array(diameter, burden, sex)
    survival = np.cumprod(1.0 - hazards, axis=1)
    event = 1.0 - survival

    quantile_grid = np.asarray(spec.quantiles, dtype=np.float64)
    diameter_bands = np.quantile(diameter, quantile_grid, axis=0).T
    burden_bands = np.quantile(burden, quantile_grid, axis=0).T
    return Forecast(
        crossing_curve=np.asarray(np.mean(crossing[:, 1:], axis=0), dtype=np.float64),
        event_curve=np.asarray(np.mean(event[:, 1:], axis=0), dtype=np.float64),
        diameter_bands=np.asarray(diameter_bands, dtype=np.float64),
        burden_bands=np.asarray(burden_bands, dtype=np.float64),
        diameters=np.asarray(diameter, dtype=np.float64),
        burdens=np.asarray(burden, dtype=np.float64),
        regime_weights=np.asarray(state.weights, dtype=np.float64),
        paths=int(spec.paths),
        spread_factor=factor,
        horizon_months=horizon,
        quantiles=tuple(float(value) for value in quantile_grid),
    )


def trajectory_forecast(
    mean: np.ndarray,
    covariance: np.ndarray,
    system: ShearSystem,
    configuration: FilterConfiguration,
    spec: ForecastSpec,
    exogenous: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic plus noise-free mixture trajectory, for the quantile read-out."""

    operators = _mode_operators(system, configuration)
    horizon = int(spec.horizon_months)
    paths = np.tile(np.asarray(mean, dtype=np.float64)[None, :], (spec.paths, 1))
    deviations = covariance_sqrt(covariance) @ rng.standard_normal((mean.shape[0], spec.paths))
    paths = paths + float(configuration.spread_factor) * deviations.T
    diameters = np.empty((spec.paths, horizon + 1), dtype=np.float64)
    burdens = np.empty_like(diameters)
    diameters[:, 0] = paths[:, 0]
    burdens[:, 0] = paths[:, 1]
    for month in range(1, horizon + 1):
        transition, gain, noise_root = operators[0]
        paths = paths @ transition.T + gain @ exogenous
        paths = paths + rng.standard_normal((spec.paths, mean.shape[0])) @ noise_root.T
        diameters[:, month] = paths[:, 0]
        burdens[:, month] = paths[:, 1]
    return diameters, burdens
