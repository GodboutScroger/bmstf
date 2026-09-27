"""Two-regime interacting-multiple-model recursion.

Ref: Algorithm 2 lines 3-15; Eq. (2)-(3); Sec. "Two-regime latent shear-state
model, filtering and forecasting". Mode sets also express the pre-specified
ablations: a single regime carrying the occupancy-weighted population drift (the
fair ablation), the quiescent drift with doubled process noise (the first
version's hedge), and the tuned scalar noise that replaces the decomposition
with the bias coordinates dropped (Table 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.inference.kalman import (
    KalmanUpdate,
    bias_augmented_operator,
    diameter_operator,
    kalman_update,
)
from bmstf.inference.mixing import collapsed_state, mix_components
from bmstf.process.cohort import CHANNEL_US, Visit
from bmstf.process.regimes import ShearSystem
from bmstf.process.transition import continuous_transition, discretise

SINGLE_REGIME_POPULATION = "population"
SINGLE_REGIME_QUIESCENT = "quiescent"


@dataclass(frozen=True, slots=True)
class ModeConfiguration:
    name: str
    drifts: tuple[np.ndarray, ...]
    inputs: tuple[np.ndarray, ...]
    noises: tuple[np.ndarray, ...]
    generator: np.ndarray
    initial_weights: np.ndarray

    @property
    def modes(self) -> int:
        return len(self.drifts)

    def transition(self, delta: float) -> np.ndarray:
        return continuous_transition(self.generator, delta)


@dataclass(frozen=True, slots=True)
class FilterConfiguration:
    name: str
    modes: ModeConfiguration
    use_ultrasound: bool = True
    bias_augmented: bool = True
    scalar_noise: np.ndarray | None = None
    spread_factor: float = 1.0
    use_event_barrier: bool = True
    hemodynamic_channels: bool = True
    channels: int = 4


@dataclass(frozen=True, slots=True)
class FilterTrace:
    """Per-step moments in visit order: collapsed for reading out, per-mode for forecasting."""

    predicted_means: np.ndarray
    predicted_covariances: np.ndarray
    updated_means: np.ndarray
    updated_covariances: np.ndarray
    mode_means: np.ndarray
    mode_covariances: np.ndarray
    mode_weights: np.ndarray
    active_probabilities: np.ndarray
    state: IMMState

    def state_at(self, index: int) -> IMMState:
        return IMMState(
            means=self.mode_means[index],
            covariances=self.mode_covariances[index],
            weights=self.mode_weights[index],
            steps=index,
            log_likelihood=0.0,
            mode_switches=0,
        )

    def as_mapping(self) -> dict[str, float]:
        return {
            "visits": float(self.predicted_means.shape[0]),
            "final_active_probability": float(self.active_probabilities[-1]),
        }


@dataclass(frozen=True, slots=True)
class IMMState:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    steps: int
    log_likelihood: float
    mode_switches: int

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    def collapsed(self) -> tuple[np.ndarray, np.ndarray]:
        return collapsed_state(self.means, self.covariances, self.weights)

    def active_probability(self) -> float:
        if self.weights.shape[0] < 2:
            return float(self.weights[0])
        return float(self.weights[1])

    def as_mapping(self) -> dict[str, float]:
        mean, covariance = self.collapsed()
        return {
            "steps": float(self.steps),
            "active_probability": float(self.active_probability()),
            "log_likelihood": float(self.log_likelihood),
            "diameter_mm": float(mean[0]),
            "burden": float(mean[1]),
            "covariance_trace": float(np.trace(covariance)),
            "mode_switches": float(self.mode_switches),
        }


def two_regime_modes(system: ShearSystem) -> ModeConfiguration:
    return ModeConfiguration(
        name="two_regime",
        drifts=(system.drift[0], system.drift[1]),
        inputs=(system.input_matrix[0], system.input_matrix[1]),
        noises=(system.noise[0], system.noise[1]),
        generator=system.chain.generator_matrix(),
        initial_weights=system.chain.stationary(),
    )


def single_regime_modes(
    system: ShearSystem, weighting: str = SINGLE_REGIME_POPULATION, noise_factor: float = 1.0
) -> ModeConfiguration:
    stationary = system.chain.stationary()
    if weighting == SINGLE_REGIME_POPULATION:
        drift = stationary[0] * system.drift[0] + stationary[1] * system.drift[1]
        inputs = stationary[0] * system.input_matrix[0] + stationary[1] * system.input_matrix[1]
        noise = stationary[0] * system.noise[0] + stationary[1] * system.noise[1]
        name = "single_regime_population"
    else:
        drift = system.drift[0].copy()
        inputs = system.input_matrix[0].copy()
        noise = system.noise[0] * float(noise_factor)
        name = "single_regime_quiescent"
    return ModeConfiguration(
        name=name,
        drifts=(drift,),
        inputs=(inputs,),
        noises=(noise,),
        generator=np.zeros((1, 1), dtype=np.float64),
        initial_weights=np.ones(1, dtype=np.float64),
    )


def augmented_modes(modes: ModeConfiguration, system: ShearSystem) -> ModeConfiguration:
    """Append the patient-persistent bias coordinates nu_i with H~ = [H, I]."""

    channels = system.observation_dimension
    dimension = system.dimension
    drifts = []
    inputs = []
    noises = []
    for index in range(modes.modes):
        drift = np.zeros((dimension + channels, dimension + channels), dtype=np.float64)
        drift[:dimension, :dimension] = modes.drifts[index]
        gain = np.zeros((dimension + channels, modes.inputs[index].shape[1]), dtype=np.float64)
        gain[:dimension] = modes.inputs[index]
        noise = np.zeros((dimension + channels, dimension + channels), dtype=np.float64)
        noise[:dimension, :dimension] = modes.noises[index]
        drifts.append(drift)
        inputs.append(gain)
        noises.append(noise)
    return ModeConfiguration(
        name=f"{modes.name}_augmented",
        drifts=tuple(drifts),
        inputs=tuple(inputs),
        noises=tuple(noises),
        generator=modes.generator,
        initial_weights=modes.initial_weights,
    )


def state_dimension(system: ShearSystem, configuration: FilterConfiguration) -> int:
    extra = system.observation_dimension if configuration.bias_augmented else 0
    return system.dimension + extra


def observation_operator(system: ShearSystem, configuration: FilterConfiguration) -> np.ndarray:
    if configuration.bias_augmented:
        return bias_augmented_operator(system.observation, system.observation_dimension)
    return np.asarray(system.observation, dtype=np.float64)


def initial_state(
    system: ShearSystem,
    configuration: FilterConfiguration,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
) -> IMMState:
    means = np.tile(
        np.asarray(prior_mean, dtype=np.float64)[None, :], (configuration.modes.modes, 1)
    )
    covariances = np.tile(
        np.asarray(prior_covariance, dtype=np.float64)[None, :, :],
        (configuration.modes.modes, 1, 1),
    )
    weights = np.asarray(configuration.modes.initial_weights, dtype=np.float64).copy()
    if weights.shape[0] != configuration.modes.modes:
        weights = np.full(configuration.modes.modes, 1.0 / float(configuration.modes.modes))
    return IMMState(
        means=means,
        covariances=covariances,
        weights=weights,
        steps=0,
        log_likelihood=0.0,
        mode_switches=0,
    )


def measurement_model(
    visit: Visit,
    system: ShearSystem,
    configuration: FilterConfiguration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (operator, measurement, noise) for the visit, or None to skip it."""

    dimension = state_dimension(system, configuration)
    if visit.channel == CHANNEL_US and not configuration.use_ultrasound:
        return None
    if visit.channel == CHANNEL_US or visit.flag == 1 or not configuration.hemodynamic_channels:
        return (
            diameter_operator(dimension),
            np.asarray([float(visit.measurement[0])], dtype=np.float64),
            np.asarray([[float(visit.covariance[0, 0])]], dtype=np.float64),
        )
    noise = np.asarray(visit.covariance, dtype=np.float64)
    if configuration.scalar_noise is not None:
        diagonal = np.asarray(configuration.scalar_noise, dtype=np.float64).reshape(-1)
        noise = np.diag(diagonal**2)
    return (
        observation_operator(system, configuration),
        np.asarray(visit.measurement, dtype=np.float64),
        noise,
    )


def imm_step(
    state: IMMState,
    system: ShearSystem,
    configuration: FilterConfiguration,
    delta: float,
    exogenous: np.ndarray,
    visit: Visit,
) -> IMMState:
    modes = configuration.modes
    mixed_means, mixed_covariances, prior_weights = mix_components(
        state.means, state.covariances, state.weights, modes.transition(delta)
    )
    updated_means = np.empty_like(mixed_means)
    updated_covariances = np.empty_like(mixed_covariances)
    likelihoods = np.zeros(modes.modes, dtype=np.float64)
    model = measurement_model(visit, system, configuration)
    for index in range(modes.modes):
        step = discretise(modes.drifts[index], modes.inputs[index], modes.noises[index], delta)
        mean = step.transition @ mixed_means[index] + step.input_gain @ exogenous
        covariance = (
            step.transition @ mixed_covariances[index] @ step.transition.T + step.covariance
        )
        if model is None:
            updated_means[index] = mean
            updated_covariances[index] = 0.5 * (covariance + covariance.T)
            continue
        operator, measurement, noise = model
        update: KalmanUpdate = kalman_update(mean, covariance, operator, measurement, noise)
        updated_means[index] = update.mean
        updated_covariances[index] = update.covariance
        likelihoods[index] = update.log_likelihood
    shifted = likelihoods - float(np.max(likelihoods))
    weights = prior_weights * np.exp(shifted)
    total = float(np.sum(weights))
    weights = weights / total if total > 0.0 else prior_weights
    switches = state.mode_switches
    if modes.modes > 1 and int(np.argmax(weights)) != int(np.argmax(state.weights)):
        switches += 1
    return IMMState(
        means=updated_means,
        covariances=updated_covariances,
        weights=weights,
        steps=state.steps + 1,
        log_likelihood=state.log_likelihood + float(np.max(likelihoods)),
        mode_switches=switches,
    )


def run_filter(
    system: ShearSystem,
    configuration: FilterConfiguration,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    months: np.ndarray,
    exogenous: np.ndarray,
    visits: tuple[Visit, ...],
) -> IMMState:
    """Propagate over the record's irregular intervals, updating at each visit."""

    return run_filter_trace(
        system, configuration, prior_mean, prior_covariance, months, exogenous, visits
    ).state


def run_filter_trace(
    system: ShearSystem,
    configuration: FilterConfiguration,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    months: np.ndarray,
    exogenous: np.ndarray,
    visits: tuple[Visit, ...],
) -> FilterTrace:
    """As run_filter, also returning the per-step predicted and updated moments."""

    state = initial_state(system, configuration, prior_mean, prior_covariance)
    times = np.asarray(months, dtype=np.float64)
    count = len(visits)
    dimension = state.dimension
    modes_count = configuration.modes.modes
    predicted_means = np.zeros((count, dimension), dtype=np.float64)
    predicted_covariances = np.zeros((count, dimension, dimension), dtype=np.float64)
    updated_means = np.zeros((count, dimension), dtype=np.float64)
    updated_covariances = np.zeros((count, dimension, dimension), dtype=np.float64)
    mode_means = np.zeros((count, modes_count, dimension), dtype=np.float64)
    mode_covariances = np.zeros((count, modes_count, dimension, dimension), dtype=np.float64)
    mode_weights = np.zeros((count, modes_count), dtype=np.float64)
    probabilities = np.zeros(count, dtype=np.float64)
    for index, visit in enumerate(visits):
        delta = 0.0 if index == 0 else float(times[index] - times[index - 1])
        propagated = propagate_without_observation(
            state, system, configuration, delta, exogenous[index]
        )
        predicted_means[index], predicted_covariances[index] = propagated.collapsed()
        state = imm_step(state, system, configuration, delta, exogenous[index], visit)
        updated_means[index], updated_covariances[index] = state.collapsed()
        mode_means[index] = state.means
        mode_covariances[index] = state.covariances
        mode_weights[index] = state.weights
        probabilities[index] = state.active_probability()
    return FilterTrace(
        predicted_means=predicted_means,
        predicted_covariances=predicted_covariances,
        updated_means=updated_means,
        updated_covariances=updated_covariances,
        mode_means=mode_means,
        mode_covariances=mode_covariances,
        mode_weights=mode_weights,
        active_probabilities=probabilities,
        state=state,
    )


def propagate_without_observation(
    state: IMMState,
    system: ShearSystem,
    configuration: FilterConfiguration,
    delta: float,
    exogenous: np.ndarray,
) -> IMMState:
    """Mean-and-covariance propagation with no measurement, used by the forecast."""

    modes = configuration.modes
    transition = modes.transition(delta)
    mixed_means, mixed_covariances, prior_weights = mix_components(
        state.means, state.covariances, state.weights, transition
    )
    means = np.empty_like(mixed_means)
    covariances = np.empty_like(mixed_covariances)
    for index in range(modes.modes):
        step = discretise(modes.drifts[index], modes.inputs[index], modes.noises[index], delta)
        means[index] = step.transition @ mixed_means[index] + step.input_gain @ exogenous
        covariances[index] = (
            step.transition @ mixed_covariances[index] @ step.transition.T + step.covariance
        )
    return IMMState(
        means=means,
        covariances=covariances,
        weights=prior_weights,
        steps=state.steps,
        log_likelihood=state.log_likelihood,
        mode_switches=state.mode_switches,
    )
