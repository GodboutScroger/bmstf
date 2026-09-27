"""Kalman update, mode mixing, the IMM recursion and the smoother."""

from __future__ import annotations

import numpy as np

from bmstf.inference.forecast import ForecastSpec, monte_carlo_forecast
from bmstf.inference.imm import (
    FilterConfiguration,
    augmented_modes,
    imm_step,
    initial_state,
    propagate_without_observation,
    run_filter_trace,
    single_regime_modes,
    state_dimension,
    two_regime_modes,
)
from bmstf.inference.kalman import (
    bias_augmented_operator,
    diameter_operator,
    gaussian_log_likelihood,
    kalman_update,
    standard_form_covariance,
)
from bmstf.inference.mixing import (
    collapsed_state,
    conditional_mixing,
    mix_components,
    mixing_weights,
    moment_match,
)
from bmstf.process.smooth import rts_smoother
from bmstf.process.transition import discretise
from bmstf.study.engines import corrected_visits, prior_moments


def test_kalman_update_matches_the_short_form() -> None:
    rng = np.random.default_rng(11)
    dimension = 5
    mean = rng.normal(size=dimension)
    root = rng.normal(size=(dimension, dimension))
    covariance = root @ root.T + np.eye(dimension) * 0.1
    operator = rng.normal(size=(3, dimension))
    measurement = rng.normal(size=3)
    noise = np.eye(3) * 0.05
    update = kalman_update(mean, covariance, operator, measurement, noise)
    assert (
        np.max(np.abs(update.covariance - standard_form_covariance(covariance, operator, noise)))
        < 1.0e-10
    )
    manual = covariance @ operator.T @ np.linalg.inv(operator @ covariance @ operator.T + noise)
    assert np.max(np.abs(update.gain - manual)) < 1.0e-10
    assert np.all(np.isfinite(update.mean))


def test_gaussian_log_likelihood_is_finite() -> None:
    value = gaussian_log_likelihood(np.asarray([0.1, -0.2]), np.eye(2) * 0.5)
    assert np.isfinite(value)
    assert value < 0.0


def test_bias_augmented_operator_reads_signal_and_bias() -> None:
    observation = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    operator = bias_augmented_operator(observation, 2)
    assert operator.shape == (2, 4)
    assert np.allclose(operator[:, :2], observation)
    assert np.allclose(operator[:, 2:], np.eye(2))
    assert diameter_operator(4).shape == (1, 4)


def test_moment_match_preserves_two_moments() -> None:
    means = np.asarray([[0.0, 1.0], [2.0, -1.0]])
    covariances = np.asarray([np.eye(2) * 0.5, np.eye(2) * 0.2])
    weights = np.asarray([0.3, 0.7])
    mean, covariance = moment_match(means, covariances, weights)
    assert np.allclose(mean, 0.3 * means[0] + 0.7 * means[1])
    expected = 0.3 * (covariances[0] + np.outer(means[0] - mean, means[0] - mean))
    expected += 0.7 * (covariances[1] + np.outer(means[1] - mean, means[1] - mean))
    assert np.max(np.abs(covariance - expected)) < 1.0e-12


def test_collapse_degenerates_to_the_single_component() -> None:
    mean = np.asarray([0.5, -0.5])
    covariance = np.asarray([[0.2, 0.01], [0.01, 0.3]])
    collapsed_mean, collapsed_covariance = collapsed_state(
        mean[None, :], covariance[None, :, :], np.asarray([1.0])
    )
    assert np.allclose(collapsed_mean, mean)
    assert np.allclose(collapsed_covariance, covariance)


def test_conditional_mixing_rows_are_distributions() -> None:
    transition = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    weights = np.asarray([0.6, 0.4])
    mix = conditional_mixing(transition, weights)
    assert np.allclose(np.sum(mix, axis=1), 1.0)
    assert np.all(mix >= 0.0)
    prior = mixing_weights(transition, weights)
    assert np.allclose(prior, transition.T @ weights)


def test_mix_components_keeps_the_mixture_moments() -> None:
    means = np.asarray([[0.0], [3.0]])
    covariances = np.asarray([np.eye(1) * 0.1, np.eye(1) * 0.4])
    weights = np.asarray([0.5, 0.5])
    transition = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    mixed_means, _, prior = mix_components(means, covariances, weights, transition)
    _, collapsed_covariance = collapsed_state(means, covariances, weights)
    assert np.allclose(prior, transition.T @ weights)
    assert mixed_means.shape == means.shape
    assert float(collapsed_covariance[0, 0]) > float(np.max(covariances))


def test_imm_step_is_finite_and_normalised(system, histories, forecast_spec) -> None:
    configuration = FilterConfiguration(
        name="test", modes=augmented_modes(two_regime_modes(system), system)
    )
    history = histories[0]
    prior_mean, prior_covariance = prior_moments(system, configuration, history)
    state = initial_state(system, configuration, prior_mean, prior_covariance)
    visits = corrected_visits(history)
    state = imm_step(state, system, configuration, 0.0, history.inputs[0], visits[0])
    state = imm_step(
        state,
        system,
        configuration,
        float(history.visit_months[1] - history.visit_months[0]),
        history.inputs[1],
        visits[1],
    )
    assert abs(float(np.sum(state.weights)) - 1.0) < 1.0e-12
    assert np.all(np.isfinite(state.means))
    assert np.all(np.isfinite(state.covariances))


def test_single_regime_configuration_has_one_mode(system) -> None:
    population = single_regime_modes(system, "population")
    quiescent = single_regime_modes(system, "quiescent", noise_factor=2.0)
    assert population.modes == 1 and quiescent.modes == 1
    assert np.allclose(quiescent.noises[0][1, 1], 2.0 * system.noise[0][1, 1])
    assert np.trace(quiescent.drifts[0]) == np.trace(system.drift[0])


def test_state_dimension_tracks_the_augmentation(system) -> None:
    plain = FilterConfiguration(name="plain", modes=two_regime_modes(system), bias_augmented=False)
    augmented = FilterConfiguration(
        name="augmented", modes=augmented_modes(two_regime_modes(system), system)
    )
    assert state_dimension(system, plain) == system.dimension
    assert state_dimension(system, augmented) == system.dimension + system.observation_dimension


def test_filter_trace_follows_the_truth_better_than_the_readings(system, histories) -> None:
    configuration = FilterConfiguration(
        name="track", modes=augmented_modes(two_regime_modes(system), system)
    )
    history = histories[1]
    prior_mean, prior_covariance = prior_moments(system, configuration, history)
    trace = run_filter_trace(
        system,
        configuration,
        prior_mean,
        prior_covariance,
        history.visit_months,
        history.inputs,
        corrected_visits(history),
    )
    truth = np.asarray([history.monthly_diameter[int(month)] for month in history.visit_months])
    filtered = float(np.mean(np.abs(trace.updated_means[:, 0] - truth)))
    raw = float(
        np.mean(np.abs(np.asarray([float(v.measurement[0]) for v in history.visits]) - truth))
    )
    assert filtered < raw
    assert trace.mode_means.shape[1] == configuration.modes.modes


def test_propagation_without_observation_grows_the_covariance(system, histories) -> None:
    configuration = FilterConfiguration(
        name="prop", modes=augmented_modes(two_regime_modes(system), system)
    )
    history = histories[0]
    prior_mean, prior_covariance = prior_moments(system, configuration, history)
    state = initial_state(system, configuration, prior_mean, prior_covariance)
    propagated = propagate_without_observation(state, system, configuration, 6.0, history.inputs[0])
    _, covariance = propagated.collapsed()
    assert float(np.trace(covariance)) > float(np.trace(state.collapsed()[1]))


def test_rts_smoother_runs_backwards() -> None:
    drift = np.asarray([[-0.5]])
    transition = discretise(drift, np.zeros((1, 1)), np.eye(1) * 0.1, 1.0)
    steps = [transition, transition]
    filtered_means = np.asarray([[0.0], [1.0], [2.0]])
    filtered_covariances = np.asarray([np.eye(1) * 0.5, np.eye(1) * 0.4, np.eye(1) * 0.3])
    predicted_means = np.asarray([[0.0], [0.0], [0.5]])
    predicted_covariances = np.asarray([np.eye(1) * 0.6, np.eye(1) * 0.5, np.eye(1) * 0.4])
    pass_result = rts_smoother(
        filtered_means, filtered_covariances, predicted_means, predicted_covariances, steps
    )
    assert pass_result.means.shape == (3, 1)
    assert (
        float(pass_result.covariances[-1][0, 0]) <= float(filtered_covariances[-1][0, 0]) + 1.0e-12
    )


def test_forecast_is_monotone_and_bounded(system, histories) -> None:
    configuration = FilterConfiguration(
        name="fc", modes=augmented_modes(two_regime_modes(system), system)
    )
    history = histories[2]
    prior_mean, prior_covariance = prior_moments(system, configuration, history)
    trace = run_filter_trace(
        system,
        configuration,
        prior_mean,
        prior_covariance,
        history.visit_months,
        history.inputs,
        corrected_visits(history),
    )
    forecast = monte_carlo_forecast(
        trace.state_at(len(history.visits) - 1),
        system,
        configuration,
        ForecastSpec(paths=16, horizon_months=24),
        history.inputs[-1],
        history.sex,
        np.random.default_rng(9),
    )
    assert np.all(np.diff(forecast.crossing_curve) >= -1.0e-12)
    assert np.all(np.diff(forecast.event_curve) >= -1.0e-12)
    assert float(forecast.crossing_curve[-1]) <= 1.0
    assert forecast.diameter_bands.shape[0] == 25
    assert forecast.diameter_bands.shape[1] == 3
