"""Minimal closed loop: the energy-score fit on a two-parameter objective.

Ref: Sec. pre-specified clinical protocol (the energy score is the criterion the
manuscript uses to select the mechanism's configuration), so a two-step fit of that
score is the smallest training loop this design has.
"""

from __future__ import annotations

import numpy as np

from bmstf.inference.forecast import ForecastSpec, monte_carlo_forecast
from bmstf.inference.imm import (
    FilterConfiguration,
    augmented_modes,
    run_filter_trace,
    two_regime_modes,
)
from bmstf.runtime.seeds import set_seed
from bmstf.scoring.energy import energy_score_samples, fit_by_energy_score
from bmstf.study.engines import corrected_visits, prior_moments


def _one_step_objective(system, history) -> object:
    def objective(parameters: np.ndarray) -> float:
        scale = float(np.exp(parameters[0]))
        configuration = FilterConfiguration(
            name="fit",
            modes=augmented_modes(two_regime_modes(system), system),
            scalar_noise=None,
        )
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
            ForecastSpec(paths=8, horizon_months=12),
            history.inputs[-1],
            history.sex,
            set_seed(97).forecast,
        )
        scores = [
            abs(
                float(
                    np.mean(forecast.diameters[:, month])
                    - history.monthly_diameter[int(history.visit_months[-1]) + month]
                )
            )
            for month in range(1, 7)
            if int(history.visit_months[-1]) + month < history.monthly_diameter.shape[0]
        ]
        return float(np.mean(scores) * scale + 1.0 / scale)

    return objective


def test_two_step_fit_reduces_the_loss(system, histories) -> None:
    objective = _one_step_objective(system, histories[0])
    report = fit_by_energy_score(objective, np.asarray([0.0]), max_iter=2)
    assert report.history[0] >= report.loss - 1.0e-12
    assert report.iterations <= 2
    assert np.isfinite(report.parameters[0])


def test_energy_score_on_the_forecast_sample_is_finite(system, histories) -> None:
    configuration = FilterConfiguration(
        name="smoke", modes=augmented_modes(two_regime_modes(system), system)
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
        ForecastSpec(paths=16, horizon_months=12),
        history.inputs[-1],
        history.sex,
        set_seed(5).forecast,
    )
    index = int(min(12, forecast.horizon_months))
    truth = np.asarray([history.monthly_diameter[0] + 0.0, history.monthly_burden[0] + 0.0])
    sample = np.stack([forecast.diameters[:, index], forecast.burdens[:, index]], axis=1)
    score = energy_score_samples(sample, truth)
    assert np.isfinite(score)
    assert score >= 0.0
