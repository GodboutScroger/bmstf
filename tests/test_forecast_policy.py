"""Forecast spread scaling, the two-barrier read-out and the band comparators."""

from __future__ import annotations

import numpy as np

from bmstf.inference.forecast import ForecastSpec, monte_carlo_forecast
from bmstf.inference.imm import (
    FilterConfiguration,
    augmented_modes,
    run_filter_trace,
    two_regime_modes,
)
from bmstf.policy.bands import (
    BAND_ORDER,
    band_from_diameter,
    calibrate_band_policy,
    extrapolate_diameter,
    literal_band_policy,
    point_crossing_probability,
)
from bmstf.policy.barriers import (
    ALPHA1,
    ALPHA2,
    action_grid,
    admissible_horizons,
    read_out,
    unconditional_read_out,
)
from bmstf.policy.wilson import BOUND_LEVEL, standard_error, wilson_bounds, wilson_upper
from bmstf.study.engines import corrected_visits, prior_moments


def test_wilson_bound_solves_the_score_equation() -> None:
    from scipy.optimize import brentq

    for successes, count in ((2, 64), (17, 40), (9, 12)):
        estimate = successes / count

        def equation(p: float, estimate: float = estimate, count: int = count) -> float:
            return (estimate - p) / float(np.sqrt(p * (1.0 - p) / count)) + BOUND_LEVEL

        root = brentq(equation, 1.0e-9, 1.0 - 1.0e-9, xtol=1.0e-14)
        assert abs(wilson_upper(estimate, count) - root) < 1.0e-11
        bounds = wilson_bounds(estimate, count)
        assert bounds.lower <= estimate <= bounds.upper
        assert (
            abs(standard_error(estimate, count) - np.sqrt(estimate * (1 - estimate) / count))
            < 1.0e-15
        )


def test_wilson_bounds_handle_the_degenerate_counts() -> None:
    assert wilson_bounds(0.0, 0).upper == 1.0
    assert wilson_bounds(0.0, 32).upper > 0.0
    assert wilson_bounds(1.0, 32).lower < 1.0


def test_action_grid_is_the_declared_one() -> None:
    grid = action_grid()
    assert grid.shape == (8,)
    assert float(grid[0]) == 3.0
    assert float(grid[-1]) == 36.0


def test_read_out_returns_the_longest_admissible_interval() -> None:
    grid = action_grid()
    crossing = np.full(grid.shape[0], 0.01)
    event = np.full(grid.shape[0], 0.001)
    decision = read_out(
        crossing, event, grid, paths=4096, gate_month=36.0, alpha1=ALPHA1, alpha2=ALPHA2
    )
    assert not decision.escalate
    assert decision.horizon_months == float(grid[-1])
    assert decision.admissible


def test_read_out_escalates_when_nothing_is_admissible() -> None:
    grid = action_grid()
    crossing = np.full(grid.shape[0], 0.5)
    event = np.full(grid.shape[0], 0.4)
    decision = read_out(crossing, event, grid, paths=512, gate_month=36.0)
    assert decision.escalate
    assert decision.horizon_months == float(grid[0])
    assert not decision.admissible


def test_gate_shortens_the_interval() -> None:
    grid = action_grid()
    crossing = np.full(grid.shape[0], 0.0)
    event = np.full(grid.shape[0], 0.0)
    decision = read_out(crossing, event, grid, paths=512, gate_month=6.0)
    assert decision.horizon_months == 6.0


def test_single_barrier_admits_more() -> None:
    grid = action_grid()
    crossing = np.full(grid.shape[0], 0.02)
    event = np.full(grid.shape[0], 0.2)
    two = read_out(crossing, event, grid, paths=2048, gate_month=36.0)
    one = unconditional_read_out(crossing, event, grid, paths=2048, gate_month=36.0)
    assert one.horizon_months >= two.horizon_months


def test_admissible_horizons_is_a_subset_of_the_grid() -> None:
    grid = action_grid()
    admissible = admissible_horizons(
        np.full(grid.shape[0], 0.01), np.full(grid.shape[0], 0.001), grid, paths=1024
    )
    assert set(admissible).issubset({float(value) for value in grid})


def test_literal_band_schedule_covers_every_band() -> None:
    policy = literal_band_policy()
    assert len(policy.schedule) == len(BAND_ORDER)
    assert policy.escalated == ()
    assert policy.interval(35.0) == 36.0
    assert policy.interval(42.0) == 12.0
    assert policy.interval(47.0) == 3.0
    assert policy.interval(58.0) is None
    assert policy.is_escalated(58.0)
    assert policy.interval(47.0) is not None


def test_band_labels_are_contiguous() -> None:
    assert band_from_diameter(31.0) == "30-40"
    assert band_from_diameter(41.0) == "40-45"
    assert band_from_diameter(47.0) == "45-55"
    assert band_from_diameter(60.0) == "55-70"


def test_band_calibration_escalates_unreachable_bands() -> None:
    labels = np.asarray(["30-40"] * 400 + ["40-45"] * 400)
    within = np.zeros((800, action_grid().shape[0]), dtype=bool)
    within[labels == "30-40", :] = True
    policy = calibrate_band_policy(labels, within, action_grid(), alpha1=ALPHA1)
    assert "30-40" in policy.escalated
    assert policy.schedule.get("40-45") == float(action_grid()[-1])


def test_extrapolation_is_linear_and_thresholded() -> None:
    months = np.asarray([0.0, 6.0, 12.0])
    diameters = np.asarray([40.0, 41.0, 42.0])
    assert abs(extrapolate_diameter(months, diameters, 24.0) - 44.0) < 1.0e-12
    assert point_crossing_probability(months, diameters, 24.0, 55.0) == 0.0
    assert point_crossing_probability(months, diameters, 24.0, 43.0) == 1.0


def test_spread_factor_widens_the_forecast(system, histories) -> None:
    narrow = FilterConfiguration(
        name="narrow", modes=augmented_modes(two_regime_modes(system), system), spread_factor=0.5
    )
    wide = FilterConfiguration(
        name="wide", modes=augmented_modes(two_regime_modes(system), system), spread_factor=2.0
    )
    history = histories[0]
    prior_mean, prior_covariance = prior_moments(system, narrow, history)
    trace = run_filter_trace(
        system,
        narrow,
        prior_mean,
        prior_covariance,
        history.visit_months,
        history.inputs,
        corrected_visits(history),
    )
    state = trace.state_at(len(history.visits) - 1)
    spec = ForecastSpec(paths=64, horizon_months=12)
    first = monte_carlo_forecast(
        state, system, narrow, spec, history.inputs[-1], history.sex, np.random.default_rng(4)
    )
    second = monte_carlo_forecast(
        state, system, wide, spec, history.inputs[-1], history.sex, np.random.default_rng(4)
    )
    narrow_width = float(first.diameter_band(12.0)[-1] - first.diameter_band(12.0)[0])
    wide_width = float(second.diameter_band(12.0)[-1] - second.diameter_band(12.0)[0])
    assert wide_width > narrow_width
