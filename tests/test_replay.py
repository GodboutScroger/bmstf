"""Visit intensity, competing-risk censoring and the IPW estimator."""

from __future__ import annotations

import numpy as np

from bmstf.replay.censoring import (
    CENSORING_FLOOR,
    apply_correction,
    censored_inside_window,
    fit_censoring,
)
from bmstf.replay.estimator import (
    ReplayVisit,
    estimate,
    intensity_from_records,
    replay_indices,
    visit_weights,
)
from bmstf.replay.intensity import (
    fit_intensity,
    intervals_from_months,
    truncated_deferral,
    weights_for_visits,
)
from bmstf.replay.landmarks import (
    LANDMARK_INDEX,
    has_minimum_history,
    inside_window,
    landmark_indices,
    landmark_summary,
    window,
)


def _visits(count: int = 40) -> list[ReplayVisit]:
    return [
        ReplayVisit(
            record=f"R{index:03d}",
            index=2,
            month=float(6 + 3 * index),
            horizon=float((3, 6, 12)[index % 3]),
            escalate=bool(index % 4 == 0),
            crossing_hit=bool(index % 7 == 0),
            event_hit=bool(index % 11 == 0),
            comparator_months=12.0,
            band=("40-45" if index % 2 else "45-55"),
            sex=("female" if index % 3 == 0 else "male"),
            repair_month=(float(20 + index) if index % 5 == 0 else None),
        )
        for index in range(count)
    ]


def test_intervals_and_intensity_are_positive() -> None:
    months = np.asarray([0.0, 8.0, 17.0, 25.0])
    intervals = intervals_from_months(months)
    assert intervals.shape == (3,)
    model = fit_intensity(intervals)
    assert model.hazard.shape[0] == 61
    assert float(np.min(model.hazard)) > 0.0
    assert np.isfinite(model.unconditional)


def test_stabilised_weights_are_positive_and_ordered() -> None:
    model = fit_intensity(np.asarray([8.0, 9.0, 10.0, 12.0]))
    months = np.asarray([0.0, 8.0, 18.0])
    gaps = np.asarray([1.0, 8.0, 10.0])
    weights = weights_for_visits(model, months, gaps)
    assert np.all(weights > 0.0)
    assert abs(float(np.mean(weights)) - 1.0) < 0.6


def test_visit_weights_shapes_the_first_gap() -> None:
    model = fit_intensity(np.asarray([7.0, 8.0, 9.0]))
    weights = visit_weights(np.asarray([0.0, 7.0, 15.0]), model)
    assert weights.shape == (3,)
    assert np.all(weights > 0.0)


def test_censoring_survival_is_non_increasing() -> None:
    model = fit_censoring(np.asarray([18.0, 24.0, 30.0, 36.0, 24.0]))
    curve = np.asarray([model.probability(float(month)) for month in range(0, 40)])
    assert np.all(np.diff(curve) <= 1.0e-12)
    assert float(np.min(curve)) >= CENSORING_FLOOR
    assert model.weight(6.0) >= 1.0


def test_competing_risk_window_detection() -> None:
    assert censored_inside_window(10.0, None, None, 6.0, 12.0)
    assert not censored_inside_window(4.0, None, None, 6.0, 12.0)
    assert not censored_inside_window(10.0, 8.0, None, 6.0, 12.0)
    assert not censored_inside_window(None, None, None, 6.0, 12.0)


def test_correction_scales_the_indicators() -> None:
    indicators = np.asarray([1.0, 0.0, 1.0])
    weights = np.asarray([2.0, 2.0, 3.0])
    adjusted = apply_correction(indicators, weights, 1.0)
    assert np.allclose(adjusted, np.asarray([2.0, 0.0, 3.0]))


def test_estimator_returns_weighted_rates_with_intervals() -> None:
    visits = _visits()
    intensity = fit_intensity(np.asarray([8.6, 9.1, 7.4, 12.0, 6.2]))
    censoring = fit_censoring(np.asarray([22.0, 28.0, 34.0]))
    estimates = estimate(
        visits, intensity, censoring, deferral_cap=0.2, resamples=200, rng=np.random.default_rng(3)
    )
    payload = estimates.as_mapping()
    assert 0.0 <= estimates.crossing_rate.estimate <= 1.0
    assert estimates.crossing_rate.lower <= estimates.crossing_rate.estimate
    assert 0.0 <= estimates.escalate_fraction <= 1.0
    assert estimates.deferral["cap"] == 0.2
    assert set(payload["by_band"]) == {"40-45", "45-55"}


def test_estimator_handles_an_empty_cohort() -> None:
    intensity = fit_intensity(np.asarray([8.0]))
    censoring = fit_censoring(np.asarray([24.0]))
    estimates = estimate([], intensity, censoring, deferral_cap=0.2, resamples=10)
    assert estimates.crossing_rate.estimate == 0.0
    assert estimates.escalate_fraction == 0.0


def test_deferral_cap_truncates() -> None:
    assert truncated_deferral(np.asarray([1.0, 1.0, 1.0]), 0.4) == 0.4
    assert abs(truncated_deferral(np.asarray([0.0, 0.0]), 0.4)) < 1.0e-12


def test_intensity_from_records_and_landmarks() -> None:
    months = [np.asarray([0.0, 8.0, 17.0]), np.asarray([0.0, 9.0, 18.0])]
    model = intensity_from_records(months)
    assert np.isfinite(model.unconditional)
    assert replay_indices(("CTA", "US", "CTA", "CTA")) == (2, 3)
    assert landmark_indices(("CTA", "US", "CTA"), start=LANDMARK_INDEX) == (2,)
    assert has_minimum_history(3)
    assert not has_minimum_history(1)
    summary = landmark_summary(np.asarray([0.0, 8.0, 17.0]), ("CTA", "US", "CTA"))
    assert len(summary) == 1
    assert summary[0].month == 17.0
    assert window(4.0, 12.0) == (4.0, 16.0)
    assert inside_window(10.0, 4.0, 12.0)
    assert not inside_window(None, 4.0, 12.0)
