"""Proper scores, calibration read-outs and discrimination."""

from __future__ import annotations

import numpy as np

from bmstf.scoring.calibration import (
    coverage,
    coverage_gap,
    expected_calibration_error,
    reliability_curve,
)
from bmstf.scoring.crps import (
    absolute_expectation,
    crps_gaussian,
    crps_mixture_gaussian,
    crps_sample,
    root_mean_square_error,
)
from bmstf.scoring.discrimination import (
    assign_risk_category,
    categorical_nri,
    net_benefit,
    operating_point,
    roc_auc,
    treat_all_net_benefit,
)
from bmstf.scoring.energy import (
    candidate_dimensions,
    energy_score_gaussian,
    energy_score_samples,
    expected_norm,
    fit_by_energy_score,
    select_latent_dimension,
)


def test_crps_sample_matches_the_gaussian_closed_form() -> None:
    rng = np.random.default_rng(3)
    draws = rng.normal(0.4, 1.7, size=200_000)
    assert abs(crps_sample(draws, 0.9) - crps_gaussian(0.4, 1.7, 0.9)) < 0.01


def test_crps_mixture_degenerates_to_one_component() -> None:
    single = crps_mixture_gaussian(np.asarray([1.0]), np.asarray([0.4]), np.asarray([1.7**2]), 0.9)
    assert abs(single - crps_gaussian(0.4, 1.7, 0.9)) < 1.0e-10


def test_crps_mixture_matches_a_weighted_sample() -> None:
    weights = np.asarray([0.4, 0.6])
    means = np.asarray([-1.0, 2.0])
    variances = np.asarray([0.09, 0.25])
    closed = crps_mixture_gaussian(weights, means, variances, 0.3)
    rng = np.random.default_rng(5)
    draws = np.concatenate(
        [
            rng.normal(means[0], 0.3, size=60_000),
            rng.normal(means[1], 0.5, size=90_000),
        ]
    )
    assert abs(closed - crps_sample(draws, 0.3)) < 0.01


def test_absolute_expectation_is_non_negative() -> None:
    for delta in (-2.0, 0.0, 1.5):
        assert absolute_expectation(delta, 0.7) > 0.0


def test_root_mean_square_error_is_zero_on_a_match() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    assert root_mean_square_error(values, values) == 0.0
    assert root_mean_square_error(values, values + 1.0) == 1.0


def test_energy_score_closed_form_matches_samples() -> None:
    mean = np.asarray([0.5, -0.4])
    covariance = np.asarray([[0.3, 0.05], [0.05, 0.2]])
    observation = np.asarray([0.9, 0.1])
    rng = np.random.default_rng(13)
    draws = rng.multivariate_normal(mean, covariance, size=8_000)
    assert (
        abs(
            energy_score_gaussian(mean, covariance, observation)
            - energy_score_samples(draws, observation)
        )
        < 0.03
    )


def test_expected_norm_grows_with_the_scale() -> None:
    small = expected_norm(np.zeros(2), np.eye(2) * 0.1)
    large = expected_norm(np.zeros(2), np.eye(2) * 2.0)
    assert large > small > 0.0


def test_dimension_selection_picks_the_index_of_the_smallest_score() -> None:
    scores = {8: 3.0, 9: 1.0, 10: 2.0}
    selection = select_latent_dimension(tuple(scores), lambda dimension: scores[dimension])
    assert selection.dimension == 9
    assert set(selection.as_mapping()) == {"score_d8", "score_d9", "score_d10", "selected"}
    assert candidate_dimensions()[0] == 8 and candidate_dimensions()[-1] == 16


def test_energy_fit_reduces_the_loss() -> None:
    def objective(parameters: np.ndarray) -> float:
        return float((parameters[0] - 3.0) ** 2 + 1.0)

    report = fit_by_energy_score(objective, np.asarray([0.0]), max_iter=6)
    assert report.loss < report.history[0]
    assert abs(float(report.parameters[0]) - 3.0) < 1.0


def test_reliability_curve_and_ece() -> None:
    probabilities = np.asarray([0.05, 0.15, 0.25, 0.45, 0.55, 0.75, 0.95])
    outcomes = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    curve = reliability_curve(probabilities, outcomes, bins=5)
    assert int(np.sum(curve.counts)) == 7
    ece = expected_calibration_error(probabilities, outcomes, bins=5)
    manual = 0.0
    for position in range(curve.counts.shape[0]):
        if curve.counts[position] > 0:
            manual += curve.counts[position] * abs(
                curve.predicted[position] - curve.observed[position]
            )
    assert abs(ece - manual / 7.0) < 1.0e-12


def test_coverage_and_gap() -> None:
    indicators = np.asarray([1.0, 1.0, 0.0, 1.0])
    assert abs(coverage(indicators) - 0.75) < 1.0e-12
    assert abs(coverage_gap(indicators, 0.9) + 0.15) < 1.0e-12


def test_roc_auc_matches_sklearn() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(23)
    labels = np.concatenate([np.ones(40), np.zeros(70)])
    scores = rng.normal(size=110) + 0.7 * labels
    scores[::9] = np.round(scores[::9], 1)
    assert abs(roc_auc(scores, labels) - float(roc_auc_score(labels, scores))) < 1.0e-12


def test_roc_auc_handles_a_single_class() -> None:
    assert roc_auc(np.asarray([0.1, 0.2]), np.asarray([1.0, 1.0])) == 0.5


def test_operating_point_is_the_youden_maximum() -> None:
    rng = np.random.default_rng(19)
    labels = np.concatenate([np.ones(50), np.zeros(90)])
    scores = rng.normal(size=140) + 0.6 * labels
    point = operating_point(scores, labels)
    assert 0.0 <= point.youden <= 1.0
    assert point.sensitivity + point.specificity - 1.0 >= point.youden - 1.0e-12


def test_risk_categories_and_nri() -> None:
    categories = assign_risk_category(np.asarray([0.05, 0.15, 0.3, 0.5]))
    assert categories.tolist() == [0, 1, 2, 3]
    outcomes = np.asarray([1.0, 0.0, 1.0, 0.0])
    nri = categorical_nri(
        np.asarray([0.05, 0.25, 0.15, 0.6]), np.asarray([0.05, 0.15, 0.25, 0.5]), outcomes
    )
    assert (
        abs(nri.nri - (nri.event_up - nri.event_down + nri.non_event_down - nri.non_event_up))
        < 1.0e-12
    )


def test_net_benefit_closed_form() -> None:
    assert abs(net_benefit(0.9, 0.9, 0.4, 0.2) - (0.9 * 0.4 - 0.1 * 0.6 * 0.25)) < 1.0e-15
    assert abs(treat_all_net_benefit(0.4, 0.2) - (0.4 - 0.6 * 0.25)) < 1.0e-15
    assert net_benefit(0.9, 0.9, 0.4, 0.0) == 0.0
