"""Bootstrap intervals, the Holm step-down, the sign-flip test and agreement."""

from __future__ import annotations

import numpy as np

from bmstf.stats.agreement import (
    cohen_kappa,
    icc_two_one,
    mean_squares,
    median_absolute_deviation,
    median_absolute_difference,
)
from bmstf.stats.bootstrap import (
    cluster_bootstrap,
    direct_percentile_interval,
    paired_difference,
    percentile_interval,
)
from bmstf.stats.multiplicity import (
    COMPARATOR_COUNT,
    adjust_within_metric,
    family_width,
    holm_bonferroni,
    significant,
)
from bmstf.stats.roc import delong, paired_contrast, placement_matrix
from bmstf.stats.signflip import enumerate_sign_flip_sum, sign_flip_p_value


def test_percentile_interval_matches_its_own_recipe() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(0.3, 0.2, size=64)
    interval = percentile_interval(values, resamples=2000, rng=np.random.default_rng(19))
    draws = np.random.default_rng(19).integers(0, 64, size=(2000, 64))
    means = np.mean(values[draws], axis=1)
    assert abs(interval.lower - float(np.quantile(means, 0.025))) < 1.0e-12
    assert abs(interval.upper - float(np.quantile(means, 0.975))) < 1.0e-12
    assert abs(interval.estimate - float(np.mean(values))) < 1.0e-12


def test_direct_interval_brackets_the_mean() -> None:
    values = np.linspace(0.0, 1.0, 101)
    interval = direct_percentile_interval(values)
    assert interval.lower <= interval.estimate <= interval.upper
    assert interval.resamples == 101


def test_cluster_bootstrap_resamples_records() -> None:
    rng = np.random.default_rng(21)
    values = rng.normal(size=90)
    labels = np.repeat(np.arange(30), 3)
    interval = cluster_bootstrap(values, labels, resamples=300, rng=rng)
    assert interval.lower <= interval.estimate <= interval.upper
    assert interval.resamples == 300


def test_paired_difference_rejects_mismatched_shapes() -> None:
    assert np.allclose(paired_difference(np.asarray([1.0, 2.0]), np.asarray([0.0, 1.0])), 1.0)
    try:
        paired_difference(np.asarray([1.0]), np.asarray([1.0, 2.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("a shape mismatch must raise")


def test_holm_step_down_matches_the_definition() -> None:
    raw = np.asarray([0.0007, 0.0012, 0.0031, 0.0036, 0.0045, 0.031, 0.09, 0.12])
    observed = holm_bonferroni(raw)
    order = np.argsort(raw, kind="mergesort")
    brute = np.empty(raw.shape[0])
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, float(raw.shape[0] - position) * float(raw[index])))
        brute[index] = running
    assert np.max(np.abs(observed - brute)) < 1.0e-15
    assert np.all(adjust_within_metric(raw) == observed)
    assert family_width() == COMPARATOR_COUNT


def test_holm_flags_only_adjusted_significance() -> None:
    raw = np.asarray([0.001, 0.2, 0.3])
    adjusted = holm_bonferroni(raw)
    flags = significant(raw, adjusted, 0.05)
    assert flags[0] and not flags[1] and not flags[2]


def test_sign_flip_matches_enumeration() -> None:
    differences = np.asarray([0.05, -0.02, 0.11, 0.003, -0.07, 0.09])
    result = sign_flip_p_value(differences)
    sums = enumerate_sign_flip_sum(differences)
    counted = float(np.count_nonzero(np.abs(sums) >= abs(float(np.sum(differences))) - 1.0e-12))
    assert result.exact
    assert result.sign_vectors == 2 ** differences.shape[0]
    assert abs(result.p_value - counted / sums.shape[0]) < 1.0e-15


def test_sign_flip_is_one_for_a_null_difference() -> None:
    result = sign_flip_p_value(np.zeros(8))
    assert result.p_value == 1.0


def test_delong_matches_the_placement_loop() -> None:
    rng = np.random.default_rng(29)
    labels = np.concatenate([np.ones(45), np.zeros(75)])
    scores = rng.normal(size=120) + 0.8 * labels
    scores[::5] = 0.0
    result = delong(scores, labels)
    matrix = placement_matrix(scores, labels)
    v10 = matrix.mean(axis=1)
    v01 = matrix.mean(axis=0)
    brute = float(np.var(v10, ddof=1) / matrix.shape[1] + np.var(v01, ddof=1) / matrix.shape[0])
    assert abs(result.variance - brute) < 1.0e-15
    assert abs(result.area - float(matrix.mean())) < 1.0e-15


def test_paired_contrast_is_finite_and_signed() -> None:
    rng = np.random.default_rng(31)
    labels = np.concatenate([np.ones(40), np.zeros(60)])
    base = rng.normal(size=100)
    first = base + 1.0 * labels
    second = base + 0.4 * labels
    contrast = paired_contrast(first, second, labels)
    assert contrast.difference > 0.0
    assert contrast.standard_error > 0.0
    assert 0.0 <= contrast.p_value <= 1.0


def test_kappa_matches_sklearn() -> None:
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(37)
    first = rng.integers(0, 2, size=300)
    second = np.where(rng.random(300) < 0.1, 1 - first, first)
    result = cohen_kappa(first, second)
    assert abs(result.kappa - float(cohen_kappa_score(first, second))) < 1.0e-12
    assert result.counts == 300


def test_icc_matches_the_mean_square_assembly() -> None:
    rng = np.random.default_rng(41)
    subjects = 30
    latent = rng.normal(size=subjects)
    matrix = np.stack([latent + rng.normal(scale=0.3, size=subjects) for _ in range(3)], axis=1)
    observed = icc_two_one(matrix)
    assembly = mean_squares(matrix)
    raters = int(assembly["raters"])
    denominator = (
        assembly["between_rows"]
        + (raters - 1) * assembly["residual"]
        + raters * (assembly["between_columns"] - assembly["residual"]) / int(assembly["subjects"])
    )
    assert abs(observed - (assembly["between_rows"] - assembly["residual"]) / denominator) < 1.0e-12


def test_reader_difference_statistics() -> None:
    first = np.asarray([1.0, 2.0, 3.0, 4.0])
    second = np.asarray([1.5, 2.5, 3.5, 4.5])
    assert abs(median_absolute_difference(first, second) - 0.5) < 1.0e-12
    assert abs(median_absolute_deviation(first) - 1.0) < 1.0e-12
