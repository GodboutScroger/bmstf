"""Continuous ranked probability score, in closed form and from samples.

Ref: Table 1 panel A (the continuous ranked probability score of the 12-month
diameter forecast); Table 2 (the CRPS is the one proper score on which the
single-regime filters beat the full mechanism, see the text).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

SQRT_PI = float(np.sqrt(np.pi))


def crps_gaussian(mean: float, standard_deviation: float, observation: float) -> float:
    """Closed form for N(mean, sd^2): sd [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]."""

    sigma = float(max(standard_deviation, 1.0e-12))
    z = (float(observation) - float(mean)) / sigma
    return float(sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / SQRT_PI))


def crps_gaussian_at_zero(scale: float) -> float:
    """CRPS of N(0, scale^2) evaluated at zero, used by the mixture cross terms."""

    return crps_gaussian(0.0, scale, 0.0)


def crps_sample(samples: np.ndarray, observation: float) -> float:
    """(1/M) sum |x_i - y| - (1/M^2) sum (2i - M - 1) x_(i) over sorted samples."""

    values = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    count = values.shape[0]
    if count == 0:
        return 0.0
    first = float(np.mean(np.abs(values - float(observation))))
    order = np.arange(1, count + 1, dtype=np.float64)
    second = float(np.sum((2.0 * order - float(count) - 1.0) * values)) / float(count**2)
    return float(first - second)


def absolute_expectation(delta: float, scale: float) -> float:
    """E|N(delta, scale^2)| = scale sqrt(2/pi) exp(-delta^2/(2 scale^2)) + delta (2 Phi(delta/scale) - 1)."""

    sigma = float(max(scale, 1.0e-12))
    ratio = float(delta) / sigma
    return float(
        sigma * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * ratio * ratio)
        + float(delta) * (2.0 * norm.cdf(ratio) - 1.0)
    )


def crps_mixture_gaussian(
    weights: np.ndarray, means: np.ndarray, variances: np.ndarray, observation: float
) -> float:
    """CRPS(F, y) = sum_i w_i h(mu_i - y, sigma_i) - 1/2 sum_ij w_i w_j h(mu_i - mu_j, s_ij)."""

    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(weight))
    weight = weight / total if total > 0.0 else weight
    centre = np.asarray(means, dtype=np.float64).reshape(-1)
    scale = np.sqrt(np.asarray(variances, dtype=np.float64).reshape(-1))
    first = float(
        np.sum(
            weight
            * np.asarray(
                [
                    absolute_expectation(centre[i] - float(observation), scale[i])
                    for i in range(centre.shape[0])
                ]
            )
        )
    )
    cross = 0.0
    for i in range(centre.shape[0]):
        for j in range(centre.shape[0]):
            spread = float(np.sqrt(scale[i] ** 2 + scale[j] ** 2))
            cross += weight[i] * weight[j] * absolute_expectation(centre[i] - centre[j], spread)
    return float(first - 0.5 * cross)


def crps_gaussian_at_zero_shifted(scale: float, shift: float) -> float:
    """CRPS of N(shift, scale^2) evaluated at zero, kept as the cross-term building block."""

    return crps_gaussian(float(shift), float(scale), 0.0)


def crps_of_samples_at(forecast: np.ndarray, observation: float) -> float:
    return crps_sample(forecast, observation)


def root_mean_square_error(predicted: np.ndarray, observed: np.ndarray) -> float:
    difference = np.asarray(predicted, dtype=np.float64) - np.asarray(observed, dtype=np.float64)
    if difference.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(difference**2)))
