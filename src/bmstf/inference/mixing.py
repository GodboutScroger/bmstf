"""Interacting-multiple-model mixing and collapse.

Ref: Algorithm 2 lines 4 and 14 (mix the modes with weights Pi^T w, then collapse
to (z_k, P_k) by moment matching); Sec. "Two-regime latent shear-state model"
(the interacting-multiple-model moment-matching step, named as the approximation
it is).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GaussianMixture:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray

    @property
    def components(self) -> int:
        return int(self.means.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    def normalise(self) -> GaussianMixture:
        total = float(np.sum(self.weights))
        if total <= 0.0:
            uniform = np.full(self.weights.shape[0], 1.0 / float(self.weights.shape[0]))
            return GaussianMixture(self.means, self.covariances, uniform)
        return GaussianMixture(self.means, self.covariances, self.weights / total)

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        return moment_match(self.means, self.covariances, self.weights)

    def as_mapping(self) -> dict[str, float]:
        mean, covariance = self.moments()
        return {
            "components": float(self.components),
            "mean_norm": float(np.linalg.norm(mean)),
            "covariance_trace": float(np.trace(covariance)),
            "weight_first": float(self.normalise().weights[0]),
        }


def mixing_weights(transition: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pi^T w: the predictive weight of each target mode at the next time step."""

    return np.asarray(
        np.asarray(transition, dtype=np.float64).T
        @ np.asarray(weights, dtype=np.float64).reshape(-1),
        dtype=np.float64,
    )


def conditional_mixing(transition: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """MIX[s, r] = Pi[r, s] w_r / (Pi^T w)_s, the weight of source r for target s."""

    matrix = np.asarray(transition, dtype=np.float64)
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    prior = mixing_weights(matrix, vector)
    numerator = matrix.T * vector[None, :]
    denominator = np.where(prior > 0.0, prior, 1.0)
    return np.asarray(numerator / denominator[:, None], dtype=np.float64)


def moment_match(
    means: np.ndarray, covariances: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """First two moments of a Gaussian mixture, preserving both by construction."""

    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(weight))
    if total <= 0.0:
        weight = np.full(weight.shape[0], 1.0 / float(weight.shape[0]))
    else:
        weight = weight / total
    mean = np.einsum("r,rd->d", weight, np.asarray(means, dtype=np.float64))
    covariance = np.zeros((mean.shape[0], mean.shape[0]), dtype=np.float64)
    for index in range(weight.shape[0]):
        delta = np.asarray(means, dtype=np.float64)[index] - mean
        covariance += weight[index] * (
            np.asarray(covariances, dtype=np.float64)[index] + np.outer(delta, delta)
        )
    return np.asarray(mean, dtype=np.float64), _stabilise(covariance)


def mix_components(
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The IMM mixing step: per-target-mode moments and the predictive weights."""

    mix = conditional_mixing(transition, weights)
    targets = mix.shape[0]
    dimension = np.asarray(means, dtype=np.float64).shape[1]
    mixed_means = np.zeros((targets, dimension), dtype=np.float64)
    mixed_covariances = np.zeros((targets, dimension, dimension), dtype=np.float64)
    for target in range(targets):
        shared, covariance = moment_match(
            np.asarray(means, dtype=np.float64),
            np.asarray(covariances, dtype=np.float64),
            mix[target],
        )
        mixed_means[target] = shared
        mixed_covariances[target] = covariance
    return mixed_means, mixed_covariances, mixing_weights(transition, weights)


def collapsed_state(
    means: np.ndarray, covariances: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return moment_match(means, covariances, weights)


def _stabilise(matrix: np.ndarray) -> np.ndarray:
    symmetry = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetry)
    floor = float(np.min(eigenvalues))
    if floor <= 1.0e-12:
        symmetry = symmetry + (1.0e-12 - floor) * np.eye(symmetry.shape[0])
    return np.asarray(symmetry, dtype=np.float64)
