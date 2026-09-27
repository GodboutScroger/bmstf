"""Regime-conditional Rauch-Tung-Striebel smoother.

Ref: Sec. "Two-regime latent shear-state model" (conditional on a regime path the
Kalman filter and smoother are exact).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.process.transition import DiscreteSystem


@dataclass(frozen=True, slots=True)
class SmootherPass:
    means: np.ndarray
    covariances: np.ndarray
    log_likelihood: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "steps": float(self.means.shape[0]),
            "final_mean_norm": float(np.linalg.norm(self.means[-1])),
            "log_likelihood": float(self.log_likelihood),
        }


def rts_smoother(
    filtered_means: np.ndarray,
    filtered_covariances: np.ndarray,
    predicted_means: np.ndarray,
    predicted_covariances: np.ndarray,
    steps: list[DiscreteSystem],
) -> SmootherPass:
    """Backward recursion over one fixed regime path."""

    count = filtered_means.shape[0]
    smoothed_means = np.array(filtered_means, dtype=np.float64, copy=True)
    smoothed_covariances = np.array(filtered_covariances, dtype=np.float64, copy=True)
    for index in range(count - 2, -1, -1):
        step = steps[index]
        prior = predicted_covariances[index + 1]
        gain = np.linalg.solve(prior, step.transition @ filtered_covariances[index].T).T
        smoothed_means[index] = filtered_means[index] + gain @ (
            smoothed_means[index + 1] - predicted_means[index + 1]
        )
        smoothed_covariances[index] = (
            filtered_covariances[index] + gain @ (smoothed_covariances[index + 1] - prior) @ gain.T
        )
        smoothed_covariances[index] = 0.5 * (
            smoothed_covariances[index] + smoothed_covariances[index].T
        )
    return SmootherPass(
        means=smoothed_means,
        covariances=smoothed_covariances,
        log_likelihood=0.0,
    )


def joint_gaussian_covariance(transition: np.ndarray, prior_covariances: np.ndarray) -> np.ndarray:
    """Joint covariance of the chain before any observation, for verification only.

    Markov structure gives Cov(z_i, z_j) = Var(z_i) (Phi^{j-i})^T for i <= j, with
    Var(z_i) = Phi Var(z_{i-1}) Phi^T + Q on the diagonal. The transpose matters: the
    left factor carries the covariance of the earlier step. Passing filtered or
    predicted (observation-conditioned) covariances here instead would be the wrong
    law: the joint a smoother is consistent with is the prior one.
    """

    step = np.asarray(transition, dtype=np.float64)
    covariances = np.asarray(prior_covariances, dtype=np.float64)
    count, dimension = int(covariances.shape[0]), int(covariances.shape[1])
    joint = np.zeros((count * dimension, count * dimension), dtype=np.float64)
    for later in range(count):
        for earlier in range(later + 1):
            block = covariances[earlier] @ np.linalg.matrix_power(step, later - earlier).T
            rows = slice(earlier * dimension, (earlier + 1) * dimension)
            columns = slice(later * dimension, (later + 1) * dimension)
            joint[rows, columns] = block
            joint[columns, rows] = block.T
    return joint


def conditional_means(
    joint_covariance: np.ndarray,
    measurement: np.ndarray,
    noise: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """E[z | y] of a zero-mean jointly Gaussian (z, y), for verification only."""

    covariance = np.asarray(joint_covariance, dtype=np.float64)
    operator = np.asarray(measurement, dtype=np.float64)
    signal = np.asarray(observations, dtype=np.float64).reshape(-1)
    gain = (
        covariance
        @ operator.T
        @ np.linalg.inv(operator @ covariance @ operator.T + np.asarray(noise, dtype=np.float64))
    )
    return np.asarray(gain @ signal, dtype=np.float64)
