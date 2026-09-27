"""Kalman update with the decomposed per-scan covariance.

Ref: the Kalman gain K_k = P_{k|k-1} H~^T (H~ P_{k|k-1} H~^T + Sigma_BC,k +
Sigma_rec,k)^{-1} in Sec. "Two-regime latent shear-state model, filtering and
forecasting"; Algorithm 2 lines 8-11.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LIKELIHOOD_FLOOR = 1.0e-300


@dataclass(frozen=True, slots=True)
class KalmanUpdate:
    mean: np.ndarray
    covariance: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    gain: np.ndarray
    log_likelihood: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "log_likelihood": float(self.log_likelihood),
            "innovation_norm": float(np.linalg.norm(self.innovation)),
            "gain_norm": float(np.linalg.norm(self.gain)),
        }


def kalman_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    observation: np.ndarray,
    measurement: np.ndarray,
    noise: np.ndarray,
) -> KalmanUpdate:
    """One Gaussian measurement update of (mean, covariance)."""

    operator = np.asarray(observation, dtype=np.float64)
    signal = np.asarray(measurement, dtype=np.float64).reshape(-1)
    innovation = signal - operator @ np.asarray(mean, dtype=np.float64)
    innovation_covariance = _stabilise(
        operator @ np.asarray(covariance, dtype=np.float64) @ operator.T
        + np.asarray(noise, dtype=np.float64)
    )
    gain = np.asarray(
        np.asarray(covariance, dtype=np.float64)
        @ operator.T
        @ np.linalg.inv(innovation_covariance),
        dtype=np.float64,
    )
    posterior_mean = np.asarray(mean, dtype=np.float64) + gain @ innovation
    identity = np.eye(covariance.shape[0], dtype=np.float64)
    residual = identity - gain @ operator
    posterior_covariance = _stabilise(
        residual @ np.asarray(covariance, dtype=np.float64) @ residual.T
        + gain @ np.asarray(noise, dtype=np.float64) @ gain.T
    )
    return KalmanUpdate(
        mean=posterior_mean,
        covariance=posterior_covariance,
        innovation=innovation,
        innovation_covariance=innovation_covariance,
        gain=gain,
        log_likelihood=gaussian_log_likelihood(innovation, innovation_covariance),
    )


def gaussian_log_likelihood(innovation: np.ndarray, covariance: np.ndarray) -> float:
    dimension = innovation.shape[0]
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        return LIKELIHOOD_FLOOR
    quadratic = float(innovation @ np.linalg.solve(covariance, innovation))
    value = -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + quadratic)
    return float(max(value, np.log(LIKELIHOOD_FLOOR)))


def standard_form_covariance(
    covariance: np.ndarray,
    observation: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """The short form I - K H applied to the prior, for verification only."""

    operator = np.asarray(observation, dtype=np.float64)
    innovation_covariance = operator @ covariance @ operator.T + np.asarray(noise, dtype=np.float64)
    gain = covariance @ operator.T @ np.linalg.inv(innovation_covariance)
    return _stabilise((np.eye(covariance.shape[0]) - gain @ operator) @ covariance)


def bias_augmented_operator(observation: np.ndarray, channels: int) -> np.ndarray:
    """H~ = [H, I] as used by Algorithm 2 line 8."""

    identity = np.eye(int(channels), dtype=np.float64)
    return np.ascontiguousarray(np.concatenate([np.asarray(observation), identity], axis=1))


def diameter_operator(dimension: int) -> np.ndarray:
    """e_1^T: a row that reads the diameter coordinate of the augmented state."""

    row = np.zeros((1, int(dimension)), dtype=np.float64)
    row[0, 0] = 1.0
    return row


def _stabilise(matrix: np.ndarray) -> np.ndarray:
    symmetry = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    eigenvalues = np.linalg.eigvalsh(symmetry)
    floor = float(np.min(eigenvalues))
    if floor <= 1.0e-12:
        symmetry = symmetry + (1.0e-12 - floor) * np.eye(symmetry.shape[0])
    return np.asarray(symmetry, dtype=np.float64)
