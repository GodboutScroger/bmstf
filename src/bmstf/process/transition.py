"""Exact discretisation of the regime-switching diffusion over an irregular interval.

Ref: Eq. (3) and the sentence that follows it: Phi_r(Delta) = exp(A_r Delta),
Gamma_r(Delta) = integral_0^Delta Phi_r(s) ds B_r and
Q_r(Delta) = integral_0^Delta Phi_r(s) Q_r Phi_r(s)^T ds, so irregular scan times
enter the model exactly rather than by resampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from bmstf.process.regimes import RegimeChain, ShearSystem


@dataclass(frozen=True, slots=True)
class DiscreteSystem:
    transition: np.ndarray
    input_gain: np.ndarray
    covariance: np.ndarray

    def as_mapping(self) -> dict[str, float]:
        return {
            "transition_trace": float(np.trace(self.transition)),
            "covariance_trace": float(np.trace(self.covariance)),
            "input_norm": float(np.linalg.norm(self.input_gain)),
        }


def regime_transition(chain: RegimeChain, delta: float) -> np.ndarray:
    """Pi(Delta) = exp(Delta L) for the two-state generator L."""

    return continuous_transition(chain.generator_matrix(), delta)


def continuous_transition(generator: np.ndarray, delta: float) -> np.ndarray:
    """exp(Delta L) for any generator, used by both the truth and the filter."""

    matrix = expm(float(delta) * np.asarray(generator, dtype=np.float64))
    return np.asarray(np.clip(matrix, 0.0, 1.0), dtype=np.float64)


def discretise(
    drift: np.ndarray, input_matrix: np.ndarray, noise: np.ndarray, delta: float
) -> DiscreteSystem:
    """Return (Phi, Gamma, Q_d) for dz = [A z + B u] dt + Q^{1/2} dW."""

    step = float(delta)
    dimension, input_dimension = input_matrix.shape
    if step <= 0.0:
        return DiscreteSystem(
            transition=np.eye(dimension, dtype=np.float64),
            input_gain=np.zeros((dimension, input_dimension), dtype=np.float64),
            covariance=np.zeros((dimension, dimension), dtype=np.float64),
        )

    augmented = np.zeros((dimension + input_dimension, dimension + input_dimension))
    augmented[:dimension, :dimension] = drift
    augmented[:dimension, dimension:] = input_matrix
    transition_block = expm(augmented * step)
    transition = np.asarray(transition_block[:dimension, :dimension], dtype=np.float64)
    input_gain = np.asarray(transition_block[:dimension, dimension:], dtype=np.float64)

    van_loan = np.zeros((2 * dimension, 2 * dimension), dtype=np.float64)
    van_loan[:dimension, :dimension] = drift
    van_loan[:dimension, dimension:] = noise
    van_loan[dimension:, dimension:] = -drift.T
    exponential = expm(van_loan * step)
    covariance = np.asarray(transition @ exponential[:dimension, dimension:], dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    return DiscreteSystem(transition=transition, input_gain=input_gain, covariance=covariance)


def discretise_regime(system: ShearSystem, regime: int, delta: float) -> DiscreteSystem:
    index = int(np.clip(int(regime), 0, 1))
    return discretise(
        system.drift[index],
        system.input_matrix[index],
        system.noise[index],
        delta,
    )


def reference_covariance(
    drift: np.ndarray, noise: np.ndarray, delta: float, substeps: int = 4000
) -> np.ndarray:
    """Independent recomputation of Q_d by a fine Riemann sum of Phi(s) Q Phi(s)^T."""

    step = float(delta)
    dimension = noise.shape[0]
    if step <= 0.0:
        return np.zeros((dimension, dimension), dtype=np.float64)
    grid = np.linspace(0.0, step, int(substeps) + 1)
    accumulator = np.zeros((dimension, dimension), dtype=np.float64)
    width = step / float(substeps)
    for position, value in enumerate(grid):
        phi = expm(drift * value)
        weight = 0.5 if position in (0, int(substeps)) else 1.0
        accumulator += weight * (phi @ noise @ phi.T)
    return np.asarray(accumulator * width, dtype=np.float64)


def reference_input_gain(
    drift: np.ndarray, input_matrix: np.ndarray, delta: float, substeps: int = 4000
) -> np.ndarray:
    """Independent recomputation of Gamma by a fine Riemann sum of Phi(s) B."""

    step = float(delta)
    dimension, input_dimension = input_matrix.shape
    if step <= 0.0:
        return np.zeros((dimension, input_dimension), dtype=np.float64)
    grid = np.linspace(0.0, step, int(substeps) + 1)
    accumulator = np.zeros((dimension, input_dimension), dtype=np.float64)
    width = step / float(substeps)
    for position, value in enumerate(grid):
        phi = expm(drift * value)
        weight = 0.5 if position in (0, int(substeps)) else 1.0
        accumulator += weight * (phi @ input_matrix)
    return np.asarray(accumulator * width, dtype=np.float64)


def covariance_sqrt(covariance: np.ndarray) -> np.ndarray:
    """Symmetric square root that tolerates the singular blocks of Q_r(Delta)."""

    symmetry = 0.5 * (
        np.asarray(covariance, dtype=np.float64) + np.asarray(covariance, dtype=np.float64).T
    )
    values, vectors = np.linalg.eigh(symmetry)
    return np.asarray(vectors * np.sqrt(np.clip(values, 0.0, None)), dtype=np.float64)


def sample_regime_path(
    chain: RegimeChain,
    start: int,
    delta: float,
    step: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sub-step boundaries and the regime held on each sub-step."""

    count = max(int(np.ceil(float(delta) / float(step))), 1)
    boundaries = np.linspace(0.0, float(delta), count + 1)
    regimes = np.empty(count, dtype=np.int64)
    current = int(start)
    for position in range(count):
        width = float(boundaries[position + 1] - boundaries[position])
        transition = regime_transition(chain, width)[current]
        current = int(rng.choice(2, p=transition / float(np.sum(transition))))
        regimes[position] = current
    return boundaries, regimes
