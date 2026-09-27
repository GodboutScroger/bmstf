"""State layout and regime-specific system matrices.

Ref: Eq. (2)-(3); Sec. "Two-regime latent shear-state model, filtering and
forecasting" (regime-specific Hurwitz drift matrices A_r, an exogenous drift
u(g_k) built from the scan-level geometry summaries, regime-specific process
noise Q_r, and a patient-level growth offset gamma_i as a constant state
coordinate whose prior variance is set from the between-patient spread of
published growth rates).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

DIAMETER_INDEX = 0
BURDEN_INDEX = 1
AREA_INDEX = 2
OSI_INDEX = 3
FIRST_MODE_INDEX = 4
MIN_STATE_DIM = 6
AREA_UNIT_MM2 = 5000.0

GROWTH_RATE_MM_PER_MONTH = 2.38 / 12.0
GROWTH_SPREAD_MM_PER_MONTH = 1.60 / 12.0

DIAMETER_REVERSION = -0.0008
QUIESCENT_BURDEN_REVERSION = -0.600
ACTIVE_BURDEN_REVERSION = -0.220
QUIESCENT_MODE_REVERSION = -0.350
ACTIVE_MODE_REVERSION = -0.080
MODE_COUPLING = 0.020

QUIESCENT_NOISE = (4.0e-3, 2.0e-3, 2.5e-3, 6.0e-4)
ACTIVE_NOISE = (9.0e-2, 1.2e-2, 2.2e-2, 2.0e-3)
MODE_NOISE = 3.0e-4

QUIESCENT_DRIFT_LOADING = 0.65
ACTIVE_DRIFT_LOADING = 1.00


class Regime(IntEnum):
    QUIESCENT = 0
    ACTIVE = 1


REGIME_NAMES: tuple[str, ...] = ("quiescent", "active")


@dataclass(frozen=True, slots=True)
class RegimeChain:
    """Continuous-time two-state Markov chain over the regime indicator."""

    rate_quiescent_to_active: float = 0.0300
    rate_active_to_quiescent: float = 0.0900

    def generator_matrix(self) -> np.ndarray:
        qa = float(self.rate_quiescent_to_active)
        aq = float(self.rate_active_to_quiescent)
        return np.asarray([[-qa, qa], [aq, -aq]], dtype=np.float64)

    def stationary(self) -> np.ndarray:
        qa = float(self.rate_quiescent_to_active)
        aq = float(self.rate_active_to_quiescent)
        total = qa + aq
        return np.asarray([aq / total, qa / total], dtype=np.float64)

    def mean_dwell_months(self) -> tuple[float, float]:
        return (
            1.0 / float(self.rate_quiescent_to_active),
            1.0 / float(self.rate_active_to_quiescent),
        )


@dataclass(frozen=True, slots=True)
class ShearSystem:
    """The full linear-Gaussian switching system that both generator and filter use."""

    dimension: int
    offset_index: int
    drift: tuple[np.ndarray, np.ndarray]
    input_matrix: tuple[np.ndarray, np.ndarray]
    noise: tuple[np.ndarray, np.ndarray]
    observation: np.ndarray
    chain: RegimeChain
    covariance_damping: float = 1.0

    @property
    def input_dimension(self) -> int:
        return int(self.input_matrix[0].shape[1])

    @property
    def observation_dimension(self) -> int:
        return int(self.observation.shape[0])

    def augment(self) -> np.ndarray:
        """H-tilde = [H, I] for the bias-augmented Kalman update (Algorithm 2 line 8)."""

        identity = np.eye(self.observation_dimension, dtype=np.float64)
        return np.concatenate([self.observation, identity], axis=1)


def build_system(
    dimension: int = 10,
    chain: RegimeChain | None = None,
    covariance_damping: float = 1.0,
    input_dimension: int = 4,
) -> ShearSystem:
    if dimension < MIN_STATE_DIM:
        raise ValueError(f"dimension must be at least {MIN_STATE_DIM}")
    offset_index = dimension - 1
    quiescent = _drift_matrix(dimension, offset_index, Regime.QUIESCENT)
    active = _drift_matrix(dimension, offset_index, Regime.ACTIVE)
    inputs = (
        _input_matrix(dimension, offset_index, Regime.QUIESCENT, input_dimension),
        _input_matrix(dimension, offset_index, Regime.ACTIVE, input_dimension),
    )
    noise = (
        _noise_matrix(dimension, offset_index, QUIESCENT_NOISE, covariance_damping),
        _noise_matrix(dimension, offset_index, ACTIVE_NOISE, covariance_damping),
    )
    observation = _observation_matrix(dimension, offset_index)
    return ShearSystem(
        dimension=dimension,
        offset_index=offset_index,
        drift=(quiescent, active),
        input_matrix=inputs,
        noise=noise,
        observation=observation,
        chain=chain if chain is not None else RegimeChain(),
        covariance_damping=float(covariance_damping),
    )


def _drift_matrix(dimension: int, offset_index: int, regime: Regime) -> np.ndarray:
    """A block-diagonal Hurwitz drift: the geometry enters only through the input."""

    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    matrix[DIAMETER_INDEX, DIAMETER_INDEX] = DIAMETER_REVERSION
    matrix[DIAMETER_INDEX, offset_index] = 1.0
    matrix[BURDEN_INDEX, BURDEN_INDEX] = (
        QUIESCENT_BURDEN_REVERSION if regime == Regime.QUIESCENT else ACTIVE_BURDEN_REVERSION
    )
    matrix[AREA_INDEX, AREA_INDEX] = (
        QUIESCENT_BURDEN_REVERSION if regime == Regime.QUIESCENT else ACTIVE_BURDEN_REVERSION
    )
    matrix[OSI_INDEX, OSI_INDEX] = -0.20 if regime == Regime.QUIESCENT else -0.05
    reversion = QUIESCENT_MODE_REVERSION if regime == Regime.QUIESCENT else ACTIVE_MODE_REVERSION
    for index in range(FIRST_MODE_INDEX, offset_index):
        matrix[index, index] = reversion
        if index + 1 < offset_index:
            matrix[index, index + 1] = MODE_COUPLING
            matrix[index + 1, index] = -MODE_COUPLING
    return matrix


def _input_matrix(
    dimension: int, offset_index: int, regime: Regime, input_dimension: int
) -> np.ndarray:
    matrix = np.zeros((dimension, input_dimension), dtype=np.float64)
    scale = QUIESCENT_DRIFT_LOADING if regime == Regime.QUIESCENT else ACTIVE_DRIFT_LOADING
    matrix[DIAMETER_INDEX, 0] = 0.055 * scale
    matrix[BURDEN_INDEX, 0] = 0.045 * scale
    matrix[BURDEN_INDEX, 1] = 0.230 * scale
    matrix[BURDEN_INDEX, 3] = 0.030 * scale
    matrix[AREA_INDEX, 1] = 0.300 * scale
    matrix[AREA_INDEX, 2] = 0.100 * scale
    matrix[OSI_INDEX, 1] = 0.070 * scale
    matrix[OSI_INDEX, 3] = 0.050 * scale
    for index in range(FIRST_MODE_INDEX, offset_index):
        matrix[index, 1] = 0.010 * scale
    return matrix


def _noise_matrix(
    dimension: int, offset_index: int, levels: tuple[float, ...], damping: float
) -> np.ndarray:
    diagonal = np.zeros(dimension, dtype=np.float64)
    diagonal[DIAMETER_INDEX] = levels[0] * damping
    diagonal[BURDEN_INDEX] = levels[1] * damping
    diagonal[AREA_INDEX] = levels[2] * damping
    diagonal[OSI_INDEX] = levels[3] * damping
    for index in range(FIRST_MODE_INDEX, offset_index):
        diagonal[index] = MODE_NOISE * damping
    diagonal[offset_index] = 0.0
    return np.diag(diagonal)


def _observation_matrix(dimension: int, offset_index: int) -> np.ndarray:
    matrix = np.zeros((4, dimension), dtype=np.float64)
    matrix[0, DIAMETER_INDEX] = 1.0
    matrix[1, AREA_INDEX] = AREA_UNIT_MM2
    matrix[2, BURDEN_INDEX] = 1.0
    matrix[3, OSI_INDEX] = 1.0
    return matrix


def offset_prior(observation_dimension: int = 4) -> np.ndarray:
    """Prior covariance of the patient-persistent surrogate error nu_i."""

    return np.full(observation_dimension, GROWTH_SPREAD_MM_PER_MONTH**2, dtype=np.float64)


def is_hurwitz(matrix: np.ndarray, offset_index: int) -> bool:
    """True when the drift restricted to the non-offset block is strictly stable."""

    kept = np.delete(np.delete(matrix, offset_index, axis=0), offset_index, axis=1)
    eigenvalues = np.linalg.eigvals(kept)
    return bool(np.all(np.real(eigenvalues) < 0.0))
