"""Patient-persistent surrogate-error covariance lookup.

Ref: Algorithm 1 lines 1 and 11 (the residual-covariance lookup Sigma_sur(kappa)
by topology class, carried as nu_i in Algorithm 2); Sec. "Here we make the
identification explicit" (the covariance decomposes into boundary-condition,
reconstruction and patient-correlated surrogate-error terms).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.geometry.perturbations import N_TOPOLOGY_CLASSES

CHANNEL_SCALE: tuple[float, ...] = (0.35, 300.0, 2.4e-2, 6.0e-2)
EQUICORRELATION = 0.25

CLASS_MULTIPLIERS: tuple[float, ...] = (1.0, 1.15, 1.35, 1.6)
FLOOR = 1.0e-12


def correlation_matrix(dimension: int, rho: float = EQUICORRELATION) -> np.ndarray:
    """Equicorrelation with a unit diagonal, positive definite for rho in (-1/(d-1), 1)."""

    matrix = np.full((dimension, dimension), rho, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    assembled: np.ndarray = matrix
    return assembled


@dataclass(frozen=True, slots=True)
class ResidualLookup:
    """Sigma_sur(kappa): a class-stratified covariance scaled by the scan's inflow swing."""

    channel_scale: tuple[float, ...] = CHANNEL_SCALE
    equicorrelation: float = EQUICORRELATION
    class_multipliers: tuple[float, ...] = CLASS_MULTIPLIERS

    @property
    def dimension(self) -> int:
        return len(self.channel_scale)

    def base(self, topology_class: int) -> np.ndarray:
        index = int(np.clip(int(topology_class), 0, N_TOPOLOGY_CLASSES - 1))
        multiplier = float(self.class_multipliers[index])
        scale = np.asarray(self.channel_scale, dtype=np.float64) * np.sqrt(multiplier)
        correlation = correlation_matrix(self.dimension, self.equicorrelation)
        assembled: np.ndarray = scale[:, None] * correlation * scale[None, :]
        return assembled

    def __call__(self, topology_class: int, kappa: float) -> np.ndarray:
        scaled = float(max(kappa, FLOOR)) * self.base(topology_class)
        symmetrised: np.ndarray = _symmetrise(scaled)
        return symmetrised


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    symmetry = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetry)
    floor = float(np.min(eigenvalues))
    if floor <= FLOOR:
        symmetry = symmetry + (FLOOR - floor) * np.eye(symmetry.shape[0])
    result: np.ndarray = symmetry
    return result


def inflate(covariance: np.ndarray, factor: float) -> np.ndarray:
    """The topology-gate inflation kappa_phi applied to the boundary-condition budget."""

    inflated: np.ndarray = _symmetrise(float(factor) * np.asarray(covariance, dtype=np.float64))
    return inflated


def covariance_trace(covariance: np.ndarray) -> float:
    return float(np.trace(np.asarray(covariance, dtype=np.float64)))


def budget_shares(
    boundary: np.ndarray, reconstruction: np.ndarray, surrogate: np.ndarray
) -> dict[str, float]:
    """Share of the marginalised trace attributable to each physical term."""

    total = (
        covariance_trace(boundary) + covariance_trace(reconstruction) + covariance_trace(surrogate)
    )
    if total <= 0.0:
        return {"boundary": 0.0, "reconstruction": 0.0, "surrogate": 0.0}
    return {
        "boundary": covariance_trace(boundary) / total,
        "reconstruction": covariance_trace(reconstruction) / total,
        "surrogate": covariance_trace(surrogate) / total,
    }
