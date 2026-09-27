"""Boundary-condition-marginalised observation for one scan.

Ref: Algorithm 1 (lines 1-16): the boundary-condition budget Var_b[E(y | g, b)],
the reconstruction budget, the patient-persistent residual and the topology gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.geometry.perturbations import (
    LONGITUDINAL_CORRELATION_MM,
    RADIAL_JITTER_MM,
    PerturbationSpec,
    boundary_condition_scale,
    mask_perturbations,
    topology_class,
    topology_flag,
)
from bmstf.geometry.summaries import GeometrySummary
from bmstf.geometry.template import ScanGeometry
from bmstf.sensor.boundary import BoundaryPrior
from bmstf.sensor.covariance import ResidualLookup, budget_shares, inflate
from bmstf.sensor.quasimc import draw_boundary_conditions, prior_mean
from bmstf.sensor.surrogate import SHEAR_CHANNELS, PhysicsAttentionSurrogate

BOUNDARY_DRAWS = 64
RECONSTRUCTION_DRAWS = 8
TOPOLOGY_INFLATION = 1.6


@dataclass(frozen=True, slots=True)
class MarginalisationSpec:
    boundary_draws: int = BOUNDARY_DRAWS
    reconstruction_draws: int = RECONSTRUCTION_DRAWS
    topology_inflation: float = TOPOLOGY_INFLATION
    jitter_mm: float = RADIAL_JITTER_MM
    correlation_mm: float = LONGITUDINAL_CORRELATION_MM


@dataclass(frozen=True, slots=True)
class Observation:
    index: str
    mean: np.ndarray
    covariance: np.ndarray
    flag: int
    sigma_boundary: np.ndarray
    sigma_reconstruction: np.ndarray
    sigma_surrogate: np.ndarray
    boundary_draws: int
    reconstruction_draws: int

    @property
    def dimension(self) -> int:
        return int(self.mean.shape[0])

    def shares(self) -> dict[str, float]:
        return budget_shares(self.sigma_boundary, self.sigma_reconstruction, self.sigma_surrogate)

    def as_mapping(self) -> dict[str, float]:
        mapping = {
            "boundary_trace": float(np.trace(self.sigma_boundary)),
            "reconstruction_trace": float(np.trace(self.sigma_reconstruction)),
            "surrogate_trace": float(np.trace(self.sigma_surrogate)),
            "flag": float(self.flag),
        }
        for name, value in zip(SHEAR_CHANNELS, self.mean, strict=True):
            mapping[f"mean_{name}"] = float(value)
        return mapping


def marginalise(
    geometry: ScanGeometry,
    summary: GeometrySummary,
    site_index: int,
    surrogate: PhysicsAttentionSurrogate,
    prior: BoundaryPrior,
    lookup: ResidualLookup,
    spec: MarginalisationSpec,
    seed: int,
    rng: np.random.Generator,
) -> Observation:
    centre = prior.mean_vector(summary, site_index)
    draws = draw_boundary_conditions(prior, centre, spec.boundary_draws, seed)
    fields = np.stack([surrogate.evaluate(geometry, draw).values for draw in draws])
    mean = np.asarray(np.mean(fields, axis=0), dtype=np.float64)
    boundary = _sample_covariance(fields)

    reference = prior_mean(prior, centre)
    perturbations = mask_perturbations(
        geometry,
        PerturbationSpec(spec.reconstruction_draws, spec.jitter_mm, spec.correlation_mm),
        rng,
    )
    rebuilt = np.stack(
        [surrogate.evaluate(perturbed, reference).values for perturbed in perturbations]
    )
    reconstruction = _sample_covariance(rebuilt)

    kappa = boundary_condition_scale(geometry)
    surrogate_error = lookup(topology_class(geometry), kappa)
    flag = topology_flag(geometry)
    if flag:
        boundary = inflate(boundary, spec.topology_inflation)

    covariance = _symmetrise(boundary + reconstruction + surrogate_error)
    return Observation(
        index=geometry.index,
        mean=mean,
        covariance=covariance,
        flag=flag,
        sigma_boundary=boundary,
        sigma_reconstruction=reconstruction,
        sigma_surrogate=surrogate_error,
        boundary_draws=int(spec.boundary_draws),
        reconstruction_draws=int(spec.reconstruction_draws),
    )


def _sample_covariance(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if values.shape[0] < 2:
        return np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    centred = values - np.mean(values, axis=0)
    return np.asarray(centred.T @ centred / (values.shape[0] - 1), dtype=np.float64)


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    symmetry = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetry)
    floor = float(np.min(eigenvalues))
    if floor <= 1.0e-12:
        symmetry = symmetry + (1.0e-12 - floor) * np.eye(symmetry.shape[0])
    result: np.ndarray = symmetry
    return result


def enumerate_boundary_budget(
    geometry: ScanGeometry,
    summary: GeometrySummary,
    site_index: int,
    surrogate: PhysicsAttentionSurrogate,
    prior: BoundaryPrior,
    spec: MarginalisationSpec,
    seed: int,
) -> np.ndarray:
    """Recompute the boundary-condition budget by a straight loop, for verification."""

    centre = prior.mean_vector(summary, site_index)
    draws = draw_boundary_conditions(prior, centre, spec.boundary_draws, seed)
    fields = np.empty((spec.boundary_draws, len(SHEAR_CHANNELS)), dtype=np.float64)
    for row in range(draws.shape[0]):
        fields[row] = surrogate.evaluate(geometry, draws[row]).values
    mean = np.sum(fields, axis=0) / float(fields.shape[0])
    covariance = np.zeros((fields.shape[1], fields.shape[1]), dtype=np.float64)
    for row in range(fields.shape[0]):
        delta = fields[row] - mean
        covariance += np.outer(delta, delta)
    return covariance / float(fields.shape[0] - 1)
