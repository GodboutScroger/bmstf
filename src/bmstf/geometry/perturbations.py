"""Segmentation perturbations and the topology gate.

Ref: Algorithm 1 lines 1, 8-14 (mask perturbations g_k^(j), the topology flag
phi_k and the inflation factor kappa_phi).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.geometry.summaries import summarise
from bmstf.geometry.template import MIN_NODE_COVERAGE, ScanGeometry, annular_volumes

RADIAL_JITTER_MM = 0.12
LONGITUDINAL_CORRELATION_MM = 9.0

N_TOPOLOGY_CLASSES = 4


@dataclass(frozen=True, slots=True)
class PerturbationSpec:
    count: int
    jitter_mm: float = RADIAL_JITTER_MM
    correlation_mm: float = LONGITUDINAL_CORRELATION_MM


def smooth_radial_field(
    nodes: int, spacing_mm: float, rng: np.random.Generator, jitter_mm: float, correlation_mm: float
) -> np.ndarray:
    """A zero-mean Gaussian field on the nodes with a squared-exponential kernel."""

    axis = np.arange(nodes, dtype=np.float64) * spacing_mm
    distance = np.abs(axis[:, None] - axis[None, :])
    covariance = np.exp(-0.5 * (distance / correlation_mm) ** 2)
    covariance += 1.0e-9 * np.eye(nodes)
    cholesky = np.linalg.cholesky(covariance)
    white = rng.standard_normal(nodes)
    field = cholesky @ white * jitter_mm
    return field - float(np.mean(field))


def perturb(
    geometry: ScanGeometry, spec: PerturbationSpec, rng: np.random.Generator
) -> ScanGeometry:
    field = smooth_radial_field(
        geometry.nodes,
        geometry.template.spacing_mm,
        rng,
        spec.jitter_mm,
        spec.correlation_mm,
    )
    lumen = np.maximum(geometry.lumen_radius + field, 0.5)
    thrombus = np.maximum(geometry.thrombus_radius + field, geometry.lumen_radius)
    wall = np.maximum(geometry.wall_radius + field, thrombus)
    return ScanGeometry(
        geometry.index,
        geometry.template,
        lumen,
        thrombus,
        wall,
        geometry.node_mask.copy(),
    )


def mask_perturbations(
    geometry: ScanGeometry, spec: PerturbationSpec, rng: np.random.Generator
) -> list[ScanGeometry]:
    return [perturb(geometry, spec, rng) for _ in range(spec.count)]


def topology_flag(geometry: ScanGeometry) -> int:
    """One when the segmented surface fails the coverage or monotonicity check."""

    coverage_ok = geometry.coverage() >= MIN_NODE_COVERAGE
    monotone_ok = bool(np.all(np.diff(geometry.lumen_radius) > -3.0))
    radius_ok = bool(np.all(geometry.wall_radius >= geometry.lumen_radius))
    return 0 if (coverage_ok and monotone_ok and radius_ok) else 1


def topology_class(geometry: ScanGeometry) -> int:
    """Bucket the scan by sac volume so the residual lookup can be stratified."""

    _, _, sheath = annular_volumes(geometry)
    summary = summarise(geometry)
    scale = sheath / 1000.0 + summary.ilt_fraction * 10.0
    return int(np.clip(int(scale // 25.0), 0, N_TOPOLOGY_CLASSES - 1))


def boundary_condition_scale(geometry: ScanGeometry) -> float:
    """How strongly this scan's inflow swings the shear field, in (0, 2]."""

    summary = summarise(geometry)
    ratio = summary.diameter_mm / 40.0
    return float(np.clip(ratio * (1.0 + summary.ilt_fraction), 0.25, 2.0))
