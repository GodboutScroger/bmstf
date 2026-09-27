"""Axisymmetric aneurysm shapes reconstructed from a latent state.

Ref: Sec. "Here we make the identification explicit" (each serial CT angiogram,
segmented into lumen, thrombus and wall, resampled onto the centreline-canonical
frame). The map from the latent diameter and low-shear burden to the three
annular surfaces is the release's own parameterisation; the manuscript states the
three-layer segmentation but not its generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.geometry.template import DEFAULT_TEMPLATE, ScanGeometry, Template

BULGE_CENTRE_FRACTION = 0.62
BULGE_WIDTH_FRACTION = 0.18
BULGE_AMPLITUDE = 0.55
WALL_THICKNESS_MM = 1.5
MIN_RADIUS_MM = 1.5
ILT_BASE_FRACTION = 0.10
ILT_BURDEN_GAIN = 0.55
ILT_CEILING = 0.70
THICKNESS_BRACKET_MM = (0.0, 26.0)


@dataclass(frozen=True, slots=True)
class ShapeSpec:
    template: Template = DEFAULT_TEMPLATE
    bulge_amplitude: float = BULGE_AMPLITUDE
    bulge_centre_fraction: float = BULGE_CENTRE_FRACTION
    bulge_width_fraction: float = BULGE_WIDTH_FRACTION
    wall_thickness_mm: float = WALL_THICKNESS_MM
    ilt_ceiling: float = ILT_CEILING


def lumen_profile(maximum_diameter_mm: float, spec: ShapeSpec) -> np.ndarray:
    """A smooth bulge whose peak equals half the requested maximum diameter."""

    nodes = spec.template.nodes
    position = np.linspace(0.0, 1.0, nodes)
    bulge = np.exp(
        -0.5 * ((position - spec.bulge_centre_fraction) / spec.bulge_width_fraction) ** 2
    )
    shape = 1.0 + spec.bulge_amplitude * bulge
    radius = 0.5 * float(maximum_diameter_mm) * shape / float(np.max(shape))
    return np.maximum(radius, MIN_RADIUS_MM)


def target_ilt_fraction(burden: float, spec: ShapeSpec) -> float:
    value = ILT_BASE_FRACTION + ILT_BURDEN_GAIN * float(burden)
    return float(np.clip(value, 0.0, spec.ilt_ceiling))


def _annular_fraction(lumen: np.ndarray, thickness: float, wall: float) -> float:
    outer = lumen + thickness
    thrombus = float(np.sum(np.pi * (outer**2 - lumen**2)))
    sheath = float(np.sum(np.pi * (outer + wall) ** 2))
    if sheath <= 0.0:
        return 0.0
    return thrombus / sheath


def solve_ilt_thickness(lumen: np.ndarray, target: float, spec: ShapeSpec) -> float:
    """Bisect the thrombus thickness that reproduces the requested ILT fraction."""

    low, high = THICKNESS_BRACKET_MM
    if _annular_fraction(lumen, high, spec.wall_thickness_mm) <= target:
        return float(high)
    for _ in range(60):
        middle = 0.5 * (low + high)
        if _annular_fraction(lumen, middle, spec.wall_thickness_mm) < target:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


def geometry_from_state(
    index: str,
    diameter_mm: float,
    burden: float,
    spec: ShapeSpec | None = None,
) -> ScanGeometry:
    active = spec if spec is not None else ShapeSpec()
    lumen = lumen_profile(diameter_mm, active)
    thickness = solve_ilt_thickness(lumen, target_ilt_fraction(burden, active), active)
    thrombus = lumen + thickness
    wall = thrombus + active.wall_thickness_mm
    mask = np.ones(active.template.nodes, dtype=bool)
    return ScanGeometry(index, active.template, lumen, thrombus, wall, mask)


def geometry_scale_mm(geometry: ScanGeometry) -> float:
    return float(np.max(geometry.wall_radius))
