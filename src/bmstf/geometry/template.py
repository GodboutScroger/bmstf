"""Centreline-canonical template frame for the aorta.

Ref: Sec. "Here we make the identification explicit" (the resampling onto the
centreline-canonical frame so that a template node is the same piece of wall at
every visit); Algorithm 1 line 1 (template geometry g_k with M nodes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_TEMPLATE_NODES = 48
MIN_NODE_COVERAGE = 0.95


@dataclass(frozen=True, slots=True)
class Template:
    """Arc-length nodes with an orthonormal frame attached to each node."""

    centers: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    binormals: np.ndarray
    spacing_mm: float

    @property
    def nodes(self) -> int:
        return int(self.centers.shape[0])


@dataclass(frozen=True, slots=True)
class ScanGeometry:
    """One serial scan expressed on the template, in either modality."""

    index: str
    template: Template
    lumen_radius: np.ndarray
    thrombus_radius: np.ndarray
    wall_radius: np.ndarray
    node_mask: np.ndarray

    @property
    def nodes(self) -> int:
        return int(self.lumen_radius.shape[0])

    def coverage(self) -> float:
        return float(np.count_nonzero(self.node_mask)) / float(self.nodes)


def build_template(length_mm: float = 110.0, nodes: int = N_TEMPLATE_NODES) -> Template:
    """A straight centreline whose node spacing is the arc-length step."""

    spacing = float(length_mm) / float(nodes - 1)
    axis = np.linspace(0.0, float(length_mm), int(nodes), dtype=np.float64)
    centers = np.stack([axis, np.zeros_like(axis), np.zeros_like(axis)], axis=1)
    tangents = np.tile(np.array([[1.0, 0.0, 0.0]]), (int(nodes), 1))
    normals = np.tile(np.array([[0.0, 1.0, 0.0]]), (int(nodes), 1))
    binormals = np.tile(np.array([[0.0, 0.0, 1.0]]), (int(nodes), 1))
    return Template(centers, tangents, normals, binormals, spacing)


DEFAULT_TEMPLATE = build_template()


def resample_radii(
    centers: np.ndarray,
    radii: np.ndarray,
    template: Template,
) -> np.ndarray:
    """Map a segmented surface onto the template by linear interpolation in arc length."""

    source = np.asarray(centers, dtype=np.float64)
    values = np.asarray(radii, dtype=np.float64)
    if source.ndim != 1 or values.shape != source.shape:
        raise ValueError("centers and radii must be one-dimensional and equally long")
    target = template.centers[:, 0]
    mapped: np.ndarray = np.interp(target, source, values, left=values[0], right=values[-1])
    return mapped


def canonicalise(
    index: str,
    centers: np.ndarray,
    lumen: np.ndarray,
    thrombus: np.ndarray,
    wall: np.ndarray,
    template: Template = DEFAULT_TEMPLATE,
) -> ScanGeometry:
    lumen_radius = resample_radii(centers, lumen, template)
    thrombus_radius = np.maximum(resample_radii(centers, thrombus, template), lumen_radius)
    wall_radius = np.maximum(resample_radii(centers, wall, template), thrombus_radius)
    node_mask = np.ones(template.nodes, dtype=bool)
    return ScanGeometry(index, template, lumen_radius, thrombus_radius, wall_radius, node_mask)


def annular_volumes(geometry: ScanGeometry) -> tuple[float, float, float]:
    """Lumen, thrombus and wall volumes in mm^3 from the axisymmetric radii."""

    step = geometry.template.spacing_mm
    lumen = float(np.sum(np.pi * geometry.lumen_radius**2) * step)
    sheath = float(np.sum(np.pi * geometry.wall_radius**2) * step)
    thrombus = float(
        np.sum(np.pi * (geometry.thrombus_radius**2 - geometry.lumen_radius**2)) * step
    )
    return lumen, thrombus, sheath
