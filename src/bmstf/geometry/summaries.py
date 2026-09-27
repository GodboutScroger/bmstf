"""Scan-level geometry summaries that drive the exogenous drift.

Ref: Algorithm 2 line 6 (the exogenous drift u(g_k) built from the scan-level
geometry summaries); Sec. "Two-regime latent shear-state model" (the drift input).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.geometry.template import ScanGeometry, annular_volumes

GEOMETRY_FEATURES: tuple[str, ...] = (
    "diameter_mm",
    "ilt_fraction",
    "sac_volume_cm3",
    "tortuosity",
)

DIAMETER_SCALE_MM = 55.0
VOLUME_SCALE_CM3 = 200.0


@dataclass(frozen=True, slots=True)
class GeometrySummary:
    diameter_mm: float
    ilt_fraction: float
    sac_volume_cm3: float
    tortuosity: float

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.diameter_mm / DIAMETER_SCALE_MM,
                self.ilt_fraction,
                self.sac_volume_cm3 / VOLUME_SCALE_CM3,
                self.tortuosity,
            ],
            dtype=np.float64,
        )

    def as_mapping(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(GEOMETRY_FEATURES, self.as_vector(), strict=True)
        }


def maximum_diameter_mm(geometry: ScanGeometry) -> float:
    return float(2.0 * np.max(geometry.lumen_radius[geometry.node_mask]))


def thrombus_fraction(geometry: ScanGeometry) -> float:
    """Intraluminal-thrombus volume over the wall volume, in [0, 1]."""

    _, thrombus, sheath = annular_volumes(geometry)
    if sheath <= 0.0:
        return 0.0
    return float(np.clip(thrombus / sheath, 0.0, 1.0))


def sac_volume_cm3(geometry: ScanGeometry) -> float:
    _, _, sheath = annular_volumes(geometry)
    return float(sheath / 1000.0)


def tortuosity(geometry: ScanGeometry) -> float:
    centers = geometry.template.centers
    arc = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    chord = float(np.linalg.norm(centers[-1] - centers[0]))
    if chord <= 0.0:
        return 1.0
    return float(arc / chord)


def summarise(geometry: ScanGeometry) -> GeometrySummary:
    return GeometrySummary(
        diameter_mm=maximum_diameter_mm(geometry),
        ilt_fraction=thrombus_fraction(geometry),
        sac_volume_cm3=sac_volume_cm3(geometry),
        tortuosity=tortuosity(geometry),
    )


def covariate_vector(summary: GeometrySummary, site_index: int) -> np.ndarray:
    """The design covariates c_k that condition the boundary-condition prior."""

    base = summary.as_vector()
    one_hot = np.zeros(3, dtype=np.float64)
    one_hot[int(site_index) % 3] = 1.0
    return np.concatenate([base, one_hot])
