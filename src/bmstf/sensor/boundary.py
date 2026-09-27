"""Boundary-condition prior over the unmeasured inflow and outflow parameters.

Ref: Algorithm 1 lines 1-2 (the prior pi(b | c_k, s) and the scrambled-Sobol draws).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bmstf.geometry.summaries import GeometrySummary, covariate_vector

BOUNDARY_FEATURES: tuple[str, ...] = (
    "inflow_amplitude_l_min",
    "cardiac_period_s",
    "outlet_resistance_ratio",
    "blood_viscosity_cp",
    "waveform_steepness",
)

BOUNDARY_LOWER: tuple[float, ...] = (2.4, 0.55, 0.35, 2.6, -0.6)
BOUNDARY_UPPER: tuple[float, ...] = (6.8, 1.25, 0.85, 4.4, 0.9)

SITE_OFFSETS: tuple[tuple[float, ...], ...] = (
    (0.00, 0.00, 0.00, 0.00, 0.00),
    (0.18, -0.03, 0.04, 0.05, 0.06),
    (-0.14, 0.02, -0.05, -0.04, -0.05),
)


@dataclass(frozen=True, slots=True)
class BoundaryPrior:
    """A Gaussian prior whose mean is a linear function of the scan covariates."""

    centre: np.ndarray = field(
        default_factory=lambda: np.asarray([4.6, 0.86, 0.60, 3.5, 0.15], dtype=np.float64)
    )
    loadings: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [
                [0.55, 0.30, 0.20, -0.10, 0.05, 0.02, -0.01],
                [-0.05, 0.02, 0.01, 0.03, -0.01, 0.00, 0.01],
                [0.10, -0.06, 0.04, 0.02, 0.03, 0.02, -0.02],
                [0.02, 0.01, -0.01, 0.00, 0.01, 0.00, 0.00],
                [0.12, 0.05, -0.02, 0.01, 0.02, -0.01, 0.01],
            ],
            dtype=np.float64,
        )
    )
    scale: np.ndarray = field(
        default_factory=lambda: np.asarray([0.62, 0.075, 0.055, 0.30, 0.10], dtype=np.float64)
    )

    @property
    def dimension(self) -> int:
        return int(self.centre.shape[0])

    def mean_vector(self, summary: GeometrySummary, site_index: int) -> np.ndarray:
        covariates = covariate_vector(summary, site_index)
        if covariates.shape[0] != self.loadings.shape[1]:
            raise ValueError("covariate vector does not match the prior loadings")
        offset = np.asarray(SITE_OFFSETS[int(site_index) % len(SITE_OFFSETS)], dtype=np.float64)
        mean = self.centre + self.loadings @ covariates + offset
        bounded: np.ndarray = np.asarray(self.clip(mean), dtype=np.float64)
        return bounded

    def clip(self, values: np.ndarray) -> np.ndarray:
        lower = np.asarray(BOUNDARY_LOWER, dtype=np.float64)
        upper = np.asarray(BOUNDARY_UPPER, dtype=np.float64)
        clipped: np.ndarray = np.clip(values, lower, upper)
        return clipped

    def covariance(self) -> np.ndarray:
        matrix: np.ndarray = np.diag(self.scale**2)
        return matrix

    def standardise(self, values: np.ndarray) -> np.ndarray:
        standardised: np.ndarray = (values - self.centre) / self.scale
        return standardised

    def sampling_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(BOUNDARY_LOWER, dtype=np.float64),
            np.asarray(BOUNDARY_UPPER, dtype=np.float64),
        )
