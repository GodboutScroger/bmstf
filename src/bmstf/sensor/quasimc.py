"""Scrambled-Sobol quasi-Monte-Carlo for the boundary-condition marginalisation.

Ref: Algorithm 1 line 2 (draw b_1..b_{N_b} from pi(b | c_k, s) by scrambled-Sobol
quasi-Monte-Carlo).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfinv
from scipy.stats import qmc

from bmstf.sensor.boundary import BoundaryPrior

TAIL_FLOOR = 1.0e-6


def sobol_uniform(count: int, dimension: int, seed: int) -> np.ndarray:
    """Scrambled Sobol points in (0, 1)^dimension, with the point count respected."""

    if count < 1 or dimension < 1:
        raise ValueError("count and dimension must be positive")
    engine = qmc.Sobol(d=int(dimension), scramble=True, seed=int(seed))
    exponent = int(np.ceil(np.log2(max(count, 2))))
    points = engine.random_base2(exponent)
    selected: np.ndarray = np.asarray(points[:count], dtype=np.float64)
    return selected


def normal_from_uniform(uniform: np.ndarray, tail_floor: float = TAIL_FLOOR) -> np.ndarray:
    """Inverse-CDF transform to a standard normal, with the tails clipped."""

    clipped = np.clip(uniform, tail_floor, 1.0 - tail_floor)
    standard: np.ndarray = np.sqrt(2.0) * np.asarray(erfinv(2.0 * clipped - 1.0), dtype=np.float64)
    return standard


def draw_boundary_conditions(
    prior: BoundaryPrior, mean: np.ndarray, count: int, seed: int
) -> np.ndarray:
    """N_b boundary-condition vectors drawn from the covariate-conditioned prior."""

    standard = normal_from_uniform(sobol_uniform(count, prior.dimension, seed))
    draws = np.asarray(mean, dtype=np.float64)[None, :] + standard * prior.scale[None, :]
    bounded: np.ndarray = np.asarray(prior.clip(draws), dtype=np.float64)
    return bounded


def prior_mean(prior: BoundaryPrior, mean: np.ndarray) -> np.ndarray:
    """The prior mean, used as the single boundary condition for the reconstruction pass."""

    bounded: np.ndarray = np.asarray(
        prior.clip(np.asarray(mean, dtype=np.float64)), dtype=np.float64
    )
    return bounded
