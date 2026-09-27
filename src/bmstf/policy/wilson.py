"""Wilson score bound for the Monte-Carlo crossing estimate.

Ref: Eq. (5): U_z(p_hat, M_mc) = (p_hat + z^2 / (2 M_mc) + z sqrt(p_hat(1-p_hat)/M_mc
+ z^2 / (4 M_mc^2))) / (1 + z^2 / M_mc), used so that the policy is checked on the
one-sided upper confidence bound rather than on the point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOUND_LEVEL = 1.645


@dataclass(frozen=True, slots=True)
class WilsonBound:
    lower: float
    upper: float
    centre: float
    half_width: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "centre": float(self.centre),
            "half_width": float(self.half_width),
        }


def wilson_bounds(proportion: float, count: int, z: float = BOUND_LEVEL) -> WilsonBound:
    """The Wilson score interval, written from the definition."""

    if count <= 0:
        return WilsonBound(lower=0.0, upper=1.0, centre=float(proportion), half_width=1.0)
    estimate = float(np.clip(proportion, 0.0, 1.0))
    level = float(z)
    denominator = 1.0 + level * level / float(count)
    centre = (estimate + level * level / (2.0 * count)) / denominator
    spread = (
        level
        * float(
            np.sqrt(
                estimate * (1.0 - estimate) / float(count)
                + level * level / (4.0 * float(count) ** 2)
            )
        )
        / denominator
    )
    return WilsonBound(
        lower=float(np.clip(centre - spread, 0.0, 1.0)),
        upper=float(np.clip(centre + spread, 0.0, 1.0)),
        centre=float(centre),
        half_width=float(spread),
    )


def wilson_upper(proportion: float, count: int, z: float = BOUND_LEVEL) -> float:
    return float(wilson_bounds(proportion, count, z).upper)


def standard_error(proportion: float, count: int) -> float:
    if count <= 0:
        return 0.0
    estimate = float(np.clip(proportion, 0.0, 1.0))
    return float(np.sqrt(estimate * (1.0 - estimate) / float(count)))
