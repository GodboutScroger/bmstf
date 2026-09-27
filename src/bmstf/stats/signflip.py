"""Exact sign-flip test over seed-paired differences.

Ref: Table 1 caption (the exact sign-flip p value Holm-adjusted within each metric).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EXACT_LIMIT = 22


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    statistic: float
    p_value: float
    exact: bool
    sign_vectors: int

    def as_mapping(self) -> dict[str, float]:
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "exact": float(self.exact),
            "sign_vectors": float(self.sign_vectors),
        }


def sign_flip_p_value(differences: np.ndarray) -> SignFlipResult:
    """Two-sided exact sign-flip p value for a paired mean difference."""

    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    count = values.shape[0]
    if count == 0:
        return SignFlipResult(statistic=0.0, p_value=1.0, exact=True, sign_vectors=0)
    observed = float(np.sum(values))
    if count <= EXACT_LIMIT:
        sums = np.zeros(1, dtype=np.float64)
        for value in values:
            sums = np.concatenate([sums + value, sums - value])
        threshold = abs(observed) - 1.0e-12
        extreme = int(np.count_nonzero(np.abs(sums) >= threshold))
        return SignFlipResult(
            statistic=observed,
            p_value=float(extreme) / float(sums.shape[0]),
            exact=True,
            sign_vectors=int(sums.shape[0]),
        )
    rng = np.random.default_rng(20260925)
    draws = rng.integers(0, 2, size=(200_000, count)) * 2.0 - 1.0
    permuted = np.abs(draws @ values)
    p_value = float((np.count_nonzero(permuted >= abs(observed)) + 1.0) / (draws.shape[0] + 1.0))
    return SignFlipResult(
        statistic=observed, p_value=p_value, exact=False, sign_vectors=int(draws.shape[0])
    )


def enumerate_sign_flip_sum(differences: np.ndarray) -> np.ndarray:
    """All 2^n signed sums, used to check the p value by direct enumeration."""

    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    sums = np.zeros(1, dtype=np.float64)
    for value in values:
        sums = np.concatenate([sums + value, sums - value])
    return sums
