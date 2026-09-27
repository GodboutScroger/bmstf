"""Percentile bootstrap intervals over seeds and over records.

Ref: Table 1 caption (every value is the mean over 20 seeds with its 95 %
percentile-bootstrap interval over seeds); Algorithm 3 line 16 (B record-level
bootstrap resamples with percentile intervals).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LEVEL = 0.95
DEFAULT_RESAMPLES = 2000


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    level: float
    resamples: int

    def as_mapping(self) -> dict[str, float]:
        return {
            "estimate": float(self.estimate),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "level": float(self.level),
        }


def percentile_interval(
    values: np.ndarray,
    level: float = DEFAULT_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> Interval:
    """Percentile bootstrap of the mean over the supplied units."""

    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size == 0:
        return Interval(estimate=0.0, lower=0.0, upper=0.0, level=level, resamples=0)
    generator = rng if rng is not None else np.random.default_rng(0)
    draws = generator.integers(0, sample.size, size=(int(resamples), sample.size))
    means = np.mean(sample[draws], axis=1)
    alpha = 0.5 * (1.0 - float(level))
    return Interval(
        estimate=float(np.mean(sample)),
        lower=float(np.quantile(means, alpha)),
        upper=float(np.quantile(means, 1.0 - alpha)),
        level=float(level),
        resamples=int(resamples),
    )


def direct_percentile_interval(values: np.ndarray, level: float = DEFAULT_LEVEL) -> Interval:
    """Percentile interval of the supplied values themselves, for cross-checks."""

    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size == 0:
        return Interval(estimate=0.0, lower=0.0, upper=0.0, level=level, resamples=0)
    alpha = 0.5 * (1.0 - float(level))
    return Interval(
        estimate=float(np.mean(sample)),
        lower=float(np.quantile(sample, alpha)),
        upper=float(np.quantile(sample, 1.0 - alpha)),
        level=float(level),
        resamples=int(sample.size),
    )


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    statistic: str = "mean",
    level: float = DEFAULT_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> Interval:
    """Record-level resampling: whole clusters are drawn with replacement."""

    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = np.asarray(clusters)
    if sample.shape[0] != labels.shape[0]:
        raise ValueError("values and clusters must share a length")
    unique = np.unique(labels)
    if unique.size == 0:
        return Interval(estimate=0.0, lower=0.0, upper=0.0, level=level, resamples=0)
    generator = rng if rng is not None else np.random.default_rng(0)
    positions = {label: np.flatnonzero(labels == label) for label in unique}
    estimates = np.empty(int(resamples), dtype=np.float64)
    for draw in range(int(resamples)):
        chosen = generator.integers(0, unique.size, size=unique.size)
        gathered = np.concatenate([positions[unique[index]] for index in chosen])
        estimates[draw] = _statistic(sample[gathered], statistic)
    alpha = 0.5 * (1.0 - float(level))
    return Interval(
        estimate=_statistic(sample, statistic),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        level=float(level),
        resamples=int(resamples),
    )


def _statistic(values: np.ndarray, name: str) -> float:
    if name == "mean":
        return float(np.mean(values))
    if name == "median":
        return float(np.median(values))
    if name == "sum":
        return float(np.sum(values))
    raise ValueError(f"unknown statistic {name!r}")


def paired_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("paired arrays must share a shape")
    return left - right
