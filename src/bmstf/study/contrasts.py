"""Seed-paired contrasts, intervals and the Holm family.

Ref: Table 1 caption (delta rows are seed-paired differences, BM-STF minus the
comparator, with the bootstrap interval and the exact sign-flip p value
Holm-adjusted within each metric across the eleven comparators of Tables 1 and 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bmstf.stats.bootstrap import percentile_interval
from bmstf.stats.multiplicity import holm_bonferroni
from bmstf.stats.signflip import sign_flip_p_value

REFERENCE_ARM = "BM-STF"


@dataclass(frozen=True, slots=True)
class SeedContrast:
    metric: str
    comparator: str
    reference: float
    comparator_value: float
    delta: float
    lower: float
    upper: float
    p_value: float
    adjusted_p: float
    anchor: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "comparator": self.comparator,
            "reference": float(self.reference),
            "comparator_value": float(self.comparator_value),
            "delta": float(self.delta),
            "interval": [float(self.lower), float(self.upper)],
            "p_value": float(self.p_value),
            "adjusted_p": float(self.adjusted_p),
            "anchor": self.anchor,
        }


def seed_values(blocks: list[dict[str, dict[str, float]]], arm: str, metric: str) -> np.ndarray:
    values = [
        float(block[arm][metric]) for block in blocks if arm in block and metric in block[arm]
    ]
    return np.asarray(values, dtype=np.float64)


def contrast_across_seeds(
    blocks: list[dict[str, dict[str, float]]],
    metric: str,
    comparator: str,
    reference: str = REFERENCE_ARM,
    anchor: str = "Table 1 caption (seed-paired delta rows)",
    resamples: int = 4000,
) -> SeedContrast:
    left = seed_values(blocks, reference, metric)
    right = seed_values(blocks, comparator, metric)
    count = min(left.shape[0], right.shape[0])
    left, right = left[:count], right[:count]
    differences = left - right
    summary = percentile_interval(differences, resamples=resamples)
    flip = sign_flip_p_value(differences)
    return SeedContrast(
        metric=metric,
        comparator=comparator,
        reference=float(np.mean(left)) if left.size else float("nan"),
        comparator_value=float(np.mean(right)) if right.size else float("nan"),
        delta=float(np.mean(differences)) if differences.size else float("nan"),
        lower=float(summary.lower),
        upper=float(summary.upper),
        p_value=float(flip.p_value),
        adjusted_p=float("nan"),
        anchor=anchor,
    )


def contrast_family(
    blocks: list[dict[str, dict[str, float]]],
    metrics: tuple[str, ...],
    comparators: tuple[str, ...],
    reference: str = REFERENCE_ARM,
) -> dict[str, list[SeedContrast]]:
    """One Holm family per metric, as the caption specifies."""

    family: dict[str, list[SeedContrast]] = {}
    for metric in metrics:
        contrasts = [
            contrast_across_seeds(blocks, metric, comparator, reference)
            for comparator in comparators
        ]
        p_values = np.asarray([contrast.p_value for contrast in contrasts], dtype=np.float64)
        adjusted = holm_bonferroni(p_values)
        family[metric] = [
            SeedContrast(
                metric=contrast.metric,
                comparator=contrast.comparator,
                reference=contrast.reference,
                comparator_value=contrast.comparator_value,
                delta=contrast.delta,
                lower=contrast.lower,
                upper=contrast.upper,
                p_value=contrast.p_value,
                adjusted_p=float(adjusted[position]),
                anchor=contrast.anchor,
            )
            for position, contrast in enumerate(contrasts)
        ]
    return family


def flatten_family(family: dict[str, list[SeedContrast]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contrasts in family.values():
        for contrast in contrasts:
            rows.append(contrast.as_mapping())
    return rows


def contrast_lookup(
    family: dict[str, list[SeedContrast]], metric: str, comparator: str
) -> SeedContrast | None:
    for contrast in family.get(metric, []):
        if contrast.comparator == comparator:
            return contrast
    return None


def component_sum(family: dict[str, list[SeedContrast]]) -> float:
    """The signed sum of the five single-component burden-coverage losses."""

    components = (
        "scalar_noise",
        "no_ultrasound",
        "single_regime_mean_drift",
        "single_barrier",
        "diameter_only",
    )
    total = 0.0
    for comparator in components:
        contrast = contrast_lookup(family, "coverage_burden", comparator)
        if contrast is not None:
            total += float(contrast.delta)
    return float(total)


def snapshot_margin(family: dict[str, list[SeedContrast]]) -> float:
    contrast = contrast_lookup(family, "coverage_burden", "snapshot")
    return float(contrast.delta) if contrast is not None else float("nan")
