"""Diameter-band comparators: the literal guideline schedule and the calibrated bands.

Ref: Table 1 panel A (the two band policies share one point extrapolation of the
diameter and are reported for transparency as a binary point forecast); Table 3
(the calibrated bands re-calibrated at each tolerance on a 6,000-history cohort,
with a band in which no interval on the grid was admissible at calibration as the
escalate state); Methods, the in-silico study (the RESCAN level).

LITERAL_SCHEDULE_MONTHS is the cited ESVS 2024 surveillance schedule, three-yearly
below 40 mm down to referral at the sex-specific threshold; the manuscript does not
print its own literal schedule, which is why its literal-band row is reported NOT_RUN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.policy.wilson import BOUND_LEVEL, wilson_upper

SCHEDULE_BAND_EDGES_MM: tuple[float, ...] = (30.0, 40.0, 45.0, 55.0)
LITERAL_SCHEDULE_MONTHS: tuple[float, ...] = (36.0, 12.0, 3.0, 0.0)
REFERRAL_MONTHS = 0.0


def band_from_diameter(diameter_mm: float) -> str:
    edges = np.asarray(SCHEDULE_BAND_EDGES_MM, dtype=np.float64)
    index = int(np.searchsorted(edges, float(diameter_mm), side="right")) - 1
    index = int(np.clip(index, 0, len(SCHEDULE_BAND_EDGES_MM) - 1))
    upper = SCHEDULE_BAND_EDGES_MM[index + 1] if index + 1 < len(SCHEDULE_BAND_EDGES_MM) else 70.0
    return f"{SCHEDULE_BAND_EDGES_MM[index]:.0f}-{upper:.0f}"


BAND_ORDER: tuple[str, ...] = tuple(
    band_from_diameter(float(edge)) for edge in (30.5, 41.0, 47.0, 56.0)
)


@dataclass(frozen=True, slots=True)
class BandPolicy:
    name: str
    schedule: dict[str, float]
    escalated: tuple[str, ...]

    def interval(self, diameter_mm: float) -> float | None:
        """The scheduled interval, or None when the band escalates or refers for repair."""

        band = band_from_diameter(diameter_mm)
        if band in self.escalated:
            return None
        months = self.schedule.get(band)
        if months is None or float(months) <= REFERRAL_MONTHS:
            return None
        return float(months)

    def is_escalated(self, diameter_mm: float) -> bool:
        band = band_from_diameter(diameter_mm)
        if band in self.escalated:
            return True
        months = self.schedule.get(band)
        return months is None or float(months) <= REFERRAL_MONTHS

    def as_mapping(self) -> dict[str, float]:
        mapping = {"escalated_bands": float(len(self.escalated))}
        for band in BAND_ORDER:
            mapping[f"schedule_{band}"] = float(self.schedule.get(band, -1.0))
        return mapping


def literal_band_policy() -> BandPolicy:
    schedule = {
        band: float(months)
        for band, months in zip(BAND_ORDER, LITERAL_SCHEDULE_MONTHS, strict=True)
    }
    return BandPolicy(name="literal_bands", schedule=schedule, escalated=())


def calibrate_band_policy(
    band_ids: np.ndarray,
    within_horizon: np.ndarray,
    grid: np.ndarray,
    alpha1: float,
    bound_level: float = BOUND_LEVEL,
    name: str = "calibrated_bands",
) -> BandPolicy:
    """Pick, per band, the longest grid interval whose Wilson upper bound stays in alpha_1."""

    labels = np.asarray(band_ids)
    hits = np.asarray(within_horizon, dtype=bool)
    schedule: dict[str, float] = {}
    escalated: list[str] = []
    for band in BAND_ORDER:
        selected = labels == band
        count = int(np.count_nonzero(selected))
        admissible: list[float] = []
        for position, horizon in enumerate(np.asarray(grid, dtype=np.float64)):
            if count == 0:
                break
            proportion = float(np.count_nonzero(hits[selected, position])) / float(count)
            if wilson_upper(proportion, count, bound_level) <= float(alpha1):
                admissible.append(float(horizon))
        if admissible:
            schedule[band] = float(max(admissible))
        else:
            escalated.append(band)
    return BandPolicy(name=name, schedule=schedule, escalated=tuple(escalated))


def extrapolate_diameter(months: np.ndarray, diameters: np.ndarray, target_month: float) -> float:
    """Linear point extrapolation of the observed diameter history to a horizon."""

    grid = np.asarray(months, dtype=np.float64)
    values = np.asarray(diameters, dtype=np.float64)
    if grid.shape[0] < 2:
        return float(values[-1])
    span = grid[-1] - grid[0]
    if abs(span) < 1.0e-9:
        return float(values[-1])
    slope = (values[-1] - values[0]) / span
    return float(values[-1] + slope * (float(target_month) - float(grid[-1])))


def point_crossing_probability(
    months: np.ndarray, diameters: np.ndarray, target_month: float, threshold_mm: float
) -> float:
    """The binary point forecast the band comparator reads out (Table 1 panel A)."""

    projected = extrapolate_diameter(months, diameters, target_month)
    return 1.0 if projected >= float(threshold_mm) else 0.0
