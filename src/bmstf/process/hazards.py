"""Crossing law, stratified event hazard and the rapid-expansion laws.

Ref: Eq. (1) (the repair-threshold crossing law at alpha_1 and the stratified
eligibility gate Lambda_strat(D_K, sex)(h) <= alpha_2 enforced on published
diameter-by-sex strata); Algorithm 2 line 18 (the per-path event law chi(D_s, B_s));
Sec. "Cohort delineation" (rapid expansion of at least 5 mm per year, the
co-secondary 2.5 mm per year and the confirmatory 10 mm per year).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEX_REPAIR_THRESHOLD_MM: dict[str, float] = {"male": 55.0, "female": 50.0}
SEX_KEYS: tuple[str, ...] = ("male", "female")

BAND_EDGES_MM: tuple[float, ...] = (30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0)
MALE_RUPTURE_RATE_PER_YEAR: tuple[float, ...] = (
    0.0020,
    0.0030,
    0.0045,
    0.0070,
    0.0120,
    0.0250,
    0.0450,
    0.0750,
)
FEMALE_TO_MALE_RATIO = 4.0

RAPID_EXPANSION_PRIMARY = 5.0
RAPID_EXPANSION_CO_SECONDARY = 2.5
RAPID_EXPANSION_CONFIRMATORY = 10.0

BURDEN_REFERENCE = 0.35
BURDEN_HAZARD_COUPLING = 1.80


@dataclass(frozen=True, slots=True)
class StratumRisk:
    """Published diameter-by-sex event strata used by the eligibility gate."""

    male_rate: tuple[float, ...] = MALE_RUPTURE_RATE_PER_YEAR
    female_ratio: float = FEMALE_TO_MALE_RATIO

    def annual_rate(self, diameter_mm: float, sex: str) -> float:
        index = (
            int(np.searchsorted(np.asarray(BAND_EDGES_MM), float(diameter_mm), side="right")) - 1
        )
        index = int(np.clip(index, 0, len(self.male_rate) - 1))
        rate = float(self.male_rate[index])
        if sex == "female":
            rate *= float(self.female_ratio)
        return rate

    def band_label(self, diameter_mm: float) -> str:
        index = (
            int(np.searchsorted(np.asarray(BAND_EDGES_MM), float(diameter_mm), side="right")) - 1
        )
        index = int(np.clip(index, 0, len(self.male_rate) - 1))
        lower = BAND_EDGES_MM[index]
        upper = BAND_EDGES_MM[index + 1] if index + 1 < len(BAND_EDGES_MM) else 70.0
        return f"{lower:.0f}-{upper:.0f}"

    def cumulative_event_probability(self, diameter_mm: float, sex: str, horizon: float) -> float:
        """Lambda_strat(h) = 1 - (1 - annual)^{h/12}."""

        annual = self.annual_rate(diameter_mm, sex)
        return float(1.0 - (1.0 - annual) ** (float(horizon) / 12.0))

    def gate_month(self, diameter_mm: float, sex: str, grid: np.ndarray, alpha2: float) -> float:
        """The largest admissible grid horizon whose stratified risk stays within alpha_2."""

        admissible = [
            float(value)
            for value in np.asarray(grid, dtype=np.float64)
            if self.cumulative_event_probability(diameter_mm, sex, float(value)) <= float(alpha2)
        ]
        return float(max(admissible)) if admissible else 0.0


def repair_threshold_mm(sex: str) -> float:
    return float(SEX_REPAIR_THRESHOLD_MM.get(sex, SEX_REPAIR_THRESHOLD_MM["male"]))


def monthly_event_rate(annual: float) -> float:
    clipped = float(np.clip(annual, 0.0, 0.999))
    return float(1.0 - (1.0 - clipped) ** (1.0 / 12.0))


def step_hazard(
    diameter_mm: float,
    burden: float,
    sex: str,
    strata: StratumRisk | None = None,
    coupling: float = BURDEN_HAZARD_COUPLING,
) -> float:
    """chi(D, B): the per-month event hazard along a path (Algorithm 2 line 18)."""

    table = strata if strata is not None else StratumRisk()
    base = monthly_event_rate(table.annual_rate(diameter_mm, sex))
    modulation = 1.0 + float(coupling) * (float(burden) - BURDEN_REFERENCE)
    return float(np.clip(base * max(modulation, 0.05), 0.0, 0.5))


def annual_rate_array(
    diameters: np.ndarray, sex: str, strata: StratumRisk | None = None
) -> np.ndarray:
    """Vectorised stratum lookup, matching StratumRisk.annual_rate element by element."""

    table = strata if strata is not None else StratumRisk()
    edges = np.asarray(BAND_EDGES_MM, dtype=np.float64)
    rates = np.asarray(table.male_rate, dtype=np.float64)
    index = np.searchsorted(edges, np.asarray(diameters, dtype=np.float64), side="right") - 1
    index = np.clip(index, 0, rates.shape[0] - 1)
    selected = rates[index]
    if sex == "female":
        selected = selected * float(table.female_ratio)
    return np.asarray(selected, dtype=np.float64)


def step_hazard_array(
    diameters: np.ndarray,
    burdens: np.ndarray,
    sex: str,
    strata: StratumRisk | None = None,
    coupling: float = BURDEN_HAZARD_COUPLING,
) -> np.ndarray:
    """Vectorised per-month hazard with the same value as step_hazard."""

    annual = annual_rate_array(diameters, sex, strata)
    base = 1.0 - (1.0 - np.clip(annual, 0.0, 0.999)) ** (1.0 / 12.0)
    modulation = 1.0 + float(coupling) * (np.asarray(burdens, dtype=np.float64) - BURDEN_REFERENCE)
    return np.asarray(np.clip(base * np.maximum(modulation, 0.05), 0.0, 0.5), dtype=np.float64)


def cumulative_hazard(
    diameters: np.ndarray,
    burdens: np.ndarray,
    sex: str,
    strata: StratumRisk | None = None,
) -> np.ndarray:
    """1 - prod_{s<=h}(1 - chi(D_s, B_s)) evaluated at every prefix."""

    diameters = np.asarray(diameters, dtype=np.float64)
    burdens = np.asarray(burdens, dtype=np.float64)
    if diameters.shape != burdens.shape:
        raise ValueError("diameters and burdens must share a shape")
    hazards = step_hazard_array(diameters, burdens, sex, strata)
    survival = np.cumprod(1.0 - hazards)
    return np.asarray(1.0 - survival, dtype=np.float64)


def first_crossing_month(months: np.ndarray, diameters: np.ndarray, sex: str) -> float | None:
    """Linear interpolation of the first grid time at which D reaches the threshold."""

    threshold = repair_threshold_mm(sex)
    values = np.asarray(diameters, dtype=np.float64)
    grid = np.asarray(months, dtype=np.float64)
    hits = np.flatnonzero(values >= threshold)
    if hits.size == 0:
        return None
    index = int(hits[0])
    if index == 0:
        return float(grid[0])
    previous = values[index - 1]
    span = values[index] - previous
    if span <= 0.0:
        return float(grid[index])
    fraction = (threshold - previous) / span
    return float(grid[index - 1] + fraction * (grid[index] - grid[index - 1]))


def rapid_expansion_month(
    months: np.ndarray, diameters: np.ndarray, rate_mm_per_year: float
) -> float | None:
    """First time the annualised between-observation slope reaches the given rate."""

    grid = np.asarray(months, dtype=np.float64)
    values = np.asarray(diameters, dtype=np.float64)
    if grid.size < 2:
        return None
    span = np.diff(grid)
    slope = np.diff(values) / np.where(span > 0.0, span, np.inf) * 12.0
    hits = np.flatnonzero(slope >= float(rate_mm_per_year))
    return float(grid[int(hits[0]) + 1]) if hits.size else None


def first_firing_month(
    diameters: np.ndarray, burdens: np.ndarray, sex: str, uniform: float
) -> int | None:
    """The first monthly index at which the cumulative event hazard passes a uniform draw."""

    cumulative = cumulative_hazard(diameters, burdens, sex)
    hits = np.flatnonzero(cumulative >= float(uniform))
    return int(hits[0]) if hits.size else None


@dataclass(frozen=True, slots=True)
class CompositeOutcome:
    crossing_month: float | None
    event_month: float | None
    rapid_primary_month: float | None
    rapid_co_secondary_month: float | None
    rapid_confirmatory_month: float | None

    def as_mapping(self) -> dict[str, float]:
        def value(entry: float | None) -> float:
            return float("nan") if entry is None else float(entry)

        return {
            "crossing_month": value(self.crossing_month),
            "event_month": value(self.event_month),
            "rapid_primary_month": value(self.rapid_primary_month),
            "rapid_co_secondary_month": value(self.rapid_co_secondary_month),
            "rapid_confirmatory_month": value(self.rapid_confirmatory_month),
        }


def composite_outcome(
    months: np.ndarray, diameters: np.ndarray, burdens: np.ndarray, sex: str, uniform: float
) -> CompositeOutcome:
    firing = first_firing_month(diameters, burdens, sex, uniform)
    event_month = float(np.asarray(months)[firing]) if firing is not None else None
    return CompositeOutcome(
        crossing_month=first_crossing_month(months, diameters, sex),
        event_month=event_month,
        rapid_primary_month=rapid_expansion_month(months, diameters, RAPID_EXPANSION_PRIMARY),
        rapid_co_secondary_month=rapid_expansion_month(
            months, diameters, RAPID_EXPANSION_CO_SECONDARY
        ),
        rapid_confirmatory_month=rapid_expansion_month(
            months, diameters, RAPID_EXPANSION_CONFIRMATORY
        ),
    )
