"""Pre-specified constants transcribed from the manuscript.

Ref: Sec. "Pre-specified clinical protocol: two-layer data design and delineation
cards" (N_b = 64, N_m = 8, the latent dimension selected between 8 and 16);
Clinical protocol criteria (equivalence margin +/-0.05 on the 90 % interval, the
descriptive +/-0.02 band, 0.03 and 0.04 margins, calibration slope 0.9-1.1);
Table 4 (outcome definitions); Code availability (the reported run command).
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALPHA1 = 0.10
ALPHA2 = 0.01
BOUND_LEVEL = 1.645

BOUNDARY_DRAWS = 64
RECONSTRUCTION_DRAWS = 8
TOPOLOGY_INFLATION = 1.6

COHORT_SEEDS = 20
COHORT_PATIENTS = 300
CALIBRATION_HISTORIES = 6000
HORIZONS_MONTHS = (6, 12, 24)
PRIMARY_HORIZON_MONTHS = 24
SECONDARY_HORIZON_MONTHS = 12
MINIMUM_INTERVAL_MONTHS = 3.0
DEFERRAL_CAP = 0.20

LATENT_DIMENSION_LOW = 8
LATENT_DIMENSION_HIGH = 16

EQUIVALENCE_MARGIN = 0.05
EQUIVALENCE_LEVEL = 0.90
DESCRIPTIVE_BAND = 0.02
MECHANISM_MARGIN = 0.03
SECONDARY_MARGIN = 0.04
CALIBRATION_SLOPE_RANGE = (0.9, 1.1)
SUBSTRATE_ERROR_TOLERANCE = 0.10
BETWEEN_SITE_GAP_ALERT = 0.05
BETWEEN_SITE_GAP_REPORTABLE = 0.10
MIN_SUBGROUP_EVENTS = 30

RAPID_EXPANSION_PRIMARY_MM_PER_YEAR = 5.0
RAPID_EXPANSION_CO_SECONDARY_MM_PER_YEAR = 2.5
RAPID_EXPANSION_CONFIRMATORY_MM_PER_YEAR = 10.0

TURNAROUND_MINUTES_PER_SCAN = 10.0
WOMEN_RUPTURE_RATE_RATIO = 4.0
RUPTURE_RATE_RATIO_PER_5MM = 1.91
POOLED_GROWTH_MM_PER_YEAR = 2.38

COMPARATOR_COUNT = 11
TOLERANCE_SWEEP = (0.05, 0.10, 0.15, 0.20)
US_PROBABILITY = 0.35
US_SITE_OFFSET_MM = (-0.65, 0.45, -0.25)
US_STANDARD_DEVIATION_MM = 1.10

REPORTED_RUN_COMMAND = "python3 bmstf_sim.py --seeds 20 --patients 300 --jobs 5"
REPORTED_CALIBRATION_COMMAND = (
    "python3 bmstf_sim.py --seeds 20 --patients 300 --jobs 5 "
    "--methods DIAMETER CAL --merge --cal-patients 6000"
)


@dataclass(frozen=True, slots=True)
class ReportedLimits:
    """Quantities the manuscript states about its own artefacts, kept verbatim."""

    reference_script_version: str = "2.1"
    reference_script_sha256: str = (
        "5df98c3a62bbb2dc" "e85d7f460630d3a620e19fdf313ca7e25f07310e240ad907"
    )
    archived_script_sha256: str = "c94263c03affece4cb409f0397b0bdf667cfd9c715a0c6491571d7c9c255d164"
    reported_python: str = "3.9.6"
    reported_numpy: str = "2.0.2"
    events_feasible: int = 75
    feasible_intervals: int = 7617
    escalation_floor_months: float = 3.0
    observer_windows: tuple[str, ...] = ("pre-window", "in-window", "silent")

    def mismatched_hashes(self) -> list[str]:
        return [
            name
            for name, value in (
                ("reference_script_sha256", self.reference_script_sha256),
                ("archived_script_sha256", self.archived_script_sha256),
            )
            if len(value) != 64
        ]


@dataclass(frozen=True, slots=True)
class ProtocolCriteria:
    """The frozen criteria the manuscript pre-specifies for the two-arm protocol."""

    equivalence_margin: float = EQUIVALENCE_MARGIN
    equivalence_level: float = EQUIVALENCE_LEVEL
    descriptive_band: float = DESCRIPTIVE_BAND
    mechanism_margin: float = MECHANISM_MARGIN
    secondary_margin: float = SECONDARY_MARGIN
    calibration_slope: tuple[float, float] = CALIBRATION_SLOPE_RANGE
    substrate_tolerance: float = SUBSTRATE_ERROR_TOLERANCE
    subgroup_events: int = MIN_SUBGROUP_EVENTS
    deferral_cap: float = DEFERRAL_CAP
    alpha1: float = ALPHA1
    alpha2: float = ALPHA2
    extras: dict[str, float] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, float]:
        mapping = {
            "equivalence_margin": self.equivalence_margin,
            "equivalence_level": self.equivalence_level,
            "descriptive_band": self.descriptive_band,
            "mechanism_margin": self.mechanism_margin,
            "secondary_margin": self.secondary_margin,
            "calibration_slope_low": self.calibration_slope[0],
            "calibration_slope_high": self.calibration_slope[1],
            "substrate_tolerance": self.substrate_tolerance,
            "deferral_cap": self.deferral_cap,
            "alpha1": self.alpha1,
            "alpha2": self.alpha2,
        }
        mapping.update({key: float(value) for key, value in self.extras.items()})
        return mapping
