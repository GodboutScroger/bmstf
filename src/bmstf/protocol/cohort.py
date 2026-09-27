"""Transcription of the retrospective supporting cohort delineation card.

Ref: Table 4 (per-site delineation of the consecutively accrued de-identified
surveillance records that form the retrospective supporting cohort and the
silent-mode arm). Counts are transcribed as printed so the arithmetic can be
recomputed from the manuscript's own numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

SITE_A = "Site A (I)"
SITE_B = "Site B (I)"
SITE_C = "Site C (II)"
RETROSPECTIVE = "Retrospective cohort"
SILENT_ARM = "Silent-mode arm"

SITE_KEYS: tuple[str, ...] = (SITE_A, SITE_B, SITE_C)
COLUMN_KEYS: tuple[str, ...] = (SITE_A, SITE_B, SITE_C, RETROSPECTIVE, SILENT_ARM)
RETROSPECTIVE_KEYS: tuple[str, ...] = (SITE_A, SITE_B, SITE_C)

AUDIT_PAIR_TARGET: tuple[int, int] = (600, 700)
AUDIT_PAIR_TARGET_FRACTION = 0.15
TOPOLOGY_GATE_FAILURE_PERCENT = 6.8
KAPPA_VALUES: dict[str, float] = {
    SITE_A: 0.94,
    SITE_B: 0.92,
    SITE_C: 0.95,
    RETROSPECTIVE: 0.94,
    SILENT_ARM: 0.93,
}
COMPOSITE_EVENT_24M: dict[str, int] = {
    SITE_A: 318,
    SITE_B: 289,
    SITE_C: 341,
    RETROSPECTIVE: 1026,
    SILENT_ARM: 121,
}
CROSSING_24M: dict[str, int] = {
    SITE_A: 236,
    SITE_B: 214,
    SITE_C: 258,
    RETROSPECTIVE: 764,
    SILENT_ARM: 79,
}
RAPID_EXPANSION_24M: dict[str, int] = {
    SITE_A: 129,
    SITE_B: 116,
    SITE_C: 141,
    RETROSPECTIVE: 404,
    SILENT_ARM: 46,
}
RUPTURE_SYMPTOMATIC_24M: dict[str, int] = {
    SITE_A: 27,
    SITE_B: 24,
    SITE_C: 31,
    RETROSPECTIVE: 86,
    SILENT_ARM: 6,
}
ELECTIVE_REPAIR_24M: dict[str, int] = {
    SITE_A: 402,
    SITE_B: 371,
    SITE_C: 429,
    RETROSPECTIVE: 1268,
    SILENT_ARM: 58,
}
INTER_READER_MEDIAN_ABSOLUTE_MM: dict[str, float] = {
    SITE_A: 1.1,
    SITE_B: 1.3,
    SITE_C: 1.2,
    RETROSPECTIVE: 1.2,
    SILENT_ARM: 1.2,
}


@dataclass(frozen=True, slots=True)
class Column:
    screened: int
    excluded_prior_repair: int
    excluded_suprarenal: int
    excluded_symptomatic: int
    excluded_non_diagnostic: int
    single_observation: int
    at_or_above_threshold: int
    two_or_more_observations: int
    ct_examinations: int
    us_observations: int
    paired_within_three_months: int
    audit_stratum_pairs: int

    @property
    def excluded_total(self) -> int:
        return int(
            self.excluded_prior_repair
            + self.excluded_suprarenal
            + self.excluded_symptomatic
            + self.excluded_non_diagnostic
        )

    @property
    def retained(self) -> int:
        return int(self.screened - self.excluded_total)

    @property
    def split_total(self) -> int:
        return int(self.single_observation + self.two_or_more_observations)


TABLE4: dict[str, Column] = {
    SITE_A: Column(5018, 812, 264, 431, 145, 1898, 987, 1468, 5842, 7214, 1102, 178),
    SITE_B: Column(4247, 704, 219, 388, 121, 1474, 892, 1341, 5196, 6433, 998, 164),
    SITE_C: Column(3968, 663, 198, 352, 108, 1188, 731, 1459, 5731, 6908, 1214, 141),
    RETROSPECTIVE: Column(13233, 2179, 681, 1171, 374, 4560, 2610, 4268, 16769, 20555, 3314, 483),
    SILENT_ARM: Column(2416, 388, 131, 214, 76, 812, 471, 1286, 4113, 5042, 986, 0),
}

INTEGER_FIELDS: tuple[str, ...] = (
    "screened",
    "excluded_prior_repair",
    "excluded_suprarenal",
    "excluded_symptomatic",
    "excluded_non_diagnostic",
    "single_observation",
    "at_or_above_threshold",
    "two_or_more_observations",
    "ct_examinations",
    "us_observations",
    "paired_within_three_months",
    "audit_stratum_pairs",
)


def column(key: str) -> Column:
    return TABLE4[key]


def pooled_sum(field: str) -> int:
    return int(sum(int(getattr(TABLE4[key], field)) for key in SITE_KEYS))


def outcome_sum(values: dict[str, int]) -> int:
    return int(sum(values[key] for key in SITE_KEYS))


def composite_event_rate(key: str) -> float:
    total = TABLE4[key].two_or_more_observations
    if total == 0:
        return 0.0
    return float(COMPOSITE_EVENT_24M[key]) / float(total)


def reported_percentages() -> dict[str, float]:
    return {
        key: round(100.0 * composite_event_rate(key), 1)
        for key in COLUMN_KEYS
        if TABLE4[key].two_or_more_observations > 0
    }
