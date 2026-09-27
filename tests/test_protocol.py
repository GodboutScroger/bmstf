"""Pre-specified constants, the reported tables and the cohort arithmetic."""

from __future__ import annotations

import numpy as np

from bmstf.protocol import cohort, criteria, margins, strata


def test_criteria_match_the_manuscript() -> None:
    assert criteria.ALPHA1 == 0.10
    assert criteria.ALPHA2 == 0.01
    assert criteria.BOUNDARY_DRAWS == 64
    assert criteria.RECONSTRUCTION_DRAWS == 8
    assert criteria.EQUIVALENCE_MARGIN == 0.05
    assert criteria.DESCRIPTIVE_BAND == 0.02
    assert criteria.MECHANISM_MARGIN == 0.03
    assert criteria.SECONDARY_MARGIN == 0.04
    assert criteria.CALIBRATION_SLOPE_RANGE == (0.9, 1.1)
    assert criteria.SUBSTRATE_ERROR_TOLERANCE == 0.10
    assert criteria.COHORT_SEEDS == 20 and criteria.COHORT_PATIENTS == 300
    assert criteria.CALIBRATION_HISTORIES == 6000
    assert criteria.TOLERANCE_SWEEP == (0.05, 0.10, 0.15, 0.20)
    assert criteria.ReportedLimits().events_feasible == 75
    assert criteria.ReportedLimits().feasible_intervals == 7617


def test_protocol_criteria_mapping_is_complete() -> None:
    mapping = criteria.ProtocolCriteria().as_mapping()
    for key in (
        "equivalence_margin",
        "descriptive_band",
        "mechanism_margin",
        "secondary_margin",
        "calibration_slope_low",
        "calibration_slope_high",
        "substrate_tolerance",
        "deferral_cap",
        "alpha1",
        "alpha2",
    ):
        assert key in mapping


def test_reported_script_digests_are_well_formed() -> None:
    limits = criteria.ReportedLimits()
    assert limits.mismatched_hashes() == []
    assert len(limits.reference_script_sha256) == 64
    assert limits.reported_python == "3.9.6" and limits.reported_numpy == "2.0.2"


def test_table1_delta_rows_recompute_within_rounding() -> None:
    for contrast in strata.TABLE1_PANEL_A + strata.TABLE1_PANEL_B + strata.TABLE1_PANEL_B_LITERAL:
        assert contrast.consistent(), contrast.label
        assert abs(contrast.recomputed() - contrast.reported_delta) <= strata.ROUNDING_TOLERANCE


def test_table2_single_component_sum_reproduces_the_printed_total() -> None:
    total = sum(value for _, value in strata.FIVE_COMPONENT_DELTAS)
    assert abs(total - strata.REPORTED_COMPONENT_SUM) < 1.0e-9


def test_comparator_count_matches_the_caption() -> None:
    assert strata.comparator_count() == criteria.COMPARATOR_COUNT == 11


def test_table4_pooled_column_is_the_site_sum() -> None:
    for field in cohort.INTEGER_FIELDS:
        assert cohort.pooled_sum(field) == getattr(
            cohort.TABLE4[cohort.RETROSPECTIVE], field
        ), field


def test_table4_retained_split_reproduces() -> None:
    for key in (*cohort.RETROSPECTIVE_KEYS, cohort.RETROSPECTIVE):
        entry = cohort.TABLE4[key]
        assert entry.retained == entry.screened - entry.excluded_total
        assert entry.split_total == entry.retained


def test_table4_pooled_outcome_counts_do_not_equal_the_site_sum() -> None:
    """The manuscript's pooled outcome counts exceed the site sums; pinned as printed."""

    assert cohort.outcome_sum(cohort.COMPOSITE_EVENT_24M) == 948
    assert cohort.COMPOSITE_EVENT_24M[cohort.RETROSPECTIVE] == 1026
    assert cohort.outcome_sum(cohort.CROSSING_24M) == 708
    assert cohort.CROSSING_24M[cohort.RETROSPECTIVE] == 764
    assert cohort.outcome_sum(cohort.ELECTIVE_REPAIR_24M) == 1202
    assert cohort.ELECTIVE_REPAIR_24M[cohort.RETROSPECTIVE] == 1268


def test_table4_silent_arm_is_internally_inconsistent() -> None:
    silent = cohort.TABLE4[cohort.SILENT_ARM]
    assert silent.split_total == 2098
    assert silent.retained == 1607
    assert silent.split_total != silent.retained


def test_manuscript_checks_report_the_pinned_discrepancies() -> None:
    checks = margins.all_manuscript_checks()
    failures = {check.name for check in checks if not check.passed()}
    assert len(checks) == 108
    expected = {
        "table4_outcome_pooled_matches_site_sum_composite_event",
        "table4_outcome_pooled_matches_site_sum_crossing",
        "table4_outcome_pooled_matches_site_sum_rapid_expansion",
        "table4_outcome_pooled_matches_site_sum_rupture_symptomatic",
        "table4_outcome_pooled_matches_site_sum_elective_repair",
        "table4_silent_arm_split",
        "table4_audit_stratum_reaches_target",
        "table5_audit_margin_geometry_only_trajectory",
        "table5_audit_margin_constant_g_null",
    }
    assert failures == expected


def test_audit_margin_is_below_the_pre_specified_minimum() -> None:
    observed = strata.TABLE5_AUDIT["BM-STF"] - strata.TABLE5_AUDIT["geometry_only_trajectory"]
    assert abs(observed - 0.01) < 1.0e-12
    assert observed < criteria.MECHANISM_MARGIN


def test_event_count_product_matches_the_printed_count() -> None:
    product = strata.TABLE1_PANEL_B[3].reference * criteria.ReportedLimits().feasible_intervals
    assert abs(round(product) - 75) <= 1


def test_banding_helpers_are_consistent() -> None:
    from bmstf.policy.bands import band_from_diameter

    assert band_from_diameter(41.0) == "40-45"
    assert strata.TABLE3_PANEL_A[0.10]["crossing_feasible"] == strata.TABLE1_PANEL_B[1].reference


def test_annual_rate_array_matches_the_scalar_lookup() -> None:
    diameters = np.linspace(28.0, 72.0, 200)
    from bmstf.process.hazards import StratumRisk, annual_rate_array

    table = StratumRisk()
    vectorised = annual_rate_array(diameters, "female", table)
    scalar = np.asarray([table.annual_rate(float(value), "female") for value in diameters])
    assert np.max(np.abs(vectorised - scalar)) < 1.0e-15
