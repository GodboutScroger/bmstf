"""Arithmetic checks recomputed from the manuscript's own reported tables.

Ref: Table 1 caption (the delta rows are seed-paired differences, BM-STF minus the
comparator; the eleven comparators of Tables 1 and 2), Table 2 (the five
single-component losses sum to 0.024 against a snapshot margin of 0.141), Table 3
(the tolerance sweep), Table 4 (the delineation card), Eq. (6) (the audit-stratum
margin of 0.03), Table 5 (the parity gate at an equivalence margin of 0.05 and a
secondary margin of 0.04).

These checks are about the manuscript, not about the release: a failure here is a
discrepancy inside the reported tables, and is reported separately from the code
verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bmstf.protocol import cohort, criteria, strata


@dataclass(frozen=True, slots=True)
class ManuscriptCheck:
    """One recomputation from the manuscript's own numbers.

    `op` is the relation the manuscript asserts: "eq" for a printed delta that must
    reproduce within rounding, "ge" or "le" for a printed bound.
    """

    name: str
    anchor: str
    observed: float
    expected: float
    tolerance: float
    detail: dict[str, Any]
    op: str = "eq"

    def passed(self) -> bool:
        observed = float(self.observed)
        expected = float(self.expected)
        if self.op == "ge":
            return observed >= expected - float(self.tolerance)
        if self.op == "le":
            return observed <= expected + float(self.tolerance)
        return abs(observed - expected) <= float(self.tolerance)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "anchor": self.anchor,
            "observed": float(self.observed),
            "expected": float(self.expected),
            "tolerance": float(self.tolerance),
            "consistent": bool(self.passed()),
            "op": self.op,
            "detail": self.detail,
        }


def _contrasts(
    contrasts: tuple[strata.Contrast, ...], group: str, tolerance: float = strata.ROUNDING_TOLERANCE
) -> list[ManuscriptCheck]:
    checks: list[ManuscriptCheck] = []
    for contrast in contrasts:
        checks.append(
            ManuscriptCheck(
                # The same metric label recurs across the panel groups, so the group is part
                # of the name; two checks may never share one.
                name=f"table1_delta_{group.lower()}_{contrast.label}",
                anchor=contrast.anchor,
                observed=contrast.recomputed(),
                expected=contrast.reported_delta,
                tolerance=tolerance,
                detail={
                    "table": group,
                    "reference": contrast.reference,
                    "comparator": contrast.comparator,
                    "reported_delta": contrast.reported_delta,
                    "recomputed_delta": contrast.recomputed(),
                    "gap": contrast.gap(),
                },
            )
        )
    return checks


def table1_checks() -> list[ManuscriptCheck]:
    return (
        _contrasts(strata.TABLE1_PANEL_A, "1A")
        + _contrasts(strata.TABLE1_PANEL_A_BANDS, "1A-bands")
        + _contrasts(strata.TABLE1_PANEL_B, "1B")
        + _contrasts(strata.TABLE1_PANEL_B_LITERAL, "1B-literal")
        + _contrasts(strata.TABLE1_PANEL_B_CALIBRATED, "1B-calibrated")
    )


def table1_comparator_count() -> ManuscriptCheck:
    observed = float(strata.comparator_count())
    return ManuscriptCheck(
        name="table1_comparator_count",
        anchor="Table 1 caption (the eleven comparators of Tables 1 and 2)",
        observed=observed,
        expected=float(criteria.COMPARATOR_COUNT),
        tolerance=0.0,
        detail={"from_tables": observed, "stated": float(criteria.COMPARATOR_COUNT)},
    )


def table1_event_count() -> ManuscriptCheck:
    observed = float(
        round(strata.TABLE1_PANEL_B[3].reference * criteria.ReportedLimits().feasible_intervals)
    )
    limits = criteria.ReportedLimits()
    return ManuscriptCheck(
        name="table1_feasible_event_count",
        anchor="Results, two-barrier policy (75 events among 7,617 feasible intervals)",
        observed=observed,
        expected=float(limits.events_feasible),
        tolerance=1.0,
        detail={
            "event_probability": strata.TABLE1_PANEL_B[3].reference,
            "feasible_intervals": limits.feasible_intervals,
            "product": observed,
        },
    )


def table2_component_sum() -> ManuscriptCheck:
    observed = float(sum(value for _, value in strata.FIVE_COMPONENT_DELTAS))
    return ManuscriptCheck(
        name="table2_five_component_sum",
        anchor="Table 2 text (the five single-component losses sum to 0.024)",
        observed=observed,
        expected=strata.REPORTED_COMPONENT_SUM,
        tolerance=1.0e-9,
        detail={"components": dict(strata.FIVE_COMPONENT_DELTAS)},
    )


def table2_snapshot_margin() -> ManuscriptCheck:
    margin = float(strata.TABLE1_PANEL_A[1].recomputed())
    return ManuscriptCheck(
        name="table2_snapshot_margin_recomputed",
        anchor="Table 1 panel A and the Table 2 text (snapshot margin 0.141 on burden coverage)",
        observed=margin,
        expected=strata.REPORTED_SNAPSHOT_BURDEN_MARGIN,
        tolerance=strata.ROUNDING_TOLERANCE,
        detail={
            "definition": "Table 1 panel A, BM-STF minus snapshot on burden coverage",
            "printed": strata.TABLE1_PANEL_A[1].reported_delta,
        },
    )


def table2_components_below_margin() -> ManuscriptCheck:
    return ManuscriptCheck(
        name="table2_components_below_snapshot_margin",
        anchor="Table 2 text (the five single-component losses do not sum to the snapshot margin)",
        observed=float(sum(value for _, value in strata.FIVE_COMPONENT_DELTAS)),
        expected=strata.REPORTED_SNAPSHOT_BURDEN_MARGIN,
        tolerance=0.0,
        op="le",
        detail={"snapshot_margin": strata.REPORTED_SNAPSHOT_BURDEN_MARGIN},
    )


def table2_checks() -> list[ManuscriptCheck]:
    checks: list[ManuscriptCheck] = [
        table2_component_sum(),
        table2_snapshot_margin(),
        table2_components_below_margin(),
    ]
    reference = strata.TABLE2_PANEL_A["BM-STF"]
    for name, row in strata.TABLE2_PANEL_A.items():
        if name == "BM-STF":
            continue
        for metric in ("cov_burden", "cov_burden_hn", "rmse_change", "crps_diameter", "auroc"):
            checks.append(
                ManuscriptCheck(
                    name=f"table2_delta_{name}_{metric}",
                    anchor=f"Table 2 panel A, row delta for {name}",
                    observed=float(reference[metric] - row[metric]),
                    expected=float(_reported_table2_delta(name, metric)),
                    tolerance=strata.ROUNDING_TOLERANCE,
                    detail={"comparator": name, "metric": metric},
                )
            )
    return checks


REPORTED_TABLE2_DELTAS: dict[tuple[str, str], float] = {
    ("scalar_noise", "cov_burden"): 0.011,
    ("scalar_noise", "cov_burden_hn"): 0.044,
    ("scalar_noise", "rmse_change"): -0.009,
    ("scalar_noise", "crps_diameter"): -0.001,
    ("scalar_noise", "auroc"): 0.001,
    ("no_ultrasound", "cov_burden"): -0.001,
    ("no_ultrasound", "cov_burden_hn"): 0.001,
    ("no_ultrasound", "rmse_change"): -0.001,
    ("no_ultrasound", "crps_diameter"): -0.008,
    ("no_ultrasound", "auroc"): 0.000,
    ("single_regime_mean_drift", "cov_burden"): 0.022,
    ("single_regime_mean_drift", "cov_burden_hn"): 0.025,
    ("single_regime_mean_drift", "rmse_change"): -0.001,
    ("single_regime_mean_drift", "crps_diameter"): 0.048,
    ("single_regime_mean_drift", "auroc"): 0.006,
    ("single_regime_quiescent", "cov_burden"): 0.012,
    ("single_regime_quiescent", "cov_burden_hn"): 0.003,
    ("single_regime_quiescent", "rmse_change"): -0.003,
    ("single_regime_quiescent", "crps_diameter"): 0.027,
    ("single_regime_quiescent", "auroc"): 0.007,
    ("single_barrier", "cov_burden"): -0.002,
    ("single_barrier", "cov_burden_hn"): 0.002,
    ("single_barrier", "rmse_change"): 0.000,
    ("single_barrier", "crps_diameter"): -0.001,
    ("single_barrier", "auroc"): 0.000,
    ("diameter_only", "cov_burden"): -0.006,
    ("diameter_only", "cov_burden_hn"): 0.007,
    ("diameter_only", "rmse_change"): -0.050,
    ("diameter_only", "crps_diameter"): -0.022,
    ("diameter_only", "auroc"): 0.004,
    ("spread_half", "cov_burden"): 0.276,
    ("spread_half", "cov_burden_hn"): 0.301,
    ("spread_half", "rmse_change"): 0.000,
    ("spread_half", "crps_diameter"): -0.034,
    ("spread_half", "auroc"): 0.002,
    ("spread_double", "cov_burden"): -0.097,
    ("spread_double", "cov_burden_hn"): -0.088,
    ("spread_double", "rmse_change"): 0.000,
    ("spread_double", "crps_diameter"): -0.101,
    ("spread_double", "auroc"): 0.000,
}


def _reported_table2_delta(name: str, metric: str) -> float:
    return float(REPORTED_TABLE2_DELTAS[(name, metric)])


def table3_checks() -> list[ManuscriptCheck]:
    checks: list[ManuscriptCheck] = []
    previous: float | None = None
    for tolerance in criteria.TOLERANCE_SWEEP:
        row = strata.TABLE3_PANEL_A[tolerance]
        if previous is not None:
            checks.append(
                ManuscriptCheck(
                    name=f"table3_crossing_increases_with_alpha1_{tolerance:g}",
                    anchor="Table 3 panel A (tolerance sweep, crossing probability rises with alpha_1)",
                    observed=float(row["crossing_feasible"] - previous),
                    expected=0.0,
                    tolerance=0.0,
                    op="ge",
                    detail={
                        "direction": "non-decreasing",
                        "from": previous,
                        "to": row["crossing_feasible"],
                    },
                )
            )
        previous = row["crossing_feasible"]
        checks.append(
            ManuscriptCheck(
                name=f"table3_interval_pooled_le_feasible_{tolerance:g}",
                anchor="Table 3 panel A (the pooled interval never exceeds the feasible-stratum interval)",
                observed=float(row["interval_pooled"]),
                expected=float(row["interval_feasible"]),
                tolerance=0.0,
                op="le",
                detail={"interval_feasible": row["interval_feasible"]},
            )
        )
    base = strata.TABLE3_PANEL_A[0.10]
    checks.append(
        ManuscriptCheck(
            name="table3_base_matches_table1",
            anchor="Table 3 panel A at alpha_1 = 0.10 against Table 1 panel B",
            observed=float(base["crossing_feasible"]),
            expected=float(strata.TABLE1_PANEL_B[1].reference),
            tolerance=1.0e-9,
            detail={
                "escalate": base["escalate"],
                "table1_escalate": strata.TABLE1_PANEL_B[0].reference,
            },
        )
    )
    return checks


def table4_checks() -> list[ManuscriptCheck]:
    checks: list[ManuscriptCheck] = []
    for field in cohort.INTEGER_FIELDS:
        pooled = cohort.pooled_sum(field)
        checks.append(
            ManuscriptCheck(
                name=f"table4_pooled_matches_site_sum_{field}",
                anchor="Table 4 (the retrospective column is the sum of Sites A, B and C)",
                observed=float(pooled),
                expected=float(getattr(cohort.TABLE4[cohort.RETROSPECTIVE], field)),
                tolerance=0.0,
                detail={"field": field, "pooled": pooled},
            )
        )
    for key in (*cohort.RETROSPECTIVE_KEYS, cohort.RETROSPECTIVE):
        entry = cohort.TABLE4[key]
        checks.append(
            ManuscriptCheck(
                name=f"table4_retained_split_{key}",
                anchor="Table 4 (single-observation stratum plus at-least-two-observation records equals retained)",
                observed=float(entry.split_total),
                expected=float(entry.retained),
                tolerance=0.0,
                detail={
                    "screened": entry.screened,
                    "excluded": entry.excluded_total,
                    "single": entry.single_observation,
                    "two_or_more": entry.two_or_more_observations,
                },
            )
        )
    for label, values in (
        ("composite_event", cohort.COMPOSITE_EVENT_24M),
        ("crossing", cohort.CROSSING_24M),
        ("rapid_expansion", cohort.RAPID_EXPANSION_24M),
        ("rupture_symptomatic", cohort.RUPTURE_SYMPTOMATIC_24M),
        ("elective_repair", cohort.ELECTIVE_REPAIR_24M),
    ):
        checks.append(
            ManuscriptCheck(
                name=f"table4_outcome_pooled_matches_site_sum_{label}",
                anchor="Table 4 (outcomes at 24 months, pooled against the site sum)",
                observed=float(cohort.outcome_sum(values)),
                expected=float(values[cohort.RETROSPECTIVE]),
                tolerance=0.0,
                detail={
                    "outcome": label,
                    "site_sum": cohort.outcome_sum(values),
                    "pooled": values[cohort.RETROSPECTIVE],
                },
            )
        )
    checks.append(
        ManuscriptCheck(
            name="table4_silent_arm_split",
            anchor="Table 4 (silent-mode arm: single-observation plus at-least-two-observation records equals retained)",
            observed=float(cohort.TABLE4[cohort.SILENT_ARM].split_total),
            expected=float(cohort.TABLE4[cohort.SILENT_ARM].retained),
            tolerance=0.0,
            detail={
                "screened": cohort.TABLE4[cohort.SILENT_ARM].screened,
                "excluded": cohort.TABLE4[cohort.SILENT_ARM].excluded_total,
            },
        )
    )
    checks.append(
        ManuscriptCheck(
            name="table4_audit_stratum_reaches_target",
            anchor="Table 4 (CFD-labelled audit stratum, target 600-700 pairs)",
            observed=float(cohort.TABLE4[cohort.RETROSPECTIVE].audit_stratum_pairs),
            expected=float(cohort.AUDIT_PAIR_TARGET[0]),
            tolerance=0.0,
            op="ge",
            detail={
                "target": list(cohort.AUDIT_PAIR_TARGET),
                "achieved": cohort.TABLE4[cohort.RETROSPECTIVE].audit_stratum_pairs,
            },
        )
    )
    checks.append(
        ManuscriptCheck(
            name="table4_composite_rate_consistency",
            anchor="Table 4 (the pooled 24.0 % composite-event rate uses the 4,268 replayed records)",
            observed=float(round(100.0 * cohort.composite_event_rate(cohort.RETROSPECTIVE), 1)),
            expected=24.0,
            tolerance=0.05,
            detail={
                "numerator": cohort.COMPOSITE_EVENT_24M[cohort.RETROSPECTIVE],
                "denominator": cohort.TABLE4[cohort.RETROSPECTIVE].two_or_more_observations,
            },
        )
    )
    return checks


def table5_checks() -> list[ManuscriptCheck]:
    checks: list[ManuscriptCheck] = []
    audit_bm = strata.TABLE5_AUDIT["BM-STF"]
    for comparator in ("geometry_only_trajectory", "constant_g_null"):
        checks.append(
            ManuscriptCheck(
                name=f"table5_audit_margin_{comparator}",
                anchor="Eq. (6) and the clinical-protocol criteria (audit-stratum margin of at least 0.03)",
                observed=float(audit_bm - strata.TABLE5_AUDIT[comparator]),
                expected=float(criteria.MECHANISM_MARGIN),
                tolerance=0.0,
                op="ge",
                detail={
                    "comparator": comparator,
                    "bm_stf": audit_bm,
                    "comparator_value": strata.TABLE5_AUDIT[comparator],
                },
            )
        )
    checks.append(
        ManuscriptCheck(
            name="table5_secondary_margin_single_scan_fusion",
            anchor="Clinical-protocol criteria (at least 0.04 over the single-scan fusion on the external region)",
            observed=float(
                strata.TABLE5_EXTERNAL["BM-STF"]
                - strata.TABLE5_EXTERNAL["single_scan_hemodynamic_index"]
            ),
            expected=float(criteria.SECONDARY_MARGIN),
            tolerance=0.0,
            op="ge",
            detail={
                "bm_stf": strata.TABLE5_EXTERNAL["BM-STF"],
                "comparator": strata.TABLE5_EXTERNAL["single_scan_hemodynamic_index"],
            },
        )
    )
    difference = strata.TABLE5_POOLED["BM-STF"] - strata.TABLE5_POOLED["fusion_equal_budget"]
    checks.append(
        ManuscriptCheck(
            name="table5_utility_parity_band",
            anchor="Clinical-protocol criteria (parity with the equal-budget fusion, descriptive +/-0.02 band)",
            observed=abs(difference),
            expected=float(criteria.DESCRIPTIVE_BAND),
            tolerance=0.0,
            op="le",
            detail={
                "point_difference": difference,
                "note": "the descriptive band is a reporting band, not a hard cutoff; the equivalence test is on the interval",
            },
        )
    )
    checks.append(
        ManuscriptCheck(
            name="table5_substrate_parity_within_tolerance",
            anchor="Table 5 panel B (substrate error within 10 % of the re-trained comparators)",
            observed=float(
                abs(
                    strata.TABLE5_SUBSTRATE_NMAE["physics_attention_substrate_cv"]
                    - strata.TABLE5_SUBSTRATE_NMAE["geometric_algebra_transformer_cv"]
                )
                / strata.TABLE5_SUBSTRATE_NMAE["geometric_algebra_transformer_cv"]
            ),
            expected=float(criteria.SUBSTRATE_ERROR_TOLERANCE),
            tolerance=1.0e-9,
            op="le",
            detail={"lower_is_better": True},
        )
    )
    return checks


def criteria_checks() -> list[ManuscriptCheck]:
    limits = criteria.ReportedLimits()
    return [
        ManuscriptCheck(
            name="reported_script_hashes_are_sha256",
            anchor="Code availability (the two reported SHA-256 digests)",
            observed=float(len(limits.mismatched_hashes())),
            expected=0.0,
            tolerance=0.0,
            detail={"mismatched": limits.mismatched_hashes()},
        ),
        ManuscriptCheck(
            name="event_barrier_sits_inside_the_gate",
            anchor="Eq. (1) and the protocol criteria (alpha_2 = 0.01 is the RESCAN-scale level)",
            observed=float(criteria.ALPHA2),
            expected=0.01,
            tolerance=0.0,
            detail={"alpha1": criteria.ALPHA1, "bound_level": criteria.BOUND_LEVEL},
        ),
    ]


def all_manuscript_checks() -> list[ManuscriptCheck]:
    return [
        *table1_checks(),
        table1_comparator_count(),
        table1_event_count(),
        *table2_checks(),
        *table3_checks(),
        *table4_checks(),
        *table5_checks(),
        *criteria_checks(),
    ]
