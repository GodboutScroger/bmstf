"""Transcription of the manuscript's reported design tables, with anchors.

Ref: Table 1 (panel A calibration and discrimination, panel B two-barrier policy),
Table 2 (ablations), Table 3 (safety-deferral frontier), Table 5 (primary
comparison and the surrogate parity gate). Every entry carries the table and the
anchor it was read from; the arithmetic checks in `margins` recompute the reported
contrasts from these numbers instead of trusting them.
"""

from __future__ import annotations

from dataclasses import dataclass

ROUNDING_TOLERANCE = 0.0101  # a delta of two two-decimal marginals carries up to 0.01 of rounding


@dataclass(frozen=True, slots=True)
class Contrast:
    label: str
    anchor: str
    reference: float
    comparator: float
    reported_delta: float
    reported_interval: tuple[float, float] | None = None

    def recomputed(self) -> float:
        return float(self.reference - self.comparator)

    def gap(self) -> float:
        return float(abs(self.recomputed() - self.reported_delta))

    def consistent(self, tolerance: float = ROUNDING_TOLERANCE) -> bool:
        return self.gap() <= float(tolerance)


TABLE1_PANEL_A: tuple[Contrast, ...] = (
    Contrast("coverage_diameter", "Table 1 panel A, row delta", 0.892, 0.629, 0.263),
    Contrast("coverage_burden", "Table 1 panel A, row delta", 0.901, 0.761, 0.141),
    Contrast("expected_calibration_error", "Table 1 panel A, row delta", 0.044, 0.101, -0.057),
    Contrast("auroc_crossing", "Table 1 panel A, row delta", 0.918, 0.903, 0.015),
)

TABLE1_PANEL_A_BANDS: tuple[Contrast, ...] = (
    Contrast(
        "expected_calibration_error", "Table 1 panel A, band extrapolation", 0.044, 0.317, -0.274
    ),
    Contrast("auroc_crossing", "Table 1 panel A, band extrapolation", 0.918, 0.685, 0.232),
)

TABLE1_PANEL_B: tuple[Contrast, ...] = (
    Contrast("escalate_fraction", "Table 1 panel B, row delta", 0.378, 0.284, 0.094),
    Contrast("crossing_feasible", "Table 1 panel B, row delta", 0.028, 0.091, -0.063),
    Contrast("crossing_pooled", "Table 1 panel B, row delta", 0.162, 0.189, -0.026),
    Contrast("event_feasible", "Table 1 panel B, row delta", 0.0098, 0.0106, -0.0008),
    Contrast("event_pooled", "Table 1 panel B, row delta", 0.0089, 0.0096, -0.0007),
    Contrast("mean_interval_pooled", "Table 1 panel B, row delta", 6.67, 7.24, -0.57),
)

TABLE1_PANEL_B_LITERAL: tuple[Contrast, ...] = (
    Contrast("escalate_fraction", "Table 1 panel B, literal bands", 0.378, 0.000, 0.378),
    Contrast("crossing_feasible", "Table 1 panel B, literal bands", 0.028, 0.300, -0.272),
    Contrast("crossing_pooled", "Table 1 panel B, literal bands", 0.162, 0.300, -0.138),
    Contrast("event_feasible", "Table 1 panel B, literal bands", 0.0098, 0.0118, -0.0020),
    Contrast("event_pooled", "Table 1 panel B, literal bands", 0.0089, 0.0118, -0.0029),
    Contrast("mean_interval_pooled", "Table 1 panel B, literal bands", 6.67, 7.86, -1.19),
)

TABLE1_PANEL_B_CALIBRATED: tuple[Contrast, ...] = (
    Contrast("escalate_fraction", "Table 1 panel B, calibrated bands", 0.378, 0.738, -0.359),
    Contrast("crossing_feasible", "Table 1 panel B, calibrated bands", 0.028, 0.018, 0.010),
    Contrast("crossing_pooled", "Table 1 panel B, calibrated bands", 0.162, 0.155, 0.007),
    Contrast("event_feasible", "Table 1 panel B, calibrated bands", 0.0098, 0.0060, 0.0039),
    Contrast("event_pooled", "Table 1 panel B, calibrated bands", 0.0089, 0.0069, 0.0020),
    Contrast("mean_interval_pooled", "Table 1 panel B, calibrated bands", 6.67, 5.63, 1.05),
)

FIVE_COMPONENT_DELTAS: tuple[tuple[str, float], ...] = (
    ("scalar_noise", 0.011),
    ("no_ultrasound", -0.001),
    ("single_regime_mean_drift", 0.022),
    ("single_barrier", -0.002),
    ("diameter_only", -0.006),
)
REPORTED_COMPONENT_SUM = 0.024
REPORTED_SNAPSHOT_BURDEN_MARGIN = 0.141

TABLE2_PANEL_A: dict[str, dict[str, float]] = {
    "BM-STF": {
        "cov_burden": 0.901,
        "cov_burden_hn": 0.912,
        "rmse_change": 0.129,
        "crps_diameter": 0.942,
        "auroc": 0.918,
    },
    "scalar_noise": {
        "cov_burden": 0.890,
        "cov_burden_hn": 0.868,
        "rmse_change": 0.138,
        "crps_diameter": 0.942,
        "auroc": 0.917,
    },
    "no_ultrasound": {
        "cov_burden": 0.903,
        "cov_burden_hn": 0.911,
        "rmse_change": 0.130,
        "crps_diameter": 0.950,
        "auroc": 0.917,
    },
    "single_regime_mean_drift": {
        "cov_burden": 0.880,
        "cov_burden_hn": 0.887,
        "rmse_change": 0.130,
        "crps_diameter": 0.894,
        "auroc": 0.912,
    },
    "single_regime_quiescent": {
        "cov_burden": 0.889,
        "cov_burden_hn": 0.909,
        "rmse_change": 0.132,
        "crps_diameter": 0.915,
        "auroc": 0.911,
    },
    "single_barrier": {
        "cov_burden": 0.903,
        "cov_burden_hn": 0.910,
        "rmse_change": 0.129,
        "crps_diameter": 0.943,
        "auroc": 0.918,
    },
    "diameter_only": {
        "cov_burden": 0.907,
        "cov_burden_hn": 0.905,
        "rmse_change": 0.179,
        "crps_diameter": 0.964,
        "auroc": 0.913,
    },
    "spread_half": {
        "cov_burden": 0.625,
        "cov_burden_hn": 0.611,
        "rmse_change": 0.129,
        "crps_diameter": 0.976,
        "auroc": 0.916,
    },
    "spread_double": {
        "cov_burden": 0.998,
        "cov_burden_hn": 1.000,
        "rmse_change": 0.129,
        "crps_diameter": 1.043,
        "auroc": 0.917,
    },
}

TABLE3_PANEL_A: dict[float, dict[str, float]] = {
    0.05: {
        "interval_pooled": 6.48,
        "interval_feasible": 9.19,
        "escalate": 0.439,
        "crossing_feasible": 0.015,
        "crossing_pooled": 0.157,
        "event_pooled": 0.0086,
    },
    0.10: {
        "interval_pooled": 6.67,
        "interval_feasible": 8.90,
        "escalate": 0.378,
        "crossing_feasible": 0.028,
        "crossing_pooled": 0.162,
        "event_pooled": 0.0089,
    },
    0.15: {
        "interval_pooled": 6.78,
        "interval_feasible": 8.72,
        "escalate": 0.340,
        "crossing_feasible": 0.039,
        "crossing_pooled": 0.166,
        "event_pooled": 0.0091,
    },
    0.20: {
        "interval_pooled": 6.86,
        "interval_feasible": 8.57,
        "escalate": 0.309,
        "crossing_feasible": 0.049,
        "crossing_pooled": 0.170,
        "event_pooled": 0.0091,
    },
}

TABLE5_AUDIT: dict[str, float] = {
    "BM-STF": 0.96,
    "geometry_only_trajectory": 0.95,
    "constant_g_null": 0.95,
    "permuted_g_null": 0.782,
}

TABLE5_POOLED: dict[str, float] = {
    "BM-STF": 0.816,
    "geometry_only_trajectory": 0.775,
    "diameter_rule_55_50": 0.712,
    "maximum_diameter": 0.719,
    "fusion_equal_budget": 0.812,
    "single_scan_hemodynamic_index": 0.764,
    "single_scan_ct_clinical_geometric": 0.788,
}

TABLE5_EXTERNAL: dict[str, float] = {
    "BM-STF": 0.846,
    "geometry_only_trajectory": 0.805,
    "fusion_equal_budget": 0.805,
    "single_scan_hemodynamic_index": 0.757,
}

TABLE5_SUBSTRATE_NMAE: dict[str, float] = {
    "physics_attention_substrate_cv": 0.809,
    "physics_attention_substrate_external": 0.839,
    "geometric_algebra_transformer_cv": 0.810,
    "geometric_algebra_transformer_external": 0.840,
    "anchored_branch_transformer_cv": 0.813,
    "anchored_branch_transformer_external": 0.843,
}

MISSING_PREDICTION_ANCHOR = "Methods, per-scan turnaround; Results, feasible intervals"


def comparator_count() -> int:
    """Table 1 contributes three comparators and Table 2 eight, as the caption states."""

    table1 = 3
    table2 = len(TABLE2_PANEL_A) - 1
    return int(table1 + table2)
