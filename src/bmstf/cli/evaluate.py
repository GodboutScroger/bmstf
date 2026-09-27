"""Render the study's run record as the manuscript's tables, in plain text.

Ref: Table 1 (panel A calibration and discrimination, panel B the two-barrier
policy), Table 2 (ablations), Table 3 (tolerance sweep), Table 4 (cohort
delineation) and the Tables 4-5 cohort card. Reports are written as .txt so the
release keeps a single markdown file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from bmstf.protocol import margins
from bmstf.protocol.cohort import (
    COMPOSITE_EVENT_24M,
    CROSSING_24M,
    ELECTIVE_REPAIR_24M,
    INTER_READER_MEDIAN_ABSOLUTE_MM,
    KAPPA_VALUES,
    RAPID_EXPANSION_24M,
    RUPTURE_SYMPTOMATIC_24M,
    TABLE4,
    outcome_sum,
    pooled_sum,
)
from bmstf.runtime.config import repository_root
from bmstf.runtime.store import read_json, write_text
from bmstf.study.contrasts import (
    REFERENCE_ARM,
    component_sum,
    contrast_family,
    snapshot_margin,
)

PANEL_A_METRICS: tuple[str, ...] = (
    "coverage_diameter",
    "coverage_burden",
    "expected_calibration_error",
    "auroc_crossing",
)
PANEL_B_METRICS: tuple[str, ...] = (
    "escalate_fraction",
    "crossing_feasible",
    "crossing_pooled",
    "event_feasible",
    "event_pooled",
    "mean_interval_pooled",
)
ABLATION_METRICS: tuple[str, ...] = (
    "coverage_burden",
    "coverage_burden_high_noise",
    "burden_change_rmse",
    "crps_diameter",
    "auroc_crossing",
)


def _table(rows: list[list[str]], width: int = 26) -> str:
    if not rows:
        return ""
    columns = max(len(row) for row in rows)
    padded = [row + [""] * (columns - len(row)) for row in rows]
    widths = [min(width, max(len(row[column]) for row in padded)) for column in range(columns)]
    lines = []
    for position, row in enumerate(padded):
        lines.append("  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)))
        if position == 0:
            lines.append("-" * (sum(widths) + 2 * (columns - 1)))
    return "\n".join(lines)


def render_panels(aggregate: dict[str, dict[str, float]], scale: dict[str, object]) -> str:
    lines = [f"Executed in-silico scale: {scale}", ""]
    lines.append("Panel A - calibration and discrimination")
    header = ["method", *PANEL_A_METRICS]
    rows = [header] + [
        [arm] + [f"{aggregate[arm].get(metric, float('nan')):.3f}" for metric in PANEL_A_METRICS]
        for arm in aggregate
    ]
    lines.append(_table(rows))
    lines.append("")
    lines.append("Panel B - two-barrier policy read-outs")
    header_b = ["method", *PANEL_B_METRICS]
    rows_b = [header_b] + [
        [arm] + [f"{aggregate[arm].get(metric, float('nan')):.4f}" for metric in PANEL_B_METRICS]
        for arm in aggregate
    ]
    lines.append(_table(rows_b))
    lines.append("")
    return "\n".join(lines)


def render_contrasts(payload: dict[str, object]) -> str:
    """Table 1 caption: seed-paired delta rows inside their Holm family.

    The caption fixes both the construction (BM-STF minus the comparator, paired by
    seed) and the multiplicity control (one Holm family per metric across the
    comparators), so the deltas printed here are not the difference of two means.
    """

    entries = cast(list[dict[str, Any]], payload.get("per_seed", []))
    blocks: list[dict[str, dict[str, float]]] = [dict(entry["metrics"]) for entry in entries]
    aggregate = cast(dict[str, dict[str, float]], payload.get("aggregate", {}))
    comparators = tuple(name for name in aggregate if name != REFERENCE_ARM)
    if not blocks or not comparators:
        return "the run record carries no per-seed blocks, so no seed-paired contrast exists\n"
    family = contrast_family(blocks, ABLATION_METRICS, comparators)
    rows = [["metric", "comparator", "delta", "95 % interval", "p", "p Holm"]]
    for metric in ABLATION_METRICS:
        for contrast in family.get(metric, []):
            rows.append(
                [
                    metric,
                    contrast.comparator,
                    f"{contrast.delta:+.4f}",
                    f"[{contrast.lower:+.4f}, {contrast.upper:+.4f}]",
                    f"{contrast.p_value:.4f}",
                    f"{contrast.adjusted_p:.4f}",
                ]
            )
    lines = [
        "Seed-paired contrasts, BM-STF minus the comparator, Holm-adjusted within each metric",
        "",
        _table(rows, width=30),
        "",
        f"five single-component burden-coverage losses sum to {component_sum(family):+.4f}",
        f"snapshot margin on burden coverage {snapshot_margin(family):+.4f}",
        "",
    ]
    return "\n".join(lines)


def render_ablations(aggregate: dict[str, dict[str, float]]) -> str:
    lines = ["Ablations - the executed arms against the full mechanism", ""]
    reference = "BM-STF"
    rows = [["variant"] + [f"delta {metric}" for metric in ABLATION_METRICS]]
    for arm in aggregate:
        if arm == reference or reference not in aggregate:
            continue
        rows.append(
            [
                arm,
                *[
                    f"{aggregate[reference].get(metric, float('nan')) - aggregate[arm].get(metric, float('nan')):+.4f}"
                    for metric in ABLATION_METRICS
                ],
            ]
        )
    lines.append(_table(rows))
    lines.append("")
    return "\n".join(lines)


def render_cohort_card() -> str:
    lines = ["Table 4 cohort delineation, recomputed from the printed counts", ""]
    rows = [["column", "screened", "excluded", "retained", "split total"]]
    for key, entry in TABLE4.items():
        rows.append(
            [
                key,
                str(entry.screened),
                str(entry.excluded_total),
                str(entry.retained),
                str(entry.split_total),
            ]
        )
    lines.append(_table(rows, width=28))
    lines.append("")
    lines.append("Pooled column against the site sum")
    rows_pooled = [["field", "site sum", "pooled column", "match"]]
    for field, pooled in (
        ("screened", TABLE4["Retrospective cohort"].screened),
        ("excluded by cause", TABLE4["Retrospective cohort"].excluded_total),
        (
            "records with at least two observations",
            TABLE4["Retrospective cohort"].two_or_more_observations,
        ),
        ("ultrasound observations", TABLE4["Retrospective cohort"].us_observations),
    ):
        key = {
            "screened": "screened",
            "excluded by cause": "excluded_prior_repair",
            "records with at least two observations": "two_or_more_observations",
            "ultrasound observations": "us_observations",
        }[field]
        site_sum = pooled_sum(key)
        rows_pooled.append(
            [field, str(site_sum), str(pooled), "yes" if site_sum == pooled else "no"]
        )
    lines.append(_table(rows_pooled, width=40))
    lines.append("")
    lines.append("Outcomes at 24 months, pooled against the site sum")
    rows_outcome = [["outcome", "site sum", "pooled column", "match"]]
    for label, values in (
        ("composite event", COMPOSITE_EVENT_24M),
        ("threshold crossing", CROSSING_24M),
        ("rapid expansion", RAPID_EXPANSION_24M),
        ("rupture or symptomatic", RUPTURE_SYMPTOMATIC_24M),
        ("elective repair", ELECTIVE_REPAIR_24M),
    ):
        total = outcome_sum(values)
        pooled = values["Retrospective cohort"]
        rows_outcome.append([label, str(total), str(pooled), "yes" if total == pooled else "no"])
    lines.append(_table(rows_outcome, width=26))
    lines.append("")
    lines.append("Reader agreement")
    rows_reader = [["column", "kappa", "median absolute difference (mm)"]]
    for key in TABLE4:
        rows_reader.append(
            [key, f"{KAPPA_VALUES[key]:.2f}", f"{INTER_READER_MEDIAN_ABSOLUTE_MM[key]:.1f}"]
        )
    lines.append(_table(rows_reader, width=34))
    lines.append("")
    return "\n".join(lines)


def render_manuscript_discrepancies() -> str:
    lines = ["Manuscript arithmetic recomputed from the printed tables", ""]
    rows = [["check", "observed", "expected", "op", "consistent"]]
    for check in margins.all_manuscript_checks():
        rows.append(
            [
                check.name,
                f"{check.observed:.6f}",
                f"{check.expected:.6f}",
                check.op,
                "yes" if check.passed() else "no",
            ]
        )
    lines.append(_table(rows, width=52))
    failures = [check for check in margins.all_manuscript_checks() if not check.passed()]
    lines.append("")
    lines.append(f"discrepancies inside the manuscript's own tables: {len(failures)}")
    for check in failures:
        lines.append(f"  {check.name} -> {check.anchor}")
    lines.append("")
    return "\n".join(lines)


def load_summary(path: Path) -> dict[str, object]:
    """Read the run record; a missing or malformed record is reported, not fatal."""

    try:
        payload = dict(read_json(path))
    except (OSError, ValueError) as error:
        return {"error": f"{type(error).__name__}: {path.name} is not a readable run record"}
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bmstf-evaluate",
        description="Render the run record and the manuscript-arithmetic audit as plain text.",
    )
    parser.add_argument("--summary", default="runs/main/summary.json")
    parser.add_argument("--output", default="runs/evaluation_report.txt")
    parser.add_argument("--root", default=None)
    arguments = parser.parse_args(argv)
    root = Path(arguments.root).resolve() if arguments.root else repository_root()
    summary_path = root / arguments.summary
    sections: list[str] = []
    payload = load_summary(summary_path)
    if payload.get("aggregate"):
        aggregate: dict[str, dict[str, float]] = dict(payload["aggregate"])  # type: ignore[call-overload]
        scale: dict[str, object] = dict(payload.get("scale", {}))  # type: ignore[call-overload]
        sections.append(render_panels(aggregate, scale))
        sections.append(render_contrasts(payload))
        sections.append(render_ablations(aggregate))
    else:
        sections.append(
            f"no run record at {arguments.summary}; run bmstf-run first. "
            "The cohort card and the manuscript audit below do not need a run.\n"
        )
    sections.append(render_cohort_card())
    sections.append(render_manuscript_discrepancies())
    write_text(root / arguments.output, "\n".join(sections))
    print(f"wrote {arguments.output} ({len(sections)} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
