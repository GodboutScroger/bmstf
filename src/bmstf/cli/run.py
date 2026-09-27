"""Run the in-silico study and write its run record.

Ref: Code availability (results.csv, results_decisions.csv, summary.json and
MANIFEST.json are released beside the run) and Table 1 (the study scale).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bmstf.harness.artefacts import write_integrity_manifest
from bmstf.process.cohort import SensorContext, patient_batch
from bmstf.process.hazards import StratumRisk
from bmstf.process.regimes import RegimeChain, build_system
from bmstf.protocol.criteria import ALPHA1, ALPHA2
from bmstf.runtime.config import Configuration, repository_root, resolve_experiment
from bmstf.runtime.journal import configure, get
from bmstf.runtime.seeds import set_seed
from bmstf.runtime.store import ensure_dir, write_json
from bmstf.sensor.boundary import BoundaryPrior
from bmstf.sensor.covariance import ResidualLookup
from bmstf.sensor.marginalise import MarginalisationSpec
from bmstf.sensor.surrogate import PhysicsAttentionSurrogate
from bmstf.study.engines import build_arm, run_arm
from bmstf.study.frontier import fixed_shift_rows, monotone_crossing, sweep
from bmstf.study.histories import StudyResult, StudySpec, calibrate_bands, run_study
from bmstf.study.selection import select_dimension

SUMMARY_NAME = "summary.json"
MANIFEST_NAME = "MANIFEST.json"
RESULTS_NAME = "results.csv"
DECISIONS_NAME = "results_decisions.csv"


def spec_from_config(configuration: Configuration) -> StudySpec:
    section = configuration.section("study")
    cohort = configuration.section("cohort")
    policy = configuration.section("policy")
    arms = configuration.get("experiment.arms")
    selected = tuple(str(name) for name in arms) if isinstance(arms, list) else None
    return StudySpec(
        seeds=section.as_int("seeds", 20),
        patients=section.as_int("patients", 300),
        calibration_patients=section.as_int("calibration_patients", 6000),
        horizon_months=section.as_int("horizon_months", 36),
        forecast_paths=section.as_int("forecast_paths", 256),
        women_share=section.as_float("women_share", 0.19),
        base_seed=section.as_int("base_seed", 20260925),
        arms=selected if selected is not None else StudySpec().arms,
        interval_median_months=cohort.as_float("interval_median_months", 8.6),
        interval_log_sigma=cohort.as_float("interval_log_sigma", 0.42),
        interval_min_months=cohort.as_int("interval_min_months", 3),
        interval_max_months=cohort.as_int("interval_max_months", 24),
        us_probability=cohort.as_float("us_probability", 0.35),
        us_standard_deviation_mm=cohort.as_float("us_standard_deviation_mm", 1.10),
        initial_diameter_band_mm=(
            cohort.as_float("initial_diameter_low_mm", 30.0),
            cohort.as_float("initial_diameter_high_mm", 54.0),
        ),
        initial_burden_band=(
            cohort.as_float("initial_burden_low", 0.18),
            cohort.as_float("initial_burden_high", 0.55),
        ),
        alpha1=policy.as_float("alpha1", ALPHA1),
        alpha2=policy.as_float("alpha2", ALPHA2),
    )


def system_from_config(configuration: Configuration) -> tuple[Any, SensorContext]:
    section = configuration.section("system")
    marginalisation = configuration.section("marginalisation")
    chain = RegimeChain(
        rate_quiescent_to_active=section.as_float("rate_quiescent_to_active", 0.03),
        rate_active_to_quiescent=section.as_float("rate_active_to_quiescent", 0.09),
    )
    system = build_system(dimension=section.as_int("dimension", 10), chain=chain)
    sensor = SensorContext(
        system,
        ResidualLookup(),
        MarginalisationSpec(
            boundary_draws=marginalisation.as_int("boundary_draws", 64),
            reconstruction_draws=marginalisation.as_int("reconstruction_draws", 8),
            topology_inflation=marginalisation.as_float("topology_inflation", 1.6),
            jitter_mm=marginalisation.as_float("jitter_mm", 0.12),
            correlation_mm=marginalisation.as_float("correlation_mm", 9.0),
        ),
        PhysicsAttentionSurrogate(),
        BoundaryPrior(),
    )
    return system, sensor


def output_directory(configuration: Configuration, root: Path) -> Path:
    configured = configuration.get("outputs.directory")
    directory = str(configured) if isinstance(configured, str) else "runs/experiment"
    return root / directory


def write_results_table(result: StudyResult, directory: Path) -> Path:
    path = directory / RESULTS_NAME
    metrics = sorted(
        {
            key
            for arm in result.aggregate.values()
            for key in arm
            if not key.endswith(("lower", "upper"))
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "metric", "estimate", "lower", "upper"])
        for arm in sorted(result.aggregate):
            for metric in metrics:
                writer.writerow(
                    [
                        arm,
                        metric,
                        f"{result.aggregate[arm].get(metric, float('nan')):.6f}",
                        f"{result.aggregate[arm].get(f'{metric}_lower', float('nan')):.6f}",
                        f"{result.aggregate[arm].get(f'{metric}_upper', float('nan')):.6f}",
                    ]
                )
    return path


DECISION_COLUMNS: tuple[str, ...] = (
    "arm",
    "patient",
    "site",
    "sex",
    "decision_month",
    "diameter_mm",
    "band",
    "gate_month",
    "interval_months",
    "escalate",
    "comparator_months",
    "crossing_probability",
    "event_probability",
    "crossing_12",
    "crossing_hit",
    "event_hit",
    "deferred",
    "under_observation",
    "crps_diameter_12",
    "burden_error",
    "active_probability",
)


def write_decisions_table(result: StudyResult, directory: Path) -> tuple[Path, int]:
    """Write the decision-level table named by the Data availability statement."""

    path = directory / DECISIONS_NAME
    rows = result.decision_rows
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(DECISION_COLUMNS)
        for row in rows:
            writer.writerow([_render(row.get(column, "")) for column in DECISION_COLUMNS])
    return path, len(rows)


def _render(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary(
    configuration: Configuration,
    result: StudyResult,
    directory: Path,
    frontier: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "experiment": configuration.name,
        "reported_result": configuration.get("experiment.reported_result", "not stated"),
        "scale": result.spec.as_mapping(),
        "environment": {
            "reported_python": configuration.get("environment.reported_python", "3.9.6"),
            "reported_numpy": configuration.get("environment.reported_numpy", "2.0.2"),
            "reported_run": configuration.get("environment.reported_run", ""),
        },
        "scalar_noise": [float(value) for value in result.scalar_noise.reshape(-1)],
        "aggregate": result.aggregate,
        "per_seed": [
            {"seed": block.seed, "records": block.records, "metrics": block.metrics}
            for block in result.seeds
        ],
    }
    if frontier:
        payload["frontier"] = frontier
    write_json(directory / SUMMARY_NAME, payload)
    return directory / SUMMARY_NAME


def write_manifest(configuration: Configuration, result: StudyResult, directory: Path) -> Path:
    keys: dict[str, str] = {}
    for arm in sorted(result.aggregate):
        for metric in sorted(result.aggregate[arm]):
            keys[f"{arm}.{metric}"] = f"aggregate.{arm}.{metric}"
    payload = {
        "summary_key": keys,
        "scale": result.spec.as_mapping(),
        "reported_result": configuration.get("experiment.reported_result", "not stated"),
        "note": (
            "every reported number of this run resolves through summary.json at the listed key"
        ),
    }
    write_json(directory / MANIFEST_NAME, payload)
    return directory / MANIFEST_NAME


def frontier_section(
    configuration: Configuration,
    sensor: SensorContext,
    system: Any,
    spec: StudySpec,
    strata: StratumRisk,
    root: Path,
) -> dict[str, Any]:
    """Table 3, when the experiment configures it.

    Panel A is the alpha_1 sweeps with the bands re-calibrated at each tolerance on the
    calibration cohort; panel B is the fixed-shift diagnostic over the spread variants.
    Neither runs unless the config names its tolerances or its shifts, so the primary
    experiments are unaffected.
    """

    tolerances = configuration.section("policy").raw("tolerances")
    shifts = configuration.section("diagnostic").raw("shifts_months")
    payload: dict[str, Any] = {}
    if not (isinstance(tolerances, (list, tuple)) and tolerances) and not (
        isinstance(shifts, (list, tuple)) and shifts
    ):
        return payload
    streams = set_seed(spec.base_seed + 4242)
    histories = patient_batch(
        spec.patients,
        sensor,
        spec.history_spec(),
        streams.generator,
        prefix="FRT",
        women_share=spec.women_share,
    )
    if isinstance(tolerances, (list, tuple)) and tolerances:
        rows = sweep(
            histories,
            system,
            spec.forecast_spec(),
            strata,
            spec.base_seed + 4243,
            lambda alpha1: calibrate_bands(sensor, system, spec, alpha1),
            tolerances=tuple(float(value) for value in tolerances),
            arms=spec.arms,
        )
        payload["tolerance_sweep"] = [row.as_mapping() for row in rows]
        payload["tolerance_sweep_monotone"] = bool(monotone_crossing(rows))
    if isinstance(shifts, (list, tuple)) and shifts:
        results: dict[str, Any] = {}
        for name in spec.arms:
            results[name] = run_arm(
                histories,
                system,
                build_arm(name, system),
                spec.forecast_spec(),
                strata,
                streams.forecast,
                alpha1=spec.alpha1,
                alpha2=spec.alpha2,
            )
        payload["fixed_shift"] = [row.as_mapping() for row in fixed_shift_rows(results)]
    return payload


def run_experiment(
    experiment: str,
    root: Path,
    overrides: Sequence[str] = (),
    write_decisions: bool = False,
    select_dimension_first: bool = False,
) -> StudyResult:
    configuration = resolve_experiment(experiment, root, overrides)
    spec = spec_from_config(configuration)
    system, sensor = system_from_config(configuration)
    strata = StratumRisk()
    if select_dimension_first:
        section = configuration.section("selection")
        streams = set_seed(spec.base_seed + 7)
        probe = patient_batch(
            section.as_int("validation_records", 48),
            sensor,
            spec.history_spec(),
            streams.generator,
            prefix="SEL",
            women_share=spec.women_share,
        )
        report = select_dimension(
            probe,
            sensor.spec,
            seed=spec.base_seed + 11,
            candidates=tuple(
                range(
                    section.as_int("latent_dimension_low", 8),
                    section.as_int("latent_dimension_high", 16) + 1,
                )
            ),
            paths=section.as_int("validation_paths", 128),
        )
        get("run").info("selected latent dimension %d", report.selection.dimension)
    result = run_study(sensor, system, spec, strata, alpha1=spec.alpha1, alpha2=spec.alpha2)
    frontier = frontier_section(configuration, sensor, system, spec, strata, root)
    directory = ensure_dir(output_directory(configuration, root))
    write_results_table(result, directory)
    write_summary(configuration, result, directory, frontier)
    if write_decisions:
        write_decisions_table(result, directory)
    write_manifest(configuration, result, directory)
    write_integrity_manifest(directory)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bmstf-run",
        description="Run the BM-STF in-silico study and write its run record.",
    )
    parser.add_argument("--experiment", default="main", help="configs/experiment/<name>.yaml")
    parser.add_argument("--config", default=None, help="path to the experiment file")
    parser.add_argument("--root", default=None, help="repository root")
    parser.add_argument("--override", action="append", default=[], help="dotted key=value")
    parser.add_argument("--decisions", action="store_true", help="also write the decision table")
    parser.add_argument(
        "--select-dimension",
        action="store_true",
        help="run the validation-fold energy-score selection before the study",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configure(arguments.log_level)
    root = Path(arguments.root).resolve() if arguments.root else repository_root()
    experiment = arguments.experiment
    if arguments.config:
        candidate = Path(arguments.config)
        experiment = candidate.stem
        if candidate.parent.name == "experiment" and candidate.parent.parent.name == "configs":
            root = candidate.resolve().parents[2]
    result = run_experiment(
        experiment,
        root,
        tuple(arguments.override),
        write_decisions=bool(arguments.decisions),
        select_dimension_first=bool(arguments.select_dimension),
    )
    get("run").info(
        "wrote %d arms over %d seeds of %d histories",
        len(result.aggregate),
        len(result.seeds),
        result.spec.patients,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
