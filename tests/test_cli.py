"""Command-line entry points write their run records and stay off the release tree."""

from __future__ import annotations

import json
from pathlib import Path

from bmstf.cli.evaluate import main as evaluate_main
from bmstf.cli.run import (
    DECISION_COLUMNS,
    main as run_main,
    run_experiment,
    spec_from_config,
)
from bmstf.harness.artefacts import verify_manifest
from bmstf.runtime.config import repository_root, resolve_experiment
from bmstf.runtime.store import read_json


def test_smoke_experiment_builds_a_small_spec() -> None:
    configuration = resolve_experiment("_smoke")
    spec = spec_from_config(configuration)
    assert spec.seeds == 2 and spec.patients == 8
    assert spec.arms == ("BM-STF",)


def test_run_experiment_writes_its_record(tmp_path) -> None:
    overrides = (
        f"outputs.directory={tmp_path}",
        "study.seeds=1",
        "study.patients=4",
        "study.calibration_patients=4",
        "study.horizon_months=24",
        "study.forecast_paths=8",
    )
    result = run_experiment("_smoke", repository_root(), overrides, write_decisions=True)
    assert result.decision_count > 0
    for name in (
        "results.csv",
        "summary.json",
        "MANIFEST.json",
        "results_decisions.csv",
        "integrity_manifest.json",
    ):
        assert (tmp_path / name).exists(), name
    summary = read_json(tmp_path / "summary.json")
    assert summary["experiment"] == "_smoke"
    assert "aggregate" in summary
    manifest = read_json(tmp_path / "MANIFEST.json")
    assert manifest["scale"]["patients"] == 4
    decisions = (tmp_path / "results_decisions.csv").read_text(encoding="utf-8").splitlines()
    assert decisions[0].split(",") == list(DECISION_COLUMNS)
    assert len(decisions) > 1
    report = verify_manifest(tmp_path)
    assert report["consistent"], report


def test_run_cli_runs_end_to_end(tmp_path) -> None:
    status = run_main(
        [
            "--experiment",
            "_smoke",
            "--override",
            f"outputs.directory={tmp_path}",
            "--override",
            "study.seeds=1",
            "--override",
            "study.patients=3",
            "--override",
            "study.calibration_patients=3",
            "--override",
            "study.horizon_months=18",
            "--override",
            "study.forecast_paths=4",
            "--log-level",
            "WARNING",
        ]
    )
    assert status == 0
    assert (tmp_path / "summary.json").exists()


def test_evaluate_cli_writes_plain_text(tmp_path) -> None:
    record = tmp_path / "summary.json"
    record.write_text(
        json.dumps(
            {
                "scale": {"seeds": 1, "patients": 4},
                "aggregate": {
                    "BM-STF": {"coverage_diameter": 0.88, "coverage_burden": 0.9},
                    "snapshot": {"coverage_diameter": 0.63, "coverage_burden": 0.76},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation_report.txt"
    status = evaluate_main(["--summary", str(record), "--output", str(output)])
    assert status == 0
    text = output.read_text(encoding="utf-8")
    assert "Panel A" in text
    assert "Table 4 cohort delineation" in text
    assert "Manuscript arithmetic" in text
    assert "discrepancies inside the manuscript's own tables" in text


def test_evaluate_cli_reports_a_missing_run(tmp_path) -> None:
    output = tmp_path / "empty.txt"
    status = evaluate_main(["--summary", "runs/absent/summary.json", "--output", str(output)])
    assert status == 0
    assert "no run record" in output.read_text(encoding="utf-8")


def test_run_record_is_reproducible(tmp_path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    overrides_first = (
        f"outputs.directory={first}",
        "study.seeds=1",
        "study.patients=3",
        "study.calibration_patients=3",
        "study.horizon_months=18",
        "study.forecast_paths=4",
    )
    overrides_second = (
        f"outputs.directory={second}",
        "study.seeds=1",
        "study.patients=3",
        "study.calibration_patients=3",
        "study.horizon_months=18",
        "study.forecast_paths=4",
    )
    run_experiment("_smoke", repository_root(), overrides_first)
    run_experiment("_smoke", repository_root(), overrides_second)
    first_summary = (first / "summary.json").read_text(encoding="utf-8")
    second_summary = (second / "summary.json").read_text(encoding="utf-8")
    assert first_summary == second_summary


def test_cli_module_paths_have_no_absolute_references() -> None:
    source = Path(repository_root(), "src", "bmstf", "cli", "run.py").read_text(encoding="utf-8")
    assert "/Users/" not in source
