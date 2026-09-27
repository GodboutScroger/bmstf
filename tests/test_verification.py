"""The shipped artefacts: shape, provenance, hygiene and self-consistency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bmstf.harness import checks
from bmstf.harness.artefacts import (
    CLAIM_NAME,
    DATASET_NAME,
    MANIFEST_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    dataset_url_lines,
    verify_manifest,
    write_claim_to_code,
    write_dataset_urls,
    write_integrity_manifest,
    write_summary,
    write_verification_report,
)
from bmstf.harness.claims import CLAIMS, DEVIATIONS, code_symbols
from bmstf.runtime.config import repository_root
from bmstf.runtime.store import read_json, sha256_mapping
from bmstf.study.histories import StudySpec

ARTEFACTS = (CLAIM_NAME, REPORT_NAME, SUMMARY_NAME, MANIFEST_NAME, DATASET_NAME)
VALID_STATUSES = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}


@pytest.fixture(scope="module")
def root() -> Path:
    return repository_root()


@pytest.fixture(scope="module")
def shipped(root: Path) -> Path:
    absent = [name for name in ARTEFACTS if not (root / name).exists()]
    if absent:
        pytest.skip(
            "the shipped artefacts are written by the verification driver; missing "
            + ", ".join(absent)
        )
    return root


@pytest.fixture(scope="module")
def report(shipped: Path) -> dict:
    return dict(read_json(shipped / REPORT_NAME))


@pytest.fixture(scope="module")
def claim_map(shipped: Path) -> dict:
    return dict(read_json(shipped / CLAIM_NAME))


def test_every_artefact_ships(shipped: Path) -> None:
    for name in ARTEFACTS:
        assert (shipped / name).exists(), name


def test_report_carries_the_full_check_set(shipped: Path, report: dict) -> None:
    statuses = {result["status"] for result in report["checks"]}
    assert statuses.issubset(VALID_STATUSES)
    assert len(report["checks"]) >= 60
    names = [result["name"] for result in report["checks"]]
    assert len(names) == len(set(names))
    assert report["overall_status"] in {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}
    assert report["code_status"] in {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}
    assert report["status_counts"] == {
        status: sum(1 for result in report["checks"] if result["status"] == status)
        for status in ("PASS", "FAIL", "NOT_RUN", "BLOCKED")
    }


def test_code_status_excludes_the_manuscript_family(report: dict) -> None:
    code_checks = [result for result in report["checks"] if result["family"] != "manuscript"]
    assert all(result["status"] != "FAIL" for result in code_checks)
    assert report["code_status"] in {"VERIFIED", "PARTIALLY_VERIFIED"}
    assert report["manuscript_discrepancies"]


def test_outstanding_claims_are_recorded_with_reasons(report: dict) -> None:
    outstanding = [
        result for result in report["checks"] if result["status"] in ("NOT_RUN", "BLOCKED")
    ]
    assert outstanding
    for result in outstanding:
        assert result["evidence"]
        assert len(result["evidence"]) > 40
    kinds = {result["status"] for result in outstanding}
    assert kinds == {"NOT_RUN", "BLOCKED"}
    assert any(
        "cohort" in result["name"] or "cohort" in result["evidence"] for result in outstanding
    )


def test_report_has_no_host_paths_or_timings(report: dict) -> None:
    blob = json.dumps(report)
    assert "/Users/" not in blob
    assert "elapsed" not in blob
    assert "timestamp" not in blob
    assert "seconds" not in report
    assert "user" not in blob.lower()


def test_claim_map_maps_every_claim_to_existing_files(root: Path, claim_map: dict) -> None:
    assert claim_map["claim_count"] == len(CLAIMS)
    assert claim_map["code_symbol_count"] == len(code_symbols())
    assert set(claim_map["claim_checks"]) == {claim["id"] for claim in CLAIMS}
    assert all(status == "PASS" for status in claim_map["claim_checks"].values())
    for claim in claim_map["claims"]:
        assert claim["paper_location"]
        assert claim["statement"]
        for symbol in claim["symbols"]:
            assert (root / symbol.split("::", 1)[0]).exists()


def test_claim_map_records_every_deviation(claim_map: dict) -> None:
    assert len(claim_map["deviations"]) == len(DEVIATIONS)
    for entry in claim_map["deviations"]:
        assert (
            entry["topic"]
            and entry["paper_location"]
            and entry["departure"]
            and entry["justification"]
        )


def test_manifest_digest_recipe_reproduces(shipped: Path) -> None:
    manifest = dict(read_json(shipped / MANIFEST_NAME))
    assert manifest["manifest_digest"] == sha256_mapping(dict(manifest["files"]))
    assert manifest["file_count"] == len(manifest["files"])
    assert MANIFEST_NAME not in manifest["files"]
    assert "root" in manifest and "/" not in str(manifest["root"])


def test_manifest_excludes_only_itself(shipped: Path) -> None:
    report = verify_manifest(shipped)
    assert report["recorded_files"] > 0
    assert report["consistent"], report


def test_dataset_urls_holds_only_the_verified_links(shipped: Path) -> None:
    listed = dataset_url_lines(shipped / DATASET_NAME)
    assert listed == [entry["url"] for entry in checks.PUBLIC_DATASETS]
    assert all(line.startswith("https://") for line in listed)
    assert all("REPO_URL_PENDING" not in line for line in listed)
    body = (shipped / DATASET_NAME).read_text(encoding="utf-8")
    assert body.startswith("#")
    assert "not redistributable" in body


def test_manuscript_discrepancies_are_the_pinned_set(report: dict) -> None:
    names = {entry["name"] for entry in report["manuscript_discrepancies"]}
    assert len(names) == 9
    assert "table4_silent_arm_split" in names
    assert "table5_audit_margin_geometry_only_trajectory" in names


def test_summary_text_has_no_markdown_and_no_absolute_path(shipped: Path) -> None:
    text = (shipped / SUMMARY_NAME).read_text(encoding="utf-8")
    assert "/Users/" not in text
    assert "```" not in text
    assert not text.lstrip().startswith("#")
    assert "overall_status" in text and "code_status" in text


def test_artefact_writers_are_root_parameterised(tmp_path: Path, shipped: Path) -> None:
    before = (shipped / REPORT_NAME).read_bytes()
    write_claim_to_code(tmp_path, {"claim_count": 0})
    write_verification_report(tmp_path, {"overall_status": "UNVERIFIED"})
    write_summary(tmp_path, "partial runner")
    write_dataset_urls(tmp_path, [checks.PUBLIC_DATASETS[0]])
    manifest = write_integrity_manifest(tmp_path)
    assert manifest["file_count"] >= 4
    assert (tmp_path / MANIFEST_NAME).exists()
    assert dataset_url_lines(tmp_path / DATASET_NAME) == [checks.PUBLIC_DATASETS[0]["url"]]
    assert (shipped / REPORT_NAME).read_bytes() == before


def test_artefact_tests_never_overwrite_the_shipped_report(report: dict, shipped: Path) -> None:
    """The shipped report must be the complete one, not a partial runner's output."""

    assert len(report["checks"]) >= 60
    assert report["integrity_manifest"]
    assert (shipped / CLAIM_NAME).exists()


def test_status_helpers_agree_with_the_report(report: dict) -> None:
    results = [
        checks.CheckResult(
            name=str(entry["name"]),
            family=str(entry["family"]),
            status=str(entry["status"]),
            evidence=str(entry["evidence"]),
            detail=dict(entry["detail"]),
        )
        for entry in report["checks"]
    ]
    from bmstf.harness.driver import code_status, overall_status, status_counts

    assert status_counts(results) == report["status_counts"]
    assert overall_status(status_counts(results)) == report["overall_status"]
    assert code_status(results) == report["code_status"]


def test_reported_scale_is_recorded_beside_the_executed_one(report: dict) -> None:
    assert report["reported_scale"]["seeds"] == 20
    assert report["reported_scale"]["patients"] == 300
    assert report["executed_scale"]["executed_scale"]["seeds"] < 20
    assert report["executed_scale"]["executed_scale"]["patients"] < 300


def test_verification_scale_is_smaller_than_the_reported_run() -> None:
    spec = StudySpec()
    assert spec.seeds == 20
    assert spec.patients == 300
    assert spec.calibration_patients == 6000
    assert spec.forecast_paths == 256


def test_report_families_cover_every_check(report: dict) -> None:
    covered = sum(sum(counts.values()) for counts in report["families"].values())
    assert covered == len(report["checks"])
    assert {"closed_form", "execution", "manuscript", "integrity", "tool", "data"} <= set(
        report["families"]
    )
