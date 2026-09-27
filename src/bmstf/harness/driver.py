"""Verification driver: runs every check and writes the root artefacts.

Ref: Data availability statement (the manifest relates every reported number to
its summary key; the run record carries the environment, the seeds and the checksums).

Order matters: the claim map and the report are written first, then the plain-text
summary, then the integrity manifest, so the manifest digests the final tree.
Nothing time-based is recorded, so two runs of the same tree produce
byte-identical artefacts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bmstf.harness import checks as registry
from bmstf.harness.artefacts import (
    write_claim_to_code,
    write_dataset_urls,
    write_integrity_manifest,
    write_summary,
    write_verification_report,
)
from bmstf.harness.checks import CheckResult
from bmstf.harness.claims import CLAIMS, DEVIATIONS, code_symbols
from bmstf.runtime.config import RELEASE_SLUG, repository_root, resolve_experiment
from bmstf.runtime.journal import get
from bmstf.runtime.store import sha256_mapping
from bmstf.study.histories import StudySpec

VERIFICATION_EXPERIMENT = "supplementary_verification"
DEFAULT_ARMS: tuple[str, ...] = (
    "BM-STF",
    "snapshot",
    "scalar_noise",
    "no_ultrasound",
    "single_regime_mean_drift",
    "single_regime_quiescent",
    "single_barrier",
    "diameter_only",
    "spread_half",
    "spread_double",
)
MANUSCRIPT_FAMILY = registry.MANUSCRIPT


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    overall_status: str
    code_status: str
    status_counts: dict[str, int]
    manuscript_failures: tuple[str, ...]
    artefacts: dict[str, str]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "code_status": self.code_status,
            "status_counts": dict(self.status_counts),
            "manuscript_failures": list(self.manuscript_failures),
            "artefacts": dict(self.artefacts),
        }


def scale_from_config(experiment: str = VERIFICATION_EXPERIMENT) -> StudySpec:
    configuration = resolve_experiment(experiment)
    section = configuration.section("verification")
    arms = section.raw("arms")
    selected = (
        tuple(str(name) for name in arms) if isinstance(arms, (list, tuple)) else DEFAULT_ARMS
    )
    return StudySpec(
        seeds=section.as_int("seeds", 3),
        patients=section.as_int("patients", 60),
        calibration_patients=section.as_int("calibration_patients", 150),
        horizon_months=section.as_int("horizon_months", 36),
        forecast_paths=section.as_int("forecast_paths", 64),
        arms=selected,
        base_seed=section.as_int("base_seed", 20260925),
    )


def status_counts(results: list[CheckResult]) -> dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "BLOCKED": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def overall_status(counts: dict[str, int]) -> str:
    if counts.get("FAIL", 0) > 0:
        return "UNVERIFIED"
    if counts.get("NOT_RUN", 0) + counts.get("BLOCKED", 0) > 0:
        return "PARTIALLY_VERIFIED"
    return "VERIFIED"


def code_status(results: list[CheckResult]) -> str:
    filtered = [result for result in results if result.family != MANUSCRIPT_FAMILY]
    return overall_status(status_counts(filtered))


def _family_breakdown(results: list[CheckResult]) -> dict[str, dict[str, int]]:
    breakdown: dict[str, dict[str, int]] = {}
    for result in results:
        entry = breakdown.setdefault(
            result.family, {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "BLOCKED": 0}
        )
        entry[result.status] = entry.get(result.status, 0) + 1
    return breakdown


def _dependency_versions() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "scikit-learn": str(sklearn.__version__),
    }


def report_dependency_versions() -> None:
    """Log the host's library versions instead of recording them in a digested file.

    The interpreter and wheel versions are inputs from the host rather than from the tree, so
    writing them into verification_report.json would make the published bytes depend on the
    machine that ran the pass. requirements.txt, environment.yml and the Dockerfile are the
    repository-side record of the same thing.
    """

    versions = ", ".join(f"{name} {value}" for name, value in _dependency_versions().items())
    get("verify").info("host interpreter and wheels: %s", versions)


def claim_payload(root: Path, executed: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, str] = {}
    for claim in CLAIMS:
        missing = [
            symbol
            for symbol in claim["symbols"]
            if not (root / str(symbol).split("::", 1)[0]).exists()
        ]
        checks[str(claim["id"])] = "PASS" if not missing else "FAIL"
    return {
        "paper_title": (
            "Deep learning forecasts hemodynamic trajectories from computational fluid dynamics "
            "for abdominal aortic aneurysm risk stratification"
        ),
        "methods_name": "Boundary-marginalised shear-trajectory forecasting (BM-STF)",
        "claim_count": len(CLAIMS),
        "code_symbol_count": len(code_symbols()),
        "claims": list(CLAIMS),
        "claim_checks": checks,
        "deviations": list(DEVIATIONS),
        "executed_scale": executed,
        "verdicts": ["PASS", "FAIL", "NOT_RUN", "BLOCKED"],
    }


def render_summary(
    outcome: VerificationOutcome,
    results: list[CheckResult],
    executed: dict[str, Any],
) -> str:
    counts = outcome.status_counts
    lines = [
        "BM-STF verification summary",
        "=" * 60,
        f"root                     : {RELEASE_SLUG}",
        f"overall_status           : {outcome.overall_status}",
        f"code_status              : {outcome.code_status} (every family except manuscript)",
        f"checks                   : {sum(counts.values())}",
        "  "
        + ", ".join(f"{name}={counts[name]}" for name in ("PASS", "FAIL", "NOT_RUN", "BLOCKED")),
        f"manuscript_discrepancies : {len(outcome.manuscript_failures)}",
        f"executed in-silico scale: {executed}",
        "",
        "By family",
        "-" * 60,
    ]
    for family, entry in sorted(_family_breakdown(results).items()):
        lines.append(
            f"  {family:<12} "
            + ", ".join(f"{name}={entry[name]}" for name in ("PASS", "FAIL", "NOT_RUN", "BLOCKED"))
        )
    failing = [result for result in results if result.status == "FAIL"]
    if failing:
        lines += ["", "Failures", "-" * 60]
        lines += [f"  {result.name} [{result.family}]: {result.evidence}" for result in failing]
    outstanding = [result for result in results if result.status in ("NOT_RUN", "BLOCKED")]
    if outstanding:
        lines += ["", "Outstanding", "-" * 60]
        lines += [f"  {result.name} [{result.status}]: {result.evidence}" for result in outstanding]
    lines += [
        "",
        "A discrepancy inside the manuscript's own tables is not a defect in this release;",
        "those checks are reported under the manuscript family and excluded from code_status.",
        "",
    ]
    return "\n".join(lines)


def run(
    root: Path | None = None,
    experiment: str = VERIFICATION_EXPERIMENT,
    skip_tools: bool = False,
    skip_study: bool = False,
    live: bool = False,
) -> VerificationOutcome:
    logger = get("verify")
    report_dependency_versions()
    base = Path(root).resolve() if root is not None else repository_root()
    spec = scale_from_config(experiment)
    logger.info("verification root %s", RELEASE_SLUG)

    results: list[CheckResult] = []
    results.extend(registry.closed_form_checks())
    results.extend(registry.sensor_checks())
    results.extend(registry.smoother_checks())
    executed: dict[str, Any] = {}
    if skip_study:
        executed = {"executed_scale": spec.as_mapping(), "skipped": True}
    else:
        study_checks, context = registry.execution_checks(base, spec, spec.arms)
        results.extend(study_checks)
        executed = registry.configuration_summary(spec)
        executed["scalar_noise"] = context["scalar_noise"]
        executed["aggregate"] = context["aggregate"]
    results.extend(registry.agreement_checks())
    results.extend(registry.manuscript_check_results())
    results.append(registry.reported_scale_check())
    results.extend(registry.dataset_checks(live=live))
    if not skip_tools:
        results.append(registry.tool_check("ruff_check", ["ruff", "check", "."], base))
        results.append(registry.tool_check("black_check", ["black", "--check", "."], base))
        results.append(registry.tool_check("isort_check", ["isort", "--check-only", "."], base))
        results.append(registry.tool_check("mypy_strict", ["mypy", "--strict", "src/bmstf"], base))
        results.append(
            registry.tool_check(
                "gate_pytest",
                [sys.executable, "-m", "pytest", "-q"],
                base,
            )
        )
    results.append(registry.docker_check())
    results.extend(registry.integrity_checks(base))

    counts = status_counts(results)
    manuscript_failures = tuple(
        result.name
        for result in results
        if result.family == MANUSCRIPT_FAMILY and result.status == "FAIL"
    )
    outcome = VerificationOutcome(
        overall_status=overall_status(counts),
        code_status=code_status(results),
        status_counts=counts,
        manuscript_failures=manuscript_failures,
        artefacts={},
    )

    payload = claim_payload(base, executed)
    write_claim_to_code(base, payload)
    report: dict[str, Any] = {
        "overall_status": outcome.overall_status,
        "code_status": outcome.code_status,
        "status_counts": counts,
        "checks": [result.as_mapping() for result in results],
        "failed_checks": [result.name for result in results if result.status == "FAIL"],
        "manuscript_discrepancies": [
            result.as_mapping()
            for result in results
            if result.family == MANUSCRIPT_FAMILY and result.status == "FAIL"
        ],
        "families": _family_breakdown(results),
        "unverified_reasons": [
            result.evidence for result in results if result.status in ("NOT_RUN", "BLOCKED")
        ],
        "executed_scale": executed,
        "reported_scale": {
            "seeds": 20,
            "patients": 300,
            "calibration_patients": 6000,
            "forecast_paths": 256,
            "run_command": "python3 bmstf_sim.py --seeds 20 --patients 300 --jobs 5",
        },
        "interpretation": (
            "A failure inside the manuscript's own arithmetic is not a defect in this release; "
            "code_status excludes the manuscript family. NOT_RUN marks claims that this host "
            "cannot settle, and BLOCKED marks artefacts that are not redistributable."
        ),
        "integrity_manifest": (
            "integrity_manifest.json is written after this report, after the plain-text summary "
            "and after dataset_urls.txt, covers every tracked file including this report, and "
            "excludes only itself"
        ),
    }
    write_verification_report(base, report)
    outcome = VerificationOutcome(
        overall_status=outcome.overall_status,
        code_status=outcome.code_status,
        status_counts=counts,
        manuscript_failures=manuscript_failures,
        artefacts={"root": str(base), "report": "verification_report.json"},
    )
    summary = render_summary(outcome, results, executed)
    write_summary(base, summary)
    write_dataset_urls(base, registry.PUBLIC_DATASETS)
    manifest = write_integrity_manifest(base)
    outcome.artefacts["manifest_digest"] = str(manifest["manifest_digest"])
    outcome.artefacts["files"] = str(manifest["file_count"])
    logger.info("overall %s, code %s", outcome.overall_status, outcome.code_status)
    return outcome


def digest_of_checks(results: list[CheckResult]) -> str:
    return sha256_mapping({result.name: result.status for result in results})


USAGE = """bmstf verification driver

  --skip-tools   skip the formatter, type-checker, test and container gates
  --skip-study   skip the executed study and use the configured scale only
  --live         probe the resource landing pages (off by default: a live probe would
                 write a host-dependent status into a digested file)
  --root=PATH    write the artefacts at PATH instead of the repository root
  --help         print this text

Exit status is 0 unless a check outside the manuscript family fails, so a
discrepancy inside the article's own tables does not read as a code defect.
"""


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--help" in arguments or "-h" in arguments:
        print(USAGE, end="")
        return 0
    skip_tools = "--skip-tools" in arguments
    skip_study = "--skip-study" in arguments
    live = "--live" in arguments
    root_argument = None
    for argument in arguments:
        if argument.startswith("--root="):
            root_argument = Path(argument.split("=", 1)[1])
    outcome = run(root_argument, skip_tools=skip_tools, skip_study=skip_study, live=live)
    print(
        f"{outcome.overall_status} (code {outcome.code_status}); "
        f"{outcome.status_counts}; manuscript discrepancies {len(outcome.manuscript_failures)}"
    )
    return 1 if outcome.code_status == "UNVERIFIED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
