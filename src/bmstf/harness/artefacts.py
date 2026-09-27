"""Writers for the root artefacts and the integrity manifest.

Ref: Data availability statement (the run record, the results tables and the
manifest that relates every reported number to its summary key).

Every writer takes an explicit root so a test can direct output to a temporary
directory and never overwrite the shipped artefacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bmstf.runtime.config import RELEASE_SLUG
from bmstf.runtime.store import (
    iter_files,
    read_json,
    sha256_file,
    sha256_mapping,
    write_json,
    write_text,
)

SKIP_DIRECTORIES: tuple[str, ...] = (
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "runs",
    "outputs",
)
MANIFEST_NAME = "integrity_manifest.json"
SUMMARY_NAME = "verification_summary.txt"
CLAIM_NAME = "claim_to_code.json"
REPORT_NAME = "verification_report.json"
DATASET_NAME = "dataset_urls.txt"

ARTEFACT_NAMES: tuple[str, ...] = (CLAIM_NAME, REPORT_NAME, DATASET_NAME, SUMMARY_NAME)


def write_claim_to_code(root: Path, payload: Mapping[str, Any]) -> Path:
    path = root / CLAIM_NAME
    write_json(path, payload)
    return path


def write_verification_report(root: Path, payload: Mapping[str, Any]) -> Path:
    path = root / REPORT_NAME
    write_json(path, payload)
    return path


def write_summary(root: Path, text: str) -> Path:
    path = root / SUMMARY_NAME
    write_text(path, text)
    return path


def write_dataset_urls(root: Path, entries: Sequence[Mapping[str, str]]) -> Path:
    """Write the resource list as comment-prefixed provenance plus bare URLs."""

    from bmstf.harness.checks import DATASET_FOOTER, DATASET_HEADER

    lines = [DATASET_HEADER.rstrip("\n")]
    for entry in entries:
        lines.append(f"# {entry['name']}")
        lines.append(f"#   terms: {entry['license']}")
        lines.append(f"#   role : {entry['role']}")
        lines.append(str(entry["url"]))
    lines.extend(DATASET_FOOTER)
    path = root / DATASET_NAME
    write_text(path, "\n".join(lines) + "\n")
    return path


def dataset_url_lines(path: Path) -> list[str]:
    """The bare URLs of a written resource file, comments excluded."""

    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped:
            lines.append(stripped)
    return lines


def write_integrity_manifest(
    root: Path,
    exclude: Iterable[str] = (MANIFEST_NAME,),
    skip_directories: Sequence[str] = SKIP_DIRECTORIES,
) -> dict[str, Any]:
    blocked = {str(name) for name in exclude}
    root = Path(root).resolve()
    digests: dict[str, str] = {}
    for path in iter_files(root, skip_dirs=skip_directories):
        relative = path.relative_to(root).as_posix()
        if relative in blocked or path.name.startswith(".tmp-") or path.name.startswith(".ckpt-"):
            continue
        digests[relative] = sha256_file(path)
    payload: dict[str, Any] = {
        "root": RELEASE_SLUG,
        "file_count": len(digests),
        "files": digests,
        "manifest_digest": sha256_mapping(digests),
    }
    write_json(root / MANIFEST_NAME, payload)
    return payload


def verify_manifest(
    root: Path,
    payload: Mapping[str, Any] | None = None,
    skip_directories: Sequence[str] = SKIP_DIRECTORIES,
) -> dict[str, Any]:
    root = Path(root).resolve()
    recorded = dict(payload) if payload is not None else dict(read_json(root / MANIFEST_NAME))
    files = dict(recorded.get("files", {}))
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, digest in files.items():
        path = root / relative
        if not path.exists():
            missing.append(relative)
            continue
        if sha256_file(path) != digest:
            mismatched.append(relative)
    live = {
        path.relative_to(root).as_posix()
        for path in iter_files(root, skip_dirs=skip_directories)
        if path.name != MANIFEST_NAME
    }
    untracked = sorted(live - set(files))
    return {
        "recorded_files": len(files),
        "missing": sorted(missing),
        "mismatched": sorted(mismatched),
        "untracked": untracked,
        "consistent": not missing and not mismatched and not untracked,
    }


def manifest_digest_matches(root: Path) -> bool:
    root = Path(root).resolve()
    payload = dict(read_json(root / MANIFEST_NAME))
    return str(payload.get("manifest_digest")) == sha256_mapping(dict(payload.get("files", {})))
