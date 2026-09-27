"""Checkpoint save and load with atomic replacement.

Ref: Code availability (the run record and its checksums), Algorithm 2 (frozen parameters).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bmstf.runtime.store import ensure_dir, sha256_array


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    path: str
    seed: int
    stage: str
    digest: str
    keys: tuple[str, ...]


def parameter_digest(arrays: dict[str, np.ndarray]) -> str:
    """Digest the numeric payload, not the container, so a rewrite is a no-op."""

    if not arrays:
        return sha256_array(np.zeros(1, dtype=np.float64))
    combined = np.concatenate(
        [np.asarray(arrays[key], dtype=np.float64).reshape(-1) for key in sorted(arrays)]
    )
    return sha256_array(combined)


def save_checkpoint(
    path: Path,
    arrays: dict[str, np.ndarray],
    seed: int,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> CheckpointRecord:
    ensure_dir(path.parent)
    payload: dict[str, np.ndarray] = {
        f"array__{key}": np.asarray(value) for key, value in arrays.items()
    }
    payload["meta__seed"] = np.asarray([seed], dtype=np.int64)
    payload["meta__count"] = np.asarray([len(arrays)], dtype=np.int64)
    if extra:
        for key, value in extra.items():
            payload[f"extra__{key}"] = np.asarray(value)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".ckpt-")
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            np.savez(handle, **payload)  # type: ignore[arg-type]
        temporary_path.chmod(0o644)
        temporary_path.replace(path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return CheckpointRecord(
        path=path.name,
        seed=seed,
        stage=stage,
        digest=parameter_digest(arrays),
        keys=tuple(sorted(arrays)),
    )


def load_checkpoint(path: Path) -> tuple[dict[str, np.ndarray], CheckpointRecord]:
    with np.load(path, allow_pickle=False) as handle:
        names = list(handle.files)
        arrays = {
            name[len("array__") :]: np.asarray(handle[name])
            for name in names
            if name.startswith("array__")
        }
        seed = int(np.asarray(handle["meta__seed"]).reshape(-1)[0])
        count = int(np.asarray(handle["meta__count"]).reshape(-1)[0])
    if count != len(arrays):
        raise ValueError(f"{path.name} declares {count} arrays but holds {len(arrays)}")
    record = CheckpointRecord(
        path=path.name,
        seed=seed,
        stage="loaded",
        digest=parameter_digest(arrays),
        keys=tuple(sorted(arrays)),
    )
    return arrays, record


def restore_into(target: dict[str, np.ndarray], loaded: dict[str, np.ndarray]) -> None:
    for key, value in loaded.items():
        if key in target:
            target[key] = np.array(value, copy=True)
