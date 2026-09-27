"""Durable file and JSON helpers.

Ref: Data availability statement (run record, manifest, checksums).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

JsonValue = Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> JsonValue:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _atomic_write(path: Path, payload: str) -> None:
    ensure_dir(path.parent)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        temporary_path.chmod(0o644)
        temporary_path.replace(path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def write_json(path: Path, payload: JsonValue, indent: int = 2) -> None:
    text = json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)
    _atomic_write(path, text + "\n")


def write_text(path: Path, text: str) -> None:
    _atomic_write(path, text if text.endswith("\n") else text + "\n")


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_mapping(payload: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(payload):
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(payload[key].encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Digest the payload of a tensor-like object, never its container metadata."""

    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}|{contiguous.shape}|".encode()
    return sha256_bytes(header + contiguous.tobytes())


def iter_files(root: Path, skip_dirs: Iterable[str] = ()) -> list[Path]:
    blocked = set(skip_dirs)
    found: list[Path] = []
    for base, directories, names in os.walk(root):
        directories[:] = sorted(directory for directory in directories if directory not in blocked)
        for name in sorted(names):
            found.append(Path(base) / name)
    return found


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".npz")
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
        temporary_path.chmod(0o644)
        temporary_path.replace(path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {name: np.asarray(handle[name]) for name in handle.files}
