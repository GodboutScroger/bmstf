"""Layered YAML configuration with dotted command-line overrides.

Ref: Sec. "Pre-specified clinical protocol" (the pre-specified constants), Code availability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from bmstf.runtime.store import read_text

LAYER_DIRECTORIES: tuple[str, ...] = ("insilico", "filter", "policy")

# The published slug. The paper title is longer than the host allows for a repository name,
# and any artefact that recorded the checkout directory's own name would then differ between
# the local tree and a clone of the published one, so the slug is the single source of truth.
RELEASE_SLUG: str = "bmstf"


class ConfigurationError(ValueError):
    """Raised when an experiment file is malformed or a key is missing."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def manifest_root(config_path: Path) -> Path:
    """configs/experiment/<name>.yaml resolves two directories up from its folder."""

    resolved = config_path.resolve()
    if resolved.parent.name != "experiment" or resolved.parent.parent.name != "configs":
        raise ConfigurationError(f"{resolved} is not a configs/experiment file")
    return resolved.parents[2]


@dataclass(frozen=True, slots=True)
class Section:
    data: Mapping[str, Any]

    def raw(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def as_float(self, key: str, default: float) -> float:
        value = self.data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(f"{key} must be numeric, found {value!r}")
        return float(value)

    def as_int(self, key: str, default: int) -> int:
        value = self.data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"{key} must be an integer, found {value!r}")
        return int(value)

    def as_bool(self, key: str, default: bool) -> bool:
        value = self.data.get(key, default)
        if not isinstance(value, bool):
            raise ConfigurationError(f"{key} must be a boolean, found {value!r}")
        return value

    def as_str(self, key: str, default: str) -> str:
        value = self.data.get(key, default)
        if not isinstance(value, str):
            raise ConfigurationError(f"{key} must be a string, found {value!r}")
        return value

    def as_tuple(self, key: str, length: int, default: Sequence[float]) -> tuple[float, ...]:
        value = self.data.get(key, list(default))
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise ConfigurationError(f"{key} must be a sequence of length {length}")
        return tuple(float(entry) for entry in value)

    def as_int_tuple(self, key: str, default: Sequence[int]) -> tuple[int, ...]:
        value = self.data.get(key, list(default))
        if not isinstance(value, (list, tuple)):
            raise ConfigurationError(f"{key} must be a sequence")
        return tuple(int(entry) for entry in value)


@dataclass(frozen=True, slots=True)
class Configuration:
    name: str
    values: Mapping[str, Any]
    sources: tuple[Path, ...]

    def section(self, prefix: str) -> Section:
        trimmed = prefix.rstrip(".")
        marker = f"{trimmed}."
        return Section(
            {
                key[len(marker) :]: value
                for key, value in self.values.items()
                if key.startswith(marker)
            }
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(cast(Mapping[str, Any], value), f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _read_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(read_text(path))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigurationError(f"{path} must hold a mapping at the top level")
    return cast(Mapping[str, Any], payload)


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def _apply_override(flat: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigurationError(f"override {override!r} must read key=value")
    key, _, raw = override.partition("=")
    flat[key.strip()] = _parse_scalar(raw)


def resolve_experiment(
    name: str, config_root: Path | None = None, overrides: Sequence[str] = ()
) -> Configuration:
    root = repository_root() if config_root is None else Path(config_root)
    experiment_path = root / "configs" / "experiment" / f"{name}.yaml"
    if not experiment_path.exists():
        raise ConfigurationError(f"no experiment named {name!r} under {root / 'configs'}")

    payload = _read_yaml(experiment_path)
    values: dict[str, Any] = {}
    sources: list[Path] = []
    layers = payload.get("extends", [])
    if not isinstance(layers, (list, tuple)):
        raise ConfigurationError("extends must be a list of layer names")
    for layer in layers:
        layer_name = str(layer)
        directory, _, stem = layer_name.partition("/")
        if directory not in LAYER_DIRECTORIES or not stem:
            raise ConfigurationError(f"layer {layer_name!r} must read <directory>/<stem>")
        layer_path = root / "configs" / directory / f"{stem}.yaml"
        if not layer_path.exists():
            raise ConfigurationError(f"layer file {layer_path} is missing")
        values.update(_flatten(_read_yaml(layer_path)))
        sources.append(layer_path)
    for key, value in payload.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping):
            values.update(_flatten(cast(Mapping[str, Any], value), f"{key}."))
        else:
            values[key] = value
    sources.append(experiment_path)
    for override in overrides:
        _apply_override(values, override)
    return Configuration(name=name, values=values, sources=tuple(sources))
