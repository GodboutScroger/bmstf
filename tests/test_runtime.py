"""Configuration resolution, seeded streams, durable IO and checkpoints."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bmstf.runtime.checkpoint import (
    load_checkpoint,
    parameter_digest,
    restore_into,
    save_checkpoint,
)
from bmstf.runtime.config import (
    ConfigurationError,
    manifest_root,
    repository_root,
    resolve_experiment,
)
from bmstf.runtime.seeds import seed_sequence, set_seed
from bmstf.runtime.store import (
    iter_files,
    load_npz,
    read_json,
    save_npz,
    sha256_array,
    sha256_mapping,
    write_json,
    write_text,
)


def test_repository_root_contains_the_configs() -> None:
    root = repository_root()
    assert (root / "configs" / "experiment" / "main.yaml").exists()


def test_manifest_root_resolves_from_an_experiment_file() -> None:
    root = repository_root()
    path = root / "configs" / "experiment" / "main.yaml"
    assert manifest_root(path) == root
    with pytest.raises(ConfigurationError):
        manifest_root(root / "pyproject.toml")


def test_layered_resolution_merges_and_overrides() -> None:
    configuration = resolve_experiment("main")
    assert configuration.get("study.seeds") == 20
    assert configuration.get("study.patients") == 300
    assert configuration.get("study.calibration_patients") == 6000
    assert configuration.get("study.forecast_paths") == 256
    assert configuration.get("policy.alpha1") == 0.10
    assert configuration.get("policy.alpha2") == 0.01
    assert configuration.get("marginalisation.boundary_draws") == 64
    assert configuration.get("marginalisation.reconstruction_draws") == 8
    assert configuration.get("environment.reported_python") == "3.9.6"
    assert configuration.get("environment.reported_numpy") == "2.0.2"
    assert len(configuration.sources) == 4


def test_overrides_replace_typed_values() -> None:
    configuration = resolve_experiment("main", overrides=("study.seeds=2", "policy.alpha1=0.15"))
    section = configuration.section("study")
    assert section.as_int("seeds", 20) == 2
    assert configuration.section("policy").as_float("alpha1", 0.10) == 0.15


def test_section_accessors_reject_wrong_types() -> None:
    configuration = resolve_experiment("main")
    section = configuration.section("study")
    with pytest.raises(ConfigurationError):
        (
            section.as_int("seeds", 1)
            if not isinstance(section.raw("seeds"), int)
            else section.as_str("seeds", "x")
        )


def test_unknown_experiment_raises() -> None:
    with pytest.raises(ConfigurationError):
        resolve_experiment("does_not_exist")


def test_smoke_experiment_is_smaller_than_main() -> None:
    smoke = resolve_experiment("_smoke")
    assert smoke.section("study").as_int("seeds", 0) == 2
    assert smoke.section("study").as_int("patients", 0) == 8


def test_seed_streams_are_independent_and_reproducible() -> None:
    first = set_seed(11)
    second = set_seed(11)
    assert np.allclose(first.generator.random(5), second.generator.random(5))
    assert not np.allclose(first.generator.random(5), first.forecast.random(5))
    assert len(seed_sequence(11, 4)) == 4


def test_durable_json_round_trip(tmp_path) -> None:
    path = tmp_path / "record.json"
    write_json(path, {"b": 2, "a": 1})
    assert read_json(path) == {"a": 1, "b": 2}
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert oct(path.stat().st_mode)[-3:] == "644"
    write_text(tmp_path / "note.txt", "verified")
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "verified\n"


def test_array_npz_round_trip(tmp_path) -> None:
    arrays = {"a": np.arange(4.0), "b": np.ones((2, 2))}
    save_npz(tmp_path / "bundle.npz", arrays)
    restored = load_npz(tmp_path / "bundle.npz")
    assert np.allclose(restored["a"], arrays["a"])
    assert np.allclose(restored["b"], arrays["b"])
    assert oct((tmp_path / "bundle.npz").stat().st_mode)[-3:] == "644"


def test_digest_helpers_are_content_addressed() -> None:
    first = sha256_array(np.arange(3.0))
    assert first == sha256_array(np.arange(3.0))
    assert first != sha256_array(np.arange(3.0) + 1.0)
    mapping = {"b": first, "a": first}
    assert sha256_mapping(mapping) == sha256_mapping({"a": first, "b": first})
    assert len(sha256_mapping(mapping)) == 64


def test_iter_files_skips_caches(tmp_path) -> None:
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "skip.pyc").write_text("x", encoding="utf-8")
    found = [path.name for path in iter_files(tmp_path, skip_dirs=("__pycache__",))]
    assert found == ["keep.txt"]


def test_checkpoint_round_trip_preserves_the_payload(tmp_path) -> None:
    arrays = {"weight": np.linspace(-1.0, 1.0, 8), "bias": np.zeros(3)}
    digest = parameter_digest(arrays)
    record = save_checkpoint(tmp_path / "state.npz", arrays, seed=29, stage="fixture")
    for value in arrays.values():
        value[:] = 0.0
    loaded, restored = load_checkpoint(tmp_path / "state.npz")
    assert parameter_digest(loaded) == digest
    assert restored.seed == 29
    assert record.digest == digest
    assert oct((tmp_path / "state.npz").stat().st_mode)[-3:] == "644"


def test_checkpoint_digest_ignores_container_metadata(tmp_path) -> None:
    arrays = {"weight": np.arange(5.0)}
    first = save_checkpoint(tmp_path / "a.npz", arrays, seed=1, stage="one")
    second = save_checkpoint(tmp_path / "b.npz", arrays, seed=1, stage="two")
    assert first.digest == second.digest


def test_restore_into_updates_only_the_known_keys() -> None:
    target = {"a": np.zeros(3), "b": np.ones(2)}
    restore_into(target, {"a": np.full(3, 7.0), "c": np.zeros(1)})
    assert np.allclose(target["a"], 7.0)
    assert np.allclose(target["b"], 1.0)
    assert "c" not in target


def test_configuration_dump_is_json_serialisable() -> None:
    configuration = resolve_experiment("ablation_spread_half")
    assert json.dumps(dict(configuration.values), default=str)
    assert configuration.get("experiment.arms") == ["spread_half"]
