"""Shared fixtures for the BM-STF test suite."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from bmstf.geometry.shapes import geometry_from_state
from bmstf.geometry.template import build_template
from bmstf.inference.forecast import ForecastSpec
from bmstf.process.cohort import HistorySpec, SensorContext, patient_batch
from bmstf.process.regimes import build_system
from bmstf.runtime.seeds import set_seed
from bmstf.sensor.boundary import BoundaryPrior
from bmstf.sensor.covariance import ResidualLookup
from bmstf.sensor.marginalise import MarginalisationSpec
from bmstf.sensor.surrogate import PhysicsAttentionSurrogate
from bmstf.study.histories import StudySpec

SMOKE_SEED = 20260925


@pytest.fixture(scope="session")
def system():
    return build_system()


@pytest.fixture(scope="session")
def sensor(system: object) -> SensorContext:
    return SensorContext(
        system,  # type: ignore[arg-type]
        ResidualLookup(),
        MarginalisationSpec(),
        PhysicsAttentionSurrogate(),
        BoundaryPrior(),
    )


@pytest.fixture(scope="session")
def template():
    return build_template()


@pytest.fixture(scope="session")
def geometry():
    return geometry_from_state("fixture", 42.0, 0.35)


@pytest.fixture(scope="session")
def histories(sensor: SensorContext) -> list:
    streams = set_seed(SMOKE_SEED)
    return patient_batch(6, sensor, HistorySpec(horizon_months=24), streams.generator, prefix="TST")


@pytest.fixture(scope="session")
def smoke_spec() -> StudySpec:
    return StudySpec(
        seeds=2,
        patients=6,
        calibration_patients=8,
        horizon_months=24,
        forecast_paths=8,
        arms=("BM-STF", "snapshot"),
    )


@pytest.fixture(scope="session")
def forecast_spec() -> ForecastSpec:
    return ForecastSpec(paths=8, horizon_months=24)


@pytest.fixture()
def rng() -> Iterator[np.random.Generator]:
    yield np.random.default_rng(SMOKE_SEED)
