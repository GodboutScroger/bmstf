"""Boundary-condition prior, quasi-Monte-Carlo draws and the marginalisation."""

from __future__ import annotations

import numpy as np

from bmstf.geometry.shapes import geometry_from_state
from bmstf.geometry.summaries import summarise
from bmstf.sensor.boundary import BOUNDARY_FEATURES, BoundaryPrior
from bmstf.sensor.covariance import ResidualLookup, budget_shares, inflate
from bmstf.sensor.marginalise import (
    MarginalisationSpec,
    enumerate_boundary_budget,
    marginalise,
)
from bmstf.sensor.quasimc import draw_boundary_conditions, normal_from_uniform, sobol_uniform
from bmstf.sensor.surrogate import SHEAR_CHANNELS, PhysicsAttentionSurrogate


def test_prior_mean_respects_the_covariates_and_the_bounds() -> None:
    prior = BoundaryPrior()
    geometry = geometry_from_state("prior", 44.0, 0.3)
    mean = prior.mean_vector(summarise(geometry), 1)
    lower, upper = prior.sampling_bounds()
    assert mean.shape == (len(BOUNDARY_FEATURES),)
    assert np.all(mean >= lower - 1.0e-12)
    assert np.all(mean <= upper + 1.0e-12)


def test_sobol_points_are_stratified_and_reproducible() -> None:
    points = sobol_uniform(64, 5, seed=11)
    assert points.shape == (64, 5)
    assert float(np.min(points)) > 0.0 and float(np.max(points)) < 1.0
    assert np.allclose(points, sobol_uniform(64, 5, seed=11))
    for column in range(5):
        assert abs(float(np.mean(points[:, column])) - 0.5) < 0.05


def test_normal_transform_is_centred() -> None:
    standard = normal_from_uniform(sobol_uniform(256, 3, seed=5))
    assert abs(float(np.mean(standard))) < 0.05
    assert abs(float(np.std(standard)) - 1.0) < 0.15


def test_boundary_draws_sit_inside_the_bounds() -> None:
    prior = BoundaryPrior()
    geometry = geometry_from_state("draws", 50.0, 0.4)
    mean = prior.mean_vector(summarise(geometry), 0)
    draws = draw_boundary_conditions(prior, mean, 64, seed=17)
    lower, upper = prior.sampling_bounds()
    assert draws.shape == (64, len(BOUNDARY_FEATURES))
    assert np.all(draws >= lower - 1.0e-12)
    assert np.all(draws <= upper + 1.0e-12)


def test_surrogate_returns_the_declared_channels() -> None:
    surrogate = PhysicsAttentionSurrogate()
    geometry = geometry_from_state("channels", 44.0, 0.3)
    output = surrogate.evaluate(geometry, np.asarray([4.6, 0.86, 0.60, 3.5, 0.15]))
    assert output.values.shape == (len(SHEAR_CHANNELS),)
    assert output.values[0] > 0.0
    assert 0.0 <= output.values[2] <= 1.0
    assert set(output.as_mapping()) == set(SHEAR_CHANNELS)


def test_surrogate_burden_matches_a_direct_enumeration() -> None:
    surrogate = PhysicsAttentionSurrogate()
    geometry = geometry_from_state("enumerate", 47.0, 0.45)
    boundary = np.asarray([4.2, 0.9, 0.55, 3.2, 0.2])
    assert (
        abs(
            surrogate.evaluate(geometry, boundary).values[2]
            - surrogate.enumerate_burden(geometry, boundary)
        )
        < 1.0e-12
    )


def test_increasing_inflow_lowers_the_burden() -> None:
    surrogate = PhysicsAttentionSurrogate()
    geometry = geometry_from_state("inflow", 47.0, 0.45)
    low = surrogate.evaluate(geometry, np.asarray([2.6, 0.9, 0.6, 3.5, 0.1])).values[2]
    high = surrogate.evaluate(geometry, np.asarray([6.6, 0.9, 0.6, 3.5, 0.1])).values[2]
    assert high <= low


def test_marginalisation_decomposes_the_covariance(sensor) -> None:
    geometry = geometry_from_state("marg", 45.0, 0.3)
    observation = marginalise(
        geometry,
        summarise(geometry),
        0,
        sensor.surrogate,
        sensor.prior,
        ResidualLookup(),
        MarginalisationSpec(boundary_draws=32, reconstruction_draws=6),
        seed=23,
        rng=np.random.default_rng(23),
    )
    summed = (
        observation.sigma_boundary + observation.sigma_reconstruction + observation.sigma_surrogate
    )
    assert np.max(np.abs(summed - observation.covariance)) < 1.0e-12
    assert float(np.min(np.linalg.eigvalsh(observation.covariance))) > 0.0
    shares = budget_shares(
        observation.sigma_boundary, observation.sigma_reconstruction, observation.sigma_surrogate
    )
    assert abs(sum(shares.values()) - 1.0) < 1.0e-9


def test_topology_gate_inflates_the_boundary_budget(geometry) -> None:
    lookup = ResidualLookup()
    inflated = inflate(lookup.base(0), 1.6)
    assert np.allclose(inflated, 1.6 * lookup.base(0))


def test_residual_lookup_is_positive_definite_across_classes() -> None:
    lookup = ResidualLookup()
    for topology in range(4):
        for scale in (0.25, 1.0, 2.0):
            assert float(np.min(np.linalg.eigvalsh(lookup(topology, scale)))) > 0.0


def test_enumerated_budget_matches_the_vectorised_path(sensor) -> None:
    geometry = geometry_from_state("enum-budget", 43.0, 0.25)
    summary = summarise(geometry)
    spec = MarginalisationSpec(boundary_draws=16, reconstruction_draws=4)
    observation = marginalise(
        geometry,
        summary,
        1,
        sensor.surrogate,
        sensor.prior,
        ResidualLookup(),
        spec,
        seed=7,
        rng=np.random.default_rng(7),
    )
    enumerated = enumerate_boundary_budget(
        geometry, summary, 1, sensor.surrogate, sensor.prior, spec, seed=7
    )
    assert np.max(np.abs(enumerated - observation.sigma_boundary)) < 1.0e-12
