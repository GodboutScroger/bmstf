"""Exact discretisation and the regime chain."""

from __future__ import annotations

import numpy as np
import pytest

from bmstf.process.regimes import RegimeChain, is_hurwitz
from bmstf.process.transition import (
    continuous_transition,
    covariance_sqrt,
    discretise,
    reference_covariance,
    reference_input_gain,
    regime_transition,
)


def test_drift_matrices_are_hurwitz(system) -> None:
    assert is_hurwitz(system.drift[0], system.offset_index)
    assert is_hurwitz(system.drift[1], system.offset_index)


def test_transition_satisfies_the_semigroup_property(system) -> None:
    single = discretise(system.drift[1], system.input_matrix[1], system.noise[1], 1.0)
    triple = discretise(system.drift[1], system.input_matrix[1], system.noise[1], 3.0)
    assert (
        np.max(
            np.abs(single.transition @ single.transition @ single.transition - triple.transition)
        )
        < 1.0e-10
    )


def test_zero_interval_is_the_identity(system) -> None:
    step = discretise(system.drift[0], system.input_matrix[0], system.noise[0], 0.0)
    assert np.allclose(step.transition, np.eye(system.dimension))
    assert np.allclose(step.covariance, 0.0)
    assert np.allclose(step.input_gain, 0.0)


def test_input_gain_matches_a_riemann_sum(system) -> None:
    observed = discretise(system.drift[1], system.input_matrix[1], system.noise[1], 2.0).input_gain
    reference = reference_input_gain(system.drift[1], system.input_matrix[1], 2.0, substeps=4000)
    assert np.max(np.abs(observed - reference)) < 1.0e-6


def test_process_covariance_matches_a_riemann_sum(system) -> None:
    observed = discretise(system.drift[0], system.input_matrix[0], system.noise[0], 2.0).covariance
    reference = reference_covariance(system.drift[0], system.noise[0], 2.0, substeps=8000)
    assert np.max(np.abs(observed - reference)) < 1.0e-5


def test_process_covariance_is_symmetric_positive_semidefinite(system) -> None:
    step = discretise(system.drift[1], system.input_matrix[1], system.noise[1], 4.0)
    assert np.allclose(step.covariance, step.covariance.T)
    assert float(np.min(np.linalg.eigvalsh(step.covariance))) > -1.0e-12


def test_regime_transition_is_row_stochastic() -> None:
    chain = RegimeChain()
    transition = regime_transition(chain, 2.0)
    assert np.allclose(np.sum(transition, axis=1), 1.0)
    assert np.all(transition >= 0.0) and np.all(transition <= 1.0)


def test_regime_transition_reaches_the_stationary_distribution() -> None:
    chain = RegimeChain()
    long_run = np.linalg.matrix_power(regime_transition(chain, 1.0), 400)[0]
    assert np.max(np.abs(long_run - chain.stationary())) < 1.0e-10


def test_generator_matrix_rows_sum_to_zero() -> None:
    chain = RegimeChain()
    assert np.allclose(np.sum(chain.generator_matrix(), axis=1), 0.0)
    assert chain.stationary()[1] > 0.0
    dwell = chain.mean_dwell_months()
    assert dwell[0] > dwell[1]


def test_continuous_transition_is_monotone_in_time() -> None:
    chain = RegimeChain()
    early = regime_transition(chain, 1.0)[0, 1]
    late = regime_transition(chain, 5.0)[0, 1]
    assert 0.0 < early < late
    between = continuous_transition(chain.generator_matrix(), 3.0)[0, 1]
    assert early < between < late


def test_covariance_square_root_tolerates_singular_blocks() -> None:
    singular = np.diag(np.asarray([2.0, 0.0, 0.5]))
    root = covariance_sqrt(singular)
    assert np.max(np.abs(root @ root.T - singular)) < 1.0e-12


@pytest.mark.parametrize("delta", [0.5, 3.0, 12.0])
def test_discretisation_scales_with_the_interval(system, delta: float) -> None:
    step = discretise(system.drift[1], system.input_matrix[1], system.noise[1], delta)
    assert np.trace(step.covariance) >= 0.0
    assert np.all(np.isfinite(step.transition))
