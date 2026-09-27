"""Latent-dimension selection by the validation-fold energy score.

Ref: Sec. "Pre-specified clinical protocol: two-layer data design and delineation
cards" (the energy score applied to the validation folds was responsible for the
selection of the latent dimension d between 8 and 16).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmstf.inference.imm import (
    imm_step,
    initial_state,
    propagate_without_observation,
)
from bmstf.process.cohort import HistorySample
from bmstf.process.regimes import build_system
from bmstf.runtime.seeds import set_seed
from bmstf.scoring.energy import (
    DimensionSelection,
    candidate_dimensions,
    energy_score_samples,
    select_latent_dimension,
)
from bmstf.sensor.marginalise import MarginalisationSpec
from bmstf.study.engines import ArmSpec, base_filter, corrected_visits, prior_moments

VALIDATION_PATHS = 128


@dataclass(frozen=True, slots=True)
class DimensionReport:
    selection: DimensionSelection
    validation_pairs: int
    anchor: str

    def as_mapping(self) -> dict[str, float]:
        mapping = self.selection.as_mapping()
        mapping["validation_pairs"] = float(self.validation_pairs)
        return mapping


def one_step_energy(
    histories: list[HistorySample],
    dimension: int,
    rng: np.random.Generator,
    paths: int = VALIDATION_PATHS,
) -> float:
    """Mean energy score of the one-step predictive law against the next observation.

    The filter is run to each observation and its one-step predictive mixture is
    scored against the following measurement, which is the validation criterion the
    protocol names for the latent dimension.
    """

    system = build_system(dimension=dimension)
    configuration = base_filter(f"energy_d{dimension}", system)
    operator = system.augment() if configuration.bias_augmented else system.observation
    scores: list[float] = []
    for history in histories:
        if len(history.visits) < 3:
            continue
        visits = corrected_visits(history)
        prior_mean, prior_covariance = prior_moments(system, configuration, history)
        state = initial_state(system, configuration, prior_mean, prior_covariance)
        for index, visit in enumerate(visits):
            delta = (
                0.0
                if index == 0
                else float(history.visit_months[index] - history.visit_months[index - 1])
            )
            if index > 0:
                propagated = propagate_without_observation(
                    state, system, configuration, delta, history.inputs[index]
                )
                mean, covariance = propagated.collapsed()
                observation = np.asarray(visit.measurement, dtype=np.float64)
                if observation.shape[0] == operator.shape[0]:
                    samples = _sample_predictive(mean, covariance, operator, paths, rng)
                    scores.append(energy_score_samples(samples, observation))
            state = imm_step(state, system, configuration, delta, history.inputs[index], visit)
    if not scores:
        return float("inf")
    return float(np.mean(scores))


def _sample_predictive(
    mean: np.ndarray,
    covariance: np.ndarray,
    operator: np.ndarray,
    paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    from bmstf.process.transition import covariance_sqrt

    root = covariance_sqrt(covariance)
    draws = mean[None, :] + (root @ rng.standard_normal((mean.shape[0], int(paths)))).T
    return np.asarray(draws @ operator.T, dtype=np.float64)


def select_dimension(
    histories: list[HistorySample],
    sensor_spec: MarginalisationSpec,
    seed: int = 20260925,
    candidates: tuple[int, ...] | None = None,
    paths: int = VALIDATION_PATHS,
) -> DimensionReport:
    del sensor_spec
    rng = set_seed(int(seed)).resample
    grid = candidates if candidates is not None else candidate_dimensions()
    pairs = sum(max(len(history.visits) - 2, 0) for history in histories)
    selection = select_latent_dimension(
        grid, lambda dimension: one_step_energy(histories, dimension, rng, paths)
    )
    return DimensionReport(
        selection=selection,
        validation_pairs=int(pairs),
        anchor="Sec. 'Pre-specified clinical protocol' (energy score on the validation folds, d in 8..16)",
    )


def arm_for_dimension(dimension: int, system_name: str) -> ArmSpec:
    system = build_system(dimension=int(dimension))
    return ArmSpec(name=f"{system_name}_d{dimension}", filter=base_filter(system_name, system))
