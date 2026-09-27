"""The verification registry: closed-form references, execution and manuscript checks.

Ref: the manuscript's own derivations (Eq. (1)-(6), Algorithms 1-3), the reported
tables (Tables 1-5) and the statements in Code availability.

Every check either compares an implementation output with a hand-derived closed
form, with a brute-force recomputation written here, or with an independent
library, or it runs the pipeline and reports what actually happened. A check that
cannot run is NOT_RUN or BLOCKED with its reason, never a success.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from bmstf.geometry.shapes import ShapeSpec, geometry_from_state, target_ilt_fraction
from bmstf.geometry.summaries import summarise, thrombus_fraction
from bmstf.harness.artefacts import verify_manifest
from bmstf.harness.claims import CLAIMS, code_symbols
from bmstf.inference.forecast import monte_carlo_forecast
from bmstf.inference.imm import (
    FilterConfiguration,
    augmented_modes,
    run_filter_trace,
    two_regime_modes,
)
from bmstf.inference.kalman import kalman_update, standard_form_covariance
from bmstf.inference.mixing import conditional_mixing, mixing_weights, moment_match
from bmstf.policy.bands import (
    BAND_ORDER,
    calibrate_band_policy,
    literal_band_policy,
    point_crossing_probability,
)
from bmstf.policy.barriers import action_grid, read_out
from bmstf.policy.wilson import wilson_bounds, wilson_upper
from bmstf.process.cohort import HistorySpec, SensorContext, patient_batch
from bmstf.process.hazards import (
    StratumRisk,
    annual_rate_array,
    cumulative_hazard,
    repair_threshold_mm,
    step_hazard,
    step_hazard_array,
)
from bmstf.process.regimes import build_system
from bmstf.process.smooth import conditional_means, joint_gaussian_covariance, rts_smoother
from bmstf.process.transition import (
    covariance_sqrt,
    discretise,
    reference_covariance,
    reference_input_gain,
    regime_transition,
)
from bmstf.protocol import margins
from bmstf.protocol.criteria import ALPHA1, ALPHA2, BOUND_LEVEL, ReportedLimits
from bmstf.replay.censoring import fit_censoring
from bmstf.replay.estimator import ReplayVisit, estimate
from bmstf.replay.intensity import fit_intensity
from bmstf.runtime.checkpoint import load_checkpoint, parameter_digest, save_checkpoint
from bmstf.runtime.journal import get
from bmstf.runtime.seeds import set_seed
from bmstf.scoring.calibration import reliability_curve
from bmstf.scoring.crps import crps_gaussian, crps_mixture_gaussian, crps_sample
from bmstf.scoring.discrimination import (
    categorical_nri,
    net_benefit,
    roc_auc,
    treat_all_net_benefit,
)
from bmstf.scoring.energy import energy_score_gaussian, energy_score_samples
from bmstf.sensor.boundary import BoundaryPrior
from bmstf.sensor.covariance import ResidualLookup
from bmstf.sensor.marginalise import MarginalisationSpec, enumerate_boundary_budget, marginalise
from bmstf.sensor.quasimc import sobol_uniform
from bmstf.sensor.surrogate import PhysicsAttentionSurrogate
from bmstf.stats.agreement import cohen_kappa, icc_two_one, mean_squares
from bmstf.stats.bootstrap import cluster_bootstrap, percentile_interval
from bmstf.stats.multiplicity import holm_bonferroni
from bmstf.stats.roc import delong, paired_contrast
from bmstf.stats.signflip import enumerate_sign_flip_sum, sign_flip_p_value
from bmstf.study.engines import corrected_visits, prior_moments
from bmstf.study.histories import StudySpec, run_study
from bmstf.study.selection import one_step_energy

# Only resources whose landing page answered from the verification host with a
# success status and whose content matched the cited work. The regenerated
# AAA-100 / AAA-L solver fields are not posted anywhere the authors could find,
# and the two clinical cohorts are held by the sites, so nothing here is a
# redistributable patient or mesh deposit.
PUBLIC_DATASETS: tuple[dict[str, str], ...] = (
    {
        "name": "Rygiel et al., wall shear stress estimation in abdominal aortic aneurysms",
        "license": "arXiv preprint, arXiv non-exclusive licence",
        "role": "source of the regenerated AAA-100 / AAA-L CFD that Tables 5 panel B is parity-tested on",
        "url": "https://arxiv.org/abs/2507.22817",
    },
    {
        "name": "Blankemeier et al., Merlin: a computed tomography vision-language foundation model and dataset",
        "license": "Nature article, publisher's terms",
        "role": "the frozen abdominal-CT embedding the hazard head takes as image context",
        "url": "https://doi.org/10.1038/s41586-026-10181-8",
    },
    {
        "name": "Lannelongue et al., physics-constrained graph network for intracranial aneurysm hemodynamics",
        "license": "npj Digital Medicine, open access",
        "role": "the public intracranial geometries of the cross-anatomy stress test, never pooled with the aortic rows",
        "url": "https://www.nature.com/articles/s41746-026-02404-z",
    },
)

DATASET_HEADER: str = (
    "# Resources this release cites. Each landing page below was fetched from the release\n"
    "# host when this list was assembled and returned a success status carrying the cited\n"
    "# work; a shipping publisher whose TLS handshake this host cannot complete over the\n"
    "# default client is still listed, because the page itself served the request.\n"
    "# Re-check them on demand with `python -m bmstf.cli.verify --live`; the default\n"
    "# verification run stays offline so its report is a function of this repository only.\n"
    "# Nothing is redistributed here; the in-silico study needs no external data.\n"
)

DATASET_FOOTER: tuple[str, ...] = (
    "#",
    "# Checked and not listed:",
    "#   AAA-100 / AAA-L regenerated CFD solver fields - no public deposit located.",
    "#   Merlin CT foundation-model weights - the model card hosts are gated, so no",
    "#     account-free link could be verified.",
    "#   Retrospective supporting cohort and silent-mode arm (Sites A, B, C) -",
    "#     de-identified surveillance records held by the sites; not redistributable.",
)

CLOSED_FORM = "closed_form"
EXECUTION = "execution"
MANUSCRIPT = "manuscript"
INTEGRITY = "integrity"
TOOL = "tool"
DATA = "data"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    family: str
    status: str
    evidence: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "status": self.status,
            "evidence": self.evidence,
            "detail": self.detail,
        }


def _verdict(name: str, family: str, ok: bool, evidence: str, **detail: Any) -> CheckResult:
    return CheckResult(name, family, "PASS" if ok else "FAIL", evidence, dict(detail))


def _not_run(name: str, family: str, reason: str, **detail: Any) -> CheckResult:
    return CheckResult(name, family, "NOT_RUN", reason, dict(detail))


def _blocked(name: str, family: str, reason: str, **detail: Any) -> CheckResult:
    return CheckResult(name, family, "BLOCKED", reason, dict(detail))


def _configuration(system: Any, name: str = "bmstf") -> FilterConfiguration:
    modes = augmented_modes(two_regime_modes(system), system)
    return FilterConfiguration(name=name, modes=modes)


def probe_sensor(dimension: int = 10) -> tuple[Any, SensorContext]:
    system = build_system(dimension=int(dimension))
    sensor = SensorContext(
        system,
        ResidualLookup(),
        MarginalisationSpec(),
        PhysicsAttentionSurrogate(),
        BoundaryPrior(),
    )
    return system, sensor


def probe_histories(count: int = 24, seed: int = 20260925) -> tuple[Any, list[Any]]:
    system, sensor = probe_sensor()
    streams = set_seed(int(seed))
    histories = patient_batch(int(count), sensor, HistorySpec(), streams.generator, prefix="VER")
    return system, histories


def smoother_checks() -> list[CheckResult]:
    """The RTS smoother against the dense joint Gaussian it approximates.

    Ref: Sec. "Two-regime latent shear-state model" - conditional on a regime path the
    Kalman filter and smoother are exact. One fixed path is enough to test that: with a
    single linear-Gaussian mode the smoothed means must equal E[z_i | y_1..y_n] read off
    the joint covariance of the chain.
    """

    drift = np.asarray([[-0.35, 0.10], [0.0, -0.60]], dtype=np.float64)
    process = np.diag(np.asarray([0.05, 0.02], dtype=np.float64))
    step = discretise(drift, np.zeros((2, 1)), process, 1.0)
    operator = np.asarray([[1.0, 0.3]], dtype=np.float64)
    measurement_noise = np.asarray([[0.03]], dtype=np.float64)
    count = 6
    prior_covariance = np.eye(2, dtype=np.float64) * 0.5

    filtered_means = np.zeros((count, 2), dtype=np.float64)
    filtered_covariances = np.zeros((count, 2, 2), dtype=np.float64)
    predicted_means = np.zeros((count, 2), dtype=np.float64)
    predicted_covariances = np.zeros((count, 2, 2), dtype=np.float64)
    mean = np.zeros(2, dtype=np.float64)
    covariance = prior_covariance
    signals: list[float] = []
    for index in range(count):
        predicted_means[index] = mean
        predicted_covariances[index] = covariance
        signal = float(np.sin(1.7 * float(index + 1)))
        signals.append(signal)
        update = kalman_update(
            mean, covariance, operator, np.asarray([signal], dtype=np.float64), measurement_noise
        )
        filtered_means[index] = update.mean
        filtered_covariances[index] = update.covariance
        mean = step.transition @ update.mean
        covariance = step.transition @ update.covariance @ step.transition.T + step.covariance

    smoothed = rts_smoother(
        filtered_means,
        filtered_covariances,
        predicted_means,
        predicted_covariances,
        [step] * (count - 1),
    )
    prior_covariances = np.zeros((count, 2, 2), dtype=np.float64)
    prior_covariances[0] = prior_covariance
    for index in range(1, count):
        prior_covariances[index] = (
            step.transition @ prior_covariances[index - 1] @ step.transition.T + step.covariance
        )
    joint = joint_gaussian_covariance(step.transition, prior_covariances)
    block_operator = np.zeros((count, count * 2), dtype=np.float64)
    for index in range(count):
        block_operator[index, index * 2 : (index + 1) * 2] = operator[0]
    exact = conditional_means(
        joint,
        block_operator,
        np.eye(count, dtype=np.float64) * float(measurement_noise[0, 0]),
        np.asarray(signals, dtype=np.float64),
    )
    gap = float(np.max(np.abs(smoothed.means.reshape(-1) - exact)))
    gain = float(np.max(np.abs(smoothed.means - filtered_means)))
    return [
        _verdict(
            "smoother_matches_dense_joint_gaussian",
            CLOSED_FORM,
            gap < 1.0e-9,
            f"the smoothed means differ from E[z | y] of the chain's joint Gaussian by "
            f"{gap:.3e} over {count} observations, and they move {gain:.3e} away from the "
            "filtered means, so the agreement is not the trivial no-op",
            gap=gap,
            filtered_gap=gain,
        )
    ]


def closed_form_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []
    system = build_system()
    chain = system.chain

    step = discretise(system.drift[1], system.input_matrix[1], system.noise[1], 3.0)
    composed = discretise(system.drift[1], system.input_matrix[1], system.noise[1], 1.0)
    product = composed.transition @ composed.transition @ composed.transition
    gap = float(np.max(np.abs(product - step.transition)))
    checks.append(
        _verdict(
            "transition_semigroup_closed_form",
            CLOSED_FORM,
            gap < 1.0e-10,
            f"Phi(3) differs from Phi(1)^3 by {gap:.3e}, the matrix-exponential semigroup property",
            gap=gap,
        )
    )

    gamma_gap = float(
        np.max(
            np.abs(
                reference_input_gain(system.drift[1], system.input_matrix[1], 2.5, substeps=20_000)
                - discretise(
                    system.drift[1], system.input_matrix[1], system.noise[1], 2.5
                ).input_gain
            )
        )
    )
    checks.append(
        _verdict(
            "input_gain_matches_riemann_sum",
            CLOSED_FORM,
            gamma_gap < 5.0e-6,
            f"Gamma(2.5) differs from a 20000-substep Riemann sum of Phi(s)B by {gamma_gap:.3e}, "
            "the trapezoidal quadrature's own error",
            gap=gamma_gap,
        )
    )

    covariance_gap = float(
        np.max(
            np.abs(
                reference_covariance(system.drift[1], system.noise[1], 2.5, substeps=20_000)
                - discretise(
                    system.drift[1], system.input_matrix[1], system.noise[1], 2.5
                ).covariance
            )
        )
    )
    checks.append(
        _verdict(
            "process_covariance_matches_riemann_sum",
            CLOSED_FORM,
            covariance_gap < 5.0e-5,
            f"Q(2.5) differs from a 20000-substep Riemann sum of Phi(s) Q Phi(s)^T by "
            f"{covariance_gap:.3e}, the trapezoidal quadrature's own error",
            gap=covariance_gap,
        )
    )

    transition = regime_transition(chain, 2.0)
    rows = float(np.max(np.abs(np.sum(transition, axis=1) - 1.0)))
    semigroup = regime_transition(chain, 1.0) @ regime_transition(chain, 1.0)
    semigroup_gap = float(np.max(np.abs(semigroup - transition)))
    stationary = np.linalg.matrix_power(regime_transition(chain, 1.0), 500)[0]
    stationary_gap = float(np.max(np.abs(stationary - chain.stationary())))
    checks.append(
        _verdict(
            "regime_transition_row_stochastic",
            CLOSED_FORM,
            rows < 1.0e-12 and semigroup_gap < 1.0e-12 and stationary_gap < 1.0e-10,
            f"row sums deviate by {rows:.3e}, Pi(1)^2 differs from Pi(2) by {semigroup_gap:.3e}, "
            f"Pi(1)^500 differs from the stationary distribution by {stationary_gap:.3e}",
            row_gap=rows,
            semigroup_gap=semigroup_gap,
            stationary_gap=stationary_gap,
        )
    )

    generator = chain.generator_matrix()
    values, vectors = np.linalg.eig(generator.T)
    index = int(np.argmin(np.abs(values)))
    null_vector = np.real(vectors[:, index])
    null_vector = null_vector / float(np.sum(null_vector))
    null_gap = float(np.max(np.abs(null_vector - chain.stationary())))
    checks.append(
        _verdict(
            "stationary_distribution_from_generator",
            CLOSED_FORM,
            null_gap < 1.0e-12,
            f"the left null vector of L differs from the stationary weights by {null_gap:.3e}",
            gap=null_gap,
        )
    )

    means = np.asarray([[0.2, -1.0], [1.4, 0.6], [-0.3, 2.1]])
    covariances = np.asarray([np.eye(2) * 0.2, np.eye(2) * 0.05, np.eye(2) * 0.9])
    weights = np.asarray([0.5, 0.3, 0.2])
    matched_mean, matched_covariance = moment_match(means, covariances, weights)
    rng = np.random.default_rng(7)
    draw_counts = [round(120_000 * float(weight)) for weight in weights]
    draws = np.concatenate(
        [
            rng.multivariate_normal(means[i], covariances[i], size=draw_counts[i])
            for i in range(means.shape[0])
        ]
    )
    empirical_mean = np.mean(draws, axis=0)
    empirical_covariance = np.cov(draws, rowvar=False)
    mean_gap = float(np.max(np.abs(empirical_mean - matched_mean)))
    covariance_gap_match = float(np.max(np.abs(empirical_covariance - matched_covariance)))
    checks.append(
        _verdict(
            "moment_match_preserves_two_moments",
            CLOSED_FORM,
            mean_gap < 0.02 and covariance_gap_match < 0.05,
            f"the collapsed mean matches a weight-proportional sample of 120k draws to {mean_gap:.3e} and the covariance to "
            f"{covariance_gap_match:.3e}",
            mean_gap=mean_gap,
            covariance_gap=covariance_gap_match,
        )
    )

    mix = conditional_mixing(regime_transition(chain, 3.0), np.asarray([0.7, 0.3]))
    mix_rows = float(np.max(np.abs(np.sum(mix, axis=1) - 1.0)))
    prior = mixing_weights(regime_transition(chain, 3.0), np.asarray([0.7, 0.3]))
    checks.append(
        _verdict(
            "conditional_mixing_rows_normalised",
            CLOSED_FORM,
            mix_rows < 1.0e-12 and bool(np.all(prior > 0.0)),
            f"the conditional mixing rows sum to one to {mix_rows:.3e} and the predictive weights "
            f"are positive ({prior[0]:.4f}, {prior[1]:.4f})",
            row_gap=mix_rows,
            prior=prior.tolist(),
        )
    )

    rng = np.random.default_rng(11)
    dimension = 4
    mean = rng.normal(size=dimension)
    root = rng.normal(size=(dimension, dimension))
    covariance = root @ root.T + np.eye(dimension) * 0.1
    operator = rng.normal(size=(3, dimension))
    measurement = rng.normal(size=3)
    noise = np.eye(3) * 0.05
    update = kalman_update(mean, covariance, operator, measurement, noise)
    alternative = standard_form_covariance(covariance, operator, noise)
    covariance_gap_kalman = float(np.max(np.abs(update.covariance - alternative)))
    residual_mean = update.mean - (mean + update.gain @ (measurement - operator @ mean))
    checks.append(
        _verdict(
            "kalman_update_matches_standard_form",
            CLOSED_FORM,
            covariance_gap_kalman < 1.0e-10
            and float(np.max(np.abs(residual_mean))) < 1.0e-12
            and float(
                np.max(
                    np.abs(
                        update.gain
                        - covariance
                        @ operator.T
                        @ np.linalg.inv(operator @ covariance @ operator.T + noise)
                    )
                )
            )
            < 1.0e-10,
            f"the Joseph-form covariance differs from the short form by {covariance_gap_kalman:.3e}; "
            "the gain and the mean satisfy their definitions",
            gap=covariance_gap_kalman,
        )
    )

    from scipy.optimize import brentq
    from scipy.stats import binomtest

    def score_root(estimate: float, count: int, level: float) -> float:
        def equation(p: float) -> float:
            return (estimate - p) / float(np.sqrt(p * (1.0 - p) / float(count))) + level

        return float(brentq(equation, 1.0e-9, 1.0 - 1.0e-9, xtol=1.0e-14, rtol=1.0e-15))

    root_gaps: list[float] = []
    library_gaps: list[float] = []
    for successes, count in ((2, 64), (4, 8), (90, 100), (0, 32), (17, 40)):
        estimate = float(successes) / float(count)
        observed = wilson_bounds(estimate, count, BOUND_LEVEL)
        root_gaps.append(abs(observed.upper - score_root(estimate, count, BOUND_LEVEL)))
        reference = binomtest(successes, count).proportion_ci(
            confidence_level=2.0 * 0.95 - 1.0, method="wilson"
        )
        library_gaps.append(abs(observed.upper - float(reference.high)))
    checks.append(
        _verdict(
            "wilson_bound_solves_the_score_equation",
            CLOSED_FORM,
            max(root_gaps) < 1.0e-11,
            f"the Wilson upper bound solves the score equation "
            f"(p_hat - p)/sqrt(p(1-p)/n) = -z to {max(root_gaps):.3e} at five (k, n) pairs",
            gap=max(root_gaps),
        )
    )
    checks.append(
        _verdict(
            "wilson_bound_corroborated_by_binomtest",
            CLOSED_FORM,
            max(library_gaps) < 1.0e-4,
            f"the same bound agrees with SciPy's binomtest Wilson interval to "
            f"{max(library_gaps):.3e}, the library's own root-finding tolerance",
            gap=max(library_gaps),
        )
    )

    samples = np.random.default_rng(3).normal(0.4, 1.7, size=200_000)
    observed_crps = crps_sample(samples, 0.9)
    closed_crps = crps_gaussian(0.4, 1.7, 0.9)
    crps_gap = abs(observed_crps - closed_crps)
    checks.append(
        _verdict(
            "crps_sample_matches_gaussian_closed_form",
            CLOSED_FORM,
            crps_gap < 0.01,
            f"the sample CRPS {observed_crps:.6f} matches the closed form {closed_crps:.6f} to "
            f"{crps_gap:.3e} on 200k standard normal draws",
            gap=crps_gap,
        )
    )

    weights_mix = np.asarray([0.4, 0.6])
    means_mix = np.asarray([-1.0, 2.0])
    variances_mix = np.asarray([0.09, 0.25])
    mixture = crps_mixture_gaussian(weights_mix, means_mix, variances_mix, 0.3)
    rng = np.random.default_rng(5)
    counts = (60_000, 90_000)
    draws = np.concatenate(
        [
            rng.normal(means_mix[0], math.sqrt(variances_mix[0]), size=counts[0]),
            rng.normal(means_mix[1], math.sqrt(variances_mix[1]), size=counts[1]),
        ]
    )
    empirical_mixture = crps_sample(draws, 0.3)
    mixture_gap = abs(mixture - empirical_mixture)
    checks.append(
        _verdict(
            "crps_mixture_matches_sample",
            CLOSED_FORM,
            mixture_gap < 0.01,
            f"the mixture CRPS {mixture:.6f} matches the sample estimate {empirical_mixture:.6f} "
            f"to {mixture_gap:.3e}",
            gap=mixture_gap,
        )
    )

    mean_energy = np.asarray([0.5, -0.4])
    covariance_energy = np.asarray([[0.3, 0.05], [0.05, 0.2]])
    observation_energy = np.asarray([0.9, 0.1])
    closed_energy = energy_score_gaussian(mean_energy, covariance_energy, observation_energy)
    rng = np.random.default_rng(13)
    draws_energy = rng.multivariate_normal(mean_energy, covariance_energy, size=8000)
    sample_energy = energy_score_samples(draws_energy, observation_energy)
    energy_gap = abs(closed_energy - sample_energy)
    checks.append(
        _verdict(
            "energy_score_closed_form_matches_samples",
            CLOSED_FORM,
            energy_gap < 0.03,
            f"the Gaussian energy score {closed_energy:.6f} matches 8000 draws {sample_energy:.6f} "
            f"to {energy_gap:.3e}",
            gap=energy_gap,
        )
    )

    raw = np.asarray([0.0007, 0.0012, 0.0031, 0.0036, 0.0045, 0.031, 0.09, 0.12, 0.4, 0.7, 0.9])
    adjusted = holm_bonferroni(raw)
    order = np.argsort(raw, kind="mergesort")
    brute = np.empty(raw.shape[0], dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, float(raw.shape[0] - position) * float(raw[index])))
        brute[index] = running
    holm_gap = float(np.max(np.abs(adjusted - brute)))
    checks.append(
        _verdict(
            "holm_step_down_matches_definition",
            CLOSED_FORM,
            holm_gap < 1.0e-15,
            f"the adjusted values differ from a from-scratch step-down by {holm_gap:.3e} over "
            f"{raw.shape[0]} contrasts",
            gap=holm_gap,
        )
    )

    differences = np.asarray([0.05, -0.02, 0.11, 0.003, -0.07, 0.09])
    flip = sign_flip_p_value(differences)
    sums = enumerate_sign_flip_sum(differences)
    counted = float(np.count_nonzero(np.abs(sums) >= abs(float(np.sum(differences))) - 1.0e-12))
    manual = counted / float(sums.shape[0])
    flip_gap = abs(flip.p_value - manual)
    checks.append(
        _verdict(
            "sign_flip_p_value_matches_enumeration",
            CLOSED_FORM,
            flip_gap < 1.0e-15 and flip.sign_vectors == 2 ** differences.shape[0],
            f"the exact p value {flip.p_value:.6f} equals the enumerated count over "
            f"{flip.sign_vectors} sign vectors",
            gap=flip_gap,
        )
    )

    rng = np.random.default_rng(17)
    values = rng.normal(0.3, 0.2, size=64)
    interval = percentile_interval(values, resamples=4000, rng=np.random.default_rng(19))
    draws = np.random.default_rng(19).integers(0, values.shape[0], size=(4000, values.shape[0]))
    means = np.mean(values[draws], axis=1)
    lower = float(np.quantile(means, 0.025))
    upper = float(np.quantile(means, 0.975))
    bootstrap_gap = max(abs(interval.lower - lower), abs(interval.upper - upper))
    checks.append(
        _verdict(
            "percentile_bootstrap_matches_manual_recipe",
            CLOSED_FORM,
            bootstrap_gap < 1.0e-12,
            f"the interval matches a hand-rolled bootstrap recipe to {bootstrap_gap:.3e}",
            gap=bootstrap_gap,
        )
    )

    rng = np.random.default_rng(23)
    labels = np.concatenate([np.ones(40), np.zeros(70)])
    scores = rng.normal(size=110) + 0.7 * labels
    scores[::9] = np.round(scores[::9], 1)
    from sklearn.metrics import roc_auc_score

    observed_auc = roc_auc(scores, labels)
    reference_auc = float(roc_auc_score(labels, scores))
    auc_gap = abs(observed_auc - reference_auc)
    checks.append(
        _verdict(
            "roc_auc_matches_sklearn",
            CLOSED_FORM,
            auc_gap < 1.0e-12,
            f"the rank-based area {observed_auc:.12f} matches scikit-learn {reference_auc:.12f}",
            gap=auc_gap,
        )
    )

    rng = np.random.default_rng(29)
    labels_delong = np.concatenate([np.ones(45), np.zeros(75)])
    scores_delong = rng.normal(size=120) + 0.8 * labels_delong
    scores_delong[::5] = 0.0
    result = delong(scores_delong, labels_delong)
    positive = scores_delong[labels_delong == 1.0]
    negative = scores_delong[labels_delong == 0.0]
    matrix = np.zeros((positive.shape[0], negative.shape[0]), dtype=np.float64)
    for row, value in enumerate(positive):
        for column, other in enumerate(negative):
            matrix[row, column] = 1.0 if value > other else (0.5 if value == other else 0.0)
    v10 = matrix.mean(axis=1)
    v01 = matrix.mean(axis=0)
    brute_variance = float(
        np.var(v10, ddof=1) / negative.shape[0] + np.var(v01, ddof=1) / positive.shape[0]
    )
    delong_gap = abs(result.variance - brute_variance)
    checks.append(
        _verdict(
            "delong_variance_matches_double_loop",
            CLOSED_FORM,
            delong_gap < 1.0e-15,
            f"the DeLong variance matches a double-loop placement computation to {delong_gap:.3e}",
            gap=delong_gap,
        )
    )

    contrast = paired_contrast(scores_delong, scores_delong * 0.98 + 0.05, labels_delong)
    checks.append(
        _verdict(
            "paired_contrast_is_finite",
            CLOSED_FORM,
            math.isfinite(contrast.standard_error) and 0.0 <= contrast.p_value <= 1.0,
            f"paired difference {contrast.difference:.6f} with standard error "
            f"{contrast.standard_error:.6e} and p {contrast.p_value:.6f}",
            difference=contrast.difference,
        )
    )

    probabilities = np.asarray([0.05, 0.15, 0.25, 0.45, 0.55, 0.75, 0.95])
    outcomes = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    curve = reliability_curve(probabilities, outcomes, bins=5)
    ece = curve.expected_calibration_error()
    manual_ece = 0.0
    total = float(np.sum(curve.counts))
    for position in range(curve.counts.shape[0]):
        if curve.counts[position] > 0:
            manual_ece += curve.counts[position] * abs(
                curve.predicted[position] - curve.observed[position]
            )
    manual_ece /= total
    ece_gap = abs(ece - manual_ece)
    checks.append(
        _verdict(
            "expected_calibration_error_matches_binning",
            CLOSED_FORM,
            ece_gap < 1.0e-15,
            f"ECE {ece:.6f} equals the weighted bin gap recomputed by hand {manual_ece:.6f}",
            gap=ece_gap,
        )
    )

    nri = categorical_nri(
        np.asarray([0.05, 0.15, 0.25, 0.45, 0.55, 0.75, 0.95]),
        np.asarray([0.06, 0.14, 0.26, 0.44, 0.56, 0.74, 0.94]),
        outcomes,
    )
    checks.append(
        _verdict(
            "net_reclassification_matches_definition",
            CLOSED_FORM,
            abs(nri.nri - (nri.event_up - nri.event_down + nri.non_event_down - nri.non_event_up))
            < 1.0e-15,
            f"NRI {nri.nri:.6f} satisfies its own definition on {nri.counts} decisions",
            nri=nri.nri,
        )
    )

    benefit = net_benefit(0.9, 0.9, 0.4, 0.2)
    hand = 0.9 * 0.4 - 0.1 * 0.6 * (0.2 / 0.8)
    treat_all = treat_all_net_benefit(0.4, 0.2)
    checks.append(
        _verdict(
            "net_benefit_matches_closed_form",
            CLOSED_FORM,
            abs(benefit - hand) < 1.0e-15 and abs(treat_all - (0.4 - 0.6 * 0.25)) < 1.0e-15,
            f"net benefit {benefit:.12f} against the hand expression {hand:.12f}",
            gap=abs(benefit - hand),
        )
    )

    diameters = np.linspace(28.0, 72.0, 400)
    vectorised = step_hazard_array(diameters, np.full_like(diameters, 0.4), "male")
    scalar = np.asarray([step_hazard(float(value), 0.4, "male") for value in diameters])
    hazard_gap = float(np.max(np.abs(vectorised - scalar)))
    rates = annual_rate_array(diameters, "female")
    scalar_rates = np.asarray(
        [StratumRisk().annual_rate(float(value), "female") for value in diameters]
    )
    rate_gap = float(np.max(np.abs(rates - scalar_rates)))
    checks.append(
        _verdict(
            "vectorised_hazard_matches_scalar_definition",
            CLOSED_FORM,
            hazard_gap < 1.0e-15 and rate_gap < 1.0e-15,
            f"the vectorised hazard and stratum lookup match the scalar definitions to "
            f"{max(hazard_gap, rate_gap):.3e} over 400 diameters",
            hazard_gap=hazard_gap,
            rate_gap=rate_gap,
        )
    )

    burden = np.linspace(0.0, 0.9, 50)
    grid_diameter = np.linspace(30.0, 60.0, 50)
    cumulative = cumulative_hazard(grid_diameter, burden, "male")
    manual_values: list[float] = []
    running = 1.0
    for value, level in zip(grid_diameter, burden, strict=True):
        running *= 1.0 - step_hazard(float(value), float(level), "male")
        manual_values.append(1.0 - running)
    cumulative_gap = float(np.max(np.abs(cumulative - np.asarray(manual_values))))
    checks.append(
        _verdict(
            "cumulative_hazard_matches_prefix_product",
            CLOSED_FORM,
            cumulative_gap < 1.0e-15,
            f"1 - prod(1 - chi) matches a hand loop to {cumulative_gap:.3e}",
            gap=cumulative_gap,
        )
    )

    table = StratumRisk()
    grid_months = action_grid()
    gate = table.gate_month(52.0, "male", grid_months, ALPHA2)
    admissible = [
        float(value)
        for value in grid_months
        if table.cumulative_event_probability(52.0, "male", float(value)) <= ALPHA2
    ]
    gate_gap = abs(gate - (max(admissible) if admissible else 0.0))
    checks.append(
        _verdict(
            "gate_month_matches_direct_evaluation",
            CLOSED_FORM,
            gate_gap < 1.0e-15,
            f"the gate returns {gate:.1f} months, equal to the largest admissible grid horizon "
            f"evaluated directly",
            gap=gate_gap,
        )
    )

    singular = np.diag(np.asarray([2.0, 0.0, 0.5]))
    root_singular = covariance_sqrt(singular)
    reconstructed = root_singular @ root_singular.T
    root_gap = float(np.max(np.abs(reconstructed - singular)))
    checks.append(
        _verdict(
            "singular_covariance_square_root_is_exact",
            CLOSED_FORM,
            root_gap < 1.0e-12,
            f"the symmetric square root reconstructs a rank-deficient covariance to "
            f"{root_gap:.3e}, which is what the zero-noise blocks of Q_r(Delta) require",
            gap=root_gap,
        )
    )

    return checks


def sensor_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []
    _, sensor = probe_sensor()
    streams = set_seed(5)
    histories = patient_batch(3, sensor, HistorySpec(), streams.generator, prefix="SNR")
    history = histories[0]
    geometry = geometry_from_state(
        "probe", float(history.monthly_diameter[6]), float(history.monthly_burden[6])
    )
    summary = summarise(geometry)
    observation = marginalise(
        geometry,
        summary,
        history.site,
        sensor.surrogate,
        sensor.prior,
        ResidualLookup(),
        MarginalisationSpec(),
        seed=3,
        rng=streams.sensor,
    )
    eigenvalues = np.linalg.eigvalsh(observation.covariance)
    decomposed = (
        observation.sigma_boundary + observation.sigma_reconstruction + observation.sigma_surrogate
    )
    additivity = float(np.max(np.abs(decomposed - observation.covariance)))
    checks.append(
        _verdict(
            "marginalised_covariance_is_positive_definite",
            CLOSED_FORM,
            float(np.min(eigenvalues)) > 0.0 and additivity < 1.0e-12,
            f"the marginalised covariance is positive definite (smallest eigenvalue "
            f"{float(np.min(eigenvalues)):.3e}) and equals the sum of its three terms to "
            f"{additivity:.3e}",
            smallest_eigenvalue=float(np.min(eigenvalues)),
        )
    )

    enumerated = enumerate_boundary_budget(
        geometry,
        summary,
        history.site,
        sensor.surrogate,
        sensor.prior,
        MarginalisationSpec(),
        seed=3,
    )
    boundary_gap = float(np.max(np.abs(enumerated - observation.sigma_boundary)))
    checks.append(
        _verdict(
            "boundary_budget_matches_explicit_loop",
            CLOSED_FORM,
            boundary_gap < 1.0e-10,
            f"the vectorised boundary-condition budget matches an explicit per-draw loop to "
            f"{boundary_gap:.3e} on {observation.boundary_draws} draws",
            gap=boundary_gap,
        )
    )

    surrogate = sensor.surrogate
    boundary = sensor.prior.mean_vector(summary, history.site)
    enumerated_burden = float(surrogate.enumerate_burden(geometry, boundary))
    summarised = float(surrogate.evaluate(geometry, boundary).values[2])
    burden_gap = abs(enumerated_burden - summarised)
    checks.append(
        _verdict(
            "surrogate_burden_matches_enumeration",
            CLOSED_FORM,
            burden_gap < 1.0e-12,
            f"the low-shear burden {summarised:.12f} equals a direct node enumeration to "
            f"{burden_gap:.3e}",
            gap=burden_gap,
        )
    )

    lookup = ResidualLookup()
    spectra = [
        float(np.min(np.linalg.eigvalsh(lookup(int(cls), float(scale)))))
        for cls in range(4)
        for scale in (0.25, 1.0, 2.0)
    ]
    checks.append(
        _verdict(
            "residual_lookup_positive_definite",
            CLOSED_FORM,
            min(spectra) > 0.0,
            f"all four topology classes at three scales give positive-definite covariances "
            f"(smallest eigenvalue {min(spectra):.3e})",
            smallest=min(spectra),
        )
    )

    points = sobol_uniform(64, 5, seed=11)
    in_unit = bool(np.all((points > 0.0) & (points < 1.0)))
    repeated = sobol_uniform(64, 5, seed=11)
    deterministic = float(np.max(np.abs(points - repeated)))
    checks.append(
        _verdict(
            "scrambled_sobol_is_deterministic_in_unit_cube",
            CLOSED_FORM,
            in_unit and deterministic < 1.0e-15 and points.shape == (64, 5),
            f"64 points in five dimensions lie strictly inside the unit cube and repeat exactly "
            f"for a fixed seed (gap {deterministic:.3e})",
            shape=list(points.shape),
        )
    )

    fractions = []
    for burden in (0.1, 0.35, 0.6, 0.85):
        sample_geometry = geometry_from_state("ilt", 44.0, burden)
        fractions.append(
            abs(thrombus_fraction(sample_geometry) - target_ilt_fraction(burden, ShapeSpec()))
        )
    checks.append(
        _verdict(
            "geometry_ilt_solver_hits_target",
            CLOSED_FORM,
            max(fractions) < 5.0e-3,
            f"the bisection reproduces the requested ILT fraction to {max(fractions):.3e} at four "
            "burden levels",
            gap=max(fractions),
        )
    )

    components = observation.shares()
    checks.append(
        _verdict(
            "budget_shares_sum_to_one",
            CLOSED_FORM,
            abs(sum(components.values()) - 1.0) < 1.0e-9,
            "the boundary, reconstruction and surrogate trace shares sum to one "
            f"({components['boundary']:.3f}, {components['reconstruction']:.3f}, "
            f"{components['surrogate']:.3f})",
            shares=components,
        )
    )
    return checks


def execution_checks(
    root: Path,
    spec: StudySpec,
    arms: tuple[str, ...],
) -> tuple[list[CheckResult], dict[str, Any]]:
    checks: list[CheckResult] = []
    system, sensor = probe_sensor()
    strata = StratumRisk()
    result = run_study(sensor, system, spec, strata, alpha1=ALPHA1, alpha2=ALPHA2)
    aggregate = result.aggregate
    context: dict[str, Any] = {
        "scale": spec.as_mapping(),
        "scalar_noise": [float(value) for value in np.asarray(result.scalar_noise).reshape(-1)],
        "aggregate": aggregate,
        "arms": list(arms),
    }

    undefined = sorted(
        f"{arm}.{key}"
        for arm, entry in aggregate.items()
        for key, value in entry.items()
        if not key.endswith(("lower", "upper")) and not math.isfinite(value)
    )
    # The snapshot carries a point state, so there is no estimated between-scan change to
    # score; the metric stays undefined for it rather than reading as perfect tracking.
    allowed = {"snapshot.burden_change_rmse"}
    unexpected = [key for key in undefined if key not in allowed]
    checks.append(
        _verdict(
            "study_execution_all_metrics_finite",
            EXECUTION,
            not unexpected,
            f"the study ran {spec.seeds} seeds of {spec.patients} histories over "
            f"{len(arms)} arms; every aggregate metric is finite except "
            f"{undefined or 'none'}, and the only undefined row is the snapshot's between-scan "
            "change, which its point state cannot estimate",
            arms=list(arms),
            undefined=undefined,
        )
    )

    reference = "BM-STF"
    reported_events = sum(
        1 for row in result.decision_rows if row.get("arm") == reference and row.get("event_hit")
    )
    reported_decisions = sum(1 for row in result.decision_rows if row.get("arm") == reference)
    if reported_events == 0:
        checks.append(
            _not_run(
                "event_barrier_resolvable_at_this_scale",
                EXECUTION,
                f"no aneurysm-related event fell inside a recommended interval among the "
                f"{reported_decisions} reported decisions of the full mechanism, so the "
                "alpha_2 = 0.01 gate carries no information here; the manuscript reports 75 "
                "events among 7,617 feasible intervals at 20 seeds of 300 histories",
                reported_decisions=reported_decisions,
            )
        )
    else:
        checks.append(
            _verdict(
                "event_barrier_resolvable_at_this_scale",
                EXECUTION,
                True,
                f"{reported_events} events fell inside a recommended interval among the "
                f"{reported_decisions} reported decisions of the full mechanism, so the "
                "alpha_2 = 0.01 gate was exercised",
                reported_events=reported_events,
                reported_decisions=reported_decisions,
            )
        )

    if reference in aggregate:
        coverage = aggregate[reference]["coverage_diameter"]
        near = abs(coverage - 0.90) <= 0.15
        checks.append(
            _verdict(
                "twelve_month_diameter_coverage_near_nominal",
                EXECUTION,
                near,
                f"the 12-month diameter coverage is {coverage:.3f} against the nominal 0.90 at "
                f"the executed scale ({spec.seeds} seeds of {spec.patients})",
                coverage=coverage,
            )
        )
        auroc = aggregate[reference]["auroc_crossing"]
        checks.append(
            _verdict(
                "crossing_discrimination_below_ceiling",
                EXECUTION,
                0.55 < auroc < 0.99,
                f"the 12-month crossing AUROC is {auroc:.3f}, inside (0.55, 0.99) so the "
                "discrimination check is not vacuous",
                auroc=auroc,
            )
        )

    if "snapshot" in aggregate and reference in aggregate:
        gap = aggregate[reference]["coverage_burden"] - aggregate["snapshot"]["coverage_burden"]
        checks.append(
            _verdict(
                "filtered_burden_coverage_exceeds_snapshot",
                EXECUTION,
                gap > 0.0,
                f"the filtered forecast covers the true low-shear burden {gap:.3f} more often than "
                "the single-scan snapshot at the executed scale",
                gap=gap,
            )
        )
        crps_gap = aggregate[reference]["crps_diameter"] - aggregate["snapshot"]["crps_diameter"]
        checks.append(
            _verdict(
                "filtered_crps_better_than_snapshot",
                EXECUTION,
                crps_gap < 0.0,
                f"the 12-month diameter CRPS is {-crps_gap:.3f} lower for the filtered forecast "
                "than for the snapshot (lower is better)",
                gap=crps_gap,
            )
        )

    if "scalar_noise" in aggregate and reference in aggregate:
        conditional = (
            aggregate[reference]["coverage_burden_high_noise"]
            - aggregate["scalar_noise"]["coverage_burden_high_noise"]
        )
        unconditional = (
            aggregate[reference]["coverage_burden"] - aggregate["scalar_noise"]["coverage_burden"]
        )
        paired = np.asarray(
            [
                (
                    block.metrics[reference]["coverage_burden_high_noise"]
                    - block.metrics["scalar_noise"]["coverage_burden_high_noise"]
                )
                - (
                    block.metrics[reference]["coverage_burden"]
                    - block.metrics["scalar_noise"]["coverage_burden"]
                )
                for block in result.seeds
                if reference in block.metrics and "scalar_noise" in block.metrics
            ],
            dtype=np.float64,
        )
        finite_paired = paired[np.isfinite(paired)]
        bounds: tuple[float, float] | None = None
        if finite_paired.size > 1:
            summary = percentile_interval(finite_paired, resamples=2000)
            bounds = (float(summary.lower), float(summary.upper))
        # The manuscript reports the loss of the scalar-noise ablation concentrating on the
        # high-noise stratum (0.044 against 0.011). Whether it does is a property of the
        # generator, which the manuscript does not print, so the ordering is only settled
        # when the seed-paired interval leaves zero on one side; otherwise the scale is
        # reported as unable to decide rather than as a defect.
        if bounds is None:
            status, reason = "NOT_RUN", "fewer than two seeds carry both arms"
        elif bounds[0] > 0.0:
            status, reason = "PASS", "the paired interval stays above zero"
        elif bounds[1] < 0.0:
            status, reason = "FAIL", "the paired interval stays below zero"
        else:
            status, reason = (
                "NOT_RUN",
                "the paired interval of the difference of differences spans zero, so this "
                "scale cannot separate the two channels",
            )
        checks.append(
            CheckResult(
                name="decomposed_covariance_helps_on_high_noise_scans",
                family=EXECUTION,
                status=status,
                evidence=(
                    f"the decomposed covariance buys {conditional:+.3f} of high-noise-stratum "
                    f"burden coverage against {unconditional:+.3f} unconditional, a difference of "
                    f"differences of {conditional - unconditional:+.3f}"
                    + (
                        f" with a seed-paired 95 % interval [{bounds[0]:+.3f}, {bounds[1]:+.3f}]"
                        if bounds is not None
                        else ""
                    )
                    + f"; {reason}"
                ),
                detail={
                    "conditional": float(conditional),
                    "unconditional": float(unconditional),
                    "per_seed_pairs": [float(value) for value in paired],
                    "interval": [bounds[0], bounds[1]] if bounds is not None else None,
                    "manuscript_reports": {"conditional": 0.044, "unconditional": 0.011},
                },
            )
        )

    if "single_regime_mean_drift" in aggregate and reference in aggregate:
        gap = (
            aggregate[reference]["coverage_burden"]
            - aggregate["single_regime_mean_drift"]["coverage_burden"]
        )
        checks.append(
            _verdict(
                "switching_latent_carries_burden_calibration",
                EXECUTION,
                gap > 0.0,
                f"the single-regime filter loses {gap:.3f} of burden coverage, the component the "
                "switching latent is built to carry",
                gap=gap,
            )
        )

    if "diameter_only" in aggregate and reference in aggregate:
        gap = (
            aggregate["diameter_only"]["burden_change_rmse"]
            - aggregate[reference]["burden_change_rmse"]
        )
        checks.append(
            _verdict(
                "hemodynamic_channel_refines_burden_tracking",
                EXECUTION,
                gap > 0.0,
                f"discarding the hemodynamic channels raises the between-scan burden RMSE by "
                f"{gap:.4f}, the one quantity the diameter channels cannot observe",
                gap=gap,
            )
        )

    if "spread_half" in aggregate and "spread_double" in aggregate and reference in aggregate:
        low = aggregate["spread_half"]["coverage_diameter"]
        middle = aggregate[reference]["coverage_diameter"]
        high = aggregate["spread_double"]["coverage_diameter"]
        checks.append(
            _verdict(
                "spread_scaling_moves_coverage_monotonically",
                EXECUTION,
                low < middle < high,
                f"halving the predictive spread covers {low:.3f}, the nominal spread {middle:.3f} "
                f"and doubling {high:.3f}",
                low=low,
                middle=middle,
                high=high,
            )
        )

    if "no_ultrasound" in aggregate and reference in aggregate:
        gap = abs(
            aggregate[reference]["coverage_diameter"]
            - aggregate["no_ultrasound"]["coverage_diameter"]
        )
        checks.append(
            _verdict(
                "ultrasound_channel_is_not_separable_at_this_density",
                EXECUTION,
                gap < 0.10,
                f"discarding the ultrasound channel moves diameter coverage by {gap:.3f}, a null "
                "at the executed observation density",
                gap=gap,
            )
        )

    if reference in aggregate:
        escalate = aggregate[reference]["escalate_fraction"]
        checks.append(
            _verdict(
                "escalate_fraction_in_unit_interval",
                EXECUTION,
                0.0 <= escalate <= 1.0,
                f"the escalate fraction is {escalate:.3f} on {int(spec.patients)} histories per seed",
                escalate=escalate,
            )
        )
        interval = aggregate[reference]["mean_interval_pooled"]
        checks.append(
            _verdict(
                "recommended_interval_on_the_action_grid",
                EXECUTION,
                float(min(action_grid())) <= interval <= float(max(action_grid())),
                f"the mean pooled recommended interval is {interval:.3f} months, inside the "
                f"[3, 36] grid",
                interval=interval,
            )
        )

    system_filter, histories = probe_histories(count=8)
    configuration = _configuration(system_filter)
    history = histories[0]

    prior_mean, prior_covariance = prior_moments(system_filter, configuration, history)

    trace = run_filter_trace(
        system_filter,
        configuration,
        prior_mean,
        prior_covariance,
        history.visit_months,
        history.inputs,
        corrected_visits(history),
    )
    truth_diameter = np.asarray(
        [history.monthly_diameter[int(month)] for month in history.visit_months]
    )
    filtered_error = float(np.mean(np.abs(trace.updated_means[:, 0] - truth_diameter)))
    raw_error = float(
        np.mean(
            np.abs(
                np.asarray([float(visit.measurement[0]) for visit in history.visits])
                - truth_diameter
            )
        )
    )
    checks.append(
        _verdict(
            "filter_reduces_diameter_error_against_raw_readings",
            EXECUTION,
            filtered_error < raw_error,
            f"the filtered diameter error is {filtered_error:.3f} mm against {raw_error:.3f} mm "
            "for the raw readings on the same record",
            filtered=filtered_error,
            raw=raw_error,
        )
    )

    forecast_spec = spec.forecast_spec()
    state = trace.state_at(len(history.visits) - 1)

    forecast = monte_carlo_forecast(
        state,
        system_filter,
        configuration,
        forecast_spec,
        history.inputs[-1],
        history.sex,
        set_seed(3).forecast,
    )
    monotone = bool(np.all(np.diff(forecast.crossing_curve) >= -1.0e-12)) and bool(
        np.all(np.diff(forecast.event_curve) >= -1.0e-12)
    )
    bounded = bool(
        np.all((forecast.crossing_curve >= 0.0) & (forecast.crossing_curve <= 1.0))
        and np.all((forecast.event_curve >= 0.0) & (forecast.event_curve <= 1.0))
    )
    checks.append(
        _verdict(
            "forecast_curves_monotone_and_bounded",
            EXECUTION,
            monotone and bounded and forecast.paths == int(forecast_spec.paths),
            f"the crossing and event curves are non-decreasing and inside [0, 1] over "
            f"{forecast.horizon_months} months from {forecast.paths} paths",
            crossing_final=float(forecast.crossing_curve[-1]),
            event_final=float(forecast.event_curve[-1]),
        )
    )

    gate = strata.gate_month(float(state.collapsed()[0][0]), history.sex, action_grid(), ALPHA2)
    decision = read_out(
        forecast.crossing_curve,
        forecast.event_curve,
        action_grid(),
        forecast.paths,
        gate,
        alpha1=ALPHA1,
        alpha2=ALPHA2,
    )
    on_grid = bool(np.any(np.isclose(action_grid(), decision.horizon_months)))
    consistent = decision.escalate == (not decision.admissible)
    checks.append(
        _verdict(
            "policy_read_out_execution",
            EXECUTION,
            on_grid and consistent,
            f"the read-out returns {decision.horizon_months:.1f} months with escalate="
            f"{decision.escalate} and admissibility {decision.admissible}",
            decision=decision.as_mapping(),
        )
    )

    energy = one_step_energy(histories, 10, set_seed(31).resample, paths=32)
    checks.append(
        _verdict(
            "energy_score_validation_execution",
            EXECUTION,
            math.isfinite(energy),
            f"the one-step validation energy score at the default latent dimension is {energy:.6f} "
            f"over {len(histories)} records",
            energy=energy,
        )
    )

    escalations = list(margins.table3_checks())
    ladder = [sum(1 for check in escalations if check.name.startswith("table3_crossing_increases"))]
    checks.append(
        _verdict(
            "manuscript_arithmetic_checks_executed",
            EXECUTION,
            ladder[0] >= 3,
            f"{len(escalations)} Table 3 arithmetic checks executed ({ladder[0]} monotonicity "
            "contrasts)",
            count=len(escalations),
        )
    )

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        arrays = {"a": np.arange(6.0), "b": np.linspace(-1.0, 1.0, 4)}
        before = parameter_digest(arrays)
        record = save_checkpoint(tmp / "state.npz", arrays, seed=11, stage="verify")
        for key in arrays:
            arrays[key] = np.zeros_like(arrays[key])
        loaded, restored = load_checkpoint(tmp / "state.npz")
        after = parameter_digest(loaded)
        mode = oct((tmp / "state.npz").stat().st_mode)[-3:]
        checks.append(
            _verdict(
                "checkpoint_round_trip_execution",
                EXECUTION,
                before == after
                and restored.seed == 11
                and record.digest == before
                and mode == "644",
                f"the parameter digest {before[:12]} survives a save and load after zeroing; the "
                f"file mode is {mode} and the stored seed is {restored.seed}",
                digest=before,
                mode=mode,
            )
        )

    rng = set_seed(41).resample
    replay_visits = [
        ReplayVisit(
            record=f"R{i:03d}",
            index=2,
            month=float(6 + 3 * i),
            horizon=float(action_grid()[i % action_grid().shape[0]]),
            escalate=bool(i % 4 == 0),
            crossing_hit=bool(i % 7 == 0),
            event_hit=bool(i % 11 == 0),
            comparator_months=12.0,
            band=("40-45" if i % 2 else "45-55"),
            sex=("female" if i % 3 == 0 else "male"),
            repair_month=(float(20 + i) if i % 5 == 0 else None),
        )
        for i in range(40)
    ]
    intensity = fit_intensity(np.asarray([8.6, 9.1, 7.4, 12.0, 6.2, 8.8, 10.4]))
    censoring = fit_censoring(np.asarray([26.0, 31.0, 22.0, 35.0, 28.0]))
    estimates = estimate(
        replay_visits, intensity, censoring, deferral_cap=0.2, resamples=200, rng=rng
    )
    payload = estimates.as_mapping()
    checks.append(
        _verdict(
            "ipw_replay_execution",
            EXECUTION,
            math.isfinite(estimates.crossing_rate.estimate)
            and math.isfinite(estimates.imaging_burden)
            and 0.0 <= estimates.escalate_fraction <= 1.0,
            f"the weighted replay returns a crossing rate of "
            f"{estimates.crossing_rate.estimate:.4f} with interval "
            f"[{estimates.crossing_rate.lower:.4f}, {estimates.crossing_rate.upper:.4f}], an "
            f"imaging burden of {estimates.imaging_burden:.3f} months and an escalate fraction of "
            f"{estimates.escalate_fraction:.3f}",
            estimates=str(payload),
        )
    )

    weights = np.asarray([intensity.stabilised_weight(visit.month) for visit in replay_visits])
    checks.append(
        _verdict(
            "visit_intensity_weights_positive",
            EXECUTION,
            bool(np.all(weights > 0.0)),
            f"all {weights.shape[0]} stabilised inverse-intensity weights are positive "
            f"(mean {float(np.mean(weights)):.4f})",
            mean_weight=float(np.mean(weights)),
        )
    )

    curves = [censoring.probability(float(month)) for month in np.arange(0.0, 40.0)]
    monotone_censoring = all(later <= earlier + 1.0e-12 for earlier, later in pairwise(curves))
    checks.append(
        _verdict(
            "censoring_survival_non_increasing",
            EXECUTION,
            monotone_censoring,
            "the fitted censoring survival curve is non-increasing over 40 months",
        )
    )

    calibration = calibrate_band_policy(
        np.asarray(["40-45", "45-55"] * 20),
        np.asarray([[i % 5 == 0 for _ in range(len(action_grid()))] for i in range(40)]),
        action_grid(),
        alpha1=ALPHA1,
    )
    # The Wilson upper bound of zero successes is itself about z^2 / n, so a band with no
    # crossing at all is still inadmissible at alpha_1 = 0.10 until n is a few hundred.
    quiet_dense = calibrate_band_policy(
        np.asarray(["40-45", "45-55"] * 200),
        np.zeros((400, len(action_grid())), dtype=bool),
        action_grid(),
        alpha1=ALPHA1,
    )
    quiet_sparse = calibrate_band_policy(
        np.asarray(["40-45", "45-55"] * 20),
        np.zeros((40, len(action_grid())), dtype=bool),
        action_grid(),
        alpha1=ALPHA1,
    )
    literal = literal_band_policy()
    longest = max(quiet_dense.schedule.values()) if quiet_dense.schedule else float("nan")
    absent = sorted(set(BAND_ORDER) - {"40-45", "45-55"})
    sparse_bound = wilson_upper(0.0, 20, BOUND_LEVEL)
    checks.append(
        _verdict(
            "band_policies_execution",
            EXECUTION,
            len(literal.schedule) == 4
            and len(calibration.escalated) == len(BAND_ORDER)
            and set(quiet_dense.escalated) == set(absent)
            and set(quiet_dense.schedule) == {"40-45", "45-55"}
            and longest == float(max(action_grid()))
            and len(quiet_sparse.escalated) == len(BAND_ORDER),
            f"on a fixture crossing at one horizon in five the calibrated policy escalates "
            f"{len(calibration.escalated)} of {len(BAND_ORDER)} bands; with no crossings and 200 "
            f"records per band it schedules {longest:.0f} months, the longest grid horizon, for "
            f"the two bands it has records for and escalates the two it does not; with no "
            f"crossings and only 20 records per band the Wilson upper bound of zero successes is "
            f"{sparse_bound:.4f}, still above alpha_1 = {ALPHA1}, so all "
            f"{len(quiet_sparse.escalated)} escalate; the literal schedule covers "
            f"{len(literal.schedule)} bands and never escalates",
            calibrated_schedule=calibration.as_mapping(),
            quiet_dense_schedule=quiet_dense.as_mapping(),
            quiet_dense_escalated=list(quiet_dense.escalated),
            quiet_sparse_escalated=list(quiet_sparse.escalated),
            wilson_upper_at_zero_of_twenty=sparse_bound,
            bands_without_records=absent,
        )
    )

    point = point_crossing_probability(
        np.asarray([0.0, 6.0, 12.0]),
        np.asarray([44.0, 45.5, 47.0]),
        12.0,
        repair_threshold_mm("male"),
    )
    checks.append(
        _verdict(
            "band_extrapolation_is_binary",
            EXECUTION,
            point in (0.0, 1.0),
            f"the point extrapolation returns {point}, the binary forecast the band comparator "
            "reads out",
            point=point,
        )
    )

    return checks, context


def agreement_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []
    rng = np.random.default_rng(47)
    first = rng.integers(0, 2, size=300)
    second = np.where(rng.random(300) < 0.12, 1 - first, first)
    kappa = cohen_kappa(first, second)
    from sklearn.metrics import cohen_kappa_score

    reference = float(cohen_kappa_score(first, second))
    gap = abs(kappa.kappa - reference)
    checks.append(
        _verdict(
            "cohen_kappa_matches_sklearn",
            CLOSED_FORM,
            gap < 1.0e-12,
            f"kappa {kappa.kappa:.12f} matches scikit-learn {reference:.12f}",
            gap=gap,
        )
    )

    subjects = 35
    latent = rng.normal(size=subjects)
    matrix = np.stack([latent + rng.normal(scale=0.35, size=subjects) for _ in range(3)], axis=1)
    observed = icc_two_one(matrix)
    assembly = mean_squares(matrix)
    raters = int(assembly["raters"])
    denominator = (
        assembly["between_rows"]
        + (raters - 1) * assembly["residual"]
        + raters * (assembly["between_columns"] - assembly["residual"]) / int(assembly["subjects"])
    )
    brute = (assembly["between_rows"] - assembly["residual"]) / denominator
    icc_gap = abs(observed - brute)
    checks.append(
        _verdict(
            "icc_two_one_matches_mean_square_assembly",
            CLOSED_FORM,
            icc_gap < 1.0e-12,
            f"ICC(2,1) {observed:.12f} matches a mean-square assembly written from the definition",
            gap=icc_gap,
        )
    )

    values = rng.normal(size=60)
    interval = cluster_bootstrap(values, np.repeat(np.arange(20), 3), resamples=300, rng=rng)
    checks.append(
        _verdict(
            "cluster_bootstrap_shapes_an_interval",
            CLOSED_FORM,
            interval.lower <= interval.estimate <= interval.upper,
            f"the record-level bootstrap brackets the point estimate "
            f"({interval.lower:.4f} <= {interval.estimate:.4f} <= {interval.upper:.4f})",
            interval=interval.as_mapping(),
        )
    )
    return checks


def manuscript_check_results() -> list[CheckResult]:
    results: list[CheckResult] = []
    for entry in margins.all_manuscript_checks():
        detail = dict(entry.as_mapping())
        consistent = bool(entry.passed())
        results.append(
            CheckResult(
                name=str(entry.name),
                family=MANUSCRIPT,
                status="PASS" if consistent else "FAIL",
                evidence=(
                    f"{entry.anchor}: observed {entry.observed:.6f} against reported "
                    f"{entry.expected:.6f} (tolerance {entry.tolerance:g})"
                ),
                detail=detail,
            )
        )
    return results


CONFIG_SECTIONS: tuple[str, ...] = (
    "study",
    "cohort",
    "marginalisation",
    "system",
    "inference",
    "selection",
    "policy",
    "diagnostic",
    "outputs",
    "experiment",
    "environment",
    "verification",
)


def _config_leaves(node: Any, prefix: tuple[str, ...] = ()) -> list[str]:
    if isinstance(node, dict):
        collected: list[str] = []
        for key, value in node.items():
            collected.extend(_config_leaves(value, (*prefix, str(key))))
        return collected
    if isinstance(node, list):
        return []
    return [".".join(prefix)]


def configuration_check(root: Path) -> CheckResult:
    """Every declared config key must be read by the code or not be declared.

    Ref: the protocol's configuration section (the reported batch, epoch and tolerance
    settings are configuration, so a key nothing reads is a stale record that will
    silently disagree with the code).
    """

    import yaml

    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((root / "src").rglob("*.py"))
    )
    declared: dict[str, str] = {}
    for path in sorted((root / "configs").rglob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        for section in CONFIG_SECTIONS:
            if section in document:
                for dotted in _config_leaves(document[section], (section,)):
                    declared.setdefault(dotted, path.name)
    unread: list[str] = []
    for dotted, source_name in sorted(declared.items()):
        leaf = dotted.rsplit(".", 1)[-1]
        markers = (f'"{dotted}"', f"'{dotted}'", f'"{leaf}"', f"'{leaf}'")
        if any(marker in sources for marker in markers):
            continue
        unread.append(f"{source_name}:{dotted}")
    return _verdict(
        "configuration_leaves_are_consumed",
        INTEGRITY,
        not unread,
        f"{len(declared)} declared config keys across configs/, and every one of them is "
        f"read by src/" + (f"; unread: {unread}" if unread else ""),
        declared=len(declared),
        unread=unread,
    )


def integrity_checks(root: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.append(configuration_check(root))
    missing = [
        symbol for symbol in code_symbols() if not (root / symbol.split("::", 1)[0]).exists()
    ]
    checks.append(
        _verdict(
            "claim_symbol_resolution",
            INTEGRITY,
            not missing,
            f"{len(code_symbols())} claim symbols across {len(CLAIMS)} claims resolve to files on disk"
            + (f"; missing {missing}" if missing else ""),
            missing=missing,
        )
    )
    if not (root / "integrity_manifest.json").exists():
        checks.append(
            _not_run(
                "integrity_manifest_consistency",
                INTEGRITY,
                "the manifest has not been written yet on this pass",
            )
        )
        return checks
    report = verify_manifest(root)
    checks.append(
        _verdict(
            "integrity_manifest_consistency",
            INTEGRITY,
            bool(report["consistent"]),
            f"{report['recorded_files']} tracked files; missing {len(report['missing'])}, "
            f"mismatched {len(report['mismatched'])}, untracked {len(report['untracked'])}",
            **report,
        )
    )
    return checks


def _tool_environment() -> dict[str, str]:
    import os

    environment = dict(os.environ)
    environment.update({"COLUMNS": "200", "LINES": "60", "NO_COLOR": "1", "PY_COLORS": "0"})
    return environment


def tool_check(
    name: str,
    command: Sequence[str],
    root: Path,
    timeout: int = 1800,
    label: str | None = None,
) -> CheckResult:
    """Run a formatter, linter or type check and record a stable verdict.

    The evidence of a passing check is a sentence this release owns, naming the tool by a
    label the caller supplies. Two things must not be quoted there: a tool's own count of
    what it skipped, because that depends on which cache directories happen to exist (an
    import-order check counts `.git` and `.mypy_cache` among them), and the path of the
    executable, because that is a host input. Either would make the published bytes depend on
    the machine and on whether the reader ran pytest first. The tool's own output goes to the
    log; a failing check does quote it, because a failure means the run is not shipping.
    """

    tool = label if label is not None else Path(str(command[0])).name
    if shutil.which(str(command[0])) is None:
        return _blocked(name, TOOL, f"{tool} is not installed on this host")
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_tool_environment(),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as error:
        return _blocked(name, TOOL, f"{type(error).__name__}: {error}")
    output = (completed.stdout + completed.stderr).strip()
    get("verify").info("%s output: %s", name, output.replace("\n", " | ")[-400:])
    if completed.returncode == 0:
        return _verdict(
            name,
            TOOL,
            True,
            f"{tool} ran over the tree and reported nothing to change",
            exit_status=0,
        )
    lines = output.splitlines()
    evidence = (
        lines[-1].split(" in ")[0].strip() if lines else f"exit status {completed.returncode}"
    )
    return _verdict(name, TOOL, False, evidence, exit_status=completed.returncode)


def docker_check() -> CheckResult:
    if shutil.which("docker") is None:
        return _blocked("docker_build", TOOL, "no container runtime is available on this host")
    return _blocked(
        "docker_build", TOOL, "a container runtime is present but the image was not built"
    )


def dataset_checks(timeout: float = 8.0, live: bool = False) -> list[CheckResult]:
    """The resource audit. Reachability is opt-in so a shipped report is a function of the tree.

    Ref: Data availability. A live probe writes a host-dependent HTTP status into the report,
    and the integrity manifest digests the report, so a default run that reached out would
    make the published bytes unreproducible whenever a host answered intermittently. The
    default run therefore records the audit as outstanding and `--live` performs it.
    """

    if not live:
        return [
            _not_run(
                "public_dataset_links",
                DATA,
                "the resource landing pages were fetched when dataset_urls.txt was assembled, "
                "and this run did not re-probe them: reachability is opt-in with --live so the "
                "shipped report stays a function of the repository and not of the network",
            ),
            _not_run(
                "surrogate_substrate_fields",
                DATA,
                "the regenerated AAA-100 / AAA-L solver fields behind the Table 5 panel B parity "
                "gate have no public deposit, so the surrogate error cannot be recomputed here "
                "and the release carries the closed-form substrate of sensor/surrogate.py instead",
            ),
            _not_run(
                "retrospective_cohort_endpoints",
                DATA,
                "the retrospective supporting cohort and the silent-mode arm are de-identified "
                "surveillance records held by Sites A, B and C under their own agreements, so no "
                "cohort-level endpoint was recomputed",
            ),
            _blocked(
                "frozen_ct_foundation_embedding",
                DATA,
                "the frozen abdominal-CT foundation-model embedding is reached through a gated "
                "model card, so the image-context comparison of the hazard head is blocked at the "
                "weight level and the release exercises the hazard head on the audit strata alone",
            ),
        ]

    answered: dict[str, int] = {}
    silent: dict[str, str] = {}
    for entry in PUBLIC_DATASETS:
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "bmstf-verification"})
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                status = int(response.status)
                if 200 <= status < 300:
                    answered[entry["url"]] = status
                else:
                    silent[entry["url"]] = f"HTTP {status}"
        except urllib.error.HTTPError as error:
            silent[entry["url"]] = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            silent[entry["url"]] = f"{type(error).__name__}: {error}"
    checks: list[CheckResult] = []
    if silent:
        checks.append(
            _not_run(
                "public_dataset_links",
                DATA,
                f"{len(answered)} of {len(PUBLIC_DATASETS)} resource landing pages answered from "
                f"this host with a success status; the rest did not complete a request here. "
                f"The per-link outcome is recorded as it stands, without treating a transport "
                f"failure of this host as evidence that the resource is retired: {silent}",
                answered=answered,
                silent=silent,
            )
        )
    else:
        checks.append(
            _verdict(
                "public_dataset_links",
                DATA,
                True,
                f"all {len(PUBLIC_DATASETS)} resource landing pages answered with a success "
                f"status: {sorted(answered.items())}",
                answered=answered,
            )
        )
    checks.append(
        _not_run(
            "surrogate_substrate_fields",
            DATA,
            "the regenerated AAA-100 / AAA-L solver fields behind the Table 5 panel B parity gate "
            "have no public deposit, so the surrogate error cannot be recomputed here and the "
            "release carries the closed-form substrate of sensor/surrogate.py instead",
        )
    )
    checks.append(
        _not_run(
            "retrospective_cohort_endpoints",
            DATA,
            "the retrospective supporting cohort and the silent-mode arm are de-identified "
            "surveillance records held by Sites A, B and C under their own agreements, so no "
            "cohort-level endpoint was recomputed",
        )
    )
    checks.append(
        _blocked(
            "frozen_ct_foundation_embedding",
            DATA,
            "the frozen abdominal-CT foundation-model embedding is reached through a gated model "
            "card, so the image-context comparison of the hazard head is blocked at the weight "
            "level and the release exercises the hazard head on the audit strata alone",
        )
    )
    return checks


def reported_scale_check() -> CheckResult:
    limits = ReportedLimits()
    return _verdict(
        "reported_script_digests_recorded",
        MANUSCRIPT,
        not limits.mismatched_hashes(),
        f"the two SHA-256 digests in Code availability are transcribed with the reported "
        f"analysis environment (Python {limits.reported_python}, NumPy {limits.reported_numpy}) "
        f"and the reported run command {limits.events_feasible} events among "
        f"{limits.feasible_intervals} feasible intervals",
        mismatched=limits.mismatched_hashes(),
    )


def configuration_summary(spec: StudySpec) -> dict[str, Any]:
    return {
        "executed_scale": spec.as_mapping(),
        "reported_scale": {
            "seeds": 20,
            "patients": 300,
            "calibration_patients": 6000,
            "horizon_months": 36,
        },
    }
