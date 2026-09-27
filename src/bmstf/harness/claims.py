"""Paper-claim to code map and the recorded deviations.

Ref: every entry names the paper location that fixes it: the abstract, Eq. (1)-(6),
Algorithm 1-3 and Tables 1-5 of the manuscript.
"""

from __future__ import annotations

from typing import Any

CLAIMS: tuple[dict[str, Any], ...] = (
    {
        "id": "C1",
        "module": "process.regimes, sensor.marginalise",
        "paper_location": "Eq. (2)",
        "statement": (
            "The scan observation is y_k = H z_k + nu_i + eps_k with the marginalised "
            "per-scan noise eps_k ~ N(0, Sigma_BC,k + Sigma_rec,k) and the patient-persistent "
            "term nu_i ~ N(0, Sigma_sur,i)."
        ),
        "symbols": [
            "src/bmstf/process/regimes.py::build_system",
            "src/bmstf/process/regimes.py::ShearSystem.augment",
            "src/bmstf/sensor/marginalise.py::marginalise",
        ],
    },
    {
        "id": "C2",
        "module": "process.regimes, process.transition",
        "paper_location": "Eq. (3), Algorithm 2 line 4",
        "statement": (
            "The regime indicator is a continuous-time two-state Markov chain with generator L "
            "built from lambda_qa and lambda_aq, so the regime transition over Delta is "
            "Pi(Delta) = exp(Delta L)."
        ),
        "symbols": [
            "src/bmstf/process/regimes.py::RegimeChain.generator_matrix",
            "src/bmstf/process/transition.py::regime_transition",
        ],
    },
    {
        "id": "C3",
        "module": "process.transition",
        "paper_location": "Eq. (3) and the sentence that follows it",
        "statement": (
            "Irregular scan times enter exactly: Phi_r(Delta) = exp(A_r Delta), "
            "Gamma_r(Delta) = integral Phi_r(s) ds B_r and "
            "Q_r(Delta) = integral Phi_r(s) Q_r Phi_r(s)^T ds."
        ),
        "symbols": [
            "src/bmstf/process/transition.py::discretise",
            "src/bmstf/process/transition.py::reference_covariance",
            "src/bmstf/process/transition.py::reference_input_gain",
        ],
    },
    {
        "id": "C4",
        "module": "inference.mixing, inference.forecast",
        "paper_location": "Eq. (4)",
        "statement": (
            "The h-step predictive law is the two-component mixture "
            "p(z_K+h | H_K) = sum_r varpi_K^(r)(h) N(mu_K+h^(r), P_K+h^(r)) whose weights follow "
            "the regime chain and whose covariances grow with the horizon."
        ),
        "symbols": [
            "src/bmstf/inference/mixing.py::moment_match",
            "src/bmstf/inference/forecast.py::monte_carlo_forecast",
        ],
    },
    {
        "id": "C5",
        "module": "inference.mixing, inference.imm",
        "paper_location": "Algorithm 2 lines 3-15; Sec. two-regime model",
        "statement": (
            "Conditional on a regime path the filter is exact; across paths the posterior is "
            "collapsed at each observation to two Gaussian components by an "
            "interacting-multiple-model moment-matching step."
        ),
        "symbols": [
            "src/bmstf/inference/mixing.py::mix_components",
            "src/bmstf/inference/imm.py::imm_step",
            "src/bmstf/inference/imm.py::run_filter_trace",
        ],
    },
    {
        "id": "C6",
        "module": "sensor.quasimc, sensor.marginalise",
        "paper_location": "Algorithm 1 lines 1-6",
        "statement": (
            "N_b = 64 boundary conditions are drawn from pi(b | c_k, s) by scrambled-Sobol "
            "quasi-Monte-Carlo and Sigma_BC,k is the covariance of the returned fields, "
            "Varb[E(y | g, b)]."
        ),
        "symbols": [
            "src/bmstf/sensor/quasimc.py::draw_boundary_conditions",
            "src/bmstf/sensor/quasimc.py::sobol_uniform",
            "src/bmstf/sensor/marginalise.py::marginalise",
            "src/bmstf/sensor/marginalise.py::enumerate_boundary_budget",
        ],
    },
    {
        "id": "C7",
        "module": "geometry.perturbations, sensor.marginalise",
        "paper_location": "Algorithm 1 lines 7-10",
        "statement": (
            "N_m = 8 mask perturbations give the reconstruction budget Sigma_rec,k as the "
            "covariance of the fields returned at the prior-mean boundary condition."
        ),
        "symbols": [
            "src/bmstf/geometry/perturbations.py::mask_perturbations",
            "src/bmstf/geometry/perturbations.py::smooth_radial_field",
        ],
    },
    {
        "id": "C8",
        "module": "sensor.covariance, geometry.perturbations",
        "paper_location": "Algorithm 1 lines 11-14",
        "statement": (
            "The patient-persistent residual covariance is a lookup by topology class, and a "
            "failed topology gate inflates the boundary-condition budget by kappa_phi and sets "
            "the flag."
        ),
        "symbols": [
            "src/bmstf/sensor/covariance.py::ResidualLookup",
            "src/bmstf/sensor/covariance.py::inflate",
            "src/bmstf/geometry/perturbations.py::topology_flag",
        ],
    },
    {
        "id": "C9",
        "module": "inference.kalman",
        "paper_location": "Sec. two-regime model (the Kalman gain)",
        "statement": (
            "Within each component the Kalman gain is K_k = P H~^T (H~ P H~^T + Sigma_BC,k + "
            "Sigma_rec,k)^{-1}; with the decomposed covariance the filter down-weights exactly "
            "the scans whose low-shear area swings with the inlet."
        ),
        "symbols": [
            "src/bmstf/inference/kalman.py::kalman_update",
            "src/bmstf/inference/kalman.py::standard_form_covariance",
        ],
    },
    {
        "id": "C10",
        "module": "process.hazards",
        "paper_location": "Eq. (1); Sec. cohort delineation (published diameter-by-sex strata)",
        "statement": (
            "The event barrier is a stratified eligibility gate: h_gate(H_K) is the largest grid "
            "horizon with Lambda_strat(D_K, sex)(h) <= alpha_2."
        ),
        "symbols": [
            "src/bmstf/process/hazards.py::StratumRisk.gate_month",
            "src/bmstf/process/hazards.py::StratumRisk.cumulative_event_probability",
        ],
    },
    {
        "id": "C11",
        "module": "policy.barriers, policy.wilson",
        "paper_location": "Eq. (1), Eq. (5), Algorithm 2 line 21",
        "statement": (
            "tau* = min{max{h : U_z(F^1(h), M_mc) <= alpha_1}, h_gate}; the event barrier is "
            "checked on F^2(h) + z_W se(F^2(h)); an empty admissible set returns the escalate "
            "state scored at the three-month floor."
        ),
        "symbols": [
            "src/bmstf/policy/barriers.py::read_out",
            "src/bmstf/policy/barriers.py::admissible_horizons",
            "src/bmstf/policy/wilson.py::wilson_upper",
        ],
    },
    {
        "id": "C12",
        "module": "inference.forecast",
        "paper_location": "Algorithm 2 lines 16-21",
        "statement": (
            "The horizon forecast draws a regime path and a state path, accumulates "
            "c1_n(h) = 1{D crosses by h} and the Rao-Blackwellised event law "
            "c2_n(h) = 1 - prod_{s<=h}(1 - chi(D_s, B_s)) frozen once the path has crossed, and "
            "reads F^1 and F^2 as path means."
        ),
        "symbols": [
            "src/bmstf/inference/forecast.py::monte_carlo_forecast",
            "src/bmstf/process/hazards.py::cumulative_hazard",
        ],
    },
    {
        "id": "C13",
        "module": "process.regimes, process.generator",
        "paper_location": "Sec. two-regime model (the patient-level growth offset)",
        "statement": (
            "A patient-level growth offset gamma_i is carried as a constant state coordinate with "
            "a prior variance set from the between-patient spread of published growth rates."
        ),
        "symbols": [
            "src/bmstf/process/regimes.py::GROWTH_SPREAD_MM_PER_MONTH",
            "src/bmstf/process/regimes.py::ShearSystem",
            "src/bmstf/process/cohort.py::build_history",
        ],
    },
    {
        "id": "C14",
        "module": "replay.estimator, replay.intensity, replay.censoring",
        "paper_location": "Algorithm 3 lines 1-17",
        "statement": (
            "The retrospective policy replay weights every visit by the inverse fitted visit "
            "intensity, treats elective repair as a competing risk with a "
            "cause-specific censoring correction, caps deferral at rho_max, and reports "
            "record-level percentile-bootstrap intervals."
        ),
        "symbols": [
            "src/bmstf/replay/estimator.py::estimate",
            "src/bmstf/replay/intensity.py::fit_intensity",
            "src/bmstf/replay/censoring.py::fit_censoring",
            "src/bmstf/replay/censoring.py::censored_inside_window",
        ],
    },
    {
        "id": "C15",
        "module": "scoring.calibration, scoring.discrimination, scoring.crps",
        "paper_location": "Table 1 panel A; Table 2 panel A",
        "statement": (
            "The read-outs are 90 % coverage of the diameter and of the low-shear burden, the "
            "expected calibration error of the 12-month crossing probability, the crossing "
            "AUROC, the continuous ranked probability score of the 12-month diameter forecast "
            "and the root-mean-square error of the between-scan change in burden."
        ),
        "symbols": [
            "src/bmstf/scoring/calibration.py::expected_calibration_error",
            "src/bmstf/scoring/discrimination.py::roc_auc",
            "src/bmstf/scoring/crps.py::crps_sample",
            "src/bmstf/scoring/crps.py::root_mean_square_error",
            "src/bmstf/study/engines.py::run_arm",
        ],
    },
    {
        "id": "C16",
        "module": "stats.bootstrap, stats.multiplicity, stats.signflip",
        "paper_location": "Table 1 caption",
        "statement": (
            "Every value is a mean over seeds with a 95 % percentile-bootstrap interval; delta "
            "rows are seed-paired with the exact sign-flip p value Holm-adjusted within each "
            "metric across the eleven comparators of Tables 1 and 2."
        ),
        "symbols": [
            "src/bmstf/stats/bootstrap.py::percentile_interval",
            "src/bmstf/stats/multiplicity.py::holm_bonferroni",
            "src/bmstf/stats/signflip.py::sign_flip_p_value",
            "src/bmstf/study/contrasts.py::contrast_family",
        ],
    },
    {
        "id": "C17",
        "module": "study.engines",
        "paper_location": "Table 2, caption",
        "statement": (
            "Each ablation removes or distorts one element: the decomposed covariance replaced by "
            "a tuned scalar noise with the bias coordinates dropped, the ultrasound channel "
            "discarded, the switching latent replaced by a single regime with the population or "
            "the quiescent drift, the event barrier dropped, the hemodynamic channels discarded, "
            "and the predictive spread halved or doubled."
        ),
        "symbols": [
            "src/bmstf/study/engines.py::build_arm",
            "src/bmstf/inference/imm.py::single_regime_modes",
            "src/bmstf/study/engines.py::tune_scalar_noise",
        ],
    },
    {
        "id": "C18",
        "module": "policy.bands",
        "paper_location": "Table 1 panel A; Table 3 caption",
        "statement": (
            "Two band comparators share one point extrapolation of the diameter: the literal "
            "guideline schedule and the calibrated bands, re-calibrated at each tolerance on a "
            "6,000-history cohort, whose escalate state is a band in which no interval on the "
            "grid was admissible at calibration."
        ),
        "symbols": [
            "src/bmstf/policy/bands.py::literal_band_policy",
            "src/bmstf/policy/bands.py::calibrate_band_policy",
            "src/bmstf/policy/bands.py::point_crossing_probability",
            "src/bmstf/study/histories.py::calibrate_bands",
        ],
    },
    {
        "id": "C19",
        "module": "study.histories",
        "paper_location": "Table 1 caption; Abstract",
        "statement": (
            "The in-silico study runs 20 seeds of 300 histories with the filters given "
            "the process's own parameters, and the filtered forecast is contrasted with the "
            "single-scan snapshot and the two band policies."
        ),
        "symbols": [
            "src/bmstf/study/histories.py::run_study",
            "src/bmstf/study/histories.py::StudySpec",
            "src/bmstf/process/cohort.py::patient_batch",
        ],
    },
    {
        "id": "C20",
        "module": "scoring.energy, study.selection",
        "paper_location": "Sec. pre-specified clinical protocol",
        "statement": (
            "The latent dimension d is selected between 8 and 16 by the energy score on the "
            "validation folds; the score is the proper score minimised against the next "
            "observation."
        ),
        "symbols": [
            "src/bmstf/scoring/energy.py::select_latent_dimension",
            "src/bmstf/scoring/energy.py::energy_score_samples",
            "src/bmstf/study/selection.py::one_step_energy",
        ],
    },
    {
        "id": "C21",
        "module": "protocol.criteria",
        "paper_location": "Clinical-protocol criteria; Table 5 caption",
        "statement": (
            "The frozen criteria are parity with the equal-budget serial-input fusion at an "
            "equivalence margin of +/-0.05 on the 90 % interval and a descriptive +/-0.02 band, a "
            "mechanism margin of at least 0.03 on the audit stratum with a one-sided 95 % lower "
            "bound above zero, a secondary margin of at least 0.04 over the single-scan fusion, a "
            "calibration slope of 0.9-1.1, categorical net reclassification with an interval "
            "excluding zero, and substrate error within 10 % of the re-trained comparators."
        ),
        "symbols": [
            "src/bmstf/protocol/criteria.py::ProtocolCriteria",
            "src/bmstf/protocol/criteria.py::EQUIVALENCE_MARGIN",
            "src/bmstf/protocol/criteria.py::MECHANISM_MARGIN",
            "src/bmstf/protocol/criteria.py::SECONDARY_MARGIN",
        ],
    },
    {
        "id": "C22",
        "module": "protocol.cohort, protocol.margins",
        "paper_location": "Table 4",
        "statement": (
            "The retrospective supporting cohort and the silent-mode arm are delineated by the "
            "recorded screened and excluded counts, the retained strata, the examination counts "
            "and the 24-month outcome counts."
        ),
        "symbols": [
            "src/bmstf/protocol/cohort.py::TABLE4",
            "src/bmstf/protocol/cohort.py::pooled_sum",
            "src/bmstf/protocol/margins.py::table4_checks",
        ],
    },
    {
        "id": "C23",
        "module": "protocol.margins, protocol.strata",
        "paper_location": "Eq. (6); Table 5 panel A",
        "statement": (
            "The audit-stratum margin of at least 0.03 over the geometry-only trajectory, and in "
            "fixed sequence over the constant-g and permuted-g nulls, is the primary mechanism "
            "criterion and is checked against the reported design rows."
        ),
        "symbols": [
            "src/bmstf/protocol/margins.py::table5_checks",
            "src/bmstf/protocol/strata.py::TABLE5_AUDIT",
        ],
    },
    {
        "id": "C24",
        "module": "protocol.strata",
        "paper_location": "Table 5 panel B",
        "statement": (
            "The boundary-condition-conditioned physics-attention substrate is reported at parity "
            "with locally re-trained equivariant comparators on the regenerated CFD, within 10 %."
        ),
        "symbols": [
            "src/bmstf/protocol/strata.py::TABLE5_SUBSTRATE_NMAE",
            "src/bmstf/sensor/surrogate.py::PhysicsAttentionSurrogate",
        ],
    },
    {
        "id": "C25",
        "module": "protocol.criteria",
        "paper_location": "Sec. here we make the identification explicit",
        "statement": (
            "A frozen abdominal-CT foundation-model embedding is image context for the two-barrier "
            "hazard head, the part of the aneurysm the mesh cannot see."
        ),
        "symbols": [
            "src/bmstf/protocol/criteria.py::ReportedLimits",
            "src/bmstf/protocol/criteria.py::TURNAROUND_MINUTES_PER_SCAN",
        ],
    },
    {
        "id": "C26",
        "module": "process.smooth",
        "paper_location": (
            "Sec. two-regime model: conditional on a regime path the Kalman filter and smoother "
            "are exact"
        ),
        "statement": (
            "Conditional on a regime path the Rauch-Tung-Striebel recursion is exact, so on a "
            "single linear-Gaussian path the smoothed means equal E[z | y] read off the joint "
            "Gaussian of the chain."
        ),
        "symbols": [
            "src/bmstf/process/smooth.py::rts_smoother",
            "src/bmstf/process/smooth.py::joint_gaussian_covariance",
            "src/bmstf/process/smooth.py::conditional_means",
        ],
    },
)

DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "topic": "generator parameterisation",
        "paper_location": "Sec. two-regime model, Eq. (2)-(3)",
        "departure": (
            "The manuscript prints the structure of the system matrices but not their entries. "
            "The entries in process/regimes.py are the release's own engineering defaults, chosen "
            "so the diameter block is Hurwitz, the geometry enters only through the exogenous "
            "input u(g_k), and the growth offset reproduces the pooled 2.38 mm per year the "
            "manuscript cites."
        ),
        "justification": "The reported numbers cannot be reproduced from a parameterisation the manuscript does not print.",
    },
    {
        "topic": "observation mean under oracle parameters",
        "paper_location": "Eq. (2); Table 1 caption (oracle-parameter filters)",
        "departure": (
            "With oracle parameters the observation mean is taken to be the latent state, so the "
            "boundary-condition and reconstruction machinery fixes the observation covariance and "
            "the patient-persistent term supplies the bias. The surrogate therefore enters the "
            "generator through the covariance it induces, not through a separate field mean."
        ),
        "justification": "This is what running the filter with the process's own parameters means; it is also the only construction in which the sensor's noise decomposition is the quantity under test.",
    },
    {
        "topic": "process noise levels and residual covariance scales",
        "paper_location": "Sec. two-regime model (small Q_q, larger Q_r); Algorithm 1 line 11",
        "departure": (
            "The regime-specific process-noise levels, the boundary-condition noise scale, the "
            "reconstruction jitter and the class-stratified residual scales are engineering "
            "defaults in process/regimes.py, geometry/perturbations.py and sensor/covariance.py."
        ),
        "justification": "The manuscript states the ordering and the decomposition but not the magnitudes.",
    },
    {
        "topic": "diameter-by-sex event strata",
        "paper_location": "Eq. (1); Sec. cohort delineation ([8, 44])",
        "departure": (
            "The stratified annual event risks are a monotone table over the 30-70 mm bands scaled "
            "by the four-fold female-to-male ratio the manuscript states, calibrated so the pooled "
            "event probability inside the recommended interval lands at the reported order."
        ),
        "justification": "The published strata are cited but not printed; the ratio is the one the manuscript does state.",
    },
    {
        "topic": "literal band schedule",
        "paper_location": "Table 1 panel B; [1] ESVS 2024 guideline",
        "departure": (
            "The literal band schedule is the cited guideline's surveillance schedule "
            "(three-yearly below 40 mm, annual at 40-44 mm, three-monthly at 45-54 mm, referral at "
            "or above the sex-specific threshold). The manuscript's own literal-band row reports an "
            "escalate fraction of 0.000, which this schedule does not produce."
        ),
        "justification": "The schedule the manuscript used is not printed, so its row is reported NOT_RUN rather than reconstructed.",
    },
    {
        "topic": "tuned scalar noise of the covariance ablation",
        "paper_location": "Table 2 caption",
        "departure": (
            "The ablation replaces the decomposed per-scan covariance with the per-channel pooled "
            "observation spread on the diagonal and drops the bias coordinates; the manuscript "
            "specifies a tuned scalar without defining the tuning."
        ),
        "justification": "A single scalar across channels of different physical units is not identifiable; the per-channel diagonal is the closest honest reading.",
    },
    {
        "topic": "observation grid and visit intervals",
        "paper_location": "Table 4 (inter-observation interval, median 8.6 months)",
        "departure": (
            "History generation and forecasting run on an integer monthly grid, with inter-visit "
            "intervals drawn from a logarithmic distribution rounded to whole months and clipped "
            "to 3-24 months, so the exact discretisation composes over the interval."
        ),
        "justification": "The intervention grid is reported in whole months; rounding keeps the interval exact and the cohort's median and spread.",
    },
    {
        "topic": "ultrasound site offset",
        "paper_location": "Algorithm 2 line 10",
        "departure": (
            "The cohort process adds a known site offset to the ultrasound diameter and the filter "
            "removes it before the update, because with oracle parameters that offset is part of "
            "the process's own parameters."
        ),
        "justification": "Algorithm 2 names delta_US,site in the measurement model; leaving it in would be a specification error rather than a test.",
    },
    {
        "topic": "reported environment",
        "paper_location": "Code availability",
        "departure": (
            "The manuscript reports Python 3.9.6 with NumPy 2.0.2. This release targets Python "
            ">= 3.10 with NumPy >= 2.0, < 3."
        ),
        "justification": "The reference implementation's interpreter is not available on the verification host.",
    },
    {
        "topic": "executed in-silico scale",
        "paper_location": "Code availability (python3 bmstf_sim.py --seeds 20 --patients 300 --jobs 5)",
        "departure": (
            "The shipped configuration defaults are the manuscript's 20 seeds of 300 histories "
            "with 256 forecast paths. The verification driver executes a reduced, explicitly "
            "labelled scale and reports those numbers as the release's own."
        ),
        "justification": "The full scale is a multi-hour run; every executed number in the report states its scale.",
    },
    {
        "topic": "visit-intensity model",
        "paper_location": "Algorithm 3 line 2",
        "departure": (
            "The fitted visit intensity is a discrete-time per-month hazard with a Gamma prior and "
            "light smoothing, stabilised by the unconditional intensity."
        ),
        "justification": "The manuscript specifies the weighting but not the intensity model.",
    },
    {
        "topic": "sample covariance in the marginalisation",
        "paper_location": "Algorithm 1 lines 6 and 10",
        "departure": (
            "Sigma_BC and Sigma_rec are sample covariances with the Bessel correction "
            "(N_b - 1 and N_m - 1); the manuscript writes cov_s and cov_j without stating the "
            "denominator."
        ),
        "justification": "With N_b = 64 and N_m = 8 the unbiased estimator is the standard reading of cov.",
    },
)


def code_symbols() -> list[str]:
    symbols: list[str] = []
    for claim in CLAIMS:
        symbols.extend(str(symbol) for symbol in claim["symbols"])
    return symbols


def claim_ids() -> list[str]:
    return [str(claim["id"]) for claim in CLAIMS]


def missing_symbols(root_exists: object) -> list[str]:
    from pathlib import Path

    base = Path(str(root_exists))
    return [symbol for symbol in code_symbols() if not (base / symbol.split("::", 1)[0]).exists()]
