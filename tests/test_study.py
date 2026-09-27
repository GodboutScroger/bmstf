"""Study orchestration, contrasts, the tolerance sweep and dimension selection."""

from __future__ import annotations

import numpy as np

from bmstf.policy.bands import literal_band_policy
from bmstf.process.hazards import StratumRisk
from bmstf.study.contrasts import (
    component_sum,
    contrast_across_seeds,
    contrast_family,
    flatten_family,
    snapshot_margin,
)
from bmstf.study.engines import (
    ARM_NAMES,
    build_arm,
    corrected_visits,
    evaluate_history,
    tune_scalar_noise,
)
from bmstf.study.frontier import (
    SWEEP_TOLERANCES,
    fixed_shift_rows,
    monotone_crossing,
    shift_increases,
    sweep,
)
from bmstf.study.histories import calibrate_bands, metric_names, run_study
from bmstf.study.selection import one_step_energy, select_dimension


def test_every_declared_arm_builds(system) -> None:
    for name in ARM_NAMES:
        arm = build_arm(name, system)
        assert arm.name == name
        assert arm.filter.modes.modes in (1, 2)


def test_scalar_noise_is_tuned_per_channel(histories) -> None:
    tuned = tune_scalar_noise(histories)
    assert tuned.shape == (4,)
    assert np.all(tuned > 0.0)


def test_evaluate_history_produces_records(system, histories, forecast_spec) -> None:
    arm = build_arm("BM-STF", system)
    records = evaluate_history(
        histories[0],
        system,
        arm,
        forecast_spec,
        literal_band_policy(),
        StratumRisk(),
        np.random.default_rng(2),
    )
    assert records
    for record in records:
        assert 0.0 <= record.crossing_probability <= 1.0
        assert 0.0 <= record.event_probability <= 1.0
        assert record.interval_months > 0.0
        assert isinstance(record.as_mapping(), dict)
        assert "coverage_diameter" in record.as_mapping()


def test_snapshot_arm_uses_a_point_state(system, histories, forecast_spec) -> None:
    arm = build_arm("snapshot", system)
    assert arm.snapshot
    records = evaluate_history(
        histories[0],
        system,
        arm,
        forecast_spec,
        literal_band_policy(),
        StratumRisk(),
        np.random.default_rng(3),
    )
    assert records
    assert all(np.isfinite(record.crossing_probability) for record in records)


def test_corrected_visits_remove_the_site_offset(histories) -> None:
    history = histories[0]
    visits = corrected_visits(history)
    for original, corrected in zip(history.visits, visits, strict=True):
        if original.channel == "US":
            assert not np.allclose(original.measurement, corrected.measurement)
        else:
            assert np.allclose(original.measurement, corrected.measurement)


def test_run_study_aggregates_every_metric(sensor, smoke_spec) -> None:
    from bmstf.process.regimes import build_system

    system = build_system()
    result = run_study(sensor, system, smoke_spec, StratumRisk())
    assert set(result.aggregate) == set(smoke_spec.arms)
    for arm in result.aggregate:
        for metric in metric_names():
            assert metric in result.aggregate[arm]
            assert f"{metric}_lower" in result.aggregate[arm]
    assert result.decision_count > 0
    assert len(result.decision_rows) <= smoke_spec.decision_row_limit
    assert result.scalar_noise.shape == (4,)
    payload = result.as_mapping()
    assert "aggregate" in payload and "spec" in payload


def test_band_calibration_returns_a_schedule(sensor, smoke_spec) -> None:
    from bmstf.process.regimes import build_system

    system = build_system()
    policy = calibrate_bands(sensor, system, smoke_spec)
    assert policy.name == "calibrated_bands"
    assert set(policy.schedule).issubset({"30-40", "40-45", "45-55", "55-70"})


def test_contrast_family_adjusts_within_each_metric() -> None:
    blocks = [
        {"BM-STF": {"coverage_burden": 0.90 + 0.01 * seed}, "snapshot": {"coverage_burden": 0.70}}
        for seed in range(6)
    ]
    family = contrast_family(blocks, ("coverage_burden",), ("snapshot",))
    rows = flatten_family(family)
    assert len(rows) == 1
    assert rows[0]["delta"] > 0.0
    assert 0.0 <= rows[0]["adjusted_p"] <= 1.0
    assert component_sum(family) == 0.0


def test_contrast_across_seeds_shapes_an_interval() -> None:
    blocks = [{"A": {"m": 0.5 + 0.01 * seed}, "B": {"m": 0.4}} for seed in range(8)]
    contrast = contrast_across_seeds(blocks, "m", "B", reference="A")
    assert contrast.lower <= contrast.delta <= contrast.upper
    assert contrast.reference > contrast.comparator_value


def test_snapshot_margin_reads_the_contrast() -> None:
    blocks = [
        {"BM-STF": {"coverage_burden": 0.9}, "snapshot": {"coverage_burden": 0.76}}
        for _ in range(4)
    ]
    family = contrast_family(blocks, ("coverage_burden",), ("snapshot",))
    assert abs(snapshot_margin(family) - 0.14) < 1.0e-12


def test_tolerance_sweep_runs_each_arm(sensor, smoke_spec) -> None:
    from bmstf.process.cohort import HistorySpec, patient_batch
    from bmstf.process.regimes import build_system
    from bmstf.runtime.seeds import set_seed

    system = build_system()
    streams = set_seed(11)
    histories = patient_batch(
        4, sensor, HistorySpec(horizon_months=24), streams.generator, prefix="SW"
    )
    rows = sweep(
        histories,
        system,
        smoke_spec.forecast_spec(),
        StratumRisk(),
        5,
        lambda alpha1: calibrate_bands(sensor, system, smoke_spec, alpha1),
        tolerances=SWEEP_TOLERANCES[:2],
        arms=("BM-STF", "calibrated_bands"),
    )
    assert len(rows) == 4
    assert {row.alpha1 for row in rows} == {0.05, 0.10}
    assert all(isinstance(row.as_mapping(), dict) for row in rows)


def test_fixed_shift_diagnostic_shapes_rows(system, histories, smoke_spec) -> None:
    from bmstf.study.engines import run_arm

    arm = build_arm("BM-STF", system)
    result = run_arm(
        histories[:2],
        system,
        arm,
        smoke_spec.forecast_spec(),
        StratumRisk(),
        np.random.default_rng(7),
    )
    rows = fixed_shift_rows({"BM-STF": result})
    assert len(rows) == 2
    assert all(row.at_interval for row in rows)
    assert all(shift_increases(row) or not row.at_interval for row in rows)
    assert monotone_crossing([]) is True


def test_dimension_selection_scores_each_candidate(sensor) -> None:
    from bmstf.process.cohort import HistorySpec, patient_batch
    from bmstf.runtime.seeds import set_seed

    streams = set_seed(13)
    histories = patient_batch(
        3, sensor, HistorySpec(horizon_months=12), streams.generator, prefix="SEL"
    )
    score = one_step_energy(histories, 10, np.random.default_rng(15), paths=16)
    assert np.isfinite(score)
    report = select_dimension(histories, sensor.spec, seed=17, candidates=(8, 10), paths=16)
    assert report.selection.dimension in (8, 10)
    assert set(report.selection.scores) == {8, 10}
    assert report.validation_pairs >= 0
