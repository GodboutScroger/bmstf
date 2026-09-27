"""History generation: visit schedule, ranges and the outcome laws."""

from __future__ import annotations

import numpy as np

from bmstf.geometry.shapes import geometry_from_state
from bmstf.process.cohort import (
    CHANNELS,
    FEMALE_DIAMETER_CEILING_MM,
    HistorySpec,
    _entry_band,
    build_history,
    draw_visit_months,
    regime_path,
)
from bmstf.process.hazards import (
    RAPID_EXPANSION_PRIMARY,
    StratumRisk,
    composite_outcome,
    cumulative_hazard,
    first_crossing_month,
    rapid_expansion_month,
    repair_threshold_mm,
)


def test_visit_months_are_integer_and_bounded() -> None:
    months = draw_visit_months(HistorySpec(horizon_months=36), np.random.default_rng(1))
    assert months[0] == 0
    assert np.all(np.diff(months) >= 3)
    assert int(months[-1]) <= 36
    assert months.shape[0] >= 3


def test_entry_band_excludes_the_female_threshold() -> None:
    assert _entry_band(HistorySpec(), "female") == (30.0, FEMALE_DIAMETER_CEILING_MM)
    assert _entry_band(HistorySpec(), "male") == (30.0, 54.0)


def test_regime_path_is_binary_and_start_anchored(system) -> None:
    path = regime_path(system, 1, 36, np.random.default_rng(3))
    assert path.shape == (37,)
    assert set(np.unique(path)).issubset({0, 1})
    assert path[0] == 1


def test_history_shapes_and_channels(sensor) -> None:
    streams = np.random.default_rng(5)
    history = build_history(
        "GEN-1", "male", 0, sensor, HistorySpec(horizon_months=24), streams, 0.4
    )
    assert history.visits[0].month == 0.0
    assert all(visit.channel in CHANNELS for visit in history.visits)
    assert history.inputs.shape[1] == 4
    assert history.monthly_diameter.shape == (25,)
    assert history.visit_states.shape[1] == 10
    assert history.boundary_trace.shape[0] == history.observations_count


def test_months_and_burdens_stay_physical(sensor) -> None:
    streams = np.random.default_rng(7)
    history = build_history(
        "GEN-2", "female", 1, sensor, HistorySpec(horizon_months=36), streams, 0.3
    )
    assert np.all(history.monthly_diameter > 0.0)
    assert np.all(history.monthly_burden >= 0.0)
    assert np.all(history.monthly_burden <= 1.0)
    assert np.all(np.isfinite(history.visit_states))


def test_growth_offset_uses_the_published_spread(sensor) -> None:
    offsets = []
    for index in range(24):
        history = build_history(
            f"G-{index}",
            "male",
            0,
            sensor,
            HistorySpec(horizon_months=12),
            np.random.default_rng(100 + index),
            0.5,
        )
        offsets.append(history.growth_offset)
    values = np.asarray(offsets)
    assert 0.05 < float(np.mean(values)) < 0.35
    assert 0.05 < float(np.std(values)) < 0.3


def test_crossing_law_is_monotone_and_interpolated() -> None:
    months = np.arange(0.0, 13.0)
    diameters = np.full(13, 40.0)
    assert first_crossing_month(months, diameters, "male") is None
    diameters[7:] = 60.0
    crossing = first_crossing_month(months, diameters, "male")
    assert crossing is not None
    assert abs(float(crossing) - 6.75) < 1.0e-12
    assert repair_threshold_mm("female") == 50.0


def test_rapid_expansion_detects_the_slope() -> None:
    months = np.arange(0.0, 13.0)
    diameters = 40.0 + 0.5 * months
    assert rapid_expansion_month(months, diameters, RAPID_EXPANSION_PRIMARY) is not None
    flat = np.full(13, 40.0)
    assert rapid_expansion_month(months, flat, RAPID_EXPANSION_PRIMARY) is None


def test_cumulative_hazard_is_a_prefix_product() -> None:
    diameters = np.linspace(40.0, 50.0, 24)
    burdens = np.linspace(0.2, 0.5, 24)
    observed = cumulative_hazard(diameters, burdens, "male")
    survival = 1.0
    for position in range(24):
        table = StratumRisk()
        annual = table.annual_rate(float(diameters[position]), "male")
        base = 1.0 - (1.0 - annual) ** (1.0 / 12.0)
        survival *= 1.0 - min(0.5, base * (1.0 + 1.8 * (float(burdens[position]) - 0.35)))
        assert abs(float(observed[position]) - (1.0 - survival)) < 1.0e-12


def test_composite_outcome_reports_every_component() -> None:
    months = np.arange(0.0, 37.0)
    diameters = 40.0 + 0.6 * months
    burdens = np.full(37, 0.4)
    outcome = composite_outcome(months, diameters, burdens, "male", 0.99)
    mapping = outcome.as_mapping()
    assert set(mapping) == {
        "crossing_month",
        "event_month",
        "rapid_primary_month",
        "rapid_co_secondary_month",
        "rapid_confirmatory_month",
    }
    assert outcome.crossing_month is not None
    assert outcome.rapid_co_secondary_month is not None


def test_geometry_and_summary_agree_with_the_state() -> None:
    geometry = geometry_from_state("state", 48.0, 0.4)
    from bmstf.geometry.summaries import maximum_diameter_mm

    assert abs(maximum_diameter_mm(geometry) - 48.0) < 1.0e-6
