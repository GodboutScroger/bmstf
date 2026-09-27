"""Run the inverse-intensity-weighted retrospective policy replay.

Ref: Algorithm 3 and Table 4 (the retrospective cohort card). The cohort is not
redistributable, so the replay runs on the cohort process's histories and every number
is labelled as the release's own.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bmstf.inference.forecast import ForecastSpec
from bmstf.policy.bands import literal_band_policy
from bmstf.process.cohort import HistorySample, HistorySpec, SensorContext, patient_batch
from bmstf.process.hazards import StratumRisk
from bmstf.process.regimes import ShearSystem, build_system
from bmstf.protocol.criteria import ALPHA1, ALPHA2, DEFERRAL_CAP
from bmstf.replay.censoring import fit_censoring
from bmstf.replay.estimator import ReplayVisit, estimate, intensity_from_records
from bmstf.replay.intensity import intervals_from_months
from bmstf.replay.landmarks import landmark_summary
from bmstf.runtime.config import repository_root
from bmstf.runtime.seeds import set_seed
from bmstf.runtime.store import write_text
from bmstf.study.engines import ArmSpec, base_filter, evaluate_history


def replay_visits_from_histories(
    histories: list[HistorySample], system: ShearSystem, forecast_paths: int
) -> list[ReplayVisit]:
    arm = ArmSpec(name="BM-STF", filter=base_filter("bmstf", system))
    strata = StratumRisk()
    policy = literal_band_policy()
    visits: list[ReplayVisit] = []
    for position, history in enumerate(histories):
        rng = set_seed(20260925 + position).forecast
        records = evaluate_history(
            history, system, arm, ForecastSpec(paths=forecast_paths), policy, strata, rng
        )
        for record in records:
            visits.append(
                ReplayVisit(
                    record=history.patient,
                    index=int(record.landmark_index),
                    month=float(record.decision_month),
                    horizon=float(record.interval_months),
                    escalate=bool(record.escalate),
                    crossing_hit=bool(record.crossing_hit),
                    event_hit=bool(record.event_hit),
                    comparator_months=float(record.comparator_months),
                    band=str(record.band),
                    sex=str(record.sex),
                    repair_month=history.outcome.event_month,
                )
            )
    return visits


def render(estimates: dict[str, object], scales: dict[str, object]) -> str:
    crossing: dict[str, float] = dict(estimates.get("crossing_rate", {}))  # type: ignore[call-overload]
    event: dict[str, float] = dict(estimates.get("event_rate", {}))  # type: ignore[call-overload]
    by_band: dict[str, dict[str, float]] = dict(estimates.get("by_band", {}))  # type: ignore[call-overload]
    lines = [
        "Inverse-intensity-weighted retrospective policy replay",
        "=" * 60,
        f"scope                    : {scales.get('scope', 'cohort histories')}",
        f"histories                : {scales.get('histories', 0)}",
        f"decisions                : {scales.get('decisions', 0)}",
        "",
        f"crossing rate (weighted) : {crossing.get('estimate', float('nan')):.4f} "
        f"[{crossing.get('lower', float('nan')):.4f}, {crossing.get('upper', float('nan')):.4f}]",
        f"event rate (weighted)    : {event.get('estimate', float('nan')):.4f} "
        f"[{event.get('lower', float('nan')):.4f}, {event.get('upper', float('nan')):.4f}]",
        f"imaging burden (months)  : {estimates.get('imaging_burden', float('nan')):.3f}",
        f"escalate fraction        : {estimates.get('escalate_fraction', float('nan')):.3f}",
        f"deferral                 : {estimates.get('deferral', {})}",
        "",
        "by 5-mm band",
        "-" * 60,
    ]
    for band, values in sorted(by_band.items()):
        lines.append(
            f"  {band:<10} n={int(values['count']):<6} escalate={values['escalate']:.3f} "
            f"crossing={values['crossing']:.3f} event={values['event']:.3f} "
            f"interval={values['interval']:.2f} months"
        )
    lines += [
        "",
        "The cohort is not redistributable, so these rates are the release's own on the",
        "the cohort process's histories and carry no claim about patients.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bmstf-replay",
        description="Run the inverse-intensity-weighted retrospective policy replay.",
    )
    parser.add_argument("--histories", type=int, default=60)
    parser.add_argument("--forecast-paths", type=int, default=64)
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--output", default="runs/replay_report.txt")
    parser.add_argument("--root", default=None)
    arguments = parser.parse_args(argv)
    root = Path(arguments.root).resolve() if arguments.root else repository_root()
    system = build_system()
    from bmstf.sensor.boundary import BoundaryPrior
    from bmstf.sensor.covariance import ResidualLookup
    from bmstf.sensor.marginalise import MarginalisationSpec
    from bmstf.sensor.surrogate import PhysicsAttentionSurrogate

    sensor = SensorContext(
        system,
        ResidualLookup(),
        MarginalisationSpec(),
        PhysicsAttentionSurrogate(),
        BoundaryPrior(),
    )
    streams = set_seed(20260925).generator
    histories = patient_batch(
        int(arguments.histories),
        sensor,
        HistorySpec(),
        streams,
        prefix="REP",
    )
    visits = replay_visits_from_histories(histories, system, int(arguments.forecast_paths))
    intervals = np.concatenate(
        [
            intervals_from_months(history.visit_months)
            for history in histories
            if history.visit_months.size > 1
        ]
    )
    intensity = intensity_from_records([history.visit_months for history in histories])
    censoring = fit_censoring(
        np.asarray(
            [
                float(history.outcome.event_month)
                for history in histories
                if history.outcome.event_month
            ]
            or [24.0]
        )
    )
    estimates = estimate(
        visits,
        intensity,
        censoring,
        deferral_cap=DEFERRAL_CAP,
        resamples=int(arguments.resamples),
        rng=set_seed(20260925).resample,
    )
    scales = {
        "scope": "generator histories; the retrospective cohort is not redistributable",
        "histories": len(histories),
        "decisions": len(visits),
        "intervals": int(intervals.size),
        "alpha1": ALPHA1,
        "alpha2": ALPHA2,
        "landmarks": len(
            landmark_summary(
                histories[0].visit_months, tuple(v.channel for v in histories[0].visits)
            )
        ),
    }
    write_text(root / arguments.output, render(estimates.as_mapping(), scales))
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
