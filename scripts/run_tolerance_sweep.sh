#!/usr/bin/env bash
# Table 3: the crossing tolerance swept at fixed alpha_2, and the band comparator
# re-calibrated at each tolerance on its own 6,000-history cohort.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
python3 -m bmstf.cli.run --experiment supplementary_tolerance_sweep "$@"
python3 -m bmstf.cli.evaluate --summary runs/tolerance_sweep/summary.json \
  --output runs/tolerance_sweep_report.txt
