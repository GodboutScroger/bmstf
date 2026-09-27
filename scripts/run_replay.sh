#!/usr/bin/env bash
# The inverse-intensity-weighted replay on the generator's histories. The
# retrospective cohort itself is not redistributable and is never downloaded here.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
python3 -m bmstf.cli.replay --histories 60 --forecast-paths 64 \
  --output runs/replay_report.txt
