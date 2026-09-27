#!/usr/bin/env bash
# The reported in-silico study: 20 seeds of 300 in-silico histories, 256 paths.
# Wall clock on a workstation-class CPU is dominated by the Monte-Carlo forecast;
# see README.md for the measured cost of the verification scale.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
python3 -m bmstf.cli.run --experiment main "$@"
