#!/usr/bin/env bash
# One arm of the Table 2 ablation matrix per invocation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
for arm in scalar_noise no_ultrasound single_regime_mean_drift single_regime_quiescent \
           single_barrier diameter_only spread_half spread_double; do
  python3 -m bmstf.cli.run --experiment "ablation_${arm}" "$@"
done
