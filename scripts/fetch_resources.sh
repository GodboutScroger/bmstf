#!/usr/bin/env bash
# Reachability re-audit of the third-party resources this release cites. Nothing
# is downloaded: the solver fields, the foundation-model weights and the clinical
# cohorts are not redistributed, and dataset_urls.txt lists only the landing pages
# that were checked from the release host.
set -euo pipefail
cd "$(dirname "$0")/.."
while read -r line; do
  case "$line" in
    ""|\#*) continue ;;
  esac
  code=$(curl -s -o /dev/null -w "%{http_code}" -L --max-time 20 "$line" || echo "000")
  printf "%s  %s\n" "$code" "$line"
done < dataset_urls.txt
