#!/usr/bin/env bash
# The verification layer, run twice, in the order the artefacts depend on each other.
#
# Pass 1 executes the study and every check, then writes the five root artefacts; its
# pytest gate sees no artefacts yet, so the tests that read them skip. Pass 2 runs the
# same work again, and now the pytest gate has pass 1's artefacts to assert against, so
# the shipped report's gate result is a real one. Pass 2 writes the manifest last, so
# the manifest digests the finished tree, and pass 1's manifest is what pass 2's
# manifest-consistency check compares against.
#
# Exit status is that of the last pass, and a pass fails only on a check outside the
# manuscript family, so a discrepancy inside the article's own tables never reads as a
# defect in this release.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
python3 -m bmstf.cli.verify "$@"
python3 -m bmstf.cli.verify "$@"
