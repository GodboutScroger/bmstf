"""Verification entry point.

Ref: Data availability statement (the run record, the manifest and the checksums).
The driver writes claim_to_code.json, verification_report.json,
verification_summary.txt and dataset_urls.txt, then the integrity manifest.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from bmstf.harness.driver import main as driver_main


def main(argv: Sequence[str] | None = None) -> int:
    return driver_main(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
