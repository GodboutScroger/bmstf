# Deep learning forecasts hemodynamic trajectories from computational fluid dynamics for abdominal aortic aneurysm risk stratification

Boundary-marginalised shear-trajectory forecasting (BM-STF) turns serial abdominal
aortic aneurysm imaging into surveillance-interval recommendations. A
boundary-condition-conditioned shear surrogate is treated as a noisy sensor whose
covariance decomposes into a boundary-condition budget, a reconstruction budget and a
patient-persistent residual; the wall-shear state is filtered by a two-regime
(quiescent/active) switching linear-Gaussian latent process whose posterior is exact
conditional on the regime path; the horizon forecast is a closed-form mixture whose
regime weight is the posterior probability that the aneurysm has entered an active
phase; and a two-barrier hazard head reads that forecast at the guideline's own
tolerances, returning the longest admissible interval plus an explicit escalate state.

The package here implements the mechanism as the manuscript specifies it: the
observation equation, the continuous-time regime chain and its exact discretisation,
the interacting-multiple-model filter and its moment-matching collapse, the
marginalised observation of Algorithm 1, the Monte-Carlo horizon forecast with the
Rao-Blackwellised event law, the two-barrier read-out with the Wilson bound, the
pre-specified ablation matrix, the band comparators, and the inverse-intensity-weighted
retrospective replay. The surrogate substrate is a parity component and is not the
claim; it is implemented as a closed-form boundary-condition-conditioned operator so
that its low-shear burden can be recomputed by enumeration.

## Installation

Three equivalent paths. The package targets Python 3.10 or newer; the manuscript's
own reference implementation reports Python 3.9.6 with NumPy 2.0.2, and that
difference is recorded as a deviation in `claim_to_code.json`.

```bash
# pip
python -m pip install -e ".[dev]"

# conda
conda env create -f environment.yml
conda activate bmstf

# container
docker build -t bmstf:2.1.0 .
docker run --rm bmstf:2.1.0 --experiment main
```

`docker build` needs a container runtime; where none is available the container step
is reported `BLOCKED` rather than assumed to work.

## Data

This release runs on its own cohort process. No patient record, imaging volume, mesh or
solver field is redistributed, and the third-party resources below are cited, not
bundled. `dataset_urls.txt` holds only the links that answered from the verification
host.

| Resource | Role | Version / identifier | Terms | Access |
| --- | --- | --- | --- | --- |
| Rygiel et al., wall shear stress estimation in abdominal aortic aneurysms | the regenerated CFD that the Table 5 panel B parity gate runs on | arXiv:2507.22817 | arXiv non-exclusive licence | `https://arxiv.org/abs/2507.22817` |
| Blankemeier et al., Merlin CT vision-language foundation model | frozen abdominal-CT embedding used as image context by the hazard head | Nature, DOI 10.1038/s41586-026-10181-8 | publisher's terms; weights gated | `https://doi.org/10.1038/s41586-026-10181-8` |
| Lannelongue et al., physics-constrained graph network for intracranial hemodynamics | public intracranial geometries of the cross-anatomy stress test, never pooled with the aortic rows | npj Digital Medicine 9, 212 | open access | `https://www.nature.com/articles/s41746-026-02404-z` |
| Retrospective supporting cohort and silent-mode arm (Sites A, B, C) | Tables 4 and 5 | de-identified surveillance records | held by the originating sites under their own agreements | not redistributable; every cohort-level cell is `NOT_RUN` |

The three landing pages above answered with a success status from the release host
and carried the cited work; the solver fields and the foundation-model weights
themselves are not posted anywhere account-free and were not downloaded.
`dataset_urls.txt` repeats that audit with its provenance lines.

Preprocessing for the registerable layers is one command, which builds the histories
and their margin blocks on the fly:

```bash
python -m bmstf.cli.run --experiment _smoke \
  --override outputs.directory=runs/smoke \
  --override study.seeds=1 --override study.patients=8 \
  --decisions
```

The cohort process writes an integer-monthly grid, three-layer annular surfaces on a
48-node centreline template, and the observation structure of Algorithm 1. Expected
on-disk size for the reported run is under 200 MB for the decision table; the run
record itself is a few hundred kilobytes. `integrity_manifest.json` carries the
SHA-256 of every tracked file.

## Reporting the study

One command per reported experiment. The defaults of `configs/experiment/*.yaml` are
the manuscript's reported values (20 seeds of 300 in-silico histories, 256 forecast
paths, 6,000-history calibration cohort, alpha_1 = 0.10, alpha_2 = 0.01, a 3-month
action floor with a 36-month ceiling).

```bash
# Table 1 and Table 2 primary result
python -m bmstf.cli.run --experiment main
python -m bmstf.cli.evaluate --summary runs/main/summary.json \
  --output runs/main_report.txt

# one arm of the Table 2 ablation matrix
bash scripts/run_ablation.sh

# Table 3 panel A (tolerance sweep, bands re-calibrated per tolerance)
bash scripts/run_tolerance_sweep.sh

# Table 3 panel B (fixed-shift diagnostic) with the dimension selection first
python -m bmstf.cli.run --experiment supplementary_shrinkage --select-dimension

# the retrospective replay, on the cohort process's histories
bash scripts/run_replay.sh
```

Every invocation writes `results.csv` (arm by metric, with the seed-level percentile
interval), `summary.json` (the aggregate, the per-seed blocks, the scale, the tuned
noise), `MANIFEST.json` (the key that resolves each reported number in `summary.json`)
and its own `integrity_manifest.json`.

## Verification layer

The verification pass has two parts and neither is optional.

```bash
bash scripts/verify_release.sh
```

The script runs the driver twice, which is not redundant. The first pass writes the
five root artefacts while its pytest gate still finds none, so the tests that read them
skip with the reason; the second pass runs the same work with the first pass's
artefacts on disk, so its gate result is a real assertion over the shipped files and
its manifest-consistency check has a manifest to compare against. A single pass of the
driver is valid on its own, and it prints its own usage:

```bash
python -m bmstf.cli.verify --help
```

It writes five root artefacts:

- `claim_to_code.json` — every paper claim with its location, the code symbols that
  implement it, the recorded deviations, and the executed scale.
- `verification_report.json` — every check with its status, its evidence and its
  detail, split by family, with `overall_status`, `code_status` and
  `manuscript_discrepancies` reported separately.
- `verification_summary.txt` — the same, in plain text.
- `dataset_urls.txt` — the resource landing pages that answered, with their terms.
- `integrity_manifest.json` — the SHA-256 of every tracked file, written last.

The default run touches no network: reachability of the cited resources is opt-in, since a
live probe would write a host-dependent HTTP status into `verification_report.json`, which
the manifest digests. Re-check the links on demand instead, from any satisfied environment:

```bash
python -m bmstf.cli.verify --live
```

The first part recomputes the manuscript's own arithmetic from the counts it prints
(the Table 1 delta rows, the Table 2 single-component sum against the snapshot margin,
the Table 3 sweep, the whole Table 4 delineation card, the Table 5 audit-stratum
margin against the pre-specified 0.03). Nine discrepancies inside the manuscript's
tables are reported as failures of the `manuscript` family and are excluded from
`code_status`, so a defect in the article never masquerades as a defect in the code.

The second part executes the pipeline: data reading, forward filtering, the
observation model, the loss, the parameter update, checkpoint save and load, plus
closed-form references written from the definitions rather than from the
implementation, and a minimal energy-score fitting loop. The status vocabulary is
exactly `PASS`, `FAIL`, `NOT_RUN`, `BLOCKED`; a check that cannot run is never a
success. `pytest -q` on its own passes before the artefacts exist — the tests that
read them skip with the reason — and asserts their content once they do.

## Scale: what was executed and what was reported

The manuscript reports a 20-seed, 300-history study with 256 forecast paths and a
6,000-history band calibration. `configs/experiment/main.yaml` carries those numbers
and `scripts/run_study.sh` runs them. The verification driver executes a smaller,
explicitly labelled scale, recorded in every artefact it writes
(`executed_scale`/`reported_scale` in the report, `executed_scale` in the claim map,
the `scale` block in `summary.json`), and the numbers it reports are the release's own
on that scale. Nothing here claims to reproduce the manuscript's table values: the
cohort process's system matrices, the substrate, the boundary-condition prior and the
event strata are not printed in the manuscript, and are recorded as engineering
defaults in the `deviations` array.

Qualitatively, on the executed scale the release reproduces the manuscript's ordering
on the contrasts it can resolve, and the report carries each measured number: the
filtered forecast covers the true low-shear burden 0.062 more often than the
single-scan snapshot, its 12-month diameter CRPS is 1.136 lower, the switching latent
carries 0.065 of burden coverage, halving the predictive spread costs coverage (0.637
against 0.871) and doubling it saturates (0.996), discarding the hemodynamic channels
raises the between-scan burden RMSE by 0.017, and the ultrasound channel is not
separable at this observation density, which is the null the manuscript reports. Two
read-outs are not settled here and are reported `NOT_RUN` with their measured values
rather than as passes: the manuscript's claim that the scalar-noise ablation loses more
coverage on the high-noise stratum than unconditionally (0.044 against 0.011) has a
seed-paired interval spanning zero at this scale, and no aneurysm-related event falls
inside a recommended interval at all, so the `alpha_2 = 0.01` gate carries no
information. Per-seed dispersion is reported beside every aggregate, so nothing reads
as more precise than it is.

## Compute budget

The verification scale (3 seeds of 60 histories, 64 forecast paths, ten arms, the
diagnostics and the full test suite) runs in a few minutes on a workstation CPU and
needs no accelerator; `scripts/verify_release.sh` runs it twice, so budget for two
passes and not one. The reported scale is 20 seeds of 300 histories with 256 paths
and a 6,000-history calibration cohort; the per-decision cost is one filter pass over
four to six visits plus one Monte-Carlo forecast of 256 paths over 36 months, and the
per-arm cost is that times the number of CTA-visit decisions, so the reported run is a
parallel-CPU job, not a GPU one. The manuscript reports the analysis environment as
Python 3.9.6 with NumPy 2.0.2 and does not report a GPU, VRAM or wall-clock budget;
this release therefore records `COMPUTE_NOT_REPORTED` rather than inventing one.

## Licence and provenance

The code is MIT-licensed (`LICENSE`). `NOTICE` names every third-party dataset,
package and model with its real licence and source, and states which artefacts are not
redistributed. The bibliography of the manuscript is the provenance record for the
cited resources.
