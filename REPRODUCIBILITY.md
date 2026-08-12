# Reproducibility and provenance

## Source and derived material

| Location | Status | Notes |
|---|---|---|
| `engine/`, `archetypes/`, `rq2/`, `paths.py` | Source | Python modules used by the experiments. |
| `data/actors_baseline.csv` | Input | Point calibration for the nine-actor model. |
| `archetypes/preferences.py` | Input | Digitised published preference specification. |
| `archetypes/country_archetypes.csv` | Context only | Country/VAT lookup; no executable module imports it. |
| `data/*.npz`, `data/headlines.json` | Derived | Written by the experiment named in the manifest. |
| `figures/fig_*` | Derived | Written by `python -m rq2.make_figures` from saved results. |
| `reference_outputs/experiment_manifest.md` | Derived | Generated experiment manifest. |
| `reference_outputs/full_run.txt` | Derived | Captured output from the full pipeline. |
| `reference_outputs/run_provenance.txt` | Derived | Environment, runtime, git state, and artifact checksums. |

Input limitations and source decisions are documented in
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md). The mapping from dissertation claims
to producers is in
[`DISSERTATION_CROSSWALK.md`](DISSERTATION_CROSSWALK.md).

## Rebuilding derived outputs

The validated environment is Python 3.13 with the exact package versions in
`requirements.txt`. Run `./run_all.sh` from the repository root. The sequence
first runs the source experiments, then writes headline data, the manifest, and
figures. The dependency relationships are documented in `rq2/make_figures.py`
and the generated `reference_outputs/experiment_manifest.md`.

To refresh the versioned run record and provenance at the same time, run:

```bash
./scripts/capture_run.sh
```

Do not edit `.npz`, JSON, figure, or reference-output files by hand. If a source
module changes, rerun its experiment and the downstream output stages. Where an
experiment uses Monte Carlo or CMA-ES, retain the stated seed and package pins
when comparing output with the dissertation.

## Verification

After a full run:

```bash
.venv/bin/python -m rq2.validate
.venv/bin/python -m rq2.validate_alpharank
.venv/bin/python scripts/check_figure_fonts.py
.venv/bin/python scripts/check_release.py
git ls-files -z '*.py' | xargs -0 .venv/bin/python -m py_compile
```

`rq2.validate` expects the derived datasets to exist. `scripts/check_release.py`
checks the release inventory, headline values, manifest coverage, and accidental
local-path leakage; it does not rerun numerical experiments.

## Known reproducibility qualifications

- The dissertation states a `[-60, 60]^2` report box, while the CMA-ES oracle
  that generated the submitted results is unbounded. This release preserves and
  explicitly labels those results; see the README.
- The archived NashConv value 3.60 was produced by a removed, superseded oracle
  and cannot be regenerated. Its provenance is retained in the manifest.
- Raw source extracts for the calibration are not redistributed, so the point
  calibration is reproducible as an input but not rebuildable from primary
  sources in this repository.
- Exact source provenance requires a git commit and release tag. The capture
  script records `UNCOMMITTED` when no commit exists.

