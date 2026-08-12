# Data provenance and calibration status

This repository contains a calibrated mechanism test environment, not a
forecasting dataset. This document distinguishes observations from modelling
choices and records which source material is not redistributed.

## Primary calibration

`data/actors_baseline.csv` is the machine-readable point calibration used by the
engine and RQ2 solvers.

| Field | Status | Basis |
|---|---|---|
| `e`, population | Observed/aggregated | EDGAR 2025 greenhouse-gas emissions excluding LULUCF and World Bank WDI population |
| `gdp_cap` | Observed/aggregated | World Bank WDI 2024 income |
| `alpha_trf` | Derived shape, chosen scale | `min(20, 20000 / GDP per capita)` |
| `alpha_base` | Anchored or chosen | OECD Effective Carbon Rates 2025 country data, checked against ICAP status information; row-specific decisions are in `source_note` |
| `alpha_cov` | Modelled | Counterfactual coverage response; no historical coalition at varying global coverage exists |
| `weight` | Derived at runtime | Recomputed as `population × emissions`, then normalised; the CSV column is descriptive |
| `source_strength` | Qualitative metadata | Strength grades for the three preference coefficients |

The common emissions benchmark `e_bar = 6.6 tCO2e/capita`, transfer scale
`20000`, transfer-sensitivity cap `20`, prior ranges, and solver/search settings
are modelling inputs described in the dissertation and code. They are not
empirical estimates produced by this repository.

## Actor-specific base-term decisions

- China uses the explicit average rate `3.27`; the OECD net effective rate
  `8.63` is retained in the notes but rejected as fuel-excise inflation.
- United States uses `3.00`, an economy-wide anchor based on subnational systems.
- European Union uses a constructed borne-cost estimate `18.50`, not the
  headline allowance price.
- India uses a development-trajectory placeholder `4.00`; it is a modelling
  adjustment, not a directly observed carbon price.
- Russia, the low-carbon frontier, and hydrocarbon rentiers use a chosen lower
  bound of zero.
- Indonesia floors a negative net effective rate to zero.
- Advanced conditional joiners use an aggregated `5.80` base term, emissions
  weighted across the bloc and adjusted for free allocation. The CSV's
  `headline_price` column for this row is population-weighted and indicative
  only; it is not the basis of `alpha_base`. Treat the implemented value as a
  fixed calibration input rather than a regenerable aggregation.

See Appendix A, Tables A.1–A.3, of the dissertation for the full narrative and
bibliographic references.

## Source material not redistributed

The repository does not include raw extracts from EDGAR, World Bank WDI, OECD,
the EU Commission, ICAP, or Reuters. It also does not include extraction queries,
access-date snapshots, or a transformation notebook from those raw sources to
the nine-actor calibration. Consequently:

- the calibration CSV is the release input and cannot be rebuilt from raw source
  files in this repository;
- the statement that represented actors cover roughly 62% of EDGAR global
  emissions cannot be regenerated here because the global denominator is not
  stored;
- users should consult the dissertation bibliography and the row-level
  `source_note` field before updating values.

These are provenance limitations, not generated-output failures.

## Archetype reproduction

- `archetypes/preferences.py` contains the five preference lines digitised from
  the published quantitative examination. Digitisation is approximate and the
  sensitivity analysis in `archetypes.reproduce_archetypes` measures its effect.
- `archetypes/country_archetypes.csv` is a contextual country/VAT lookup. No
  executable module imports it, and it is not the machine-readable source of the
  dissertation's Table A.3 mapping.
- The actual nine-actor-to-seven-archetype relationship is recorded in Table
  A.3 of the dissertation and summarised in `DISSERTATION_CROSSWALK.md`.

## UK aggregation check

`engine/aggregation_check.py` contains an illustrative UK split with population,
emissions, income, and base-price values hard-coded in that module. They are not
part of `data/actors_baseline.csv` and do not have repository-level raw-source
provenance. The script sweeps the least certain base term, but the result should
not be treated as independently reproducible source-data analysis.

## Derived artifacts

The `.npz` and JSON files in `data/`, all files in `figures/`, and the files in
`reference_outputs/` are generated artifacts. Their producer modules are listed
in `reference_outputs/experiment_manifest.md` and
`DISSERTATION_CROSSWALK.md`. Regenerate them through `./run_all.sh`; do not edit
them manually.
