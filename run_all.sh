#!/usr/bin/env bash
# Reproduce all derived dissertation outputs from a clean checkout.
# Usage: ./run_all.sh
# Set PY to use an existing interpreter (for example, PY=.venv/bin/python).
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-themis}"
mkdir -p "$MPLCONFIGDIR"

PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "Creating .venv and installing requirements..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  PY=".venv/bin/python"
fi

run () {           # run <label> <module...>
  printf '\n== %s ==\n' "$1"
  shift
  "$PY" -m "$@"
}

# ---- engine / RQ1
run "engine.themis (RQ1 point)"                engine.themis
run "engine.scenario_prior"                    engine.scenario_prior
run "engine.transfer_parameterisation"         engine.transfer_parameterisation
run "engine.aggregation_check"                 engine.aggregation_check
run "archetypes.reproduce_archetypes"          archetypes.reproduce_archetypes

# ---- RQ2 core
run "rq2.report_semantics"                     rq2.report_semantics
run "rq2.fixed_coverage"                       rq2.fixed_coverage
run "rq2.endogenous_coverage"                  rq2.endogenous_coverage
run "rq2.sqloss_check"                         rq2.sqloss_check
run "rq2.case_study"                           rq2.case_study

# ---- RQ2 adversarial / practical
run "rq2.obstruction_voteselling"              rq2.obstruction_voteselling
run "rq2.climate_benefit"                      rq2.climate_benefit
run "rq2.information_and_risk"                 rq2.information_and_risk
run "rq2.exinterim_guardrails (slow)"          rq2.exinterim_guardrails
run "rq2.psro_lite"                            rq2.psro_lite
run "rq2.validate_alpharank"                   rq2.validate_alpharank
run "rq2.sensitivity_alpharank"                rq2.sensitivity_alpharank
run "rq2.ablation_alpharank"                   rq2.ablation_alpharank
run "rq2.sensitivity_oracles"                  rq2.sensitivity_oracles

# ---- RQ2 structure / robustness
run "rq2.vote_structure"                       rq2.vote_structure
run "rq2.bruteforce"                           rq2.bruteforce
run "rq2.robustness"                           rq2.robustness

# ---- outputs
run "rq2.collect_headlines"                    rq2.collect_headlines
run "rq2.validate"                             rq2.validate
run "rq2.manifest"                             rq2.manifest
run "rq2.make_figures"                         rq2.make_figures

printf '\nCompleted. Derived outputs:\n'
printf '  data/               numeric results consumed by figures\n'
printf '  figures/            PDF and PNG figures\n'
printf '  reference_outputs/  experiment manifest\n'
