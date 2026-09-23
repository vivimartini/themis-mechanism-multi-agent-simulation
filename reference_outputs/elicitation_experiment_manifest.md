# Elicitation experiment manifest

This experiment is a post-dissertation extension. It is not run by `run_all.sh`
and does not alter any submitted result.

| module | question | sampling | seed / search | utility | outputs |
|---|---|---|---|---|---|
| `elicit.run_experiment` | How does limiting the report format trade truthful welfare against strategic regret? | deterministic scenario sweep | peak fractions 0.35, 0.50, 0.65; format-specific global search, seed 42 | normalised quadratic true surplus; zero outside option | `data/elicitation_tradeoff.npz`, `figures/fig_elicitation_tradeoff.{pdf,png}` |

## Modelling assumption

At fixed coverage and with transfers off, the existing affine willingness is the
upper zero of a quadratic surplus. `peak_fraction` places the interior peak as a
fraction of that upper zero. No data identify this quantity, so all headline
outputs show the three-point scenario sweep.

## Report formats

- **Full:** peak, upper zero, and scale. Reported scale is normalised to one
  before welfare maximisation; the raw-scale validation reproduces utility
  hiking.
- **Peak:** one preferred price, aggregated by the fixed weighted quantile; the
  common scenario shape implies membership from that scalar report.
- **Region:** a closed acceptable interval, evaluated under both the weighted
  midpoint of the greatest-emissions intersection and a maximum-participating-
  emissions rule.

Membership follows the report; realised utility uses the true quadratic surplus;
non-members receive the zero outside option.

Reported regrets are lower bounds found by a deliberately redundant search.
Membership makes the objective discontinuous, so the runner does not present
CMA convergence as an equilibrium or exact-regret certificate. Phase B is exact
only on its declared report grid `[0, 42]` in two-unit steps. Phase A uses:

- full reports: coarse grid, 4096 seeded global reports, explicit five-cent
  narrow-report probes, then CMA;
- region reports: coarse grid, 4096 seeded global reports, then CMA;
- peak reports: the quantile rule's exact report-crossing candidates.
