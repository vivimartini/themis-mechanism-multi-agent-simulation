# Report-format elicitation experiment

This package is a post-dissertation extension motivated by the distinction
between asking nations for a full utility function, a preferred price, or an
acceptable region. It is deliberately excluded from `run_all.sh`; nothing in
the submitted evidence pipeline depends on it.

## Modelling choice

At fixed coverage and with transfers off, each actor's existing affine
willingness becomes the **upper zero** of a quadratic price surplus. The
quadratic's peak is placed at a fixed fraction of that threshold. No data
identify this fraction, so the experiment reports a scenario sweep over 0.35,
0.50, and 0.65 rather than treating one position as estimated.

In one sentence: *the only new preference assumption is where each nation's
ideal price lies inside its calibrated acceptable region, and the results sweep
that assumption explicitly.*

## Report formats

- **Full utility:** quadratic peak, upper zero, and scale. Reported scale is
  normalised to unit peak before aggregation, preventing an actor from buying
  influence merely by multiplying its utility.
- **Preferred value:** one peak, aggregated by the fixed weighted-quantile rule.
  The scenario's common quadratic shape implies the reported acceptance region
  from that scalar; the fixed-membership variant is used only for the Moulin
  strategy-proofness anchor.
- **Acceptable region:** one interval, evaluated with both a weighted midpoint
  of the greatest-emissions common intersection and a rule that maximises
  participating emissions.

Membership follows the report. Payoff is always scored using the actor's true
quadratic surplus, with zero for staying out.

Because membership changes discontinuously, Phase A reports **found regret lower
bounds**, not exact continuous-game regrets. Full reports use a coarse grid,
4,096 seeded global reports, explicit five-cent narrow-report probes, and CMA
refinement; region reports use the grid, global reports, and CMA; scalar peaks
use the quantile rule's exact report-crossing candidates. Phase B is exhaustive
only on its declared `[0, 42]` two-unit deviation grid.

## Run

```bash
.venv/bin/python -m elicit.validate
.venv/bin/python -m elicit.run_experiment
```

The runner refuses to print headline results until the Moulin, approval, and
utility-hiking anchors pass. It writes:

- `data/elicitation_tradeoff.npz`
- `figures/fig_elicitation_tradeoff.pdf`
- `figures/fig_elicitation_tradeoff.png`
- `reference_outputs/elicitation_experiment_manifest.md`

Phase B repeats the central scenario on the China–US–India–Indonesia four-actor
case using report coordinates from 0 to 42 in two-unit steps and no optimiser.

