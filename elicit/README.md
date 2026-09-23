# Report-format elicitation experiment

This experiment compares three ways of eliciting price preferences: a full
utility curve, a preferred price, and an acceptable interval. It was added
after the dissertation was submitted and runs separately from `run_all.sh`.

## Model

Coverage is fixed and transfers are disabled. For each actor, the calibrated
affine willingness is used as the upper zero of a quadratic price surplus. The
quadratic's peak is placed at 0.35, 0.50, or 0.65 of that threshold. This peak
location is a scenario assumption, not an estimated parameter.

Membership is determined by the submitted report. Realised utility is evaluated
against the actor's true quadratic, with an outside option of zero.

## Reports and selection rules

- **Full utility:** peak, upper zero, and scale; scale is normalised before
  reported welfare is maximised.
- **Preferred price:** one scalar, aggregated by a weighted quantile at either
  the Themis coverage level (`0.869`) or the median. A scalar carries no shape
  information, so membership is inferred from the population-average peak
  fraction.
- **Acceptable interval:** lower and upper bounds, evaluated with four rules:
  greatest-overlap midpoint, maximum participating emissions, weighted median
  of midpoints, and the Themis objective (participating emissions × price).

Regret is reported unweighted, weighted by emissions (the welfare weights), and
as the largest single-actor regret.

## Results

Full tables: `reference_outputs/elicitation_rule_comparison.md`.

- **The aggregation rule matters as much as the report format.** Whenever
  peaks sit at or above the middle of the acceptable interval, the interval
  midpoint equals the peak. Interval reports with a weighted median are then
  the same rule as peak reports with a median. The apparent advantage of
  interval reports in that comparison comes from the quantile level (`0.869`
  against `0.5`), not from the format.
- **Coverage-first interval rules act almost as a veto.** Every true interval
  starts at zero when peaks sit at or below the middle. The midpoint and
  maximum-coverage rules then include every actor, and the lowest ceiling sets
  the price. This produces the large regrets (`2.97`, `2.99`), a third of which
  comes from one actor holding 2.9% of emissions.
- **Under the Themis objective, interval reports remain manipulable because
  members narrow their intervals to steer the price toward their own peak.**
  In the central scenario, the truthful price is `27.32` with China, the United
  States, the EU, and the conditional joiners as members. Found regret is
  `0.75` unweighted and `0.19` emissions-weighted, all held by those four.
  China reports `[7.33, 16.84]` instead of `[0, 33.67]`; cutting its upper
  bound to its peak moves the price to `16.84`, its peak, and India joins. The
  United States and the joiners also lower their upper bounds. The EU, whose
  peak lies above the truthful price, raises its lower bound above the US
  ceiling instead, which removes the United States and raises the price to
  `33.67`. Full-utility reports have `0.90` unweighted and `0.08` weighted, so
  the ranking of the two formats depends on the weighting.
- **Zero regret for peak reports depends on how membership is decided.** If the
  price is set first and each actor then joins only if it gains, the payoff
  `max(0, s(p))` stays single-peaked. The fixed quantile is then strategy-proof
  (Moulin, 1980) even when peak fractions differ across actors; a grid check
  over all four peak scenarios finds no profitable report at either quantile
  level. If instead membership is inferred from the reported peak, using the
  population-average peak fraction, the `0.869` quantile has found regret
  `0.42`–`0.92` in the heterogeneous scenarios, and the median `0.41` in one of
  them. This is the same weakness as membership decided by reports in the
  dissertation.

Continuous-format regrets are search lower bounds, not exact optima. The
experiment holds coverage fixed and disables transfers, so it compares report
formats in a stylised setting rather than testing Themis itself.

## Reproduction

```bash
.venv/bin/python -m elicit.validate
.venv/bin/python -m elicit.run_experiment
.venv/bin/python -m elicit.hand_check
.venv/bin/python -m elicit.enumerate_small
.venv/bin/python -m elicit.differential_check
.venv/bin/python -m elicit.robustness
```

`run_experiment` first checks the quadratic calibration, the weighted-quantile
strategy-proofness anchor, interval narrowing under dichotomous approval, and
the unnormalised utility-hiking example.

The continuous search combines a coarse grid, 4,096 seeded reports per actor,
targeted narrow reports, and CMA refinement. The four-actor check uses the
declared `[0, 42]` grid in steps of two. The three-actor check enumerates every
profile in its finite report sets. The differential check compares separate
implementations on 500 seeded random profiles.

## Outputs

- `data/elicitation_tradeoff.npz`
- `data/elicitation_joint_enumeration.json`
- `data/elicitation_robustness.npz`
- `figures/fig_elicitation_tradeoff.{pdf,png}`
- `figures/fig_elicitation_robustness.{pdf,png}`
- `reference_outputs/elicitation_experiment_manifest.md`
- `reference_outputs/elicitation_hand_check.md`
- `reference_outputs/elicitation_differential_check.md`
- `reference_outputs/elicitation_rule_comparison.md`

