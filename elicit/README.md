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
- **Preferred price:** one scalar, aggregated by the fixed weighted-quantile
  rule.
- **Acceptable interval:** lower and upper bounds, evaluated using either the
  greatest-overlap midpoint or maximum participating emissions.

The robustness run also evaluates the emission-weighted median of interval
midpoints.

## Main result

For the central peak-location scenario:

- Full utility has zero truthful welfare loss and total found regret `0.9031`.
- Preferred-price reporting has truthful welfare loss `0.1988` and zero regret.
- The two original interval rules have welfare loss `0.3055` and total found
  regret `2.9695` and `2.9925`.
- Replacing those rules with the weighted median reduces interval regret to
  zero in the homogeneous case and `0.013`–`0.053` in three heterogeneous
  cases.

The large interval-report regrets are therefore specific to the selection rule,
not to interval reports alone. Continuous-format regrets are search lower
bounds, not exact optima.

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

