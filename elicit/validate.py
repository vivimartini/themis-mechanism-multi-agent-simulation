"""Checks for the elicitation model and report rules."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from elicit.formats import PeakReport, truthful_full, truthful_peaks
from elicit.measures import SPECS, best_response
from elicit.rules import full_welfare, peak_quantile
from elicit.utilities import DEFAULT_PEAK_FRACTIONS, load_model


def validate_utility_construction() -> None:
    for fraction in DEFAULT_PEAK_FRACTIONS:
        model = load_model(peak_fraction=fraction)
        peak_values = [
            model.surplus(i, model.peak(i)) for i in range(len(model.names))
        ]
        upper_values = [
            model.surplus(i, model.region(i)[1]) for i in range(len(model.names))
        ]
        assert np.allclose(peak_values, 1.0)
        assert np.allclose(upper_values, 0.0)


def validate_moulin_anchor() -> None:
    """Truthful peaks have zero regret under the fixed weighted quantile."""
    for fraction in DEFAULT_PEAK_FRACTIONS:
        model = load_model(peak_fraction=fraction)
        reports = truthful_peaks(model)
        truthful = peak_quantile(
            reports, model.weights, model.coverage, model.peak_fraction,
            fixed_membership=True,
        )
        crossings = [
            shifted
            for report in reports
            for shifted in (
                max(0.0, report.peak - 1e-8),
                report.peak,
                report.peak + 1e-8,
            )
        ]
        grid = np.unique(np.r_[
            np.linspace(0, model.thresholds.max() * 1.25, 201),
            model.peaks, crossings,
        ])
        regrets = []
        for actor in range(len(model.names)):
            truthful_u = model.surplus(actor, truthful.price)
            best = truthful_u
            for value in grid:
                deviated = list(reports)
                deviated[actor] = PeakReport(float(value))
                outcome = peak_quantile(
                    deviated, model.weights, model.coverage,
                    model.peak_fraction, fixed_membership=True,
                )
                best = max(best, model.surplus(actor, outcome.price))
            regrets.append(max(0.0, best - truthful_u))
        assert max(regrets) < 1e-10, (fraction, regrets)


def validate_approval_anchor() -> None:
    """A truthful approver cannot improve on utility one by narrowing."""
    model = load_model(peak_fraction=0.50)
    for spec in SPECS[2:]:
        # Under truthful reports every actor approves the selected price, hence
        # receives the maximum dichotomous utility one. No interval narrowing,
        # sampled or otherwise, can improve on that upper bound.
        truthful = best_response(
            model, spec, 0, refine=False, global_draws=0,
            dichotomous=True, narrowing_only=True,
        ).best_outcome
        assert all(
            truthful.members[i] and model.dichotomous_surplus(i, truthful.price) == 1
            for i in range(len(model.names))
        )
        regrets = [
            best_response(
                model, spec, i, refine=False, global_draws=0, dichotomous=True,
                narrowing_only=True,
            ).regret
            for i in range(len(model.names))
        ]
        assert max(regrets) < 1e-10, (spec, regrets)


def validate_hiking_anchor() -> None:
    """Raw scale buys unbounded reported influence; normalisation removes it."""
    model = load_model(peak_fraction=0.50)
    base = truthful_full(model)
    actor = model.names.index("EUROPEAN UNION")
    objectives = []
    distances = []
    for scale in (1.0, 10.0, 100.0, 1000.0):
        reports = list(base)
        reports[actor] = replace(reports[actor], scale=scale)
        outcome = full_welfare(reports, model.weights, normalise=False)
        objectives.append(outcome.reported_objective)
        distances.append(abs(outcome.price - model.peak(actor)))
    assert all(b > a for a, b in zip(objectives, objectives[1:])), objectives
    assert objectives[-1] > 100 * objectives[0], objectives
    assert distances[-1] < distances[0], distances

    normalised_a = full_welfare(base, model.weights, normalise=True)
    hiked = list(base)
    hiked[actor] = replace(hiked[actor], scale=1e9)
    normalised_b = full_welfare(hiked, model.weights, normalise=True)
    assert abs(normalised_a.price - normalised_b.price) < 1e-12


def main() -> None:
    checks = (
        ("quadratic anchoring", validate_utility_construction),
        ("Moulin peak/quantile", validate_moulin_anchor),
        ("approval narrowing", validate_approval_anchor),
        ("unnormalised utility hiking", validate_hiking_anchor),
    )
    for label, check in checks:
        check()
        print(f"  [PASS] {label}")


if __name__ == "__main__":
    main()

