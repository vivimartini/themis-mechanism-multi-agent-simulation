"""Compare the primary rules with separate reference implementations."""
from __future__ import annotations

import numpy as np

from elicit.formats import FullReport, PeakReport, RegionReport
from elicit.rules import (
    full_welfare,
    peak_quantile,
    region_intersection_midpoint,
    region_max_coverage,
)
from elicit.utilities import load_model
from paths import ELICIT_DIFFERENTIAL

SEED = 20260923
N_PROFILES = 500


def independent_full(reports, weights):
    zeros = sorted({0.0, *(r.lower for r in reports), *(r.upper for r in reports)})
    candidate_prices = set(zeros)
    for left, right in zip(zeros[:-1], zeros[1:]):
        middle = (left + right) / 2
        active = [i for i, r in enumerate(reports)
                  if r.lower < middle < r.upper]
        if not active:
            continue
        inverse_variance = np.array([
            weights[i] / reports[i].half_width**2 for i in active
        ])
        centre = np.average(
            [reports[i].peak for i in active], weights=inverse_variance
        )
        candidate_prices.add(float(np.clip(centre, left, right)))

    def reported_value(price):
        values = [
            max(0.0, 1.0 - ((price - r.peak) / r.half_width) ** 2)
            for r in reports
        ]
        return float(np.dot(weights, values))

    price = min(candidate_prices, key=lambda p: (-reported_value(p), p))
    raw = np.array([
        1.0 - ((price - r.peak) / r.half_width) ** 2 for r in reports
    ])
    return price, raw >= -1e-12


def independent_peak(reports, weights, coverage, fractions):
    values = np.array([r.peak for r in reports])
    order = np.argsort(-values, kind="stable")
    pivot = np.searchsorted(np.cumsum(weights[order]), coverage - 1e-12)
    price = float(values[order[pivot]])
    upper = np.divide(
        values, fractions, out=np.zeros_like(values), where=values > 0
    )
    lower = np.maximum(0.0, 2 * values - upper)
    members = (lower - 1e-12 <= price) & (price <= upper + 1e-12)
    return price, members


def independent_region_midpoint(reports, weights):
    intervals = np.array([(r.lower, r.upper) for r in reports])
    points = sorted(set(intervals[:, 0]) | set(intervals[:, 1]))
    candidates = []
    for point in points:
        members = ((intervals[:, 0] - 1e-12 <= point)
                   & (point <= intervals[:, 1] + 1e-12))
        candidates.append((weights[members].sum(), point, point, members))
    for left, right in zip(points[:-1], points[1:]):
        middle = (left + right) / 2
        members = ((intervals[:, 0] - 1e-12 <= middle)
                   & (middle <= intervals[:, 1] + 1e-12))
        candidates.append((weights[members].sum(), left, right, members))
    _, left, right, members = max(
        candidates, key=lambda row: (row[0], row[2] - row[1], -row[1])
    )
    price = float(np.clip(
        np.average(intervals[members].mean(axis=1), weights=weights[members]),
        left, right,
    ))
    realised = ((intervals[:, 0] - 1e-12 <= price)
                & (price <= intervals[:, 1] + 1e-12))
    return price, realised


def independent_region_coverage(reports, weights):
    candidates = {
        value for r in reports for value in (r.lower, r.midpoint, r.upper)
    }

    def covered(price):
        return float(sum(
            weights[i] for i, r in enumerate(reports)
            if r.lower - 1e-12 <= price <= r.upper + 1e-12
        ))

    price = max(candidates, key=lambda p: (covered(p), p))
    members = np.array([
        r.lower - 1e-12 <= price <= r.upper + 1e-12 for r in reports
    ])
    return float(price), members


def independent_welfare(model, price, members):
    peaks = np.asarray(model.peak_fractions) * model.thresholds
    widths = model.thresholds - peaks
    utilities = 1.0 - ((price - peaks) / widths) ** 2
    return float(np.dot(model.weights, np.where(members, utilities, 0.0)))


def random_profiles(model, rng):
    limit = float(model.thresholds.max() * 1.25)
    for _ in range(N_PROFILES):
        pairs = np.sort(rng.uniform(0, limit, size=(len(model.names), 2)), axis=1)
        full = [
            FullReport(float(peak), float(upper))
            for peak, upper in pairs
            if upper > peak
        ]
        if len(full) != len(model.names):
            continue
        peaks = [PeakReport(float(value)) for value in rng.uniform(
            0, limit, size=len(model.names)
        )]
        intervals = np.sort(
            rng.uniform(0, limit, size=(len(model.names), 2)), axis=1
        )
        regions = [RegionReport(float(lo), float(hi)) for lo, hi in intervals]
        yield full, peaks, regions


def main() -> None:
    model = load_model(peak_fraction=0.50)
    rng = np.random.default_rng(SEED)
    stats = {
        name: {"price": 0.0, "welfare": 0.0, "member_mismatches": 0}
        for name in ("full", "peak", "region midpoint", "region max coverage")
    }
    count = 0
    for full, peaks, regions in random_profiles(model, rng):
        comparisons = (
            ("full", full_welfare(full, model.weights),
             independent_full(full, model.weights)),
            ("peak", peak_quantile(
                peaks, model.weights, model.coverage, model.peak_fractions
             ), independent_peak(
                peaks, model.weights, model.coverage, model.peak_fractions
             )),
            ("region midpoint", region_intersection_midpoint(
                regions, model.weights
             ), independent_region_midpoint(regions, model.weights)),
            ("region max coverage", region_max_coverage(
                regions, model.weights
             ), independent_region_coverage(regions, model.weights)),
        )
        for name, primary, independent in comparisons:
            price, members = independent
            primary_welfare = independent_welfare(
                model, primary.price, primary.members
            )
            independent_value = independent_welfare(model, price, members)
            stats[name]["price"] = max(
                stats[name]["price"], abs(primary.price - price)
            )
            stats[name]["welfare"] = max(
                stats[name]["welfare"],
                abs(primary_welfare - independent_value),
            )
            stats[name]["member_mismatches"] += int(
                not np.array_equal(primary.members, members)
            )
        count += 1

    for name, values in stats.items():
        assert values["price"] < 1e-9, (name, values)
        assert values["welfare"] < 1e-9, (name, values)
        assert values["member_mismatches"] == 0, (name, values)

    rows = "\n".join(
        f"| {name} | {values['price']:.2e} | {values['welfare']:.2e} | "
        f"{values['member_mismatches']} |"
        for name, values in stats.items()
    )
    text = f"""# Rule and scoring cross-check

- Profiles: {count}
- Seed: {SEED}
- Reference implementations: `elicit/differential_check.py`

| Rule | max absolute price difference | max absolute welfare difference | membership mismatches |
|---|---:|---:|---:|
{rows}

All comparisons passed at tolerance `1e-9`.
"""
    ELICIT_DIFFERENTIAL.parent.mkdir(parents=True, exist_ok=True)
    ELICIT_DIFFERENTIAL.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

