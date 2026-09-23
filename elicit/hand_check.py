"""Recompute one profitable region deviation directly from the calibration.

The calculation does not import the primary format, rule, or scoring modules.
It rebuilds the central scenario from the calibration CSV and checks a saved
best response from ``elicitation_tradeoff.npz``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from paths import ACTORS_CSV, ELICIT_HAND_CHECK, ELICIT_NPZ

COVERAGE = 0.8686437102493683
PEAK_FRACTION = 0.50
SCENARIO_INDEX = 1
SPEC_INDEX = 2  # region: intersection-midpoint


def select_intersection_midpoint(
    intervals: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray, float, float]:
    """Independent implementation of the greatest-weight intersection rule."""
    points = sorted(set(intervals[:, 0]) | set(intervals[:, 1]))
    candidates = []
    for point in points:
        members = ((intervals[:, 0] - 1e-12 <= point)
                   & (point <= intervals[:, 1] + 1e-12))
        candidates.append((weights[members].sum(), point, point, members))
    for left, right in zip(points[:-1], points[1:]):
        if right <= left:
            continue
        middle = 0.5 * (left + right)
        members = ((intervals[:, 0] - 1e-12 <= middle)
                   & (middle <= intervals[:, 1] + 1e-12))
        candidates.append((weights[members].sum(), left, right, members))
    _, left, right, members = max(
        candidates, key=lambda row: (row[0], row[2] - row[1], -row[1])
    )
    midpoints = intervals.mean(axis=1)
    price = float(np.clip(
        np.average(midpoints[members], weights=weights[members]), left, right
    ))
    realised = ((intervals[:, 0] - 1e-12 <= price)
                & (price <= intervals[:, 1] + 1e-12))
    return price, realised, float(left), float(right)


def true_surplus(peak: float, upper: float, price: float) -> float:
    return float(1.0 - ((price - peak) / (upper - peak)) ** 2)


def write_hand_check() -> None:
    frame = pd.read_csv(ACTORS_CSV)
    names = frame["name"].astype(str).tolist()
    emissions = frame["pop_m"].to_numpy(float) * frame["e"].to_numpy(float)
    weights = emissions / emissions.sum()
    upper = (
        frame["alpha_base"].to_numpy(float)
        + frame["alpha_cov"].to_numpy(float) * COVERAGE
    )
    peaks = PEAK_FRACTION * upper
    lower = np.maximum(0.0, 2.0 * peaks - upper)
    truthful_intervals = np.column_stack([lower, upper])

    with np.load(ELICIT_NPZ) as saved:
        regrets = saved["actor_regret"][SCENARIO_INDEX, SPEC_INDEX]
        actor = int(np.argmax(regrets))
        reported = saved["best_report"][SCENARIO_INDEX, SPEC_INDEX, actor, :2]
        stored_price = float(
            saved["best_response_price"][SCENARIO_INDEX, SPEC_INDEX, actor]
        )
        stored_truth_u = float(
            saved["truthful_utility"][SCENARIO_INDEX, SPEC_INDEX, actor]
        )
        stored_best_u = float(
            saved["best_utility"][SCENARIO_INDEX, SPEC_INDEX, actor]
        )
        stored_members = saved[
            "best_response_members"
        ][SCENARIO_INDEX, SPEC_INDEX, actor]

    truthful_price, truthful_members, _, _ = select_intersection_midpoint(
        truthful_intervals, weights
    )
    truthful_u = (
        true_surplus(peaks[actor], upper[actor], truthful_price)
        if truthful_members[actor] else 0.0
    )

    deviated = truthful_intervals.copy()
    deviated[actor] = reported
    dev_price, dev_members, overlap_lo, overlap_hi = (
        select_intersection_midpoint(deviated, weights)
    )
    best_u = (
        true_surplus(peaks[actor], upper[actor], dev_price)
        if dev_members[actor] else 0.0
    )
    regret = best_u - truthful_u

    assert abs(truthful_u - stored_truth_u) < 1e-8
    assert abs(dev_price - stored_price) < 1e-8
    assert abs(best_u - stored_best_u) < 1e-8
    assert abs(regret - regrets[actor]) < 1e-8
    assert np.array_equal(dev_members, stored_members)

    member_names = [name for name, member in zip(names, dev_members) if member]
    text = f"""# Arithmetic check: profitable region report

The calculation implements the interval rule and quadratic scoring directly from
`data/actors_baseline.csv`; it does not call `elicit.rules`,
`elicit.formats`, or `elicit.measures`.

## Case

- Scenario peak fraction: `{PEAK_FRACTION:.2f}`
- Rule: `region: intersection-midpoint`
- Deviator: `{names[actor]}`
- True quadratic: `1 - ((p - {peaks[actor]:.6f}) / ({upper[actor]:.6f} - {peaks[actor]:.6f}))^2`
- True acceptable interval: `[{lower[actor]:.6f}, {upper[actor]:.6f}]`

## Truthful profile

- Selected price: `{truthful_price:.6f}`
- Deviator participates: `{bool(truthful_members[actor])}`
- True utility (zero if out): `{truthful_u:.6f}`

## Constructive deviation

- Reported interval: `[{reported[0]:.6f}, {reported[1]:.6f}]`
- Greatest-weight common intersection selected by the rule:
  `[{overlap_lo:.6f}, {overlap_hi:.6f}]`
- Realised price after weighted-midpoint projection: `{dev_price:.6f}`
- Report-implied members: {", ".join(member_names)}
- Deviator participates: `{bool(dev_members[actor])}`
- True utility at realised price: `{best_u:.6f}`
- Profitable gain: `{best_u:.6f} - {truthful_u:.6f} = {regret:.6f}`

The recomputed price, membership, utility, and regret agree with the saved
search output to better than `1e-8`. This profitable deviation is sufficient to
show that the region rule is not strategy-proof; it does not require the search
to have found the global optimum.
"""
    ELICIT_HAND_CHECK.parent.mkdir(parents=True, exist_ok=True)
    ELICIT_HAND_CHECK.write_text(text)


if __name__ == "__main__":
    write_hand_check()
    print(f"wrote {ELICIT_HAND_CHECK}")

