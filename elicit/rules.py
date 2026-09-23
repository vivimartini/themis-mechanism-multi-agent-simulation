"""Outcome rules for full, peak, and acceptable-region reports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from elicit.formats import FullReport, PeakReport, RegionReport


@dataclass(frozen=True)
class Outcome:
    price: float
    members: np.ndarray
    reported_objective: float


def full_welfare(
    reports: Sequence[FullReport],
    weights: np.ndarray,
    *,
    normalise: bool = True,
) -> Outcome:
    """Maximise weighted positive reported surplus.

    Piecewise between reported zeros, the objective is a concave quadratic, so
    its maximiser and every participation breakpoint form a finite exact
    candidate set.
    """
    weights = np.asarray(weights, float)
    breaks = sorted({0.0, *(r.lower for r in reports), *(r.upper for r in reports)})
    candidates = set(breaks)
    amplitudes = np.ones(len(reports)) if normalise else np.array(
        [r.scale for r in reports], float
    )

    for left, right in zip(breaks[:-1], breaks[1:]):
        if right - left <= 1e-12:
            continue
        middle = 0.5 * (left + right)
        active = np.array([r.lower < middle < r.upper for r in reports])
        if not active.any():
            continue
        precision = np.array([
            weights[i] * amplitudes[i] / reports[i].half_width**2
            for i in range(len(reports))
        ])
        optimum = float(
            sum(precision[i] * reports[i].peak for i in np.flatnonzero(active))
            / precision[active].sum()
        )
        candidates.add(float(np.clip(optimum, left, right)))

    def objective(price: float) -> float:
        values = np.array([
            report.value(price, normalise=normalise) for report in reports
        ])
        return float(np.dot(weights, np.maximum(values, 0.0)))

    price = min(candidates, key=lambda p: (-objective(p), p))
    values = np.array([r.value(price, normalise=normalise) for r in reports])
    return Outcome(
        price=float(price),
        members=values >= -1e-12,
        reported_objective=objective(price),
    )


def peak_quantile(
    reports: Sequence[PeakReport],
    weights: np.ndarray,
    target_coverage: float,
    peak_fraction: float | np.ndarray,
    *,
    fixed_membership: bool = False,
) -> Outcome:
    """Lower weighted quantile with membership implied by the reported peak.

    The scenario's common peak fraction maps a scalar peak back to the
    corresponding normalised quadratic's positive region. ``fixed_membership``
    is retained only for the Moulin verification anchor.
    """
    values = np.array([r.peak for r in reports], float)
    weights = np.asarray(weights, float)
    order = np.argsort(-values, kind="stable")
    cumulative = np.cumsum(weights[order])
    pivot = int(np.searchsorted(cumulative, target_coverage - 1e-12))
    price = float(values[order[pivot]])
    objective = -float(np.dot(weights, np.abs(values - price)))
    if fixed_membership:
        members = np.ones(len(reports), dtype=bool)
    else:
        uppers = np.divide(
            values, peak_fraction,
            out=np.zeros_like(values), where=values > 0,
        )
        lowers = np.maximum(0.0, 2.0 * values - uppers)
        members = (lowers - 1e-12 <= price) & (price <= uppers + 1e-12)
    return Outcome(price, members, objective)


def _overlap_candidates(
    reports: Sequence[RegionReport],
    weights: np.ndarray,
) -> list[tuple[float, float, float, np.ndarray]]:
    """Return ``(coverage, left, right, members)`` overlap plateaus and points."""
    points = sorted({r.lower for r in reports} | {r.upper for r in reports})
    candidates: list[tuple[float, float, float, np.ndarray]] = []

    for point in points:
        members = np.array([r.accepts(point) for r in reports])
        candidates.append((float(weights[members].sum()), point, point, members))

    for left, right in zip(points[:-1], points[1:]):
        if right - left <= 1e-12:
            continue
        middle = 0.5 * (left + right)
        members = np.array([r.accepts(middle) for r in reports])
        candidates.append((float(weights[members].sum()), left, right, members))
    return candidates


def region_intersection_midpoint(
    reports: Sequence[RegionReport],
    weights: np.ndarray,
) -> Outcome:
    """Choose a weighted midpoint in the greatest-weight common intersection.

    If all intervals intersect, this is their common intersection. Otherwise,
    the rule uses the greatest-emissions compatible coalition. Ties prefer the
    widest intersection and then the lower interval.
    """
    weights = np.asarray(weights, float)
    coverage, left, right, members = max(
        _overlap_candidates(reports, weights),
        key=lambda row: (row[0], row[2] - row[1], -row[1]),
    )
    if members.any():
        mids = np.array([r.midpoint for r in reports])
        midpoint = float(np.average(mids[members], weights=weights[members]))
        price = float(np.clip(midpoint, left, right))
    else:
        price = float(left)
    realised = np.array([r.accepts(price) for r in reports])
    return Outcome(price, realised, float(weights[realised].sum()))


def region_max_coverage(
    reports: Sequence[RegionReport],
    weights: np.ndarray,
) -> Outcome:
    """Maximise participating emissions, breaking coverage ties toward price."""
    weights = np.asarray(weights, float)
    candidates = {
        value
        for report in reports
        for value in (report.lower, report.midpoint, report.upper)
    }

    def coverage(price: float) -> float:
        members = np.array([r.accepts(price) for r in reports])
        return float(weights[members].sum())

    price = max(candidates, key=lambda p: (coverage(p), p))
    members = np.array([r.accepts(price) for r in reports])
    return Outcome(float(price), members, float(weights[members].sum()))


def region_weighted_median_midpoints(
    reports: Sequence[RegionReport],
    weights: np.ndarray,
) -> Outcome:
    """Select the emission-weighted median of reported interval midpoints."""
    weights = np.asarray(weights, float)
    midpoints = np.array([report.midpoint for report in reports])
    order = np.argsort(midpoints, kind="stable")
    pivot = int(np.searchsorted(
        np.cumsum(weights[order]), 0.5 * weights.sum() - 1e-12
    ))
    price = float(midpoints[order[pivot]])
    members = np.array([report.accepts(price) for report in reports])
    return Outcome(price, members, float(weights[members].sum()))

