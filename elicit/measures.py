"""Welfare loss and unilateral regret for each report format."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from elicit.formats import (
    FullReport,
    PeakReport,
    RegionReport,
    truthful_full,
    truthful_peaks,
    truthful_regions,
)
from elicit.rules import (
    Outcome,
    full_welfare,
    peak_quantile,
    region_intersection_midpoint,
    region_max_coverage,
)
from elicit.utilities import QuadraticSurplusModel
from rq2.oracle import cma_minimize, DEFAULT_SEED

FormatName = Literal["full", "peak", "region"]
RuleName = Literal["welfare", "quantile", "intersection-midpoint", "max-coverage"]


@dataclass(frozen=True)
class ExperimentSpec:
    format: FormatName
    rule: RuleName

    @property
    def label(self) -> str:
        return f"{self.format}: {self.rule}"


@dataclass(frozen=True)
class RegretResult:
    regret: float
    truthful_utility: float
    best_utility: float
    best_report: object
    best_outcome: Outcome


SPECS = (
    ExperimentSpec("full", "welfare"),
    ExperimentSpec("peak", "quantile"),
    ExperimentSpec("region", "intersection-midpoint"),
    ExperimentSpec("region", "max-coverage"),
)


def truthful_reports(model: QuadraticSurplusModel, spec: ExperimentSpec) -> list:
    if spec.format == "full":
        return truthful_full(model)
    if spec.format == "peak":
        return truthful_peaks(model)
    if spec.format == "region":
        return truthful_regions(model)
    raise ValueError(spec.format)


def select(
    reports: Sequence,
    model: QuadraticSurplusModel,
    spec: ExperimentSpec,
    *,
    normalise_full: bool = True,
) -> Outcome:
    if spec == ExperimentSpec("full", "welfare"):
        return full_welfare(reports, model.weights, normalise=normalise_full)
    if spec == ExperimentSpec("peak", "quantile"):
        return peak_quantile(
            reports, model.weights, model.coverage, model.peak_fraction
        )
    if spec == ExperimentSpec("region", "intersection-midpoint"):
        return region_intersection_midpoint(reports, model.weights)
    if spec == ExperimentSpec("region", "max-coverage"):
        return region_max_coverage(reports, model.weights)
    raise ValueError(f"unsupported format/rule combination: {spec}")


def actor_utility(
    model: QuadraticSurplusModel,
    actor: int,
    outcome: Outcome,
    *,
    dichotomous: bool = False,
) -> float:
    if not outcome.members[actor]:
        return 0.0
    if dichotomous:
        return model.dichotomous_surplus(actor, outcome.price)
    return float(model.surplus(actor, outcome.price))


def realised_welfare(
    model: QuadraticSurplusModel,
    outcome: Outcome,
    *,
    dichotomous: bool = False,
) -> float:
    utilities = np.array([
        actor_utility(model, i, outcome, dichotomous=dichotomous)
        for i in range(len(model.names))
    ])
    return float(np.dot(model.weights, utilities))


def truthful_metrics(
    model: QuadraticSurplusModel,
    spec: ExperimentSpec,
) -> tuple[Outcome, float, float]:
    """Return truthful outcome, welfare, and loss against the full benchmark."""
    benchmark_out = select(truthful_full(model), model, SPECS[0])
    benchmark = realised_welfare(model, benchmark_out)
    outcome = select(truthful_reports(model, spec), model, spec)
    welfare = realised_welfare(model, outcome)
    return outcome, welfare, max(0.0, benchmark - welfare)


def _price_grid(limit: float, step: float | None) -> np.ndarray:
    if step is None:
        return np.linspace(0.0, limit, 17)
    return np.arange(0.0, limit + 0.5 * step, step)


def best_response(
    model: QuadraticSurplusModel,
    spec: ExperimentSpec,
    actor: int,
    *,
    budget: int = 300,
    seed: int = DEFAULT_SEED,
    grid_step: float | None = None,
    refine: bool = True,
    global_draws: int = 4096,
    strict_grid: bool = False,
    dichotomous: bool = False,
    narrowing_only: bool = False,
) -> RegretResult:
    """Global structured search followed by CMA refinement.

    Report rules are discontinuous when membership changes, so CMA is only a
    refinement. Dense narrow-report probes and seeded global draws are included
    before it. Returned regret remains a found lower bound, not an equilibrium
    certificate.
    """
    reports = truthful_reports(model, spec)
    truthful_outcome = select(reports, model, spec)
    truthful_u = actor_utility(
        model, actor, truthful_outcome, dichotomous=dichotomous
    )
    price_limit = max(float(model.thresholds.max()) * 1.25, 10.0)
    grid = _price_grid(price_limit, grid_step)
    best_u = truthful_u
    best_report = reports[actor]
    best_outcome = truthful_outcome

    def evaluate(report) -> float:
        nonlocal best_u, best_report, best_outcome
        profile = list(reports)
        profile[actor] = report
        outcome = select(profile, model, spec)
        utility = actor_utility(
            model, actor, outcome, dichotomous=dichotomous
        )
        if utility > best_u + 1e-12:
            best_u, best_report, best_outcome = utility, report, outcome
        return utility

    if spec.format == "peak":
        # A quantile changes only when the report crosses another report; include
        # those breakpoints and the actor's true peak in addition to the grid.
        candidates = set(float(x) for x in grid)
        if not strict_grid:
            candidates.add(model.peak(actor))
            for report in reports:
                candidates.update((max(0.0, report.peak - 1e-8), report.peak,
                                   min(price_limit, report.peak + 1e-8)))
        for price in sorted(candidates):
            evaluate(PeakReport(price))

    elif spec.format == "region":
        true_lo, true_hi = model.region(actor)
        region_grid = grid
        if narrowing_only:
            region_grid = np.unique(np.r_[
                np.linspace(true_lo, true_hi, 17), true_lo, true_hi
            ])
        for j, lower in enumerate(region_grid):
            for upper in region_grid[j:]:
                if narrowing_only and (
                    lower < true_lo - 1e-12 or upper > true_hi + 1e-12
                ):
                    continue
                evaluate(RegionReport(float(lower), float(upper)))

        if global_draws and not narrowing_only:
            rng = np.random.default_rng(seed + 1000 * SPECS.index(spec) + actor)
            draws = np.sort(
                rng.uniform(0.0, price_limit, size=(global_draws, 2)), axis=1
            )
            for lower, upper in draws:
                evaluate(RegionReport(float(lower), float(upper)))

        if refine and not narrowing_only:
            x0 = np.array([true_lo, true_hi])

            def objective(x):
                pair = np.clip(np.asarray(x, float), 0.0, price_limit)
                lower, upper = sorted(pair)
                return -evaluate(RegionReport(float(lower), float(upper)))

            cma_minimize(
                objective, budget, sigma0=max(1.0, price_limit / 8),
                seed=seed + actor, x0=x0,
                warm_starts=[np.array([best_report.lower, best_report.upper])],
            )

    elif spec.format == "full":
        scales = (0.25, 1.0, 4.0)
        for peak in grid:
            for upper in grid[grid > peak + 1e-9]:
                for scale in scales:
                    evaluate(FullReport(float(peak), float(upper), scale))

        if global_draws:
            rng = np.random.default_rng(seed + 1000 * SPECS.index(spec) + actor)
            draws = np.sort(
                rng.uniform(0.0, price_limit, size=(global_draws, 2)), axis=1
            )
            for peak, upper in draws:
                if upper - peak > 1e-9:
                    evaluate(FullReport(float(peak), float(upper), 1.0))

        if not strict_grid:
            # Narrow reports create membership discontinuities that a coarse
            # grid and local CMA can both miss. Probe every target price at
            # roughly five-cent resolution over several narrow widths.
            targets = np.linspace(
                0.0, price_limit, int(np.ceil(price_limit / 0.05)) + 1
            )
            for peak in targets:
                for width in (0.05, 0.10, 0.25, 0.50, 1.00):
                    upper = min(price_limit, peak + width)
                    if upper > peak + 1e-9:
                        evaluate(FullReport(float(peak), float(upper), 1.0))

        if refine:
            truth = reports[actor]
            x0 = np.array([truth.peak, truth.upper, 0.0])

            def decode(x) -> FullReport:
                pair = np.clip(np.asarray(x[:2], float), 0.0, price_limit)
                peak, upper = sorted(pair)
                if upper - peak < 1e-6:
                    upper = min(price_limit, peak + 1e-6)
                    if upper <= peak:
                        peak = max(0.0, upper - 1e-6)
                scale = float(np.exp(np.clip(x[2], -8.0, 8.0)))
                return FullReport(float(peak), float(upper), scale)

            best_full = best_report
            warm = np.array([
                best_full.peak, best_full.upper, np.log(best_full.scale)
            ])
            cma_minimize(
                lambda x: -evaluate(decode(x)),
                budget, sigma0=max(1.0, price_limit / 8),
                seed=seed + actor, x0=x0, warm_starts=[warm],
            )
    else:
        raise ValueError(spec.format)

    return RegretResult(
        regret=max(0.0, best_u - truthful_u),
        truthful_utility=truthful_u,
        best_utility=best_u,
        best_report=best_report,
        best_outcome=best_outcome,
    )


def evaluate_spec(
    model: QuadraticSurplusModel,
    spec: ExperimentSpec,
    **best_response_kwargs,
) -> tuple[Outcome, float, float, list[RegretResult]]:
    outcome, welfare, welfare_loss = truthful_metrics(model, spec)
    regrets = [
        best_response(model, spec, i, **best_response_kwargs)
        for i in range(len(model.names))
    ]
    return outcome, welfare, welfare_loss, regrets

