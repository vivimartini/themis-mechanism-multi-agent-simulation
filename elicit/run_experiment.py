"""Run the seeded price-only report-format experiment and draw its trade-off."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elicit.formats import FullReport, PeakReport, RegionReport
from elicit.manifest import write_manifest
from elicit.measures import SPECS, evaluate_spec
from elicit.utilities import DEFAULT_PEAK_FRACTIONS, load_model
from elicit.validate import main as validate
from paths import ELICIT_FIGURE_PDF, ELICIT_FIGURE_PNG, ELICIT_NPZ

SEED = 42
FOUR_ACTORS = ("CHINA", "UNITED STATES", "INDIA", "INDONESIA")
FOUR_GRID_MIN = 0.0
FOUR_GRID_STEP = 2.0


def run_nine_actor(budget: int, global_draws: int):
    losses = np.zeros((len(DEFAULT_PEAK_FRACTIONS), len(SPECS)))
    regrets = np.zeros_like(losses)
    prices = np.zeros_like(losses)
    actor_regrets = np.zeros((
        len(DEFAULT_PEAK_FRACTIONS), len(SPECS), 9
    ))
    best_report = np.full((len(DEFAULT_PEAK_FRACTIONS), len(SPECS), 9, 3), np.nan)
    truthful_utility = np.zeros_like(actor_regrets)
    best_utility = np.zeros_like(actor_regrets)
    best_price = np.zeros_like(actor_regrets)
    best_members = np.zeros((
        len(DEFAULT_PEAK_FRACTIONS), len(SPECS), 9, 9
    ), dtype=bool)

    rows = []
    for s, fraction in enumerate(DEFAULT_PEAK_FRACTIONS):
        model = load_model(peak_fraction=fraction)
        for k, spec in enumerate(SPECS):
            outcome, welfare, loss, responses = evaluate_spec(
                model, spec, budget=budget, seed=SEED,
                global_draws=global_draws,
            )
            values = np.array([response.regret for response in responses])
            losses[s, k] = loss
            regrets[s, k] = values.sum()
            prices[s, k] = outcome.price
            actor_regrets[s, k] = values
            for i, response in enumerate(responses):
                report = response.best_report
                if isinstance(report, FullReport):
                    best_report[s, k, i] = (
                        report.peak, report.upper, report.scale
                    )
                elif isinstance(report, PeakReport):
                    best_report[s, k, i, 0] = report.peak
                elif isinstance(report, RegionReport):
                    best_report[s, k, i, :2] = (
                        report.lower, report.upper
                    )
                truthful_utility[s, k, i] = response.truthful_utility
                best_utility[s, k, i] = response.best_utility
                best_price[s, k, i] = response.best_outcome.price
                best_members[s, k, i] = response.best_outcome.members
            rows.append({
                "peak fraction": fraction,
                "format / rule": spec.label,
                "truthful price": outcome.price,
                "truthful welfare": welfare,
                "welfare loss": loss,
                "total found regret": values.sum(),
                "weighted regret": float(np.dot(model.weights, values)),
                "max regret": values.max(),
            })
    details = {
        "best_report": best_report,
        "truthful_utility": truthful_utility,
        "best_utility": best_utility,
        "best_price": best_price,
        "best_members": best_members,
    }
    return losses, regrets, prices, actor_regrets, details, pd.DataFrame(rows)


def run_four_actor():
    base = load_model(peak_fraction=0.50)
    indices = [base.names.index(name) for name in FOUR_ACTORS]
    model = base.subset(indices)
    continuous_limit = max(float(model.thresholds.max()) * 1.25, 10.0)
    grid = np.arange(
        FOUR_GRID_MIN, continuous_limit + 0.5 * FOUR_GRID_STEP, FOUR_GRID_STEP
    )
    grid_max = float(grid[-1])
    actor_regrets = np.zeros((len(SPECS), len(FOUR_ACTORS)))
    prices = np.zeros(len(SPECS))
    welfares = np.zeros(len(SPECS))
    losses = np.zeros(len(SPECS))
    rows = []
    for k, spec in enumerate(SPECS):
        outcome, welfare, loss, responses = evaluate_spec(
            model, spec, grid_step=FOUR_GRID_STEP, refine=False, global_draws=0,
            strict_grid=True, seed=SEED,
        )
        values = np.array([response.regret for response in responses])
        actor_regrets[k] = values
        prices[k] = outcome.price
        welfares[k] = welfare
        losses[k] = loss
        rows.append({
            "format / rule": spec.label,
            "truthful price": outcome.price,
            "truthful welfare": welfare,
            "welfare loss": loss,
            "total grid regret": values.sum(),
        })
    return (
        actor_regrets, prices, welfares, losses,
        (FOUR_GRID_MIN, grid_max, FOUR_GRID_STEP), pd.DataFrame(rows),
    )


def draw(losses: np.ndarray, regrets: np.ndarray) -> None:
    labels = ("full", "peak", "region midpoint", "region max coverage")
    markers = ("o", "s", "^", "D")
    colours = ("#355f7d", "#4f8a63", "#b86b45", "#7d5a91")
    fig, axes = plt.subplots(
        1, len(DEFAULT_PEAK_FRACTIONS), figsize=(9.2, 3.1),
        sharex=True, sharey=True,
    )
    for s, (ax, fraction) in enumerate(zip(axes, DEFAULT_PEAK_FRACTIONS)):
        for k, label in enumerate(labels[:2]):
            ax.scatter(
                losses[s, k], regrets[s, k], marker=markers[k],
                color=colours[k], s=52, label=label, zorder=3,
            )
            ax.annotate(
                label, (losses[s, k], regrets[s, k]),
                xytext=(5, 4), textcoords="offset points", fontsize=7,
            )
        if np.allclose(losses[s, 2:], losses[s, 2]) and np.allclose(
            regrets[s, 2:], regrets[s, 2]
        ):
            ax.scatter(
                losses[s, 2], regrets[s, 2], marker="D",
                color=colours[3], s=52, zorder=3,
            )
            ax.annotate(
                "region rules (coincident)", (losses[s, 2], regrets[s, 2]),
                xytext=(5, -11), textcoords="offset points", fontsize=7,
            )
        else:
            region_top = max(regrets[s, 2], regrets[s, 3])
            near_top = region_top > 0.9 * regrets[s].max()
            offsets = ((20, -4), (20, -18)) if near_top else (
                (20, 8), (20, -10)
            )
            for k, offset in zip((2, 3), offsets):
                ax.scatter(
                    losses[s, k], regrets[s, k], marker=markers[k],
                    color=colours[k], s=52, label=labels[k], zorder=3,
                )
                ax.annotate(
                    labels[k], (losses[s, k], regrets[s, k]),
                    xytext=offset, textcoords="offset points", fontsize=7,
                )
        ax.set_title(f"peak fraction = {fraction:.2f}", fontsize=9)
        ax.set_xlabel("truthful welfare loss")
        ax.grid(color="#e5e5e5", linewidth=0.7, zorder=0)
    axes[0].set_ylabel("total found regret (lower bound)")
    fig.suptitle(
        "Report format trades information against manipulability",
        fontsize=10,
    )
    fig.tight_layout()
    ELICIT_FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ELICIT_FIGURE_PDF, bbox_inches="tight")
    fig.savefig(ELICIT_FIGURE_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget", type=int, default=300,
        help="CMA objective evaluations after each format-specific grid search",
    )
    parser.add_argument(
        "--global-draws", type=int, default=4096,
        help="seeded global report draws per actor before CMA refinement",
    )
    args = parser.parse_args(argv)

    print("=== model checks ===")
    validate()

    print("\n=== Phase A: calibrated nine-actor price-only game ===")
    losses, regrets, prices, per_actor, details, nine_table = run_nine_actor(
        args.budget, args.global_draws
    )
    print(nine_table.round(4).to_string(index=False))

    print("\nPer-actor regret at peak fraction 0.50:")
    central = pd.DataFrame(
        per_actor[1].T,
        index=load_model().names,
        columns=[spec.label for spec in SPECS],
    )
    print(central.round(4).to_string())

    four = run_four_actor()
    (four_regrets, four_prices, four_welfare, four_loss,
     four_grid, four_table) = four
    print(
        "\n=== Phase B: four-actor grid check "
        f"({four_grid[0]:g}–{four_grid[1]:g}, step {four_grid[2]:g}) ==="
    )
    print(four_table.round(4).to_string(index=False))

    ELICIT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ELICIT_NPZ,
        peak_fractions=np.array(DEFAULT_PEAK_FRACTIONS),
        spec_labels=np.array([spec.label for spec in SPECS], dtype="U40"),
        actor_names=np.array(load_model().names, dtype="U48"),
        welfare_loss=losses,
        total_regret=regrets,
        weighted_regret=per_actor @ load_model().weights,
        max_regret=per_actor.max(axis=-1),
        truthful_price=prices,
        actor_regret=per_actor,
        best_report=details["best_report"],
        truthful_utility=details["truthful_utility"],
        best_utility=details["best_utility"],
        best_response_price=details["best_price"],
        best_response_members=details["best_members"],
        four_actor_names=np.array(FOUR_ACTORS, dtype="U24"),
        four_actor_regret=four_regrets,
        four_actor_price=four_prices,
        four_actor_welfare=four_welfare,
        four_actor_welfare_loss=four_loss,
        four_actor_grid_min=np.array([four_grid[0]]),
        four_actor_grid_max=np.array([four_grid[1]]),
        four_actor_grid_step=np.array([four_grid[2]]),
        seed=np.array([SEED]),
        cma_budget=np.array([args.budget]),
        global_draws=np.array([args.global_draws]),
        regret_is_lower_bound=np.array([True]),
        full_scale_normalised=np.array([True]),
    )
    draw(losses, regrets)
    write_manifest()
    from elicit.hand_check import write_hand_check
    write_hand_check()
    print(f"\nwrote {ELICIT_NPZ}")
    print(f"wrote {ELICIT_FIGURE_PDF} and {ELICIT_FIGURE_PNG}")


if __name__ == "__main__":
    main()

