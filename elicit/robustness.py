"""Region-rule and heterogeneous-peak robustness checks."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elicit.measures import ExperimentSpec, evaluate_spec
from elicit.utilities import load_model
from paths import (
    ELICIT_ROBUSTNESS_NPZ,
    ELICIT_ROBUSTNESS_PDF,
    ELICIT_ROBUSTNESS_PNG,
)

SEED = 42
GLOBAL_DRAWS = 4096
BUDGET = 300
RULES = (
    ExperimentSpec("region", "intersection-midpoint"),
    ExperimentSpec("region", "max-coverage"),
    ExperimentSpec("region", "weighted-median"),
)


def scenarios(n: int):
    values = np.linspace(0.35, 0.65, n)
    shuffled = values.copy()
    np.random.default_rng(SEED).shuffle(shuffled)
    return (
        ("homogeneous 0.50", np.full(n, 0.50)),
        ("heterogeneous ascending", values),
        ("heterogeneous descending", values[::-1]),
        ("heterogeneous seeded", shuffled),
    )


def draw(labels, welfare_loss, regret):
    x = np.arange(len(labels))
    width = 0.24
    colours = ("#7d5a91", "#b86b45", "#4f8a63")
    short = ("intersection midpoint", "max coverage", "weighted median")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for k, (name, colour) in enumerate(zip(short, colours)):
        shift = (k - 1) * width
        axes[0].bar(x + shift, welfare_loss[:, k], width, color=colour, label=name)
        axes[1].bar(x + shift, regret[:, k], width, color=colour, label=name)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    axes[0].set_ylabel("truthful welfare loss")
    axes[0].set_title("information loss")
    axes[1].set_ylabel("total found regret (lower bound)")
    axes[1].set_title("manipulability")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Region-report conclusion under rule and peak heterogeneity")
    fig.tight_layout()
    ELICIT_ROBUSTNESS_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ELICIT_ROBUSTNESS_PDF, bbox_inches="tight")
    fig.savefig(ELICIT_ROBUSTNESS_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    base = load_model()
    cases = scenarios(len(base.names))
    losses = np.zeros((len(cases), len(RULES)))
    regrets = np.zeros_like(losses)
    prices = np.zeros_like(losses)
    rows = []
    for s, (label, fractions) in enumerate(cases):
        model = load_model(peak_fraction=fractions)
        for k, spec in enumerate(RULES):
            outcome, _, loss, responses = evaluate_spec(
                model, spec, budget=BUDGET, seed=SEED,
                global_draws=GLOBAL_DRAWS,
            )
            total = sum(response.regret for response in responses)
            losses[s, k] = loss
            regrets[s, k] = total
            prices[s, k] = outcome.price
            rows.append((label, spec.rule, outcome.price, loss, total))

    table = pd.DataFrame(rows, columns=[
        "peak scenario", "region rule", "truthful price",
        "welfare loss", "total found regret",
    ])
    print(table.round(4).to_string(index=False))

    ELICIT_ROBUSTNESS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ELICIT_ROBUSTNESS_NPZ,
        scenario_labels=np.array([case[0] for case in cases], dtype="U32"),
        peak_fractions=np.stack([case[1] for case in cases]),
        rule_labels=np.array([spec.rule for spec in RULES], dtype="U32"),
        truthful_price=prices,
        welfare_loss=losses,
        total_regret=regrets,
        regret_is_lower_bound=np.array([True]),
        seed=np.array([SEED]),
        global_draws=np.array([GLOBAL_DRAWS]),
        cma_budget=np.array([BUDGET]),
    )
    draw([case[0] for case in cases], losses, regrets)
    print(f"wrote {ELICIT_ROBUSTNESS_NPZ}")
    print(f"wrote {ELICIT_ROBUSTNESS_PDF} and {ELICIT_ROBUSTNESS_PNG}")


if __name__ == "__main__":
    main()

