"""Every format and rule under homogeneous and heterogeneous peak locations.

Each rule is scored three ways: unweighted total regret, emissions-weighted
regret (the same weights as welfare), and the largest single-actor regret.
"""
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
    ELICIT_RULE_TABLE,
)

SEED = 42
GLOBAL_DRAWS = 4096
BUDGET = 300
RULES = (
    ExperimentSpec("full", "welfare"),
    ExperimentSpec("peak", "quantile"),
    ExperimentSpec("peak", "median"),
    ExperimentSpec("region", "intersection-midpoint"),
    ExperimentSpec("region", "max-coverage"),
    ExperimentSpec("region", "weighted-median"),
    ExperimentSpec("region", "themis"),
)
SHORT = (
    "full", "peak q=0.869", "peak median", "region midpoint",
    "region max coverage", "region median", "region c×p",
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


def draw(labels, welfare_loss, weighted, unweighted):
    x = np.arange(len(labels))
    width = 0.8 / len(RULES)
    colours = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharex=True)
    panels = (
        (welfare_loss, "truthful welfare loss"),
        (weighted, "emissions-weighted regret"),
        (unweighted, "unweighted total regret"),
    )
    for ax, (values, ylabel) in zip(axes, panels):
        for k, name in enumerate(SHORT):
            shift = (k - (len(RULES) - 1) / 2) * width
            ax.bar(x + shift, values[:, k], width, color=colours[k], label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, ncol=len(RULES),
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Report formats and rules at fixed coverage "
        "(regrets are found lower bounds)", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    ELICIT_ROBUSTNESS_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ELICIT_ROBUSTNESS_PDF, bbox_inches="tight")
    fig.savefig(ELICIT_ROBUSTNESS_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_table(labels, names, prices, losses, unweighted, weighted, worst,
                actor_regret) -> None:
    lines = [
        "# Elicitation rule comparison",
        "",
        "Fixed coverage, transfers off. Regrets are found lower bounds for the "
        "continuous formats. Peak rules infer membership from the "
        "population-average peak fraction.",
        "",
    ]
    for s, label in enumerate(labels):
        lines += [
            f"## {label}",
            "",
            "| rule | price | welfare loss | unweighted regret "
            "| weighted regret | max regret | largest-regret actor |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for k, name in enumerate(SHORT):
            top = int(np.argmax(actor_regret[s, k]))
            who = names[top] if actor_regret[s, k, top] > 1e-9 else "—"
            lines.append(
                f"| {name} | {prices[s, k]:.4f} | {losses[s, k]:.4f} "
                f"| {unweighted[s, k]:.4f} | {weighted[s, k]:.4f} "
                f"| {worst[s, k]:.4f} | {who} |"
            )
        lines.append("")
    ELICIT_RULE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    ELICIT_RULE_TABLE.write_text("\n".join(lines))


def main():
    base = load_model()
    cases = scenarios(len(base.names))
    shape = (len(cases), len(RULES))
    losses, prices = np.zeros(shape), np.zeros(shape)
    unweighted, weighted, worst = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    actor_regret = np.zeros(shape + (len(base.names),))
    rows = []
    for s, (label, fractions) in enumerate(cases):
        model = load_model(peak_fraction=fractions)
        for k, spec in enumerate(RULES):
            outcome, _, loss, responses = evaluate_spec(
                model, spec, budget=BUDGET, seed=SEED,
                global_draws=GLOBAL_DRAWS,
            )
            values = np.array([response.regret for response in responses])
            losses[s, k] = loss
            prices[s, k] = outcome.price
            actor_regret[s, k] = values
            unweighted[s, k] = values.sum()
            weighted[s, k] = float(np.dot(model.weights, values))
            worst[s, k] = values.max()
            rows.append((label, SHORT[k], outcome.price, loss, unweighted[s, k],
                         weighted[s, k], worst[s, k]))

    table = pd.DataFrame(rows, columns=[
        "peak scenario", "rule", "truthful price", "welfare loss",
        "unweighted regret", "weighted regret", "max regret",
    ])
    print(table.round(4).to_string(index=False))

    labels = [case[0] for case in cases]
    ELICIT_ROBUSTNESS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ELICIT_ROBUSTNESS_NPZ,
        scenario_labels=np.array(labels, dtype="U32"),
        peak_fractions=np.stack([case[1] for case in cases]),
        rule_labels=np.array([spec.label for spec in RULES], dtype="U40"),
        actor_names=np.array(base.names, dtype="U48"),
        weights=base.weights,
        truthful_price=prices,
        welfare_loss=losses,
        total_regret=unweighted,
        weighted_regret=weighted,
        max_regret=worst,
        actor_regret=actor_regret,
        regret_is_lower_bound=np.array([True]),
        seed=np.array([SEED]),
        global_draws=np.array([GLOBAL_DRAWS]),
        cma_budget=np.array([BUDGET]),
    )
    draw(labels, losses, weighted, unweighted)
    write_table(labels, base.names, prices, losses, unweighted, weighted,
                worst, actor_regret)
    print(f"wrote {ELICIT_ROBUSTNESS_NPZ}")
    print(f"wrote {ELICIT_ROBUSTNESS_PDF} and {ELICIT_ROBUSTNESS_PNG}")
    print(f"wrote {ELICIT_RULE_TABLE}")


if __name__ == "__main__":
    main()
