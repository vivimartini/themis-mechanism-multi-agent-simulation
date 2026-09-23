"""Exhaust every report profile in a three-actor finite report game."""
from __future__ import annotations

import itertools
import json

import numpy as np

from elicit.formats import (
    FullReport,
    PeakReport,
    RegionReport,
    truthful_full,
    truthful_peaks,
    truthful_regions,
)
from elicit.measures import SPECS, actor_utility, select
from elicit.utilities import load_model
from paths import ELICIT_JOINT_JSON

ACTORS = ("CHINA", "INDIA", "INDONESIA")
PRICE_GRID = (0.0, 10.0, 20.0, 30.0)


def unique(items):
    return list(dict.fromkeys(items))


def action_sets(model, spec):
    if spec.format == "peak":
        truth = truthful_peaks(model)
        return [
            unique([truth[i], *(PeakReport(p) for p in PRICE_GRID)])
            for i in range(len(model.names))
        ]
    if spec.format == "region":
        truth = truthful_regions(model)
        grid_actions = [
            RegionReport(lower, upper)
            for lower in PRICE_GRID
            for upper in PRICE_GRID
            if lower <= upper
        ]
        return [
            unique([truth[i], *grid_actions]) for i in range(len(model.names))
        ]
    if spec.format == "full":
        truth = truthful_full(model)
        grid_actions = [
            FullReport(peak, upper)
            for peak in PRICE_GRID
            for upper in PRICE_GRID
            if peak < upper
        ]
        return [
            unique([truth[i], *grid_actions]) for i in range(len(model.names))
        ]
    raise ValueError(spec.format)


def dominant_truth_regret(model, spec):
    """Maximum gain from lying, over every finite profile of the other actors."""
    actions = action_sets(model, spec)
    n = len(model.names)
    max_regret = np.zeros(n)
    witness = [None] * n

    for actor in range(n):
        others = [j for j in range(n) if j != actor]
        for other_indices in itertools.product(
            *(range(len(actions[j])) for j in others)
        ):
            profile = [action[0] for action in actions]
            for j, index in zip(others, other_indices):
                profile[j] = actions[j][index]
            truthful_outcome = select(profile, model, spec)
            truthful_u = actor_utility(model, actor, truthful_outcome)
            for report in actions[actor]:
                deviated = list(profile)
                deviated[actor] = report
                outcome = select(deviated, model, spec)
                utility = actor_utility(model, actor, outcome)
                gain = utility - truthful_u
                if gain > max_regret[actor] + 1e-12:
                    max_regret[actor] = gain
                    witness[actor] = {
                        "others": list(other_indices),
                        "report": repr(report),
                        "truthful_price": truthful_outcome.price,
                        "deviated_price": outcome.price,
                    }
    return actions, max_regret, witness


def main() -> None:
    full = load_model(peak_fraction=0.50)
    indices = [full.names.index(name) for name in ACTORS]
    model = full.subset(indices)
    results = {
        "actors": list(model.names),
        "price_grid": list(PRICE_GRID),
        "peak_fraction": 0.50,
        "interpretation": (
            "Maximum truth-telling regret over every finite report profile of "
            "the other two actors; no optimiser is used."
        ),
        "formats": {},
    }
    for spec in SPECS:
        actions, regrets, witness = dominant_truth_regret(model, spec)
        profile_count = int(np.prod([len(a) for a in actions]))
        results["formats"][spec.label] = {
            "actions_per_actor": [len(a) for a in actions],
            "joint_profiles": profile_count,
            "max_truth_regret_by_actor": regrets.tolist(),
            "max_truth_regret": float(regrets.max()),
            "sum_max_truth_regret": float(regrets.sum()),
            "witness": witness,
        }
        print(
            f"{spec.label:<32} profiles={profile_count:4d} "
            f"max truth-regret={regrets.max():.6f} "
            f"sum={regrets.sum():.6f}"
        )
    ELICIT_JOINT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ELICIT_JOINT_JSON.write_text(json.dumps(results, indent=2))
    print(f"wrote {ELICIT_JOINT_JSON}")


if __name__ == "__main__":
    main()

