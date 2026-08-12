"""Leave-one-attack-out Alpha-Rank ablation."""
import numpy as np

from paths import ALPHARANK_ABLATION_NPZ
from rq2 import psro_lite as psro


ALPHA = 5.0
M_POP = 50


def outcome(profile, policies):
    ab = psro.R.AB.copy()
    ac = psro.R.AC.copy()
    for actor, policy in enumerate(profile):
        ab[actor] += policies[actor][policy][0]
        ac[actor] += policies[actor][policy][1]
    return psro.F.solve(ab, ac)


def main():
    policies = psro.seed_policies({})
    profiles, utilities, shapes = psro.payoff_tensor(policies, {})
    active = [i for i, n_policies in enumerate(shapes) if n_policies > 1]
    rows = []

    print(f"{'removed attack':<20} {'truthful mass':>14} {'c':>8} {'p':>8}  top profile")
    for actor in active:
        keep = np.array([profile[actor] == 0 for profile in profiles])
        sub_profiles = [profile for profile, include in zip(profiles, keep) if include]
        sub_utilities = utilities[keep]
        sub_shapes = list(shapes)
        sub_shapes[actor] = 1
        pi = psro.alpha_rank(sub_profiles, sub_utilities, sub_shapes, ALPHA, m=M_POP)
        truthful = sub_profiles.index((0,) * len(shapes))
        top = int(np.argmax(pi))
        top_profile = sub_profiles[top]
        top_outcome = outcome(top_profile, policies)
        label = psro.profile_label(sub_profiles, top)
        rows.append((psro.R.names[actor], pi[truthful], top_outcome["c"],
                     top_outcome["p"], label))
        print(f"{psro.R.names[actor][:20]:<20} {pi[truthful]:14.6f} "
              f"{top_outcome['c']:8.6f} {top_outcome['p']:8.6f}  {label}")

    ALPHARANK_ABLATION_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ALPHARANK_ABLATION_NPZ,
             actors=np.array([row[0] for row in rows], dtype="U40"),
             truthful_mass=np.array([row[1] for row in rows]),
             top_c=np.array([row[2] for row in rows]),
             top_p=np.array([row[3] for row in rows]),
             top_labels=np.array([row[4] for row in rows], dtype="U80"))
    print(f"saved {ALPHARANK_ABLATION_NPZ}")


if __name__ == "__main__":
    main()
