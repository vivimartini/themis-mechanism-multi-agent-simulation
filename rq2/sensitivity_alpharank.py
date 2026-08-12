"""Alpha-Rank population-size sensitivity."""
import numpy as np

from rq2 import psro_lite as psro


M_VALUES = (5, 10, 25, 50, 100, 200)
ALPHAS = (0.05, 0.5, 5.0)


def main():
    policies = psro.seed_policies({})
    profiles, utilities, shapes = psro.payoff_tensor(policies, {})
    truthful = profiles.index((0,) * len(shapes))

    print(f"{'m':>5} {'alpha':>7} {'truthful':>10} {'top mass':>10}  top profile")
    for m in M_VALUES:
        for alpha in ALPHAS:
            pi = psro.alpha_rank(profiles, utilities, shapes, alpha, m=m)
            top = int(np.argmax(pi))
            print(f"{m:5d} {alpha:7.2f} {pi[truthful]:10.6f} {pi[top]:10.6f}  "
                  f"{psro.profile_label(profiles, top)}")


if __name__ == "__main__":
    main()
