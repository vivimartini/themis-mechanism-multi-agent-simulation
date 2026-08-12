"""Equal-budget random-search guardrail sensitivity."""
import numpy as np

from rq2 import coarse_solver as solver
from rq2 import endogenous_coverage as model


N_EVALUATIONS = 900
SEED = 3
DESIGNS = (
    ("baseline", {}),
    ("pool cap", {"pool_cap": 150000.0}),
    ("all guardrails", {"tminus_cap": 1.0, "cov_floor": 0.5,
                        "pool_cap": 150000.0}),
)


def random_regret(actor, options, rng):
    baseline = solver.solve(model.AB, model.AC, **options)
    utility = solver.u_peaked(actor, baseline, model.AB, model.AC)
    best = utility
    for _ in range(N_EVALUATIONS):
        delta = rng.uniform(-60, 60, 2)
        ab = model.AB.copy()
        ac = model.AC.copy()
        ab[actor] += delta[0]
        ac[actor] += delta[1]
        best = max(best, solver.u_peaked(actor, solver.solve(ab, ac, **options),
                                         model.AB, model.AC))
    return max(0.0, best - utility)


def main():
    print(f"{'design':<16} {'random NashConv':>16}")
    for name, options in DESIGNS:
        rng = np.random.default_rng(SEED)
        regrets = [random_regret(i, options, rng) for i in range(len(model.names))]
        print(f"{name:<16} {sum(regrets):16.3f}")


if __name__ == "__main__":
    main()
