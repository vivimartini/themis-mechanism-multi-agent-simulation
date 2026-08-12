"""Squared-loss robustness check for the claim in Section 4.5."""
import numpy as np

import rq2.endogenous_coverage as R
from rq2.oracle import cma_minimize, DEFAULT_SEED, DEFAULT_SIGMA0, WARM_STARTS

BUDGET = 900


def regrets(kind):
    out = []
    for i in range(9):
        u0 = R.utility(i, R.TRUTHFUL, kind)

        def neg_u(x, i=i, kind=kind):
            ab = R.AB.copy(); ac = R.AC.copy()
            ab[i] += x[0]; ac[i] += x[1]
            return -R.utility(i, R.solve_full(ab, ac), kind)

        f, _ = cma_minimize(neg_u, BUDGET, sigma0=DEFAULT_SIGMA0,
                            seed=DEFAULT_SEED, warm_starts=WARM_STARTS)
        out.append(max(0.0, -f - u0))
    return np.array(out)


if __name__ == "__main__":
    pk, sq = regrets("peaked"), regrets("sqloss")
    tol = 1e-6
    print(f"{'actor':<40}{'peaked':>10}{'sqloss':>12}")
    for i in range(9):
        print(f"{R.names[i]:<40}{pk[i]:>10.4f}{sq[i]:>12.4f}")
    set_pk = {R.names[j] for j in np.where(pk > tol)[0]}
    set_sq = {R.names[j] for j in np.where(sq > tol)[0]}
    rank_pk = [R.names[j] for j in np.argsort(-pk) if pk[j] > tol]
    rank_sq = [R.names[j] for j in np.argsort(-sq) if sq[j] > tol]
    print(f"\nexploitable set identical : {set_pk == set_sq}")
    print(f"ordering identical        : {rank_pk == rank_sq}")
    print(f"peaked ordering : {rank_pk}")
    print(f"sqloss ordering : {rank_sq}")
    print(f"pivot (US) sqloss regret  : {sq[R.names.index('UNITED STATES')]:.6f}")
