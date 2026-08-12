"""RQ2 obstruction and vote-selling experiment."""
import numpy as np
import pandas as pd
import rq2.endogenous_coverage as R
from rq2.oracle import (cma_minimize, DEFAULT_SEED, DEFAULT_SIGMA0,
                        WARM_STARTS, EXIT_SHRINK_STARTS)

SEED = DEFAULT_SEED
SIGMA0 = DEFAULT_SIGMA0


# ------------------------------------------------------------------ obstruction
def obstruction(i, budget=900):
    """Minimise the mechanism objective c*p over actor i's report."""
    def neg_obj(x):                      # we minimise c*p, so neg_obj = +c*p
        ab = R.AB.copy(); ac = R.AC.copy()
        ab[i] += x[0]; ac[i] += x[1]
        out = R.solve_full(ab, ac)
        return 0.0 if out is None else out["c"] * out["p"]

    f, x = cma_minimize(neg_obj, budget, sigma0=SIGMA0, seed=SEED + i,
                        warm_starts=WARM_STARTS + EXIT_SHRINK_STARTS)
    ab = R.AB.copy(); ac = R.AC.copy(); ab[i] += x[0]; ac[i] += x[1]
    out = R.solve_full(ab, ac)
    return f, x, out


# ----------------------------------------------------------------- vote-selling
def collusive_surplus(a, b, budget=450):
    """max over B's report of Delta U_A + Delta U_B (transfer utility)."""
    uA0 = R.utility(a, R.TRUTHFUL, "transfer")
    uB0 = R.utility(b, R.TRUTHFUL, "transfer")

    def neg_joint(x):
        ab = R.AB.copy(); ac = R.AC.copy()
        ab[b] += x[0]; ac[b] += x[1]          # only the seller misreports
        out = R.solve_full(ab, ac)
        return -((R.utility(a, out, "transfer") - uA0)
                 + (R.utility(b, out, "transfer") - uB0))

    # Shared seed and no warm starts: the exact configuration behind the reported
    # 53-positive-pair / frontier-China 949.6 results (see reproducibility appendix).
    f, x = cma_minimize(neg_joint, budget, sigma0=SIGMA0, seed=SEED)
    ab = R.AB.copy(); ac = R.AC.copy(); ab[b] += x[0]; ac[b] += x[1]
    out = R.solve_full(ab, ac)
    dA = R.utility(a, out, "transfer") - uA0
    dB = R.utility(b, out, "transfer") - uB0
    return max(0.0, -f), dA, dB, x, out


if __name__ == "__main__":
    t = R.TRUTHFUL
    cp0 = t["c"] * t["p"]
    print(f"truthful objective c*p = {cp0:.3f}  (c={t['c']:.4f}, p={t['p']:.2f})\n")

    print("=== step 3: obstruction (minimise c*p) ===")
    rows = []
    for i, n in enumerate(R.names):
        f, x, out = obstruction(i)
        if out is None:
            desc, cost_pk, cost_tr = "COLLAPSE (no s.c. point)", np.nan, np.nan
        else:
            desc = f"p={out['p']:.2f}, c={out['c']:.3f}"
            cost_pk = R.utility(i, out, "peaked") - R.utility(i, t, "peaked")
            cost_tr = R.utility(i, out, "transfer") - R.utility(i, t, "transfer")
        rows.append((n, round(cp0 - f, 3), round(100 * (cp0 - f) / cp0, 1),
                     desc, None if np.isnan(cost_pk) else round(cost_pk, 2),
                     None if np.isnan(cost_tr) else round(cost_tr, 2)))
    tb = pd.DataFrame(rows, columns=["actor", "damage to c*p", "damage %",
                                     "worst outcome", "self dU peaked",
                                     "self dU transfer"])
    print(tb.to_string(index=False), "\n")

    print("=== step 4: pairwise vote-selling (transfer utility, TU condition) ===")
    rows = []
    for a in range(9):
        for b in range(9):
            if a == b:
                continue
            s, dA, dB, x, out = collusive_surplus(a, b)
            if s > 1e-6:
                rows.append((R.names[a], R.names[b], round(s, 2), round(dA, 2),
                             round(dB, 2),
                             # dB >= 0: seller already wants to lie; no side payment.
                             "aligned (no bribe needed)" if dB >= -1e-9
                             else f"purchase, min bribe {-dB:.2f}"))
    if rows:
        vb = pd.DataFrame(rows, columns=["buyer A", "seller B", "joint surplus",
                                         "dU_A", "dU_B", "type"])
        vb = vb.sort_values("joint surplus", ascending=False)
        print(vb.to_string(index=False))
        n_true = (vb["dU_B"] < -1e-9).sum()
        print(f"\npairs with positive joint surplus: {len(vb)} / 72; "
              f"true purchases (seller must be bribed): {n_true}")
    else:
        print("no pair with positive joint surplus: robust to pairwise "
              "vote-selling in the TU sense")
