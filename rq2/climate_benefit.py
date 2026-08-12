"""RQ2 climate-benefit sensitivity analysis."""
import numpy as np
import pandas as pd

import rq2.endogenous_coverage as R
import rq2.coarse_solver as F
from rq2.oracle import (cma_minimize, DEFAULT_SEED, DEFAULT_SIGMA0,
                        WARM_STARTS, EXIT_SHRINK_STARTS)
from paths import CLIMATE_NPZ

B_GRID = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
POP = R.pop                      # millions
CP0 = R.TRUTHFUL["c"] * R.TRUTHFUL["p"]


def cp(out):
    return 0.0 if out is None else out["c"] * out["p"]


def u_climate(i, out, B):
    """Transfer utility plus B * (c*p), in EUR per capita."""
    if out is None:
        return -1e6
    return R.utility(i, out, "transfer") + B * cp(out)


# ------------------------------------------------------- A: analytic break-even
def obstruction_breakeven():
    """B* for the pure obstruction deviation (minimise c*p), per actor."""
    rows = []
    for i, n in enumerate(R.names):
        def neg_cp(x, i=i):
            ab = R.AB.copy(); ac = R.AC.copy()
            ab[i] += x[0]; ac[i] += x[1]
            return cp(R.solve_full(ab, ac))

        f, x = cma_minimize(neg_cp, 900, sigma0=DEFAULT_SIGMA0, seed=DEFAULT_SEED + i,
                            warm_starts=WARM_STARTS + EXIT_SHRINK_STARTS)
        ab = R.AB.copy(); ac = R.AC.copy(); ab[i] += x[0]; ac[i] += x[1]
        out = R.solve_full(ab, ac)
        d_cp = cp(out) - CP0
        d_tr = R.utility(i, out, "transfer") - R.utility(i, R.TRUTHFUL, "transfer")
        if d_cp < -1e-9 and d_tr > 0:
            bstar = d_tr / (-d_cp)
            verdict = f"{bstar:.3f}"
        elif d_tr <= 0:
            bstar = 0.0
            verdict = "never (costs transfers too)"
        else:
            bstar = np.inf
            verdict = "always (no climate cost)"
        rows.append((n, round(100 * (-d_cp) / CP0, 1), round(d_tr, 2),
                     bstar, verdict))
    return pd.DataFrame(rows, columns=[
        "actor", "damage to c*p (%)", "transfer gain (EUR/cap)",
        "B* raw", "break-even B* (EUR/cap per unit c*p)"])


# ------------------------------------------- B: re-optimised attack under climate
def reoptimised(B, budget=300):
    """Best deviation for each actor under u = transfer + B*(c*p), coarse grid."""
    t0 = F.solve(R.AB, R.AC)
    rows = []
    for i in range(9):
        u0 = R.utility(i, t0, "transfer") + B * cp(t0)

        def neg_u(x, i=i):
            ab = R.AB.copy(); ac = R.AC.copy()
            ab[i] += x[0]; ac[i] += x[1]
            return -u_climate(i, F.solve(ab, ac), B)

        f, x = cma_minimize(neg_u, budget, sigma0=DEFAULT_SIGMA0,
                            seed=DEFAULT_SEED + i,
                            warm_starts=WARM_STARTS + EXIT_SHRINK_STARTS)
        ab = R.AB.copy(); ac = R.AC.copy(); ab[i] += x[0]; ac[i] += x[1]
        o = F.solve(ab, ac)
        rows.append(dict(actor=R.names[i], regret=max(0.0, -f - u0),
                         d_cp=cp(o) - cp(t0)))
    return pd.DataFrame(rows)


# --------------------------------------------------- C: vote-selling with climate
def deal(a, b, B):
    """Joint surplus of buyer a / seller b, per capita and population-weighted.

    The per-capita search is delegated to obstruction_voteselling.collusive_surplus
    so the EUR/cap column reproduces the published table exactly; this module only
    re-weights it and adds the climate term.
    """
    from rq2.obstruction_voteselling import collusive_surplus
    percap, dA, dB, x, o = collusive_surplus(a, b)
    d_cp = cp(o) - CP0
    total = (dA * POP[a] + dB * POP[b]) / 1000.0           # EUR bn
    return dict(percap=percap, total=total, dA=dA, dB=dB, d_cp=d_cp,
                percap_climate=percap + B * d_cp * 2,
                total_climate=total + B * d_cp * (POP[a] + POP[b]) / 1000.0)


def main():
    t = R.TRUTHFUL
    print("=== A. break-even climate valuation for obstruction ===")
    print(f"truthful objective c*p = {CP0:.3f}\n")
    tb = obstruction_breakeven()
    print(tb.drop(columns=["B* raw"]).to_string(index=False))
    print()

    print("=== B. re-optimised attacks under a climate term (coarse grid) ===")
    print("A deviator facing a real climate cost does not stop attacking; it")
    print("switches to attacks that spare the objective. Regret is in EUR/cap.\n")
    recs = []
    for B in B_GRID:
        d = reoptimised(B)
        n_att = int((d["regret"] > 0.01).sum())
        n_dam = int(((d["regret"] > 0.01) & (d["d_cp"] < -0.01)).sum())
        recs.append((B, n_att, n_dam, round(d["regret"].sum(), 2),
                     round(d.loc[d["regret"].idxmax(), "regret"], 2),
                     d.loc[d["regret"].idxmax(), "actor"][:16]))
    rb = pd.DataFrame(recs, columns=["B", "actors with regret", "of which damaging",
                                     "total regret", "max regret", "worst actor"])
    print(rb.to_string(index=False))
    print()

    print("=== C. vote-selling: per-capita vs population-weighted ===")
    print("The published joint surplus sums EUR/capita across countries of very")
    print("different size. Totals below are EUR bn and are the meaningful figure.\n")
    pairs = [("LOW-CARBON FRONTIER", "CHINA"), ("LOW-CARBON FRONTIER", "INDIA"),
             ("UNITED STATES", "INDIA"), ("LOW-CARBON FRONTIER", "EUROPEAN UNION")]
    rows = []
    for an, bn in pairs:
        a, b = R.names.index(an), R.names.index(bn)
        d = deal(a, b, B=1.0)
        rows.append((an[:14], bn[:14], round(d["percap"], 1), round(d["total"], 1),
                     round(d["d_cp"], 2), round(d["percap_climate"], 1),
                     round(d["total_climate"], 1)))
    cb = pd.DataFrame(rows, columns=["buyer", "seller", "surplus EUR/cap",
                                     "surplus EUR bn", "d(c*p)",
                                     "EUR/cap at B=1", "EUR bn at B=1"])
    print(cb.to_string(index=False))
    print()

    np.savez(CLIMATE_NPZ,
             names=np.array(R.names, dtype="U40"),
             bstar=tb["B* raw"].to_numpy(float),
             damage_pct=tb["damage to c*p (%)"].to_numpy(float),
             transfer_gain=tb["transfer gain (EUR/cap)"].to_numpy(float),
             B_grid=B_GRID,
             n_attack=rb["actors with regret"].to_numpy(float),
             n_damaging=rb["of which damaging"].to_numpy(float),
             total_regret=rb["total regret"].to_numpy(float),
             deal_percap=cb["surplus EUR/cap"].to_numpy(float),
             deal_total=cb["surplus EUR bn"].to_numpy(float),
             deal_labels=np.array([f"{r[0]}<-{r[1]}" for r in rows], dtype="U32"))
    print(f"saved {CLIMATE_NPZ}")


if __name__ == "__main__":
    main()
