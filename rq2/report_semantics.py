"""RQ2 report-semantics experiment."""
import numpy as np
import pandas as pd

import rq2.endogenous_coverage as R
import rq2.fixed_coverage as FC
from rq2.oracle import cma_minimize, DEFAULT_SEED, DEFAULT_SIGMA0, WARM_STARTS
from paths import SEMANTICS_NPZ

BUDGET = 900
SEMANTICS = ("peak", "cap", "transfer")


# --------------------------------------------------------------- fixed coverage
def u_fixed(i, reports, kind):
    """Utility at fixed (c*, T+, T-); the mechanism is FC's quantile price rule.

    `cap` membership is decided by the REPORT (as the mechanism must), while the
    surplus is measured against the TRUE cap FC.TRUTH[i]. A liar can therefore be
    dragged into a deal above its true acceptable price and score negative.
    """
    p, members = FC.outcome(reports)
    if kind == "peak":
        return -abs(p - FC.TRUTH[i])
    if kind == "cap":
        return (FC.TRUTH[i] - p) if members[i] else 0.0
    if kind == "transfer":
        return FC.utility(i, reports, "transfer")
    raise ValueError(kind)


def regret_fixed(i, kind, budget=BUDGET):
    base = FC.TRUTH.copy()
    u0 = u_fixed(i, base, kind)

    def neg_u(x):
        r = base.copy()
        r[i] = FC.scalar_report(i, x[0], x[1])
        return -u_fixed(i, r, kind)

    f, x = cma_minimize(neg_u, budget, sigma0=DEFAULT_SIGMA0,
                        seed=DEFAULT_SEED, warm_starts=WARM_STARTS)
    return max(0.0, -f - u0), x, u0, -f


# ---------------------------------------------------------- endogenous coverage
def u_endog(i, out, kind):
    if out is None:
        return -1e6
    if kind in ("peak", "transfer"):
        return R.utility(i, out, "peaked" if kind == "peak" else "transfer")
    if kind == "cap":
        if not out["members"][i]:
            return 0.0
        w = R.true_willingness(i, out["c"], out["Tplus"], out["Tminus"])
        return w - out["p"]
    raise ValueError(kind)


def regret_endog(i, kind, budget=BUDGET):
    u0 = u_endog(i, R.TRUTHFUL, kind)

    def neg_u(x):
        ab = R.AB.copy(); ac = R.AC.copy()
        ab[i] += x[0]; ac[i] += x[1]
        return -u_endog(i, R.solve_full(ab, ac), kind)

    f, x = cma_minimize(neg_u, budget, sigma0=DEFAULT_SIGMA0,
                        seed=DEFAULT_SEED, warm_starts=WARM_STARTS)
    return max(0.0, -f - u0), x, u0, -f


# ------------------------------------------------- price-direction of preference
def dudp_table():
    """d(transfer utility)/dp per actor at the truthful operating point.

    Transfers are t+- = T+- * p * |e_i - ebar|, so the derivative is exact and
    constant in p: contributors -T+ (e_i - ebar), beneficiaries +T- (ebar - e_i).
    """
    t = R.TRUTHFUL
    Tp, Tm, p = t["Tplus"], t["Tminus"], t["p"]
    rows = []
    for i, n in enumerate(R.names):
        if R.contrib[i]:
            d = -Tp * (R.e[i] - R.EBAR)
            role = "contributor"
        else:
            d = Tm * (R.EBAR - R.e[i])
            role = "beneficiary"
        member = bool(t["members"][i])
        rows.append((n, role, "in" if member else "out",
                     round(R.e[i], 2),
                     round(d if member else 0.0, 3),
                     round(R.utility(i, t, "transfer"), 2),
                     "lower p" if member and d < 0 else
                     ("higher p" if member and d > 0 else "indifferent")))
    return pd.DataFrame(rows, columns=[
        "actor", "role", "status", "e (t/cap)", "du/dp (EUR/cap per EUR/t)",
        "transfer at p* (EUR/cap)", "wants"]), p


# --------------------------------------------------------------------------- main
def main():
    t = R.TRUTHFUL
    print("=== report semantics: which reading of p_i(c) supports the DSIC claim? ===")
    print(f"operating point: p = {t['p']:.2f}, c = {t['c']:.4f}, "
          f"T+ = {t['Tplus']:.2f}, T- = {t['Tminus']:.4f}\n")

    grid = {}
    per_actor = {}
    for regime, fn in (("fixed", regret_fixed), ("endogenous", regret_endog)):
        for kind in SEMANTICS:
            regs = []
            for i in range(9):
                reg, x, u0, ub = fn(i, kind)
                regs.append(reg)
            grid[(regime, kind)] = float(np.sum(regs))
            per_actor[(regime, kind)] = np.array(regs)

    tb = pd.DataFrame(
        [[round(grid[(rg, k)], 3) for k in SEMANTICS] for rg in ("fixed", "endogenous")],
        index=["fixed coverage (DSIC control)", "endogenous coverage"],
        columns=[f"NashConv | {k}" for k in SEMANTICS])
    print(tb.to_string(), "\n")

    nc_peak = grid[("fixed", "peak")]
    nc_cap = grid[("fixed", "cap")]
    print(f"At FIXED coverage the peaked reading gives NashConv = {nc_peak:.4f} "
          f"and the cap reading gives {nc_cap:.4f}.")
    print("Per-actor regret at fixed coverage (who gains from which reading):")
    pa = pd.DataFrame({"actor": R.names,
                       "peak": per_actor[("fixed", "peak")].round(3),
                       "cap": per_actor[("fixed", "cap")].round(3),
                       "transfer": per_actor[("fixed", "transfer")].round(3)})
    print(pa.to_string(index=False), "\n")

    print("=== direction of price preference under T+- = t+-/p ===")
    dd, p_star = dudp_table()
    print(dd.to_string(index=False))
    ben_in = dd[(dd["role"] == "beneficiary") & (dd["status"] == "in")]
    con_in = dd[(dd["role"] == "contributor") & (dd["status"] == "in")]
    print(f"\nMembers wanting a HIGHER price: {len(ben_in)} beneficiaries; "
          f"wanting a LOWER price: {len(con_in)} contributors.")
    print("Beneficiary transfer utility is linear and strictly increasing in p, so it")
    print("has no interior maximum: their acceptable set is bounded BELOW, not above.")
    print("A single 'highest acceptable price' elicitation cannot represent both sides.\n")

    np.savez(SEMANTICS_NPZ,
             semantics=np.array(SEMANTICS, dtype="U10"),
             regimes=np.array(["fixed", "endogenous"], dtype="U12"),
             nashconv=np.array([[grid[(rg, k)] for k in SEMANTICS]
                                for rg in ("fixed", "endogenous")]),
             per_actor_fixed=np.array([per_actor[("fixed", k)] for k in SEMANTICS]),
             per_actor_endog=np.array([per_actor[("endogenous", k)] for k in SEMANTICS]),
             names=np.array(R.names, dtype="U40"),
             dudp=dd["du/dp (EUR/cap per EUR/t)"].to_numpy(float))
    print(f"saved {SEMANTICS_NPZ}")


if __name__ == "__main__":
    main()
