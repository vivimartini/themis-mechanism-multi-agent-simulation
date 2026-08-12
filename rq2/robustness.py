"""Attack locality, self-consistency slack sweep, and reference-vs-vectorised solver agreement."""
import numpy as np
import pandas as pd

import rq2.endogenous_coverage as R
from paths import LOCALITY_NPZ, SLACK_NPZ

SLACKS = [0.0, 1e-4, 1e-3, 0.01, 0.1, 0.5]
BR_BUDGET = 300          # smaller than the 900 headline budget; this is a sweep


# ------------------------------------------------------------------ A: locality
def locality(actor="INDONESIA", lo=-40.0, hi=40.0, step=0.5):
    i = R.names.index(actor)
    db = np.arange(lo, hi + 1e-9, step)
    p = np.empty(len(db)); c = np.empty(len(db)); n = np.empty(len(db), int)
    for k, d in enumerate(db):
        ab = R.AB.copy(); ab[i] += d
        o = R.solve_full(ab, R.AC)
        p[k], c[k], n[k] = o["p"], o["c"], int(o["members"].sum())
    return db, p, c, n


# --------------------------------------------------------------------- B: slack
def slack_sweep():
    rows = []
    for s in SLACKS:
        t = R.solve_full(R.AB, R.AC, slack=s)
        if t is None:
            rows.append((s, None, None, None, None, None))
            continue
        regs = []
        for i in range(9):
            u0 = R.utility(i, t, "peaked")

            def neg_u(x, i=i, s=s):
                ab = R.AB.copy(); ac = R.AC.copy()
                ab[i] += x[0]; ac[i] += x[1]
                return -R.utility(i, R.solve_full(ab, ac, slack=s), "peaked")

            from rq2.oracle import cma_minimize, DEFAULT_SEED, DEFAULT_SIGMA0, WARM_STARTS
            f, _ = cma_minimize(neg_u, BR_BUDGET, sigma0=DEFAULT_SIGMA0,
                                seed=DEFAULT_SEED, warm_starts=WARM_STARTS)
            regs.append(max(0.0, -f - u0))
        regs = np.array(regs)
        rows.append((s, round(t["p"], 2), round(t["c"], 4),
                     int(t["members"].sum()), round(regs.sum(), 2),
                     round(regs[R.names.index("INDONESIA")], 2)))
    return pd.DataFrame(rows, columns=["slack (EUR/t)", "p", "c", "members",
                                       "NashConv", "Indonesia regret"])


# ------------------------------------------------------- C: solver cross-check
def solver_agreement(n_profiles=24, seed=5):
    """Compare engine.themis against the vectorised solver on deviated profiles."""
    import pandas as _pd
    from engine.themis import run_mechanism_selfconsistent, EngineConfig
    from paths import ACTORS_CSV

    base = _pd.read_csv(ACTORS_CSV)
    rng = np.random.default_rng(seed)
    cfg = EngineConfig(t_steps=101)
    rows = []
    for k in range(n_profiles):
        i = int(rng.integers(0, 9))
        dx = rng.uniform(-40, 40, 2)
        ab = R.AB.copy(); ac = R.AC.copy()
        ab[i] += dx[0]; ac[i] += dx[1]
        fast = R.solve_full(ab, ac)
        df = base.copy()
        df["alpha_base"] = ab
        df["alpha_cov"] = ac
        ref = run_mechanism_selfconsistent(df, config=cfg)
        if fast is None:
            rows.append((k, R.names[i][:12], None, None, None, "fast: no s.c. point"))
            continue
        dp = abs(ref["p_star"] - fast["p"])
        dc = abs(ref["c_star"] - fast["c"])
        rows.append((k, R.names[i][:12], round(fast["p"], 3), round(ref["p_star"], 3),
                     round(dp, 6), "OK" if dp < 1e-3 and dc < 1e-6 else "MISMATCH"))
    return pd.DataFrame(rows, columns=["#", "deviator", "p (vectorised)",
                                       "p (engine.themis)", "|dp|", "verdict"])


def main():
    print("=" * 78)
    print("A. LOCALITY OF THE INDONESIA ENTRY ATTACK")
    print("=" * 78)
    db, p, c, n = locality()
    jump = np.where(np.diff(n) != 0)[0]
    print(f"sweeping Indonesia's reported alpha_base over [{db[0]:.0f}, {db[-1]:.0f}] "
          f"in steps of {db[1]-db[0]}")
    if len(jump):
        k = int(jump[0])
        print(f"coalition changes between db = {db[k]:+.1f} and {db[k+1]:+.1f}: "
              f"{n[k]} -> {n[k+1]} members")
        print(f"  price {p[k]:.3f} -> {p[k+1]:.3f}, coverage {c[k]:.4f} -> {c[k+1]:.4f}")
        print(f"  the operating point is piecewise constant in the report and jumps")
        print(f"  only at the boundary — which is why the regret is a step, not a slope.")
    else:
        print("no coalition change over this range")
    print(f"distinct operating points visited: {len(np.unique(p.round(6)))}")
    np.savez(LOCALITY_NPZ, db=db, p=p, c=c, n=n)
    print(f"saved {LOCALITY_NPZ}\n")

    print("=" * 78)
    print("B. SELF-CONSISTENCY SLACK")
    print("=" * 78)
    print("An outsider within `slack` EUR/t of the price is treated as unwilling.")
    print(f"The reported results use slack = {R.SLACK}.\n")
    sw = slack_sweep()
    print(sw.to_string(index=False))
    base_row = sw[sw["slack (EUR/t)"] == R.SLACK].iloc[0]
    same_p = (sw["p"] == base_row["p"]).sum()
    print(f"\n{same_p}/{len(sw)} slack settings give the same truthful price.")
    rng_nc = sw["NashConv"].dropna()
    print(f"NashConv over the sweep: {rng_nc.min():.2f} to {rng_nc.max():.2f} "
          f"(reported {base_row['NashConv']:.2f} at slack = {R.SLACK}).")
    print("The tolerance sets which side of a tie an indifferent outsider falls on;")
    print("it does not create the exploitability.\n")

    print("=" * 78)
    print("C. SOLVER AGREEMENT UNDER MISREPORTS")
    print("=" * 78)
    ag = solver_agreement()
    print(ag.to_string(index=False))
    bad = (ag["verdict"] == "MISMATCH").sum()
    print(f"\n{len(ag) - bad}/{len(ag)} deviated profiles agree between the pandas")
    print("reference engine and the vectorised solver that produces every headline.")
    if bad:
        print("MISMATCHES PRESENT — investigate before quoting the affected numbers.")

    np.savez(SLACK_NPZ, slacks=np.array(SLACKS, float),
             p=sw["p"].to_numpy(dtype=object).astype(float),
             nashconv=sw["NashConv"].to_numpy(dtype=object).astype(float),
             indonesia=sw["Indonesia regret"].to_numpy(dtype=object).astype(float))
    print(f"\nsaved {SLACK_NPZ}")


if __name__ == "__main__":
    main()
