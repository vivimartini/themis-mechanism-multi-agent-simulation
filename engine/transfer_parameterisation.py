"""Transfer parameterisation experiment."""
import numpy as np
import pandas as pd

import rq2.endogenous_coverage as R
from paths import TRANSFERPARAM_NPZ

EBAR = R.EBAR
EXCESS = np.maximum(R.e - EBAR, 0.0)
DEFICIT = np.maximum(EBAR - R.e, 0.0)

# t+ grid in EUR/tonne. The truthful ratio point sits at T+ = 0.24, p = 26.70,
# i.e. t+ = 6.41 EUR/t, so this grid brackets the calibration comfortably.
T_ABS = np.round(np.linspace(0.5, 25.0, 25), 3)
P_SCAN = np.linspace(1e-3, 300.0, 4000)


def willingness_abs(k, t_plus, p):
    """w_i(c, p) for coalition index k under absolute transfers. p may be a vector."""
    ex, de = R.EX[k], R.DE[k]
    if de <= 0 or ex <= 0:
        return None
    t_minus = t_plus * ex / de
    per_cap = np.where(R.contrib, -t_plus * EXCESS, t_minus * DEFICIT)   # EUR/cap
    p = np.atleast_1d(np.asarray(p, float))
    shift = R.AT[None, :] * per_cap[None, :] / p[:, None]
    base = (R.AB + R.AC * R.COV[k])[None, :]
    return np.maximum(0.0, base + shift)                                  # [n_p, N]


def fixed_points(k, t_plus, p_scan=P_SCAN):
    """Positive roots of p - Phi(p) for coalition k under absolute transfers."""
    W = willingness_abs(k, t_plus, p_scan)
    if W is None:
        return []
    mem = R.M[k]
    phi = W[:, mem].min(1)
    nonmax = W[:, ~mem].max(1) if (~mem).any() else np.full(len(p_scan), -np.inf)
    g = phi - p_scan
    roots = []
    for a in range(len(p_scan) - 1):
        if g[a] == 0.0 or g[a] * g[a + 1] < 0:
            # linear interpolation onto the root
            w = g[a] / (g[a] - g[a + 1]) if g[a] != g[a + 1] else 0.0
            p_root = p_scan[a] + w * (p_scan[a + 1] - p_scan[a])
            if p_root <= 1e-6:
                continue
            Wr = willingness_abs(k, t_plus, p_root)
            selfcons = bool(Wr[0, ~mem].max() < p_root + 0.01) if (~mem).any() else True
            roots.append((float(p_root), selfcons))
    return roots


def main():
    print("=" * 78)
    print("A. WELL-POSEDNESS: how many operating prices does each design admit?")
    print("=" * 78)
    n_designs = 0
    hist = {0: 0, 1: 0, 2: 0}          # key 2 means ">=2"
    sc_hist = {0: 0, 1: 0, 2: 0}
    multi_examples = []
    for k in range(len(R.M)):
        if R.EX[k] <= 0 or R.DE[k] <= 0:
            continue
        for tp in T_ABS:
            n_designs += 1
            roots = fixed_points(k, tp)
            n = len(roots)
            hist[min(n, 2)] += 1
            n_sc = sum(1 for _, s in roots if s)
            sc_hist[min(n_sc, 2)] += 1
            if n_sc >= 2 and len(multi_examples) < 5:
                multi_examples.append((k, tp, [round(p, 2) for p, s in roots if s]))

    tb = pd.DataFrame([
        ("absolute t+-", hist[0], hist[1], hist[2]),
        ("ratio T+- = t+-/p", 0, n_designs, 0),
    ], columns=["design", "no operating price", "exactly one", "two or more"])
    print(f"{n_designs} (coalition, transfer-level) designs enumerated.\n")
    print(tb.to_string(index=False))
    print(f"\nrestricting to SELF-CONSISTENT roots (no willing outsider): "
          f"{sc_hist[0]} none, {sc_hist[1]} one, {sc_hist[2]} two or more")
    print()

    if multi_examples:
        print("Examples of multiplicity (self-consistent roots, EUR/t):")
        for k, tp, ps in multi_examples:
            mem = [n[:12] for n, m in zip(R.names, R.M[k]) if m]
            print(f"  t+ = {tp:5.2f}, c = {R.COV[k]:.3f}, roots {ps}")
            print(f"      coalition: {', '.join(mem)}")
        print()

    print("=" * 78)
    print("B. DOES THE SELECTION RULE MATTER?")
    print("=" * 78)
    best = {}
    for rule in ("lowest root", "highest root"):
        bo, bk, bt, bp = -1.0, None, None, None
        for k in range(len(R.M)):
            if R.EX[k] <= 0 or R.DE[k] <= 0:
                continue
            for tp in T_ABS:
                ps = [p for p, s in fixed_points(k, tp) if s]
                if not ps:
                    continue
                p = min(ps) if rule == "lowest root" else max(ps)
                if R.COV[k] * p > bo:
                    bo, bk, bt, bp = R.COV[k] * p, k, tp, p
        best[rule] = (bo, bk, bt, bp)
        mem = [n[:12] for n, m in zip(R.names, R.M[bk]) if m]
        print(f"  {rule:<14}: p = {bp:6.2f}, c = {R.COV[bk]:.3f}, t+ = {bt:5.2f}, "
              f"c*p = {bo:.2f}")
        print(f"                  coalition ({len(mem)}): {', '.join(mem)}")
    lo, hi = best["lowest root"], best["highest root"]
    if abs(lo[0] - hi[0]) > 1e-6:
        print(f"\nThe two rules disagree: c*p = {lo[0]:.2f} vs {hi[0]:.2f}, prices "
              f"{lo[3]:.2f} vs {hi[3]:.2f} EUR/t.")
    print()

    print("\n" + "=" * 78)
    print("C. THE RATIO DESIGN, FOR COMPARISON")
    print("=" * 78)
    t = R.TRUTHFUL
    print(f"one pass, no fixed point required: p = {t['p']:.2f}, c = {t['c']:.4f}, "
          f"T+ = {t['Tplus']:.2f}, T- = {t['Tminus']:.4f}")
    print(f"equivalent absolute rate at this price: t+ = T+ * p = "
          f"{t['Tplus'] * t['p']:.2f} EUR/t")

    np.savez(TRANSFERPARAM_NPZ,
             n_designs=np.array([n_designs]),
             abs_hist=np.array([hist[0], hist[1], hist[2]]),
             abs_selfcons_hist=np.array([sc_hist[0], sc_hist[1], sc_hist[2]]),
             t_abs=T_ABS,
             sel_low=np.array([lo[0], lo[3]]), sel_high=np.array([hi[0], hi[3]]),
             ratio_point=np.array([t["p"], t["c"], t["Tplus"], t["Tminus"]]))
    print(f"\nsaved {TRANSFERPARAM_NPZ}")


if __name__ == "__main__":
    main()
