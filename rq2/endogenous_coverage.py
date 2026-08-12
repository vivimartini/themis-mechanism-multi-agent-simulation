"""RQ2 endogenous-coverage experiment."""
import numpy as np
import pandas as pd
from paths import ACTORS_CSV
from rq2.oracle import cma_minimize, DEFAULT_SEED, DEFAULT_SIGMA0, WARM_STARTS

EBAR = 6.6
SEED = DEFAULT_SEED
SIGMA0 = DEFAULT_SIGMA0
BUDGET = 900            # full-solver evaluations per actor, split over restarts
RESTARTS = 3

df = pd.read_csv(ACTORS_CSV)
names = df["name"].tolist()
e = df["e"].to_numpy(float)
pop = df["pop_m"].to_numpy(float)
w = pop * e; w = w / w.sum()
AB = df["alpha_base"].to_numpy(float)
AC = df["alpha_cov"].to_numpy(float)
AT = df["alpha_trf"].to_numpy(float)
N = 9
contrib = e > EBAR

# ------------------------------------------------ vectorised self-consistent solver
# Enumerate all 512 coalitions; keep only those with >=2 members, at least one
# contributor and one beneficiary (otherwise T+ or T- is undefined).
masks = []
for m in range(1, 2 ** N):
    mem = np.array([(m >> j) & 1 for j in range(N)], bool)
    if mem.sum() < 2 or not (mem & contrib).any() or not (mem & ~contrib).any():
        continue
    masks.append(mem)
M = np.array(masks)
COV = (M * w).sum(1)
EX = (M * np.maximum(e - EBAR, 0) * pop).sum(1)
DE = (M * np.maximum(EBAR - e, 0) * pop).sum(1)
T_GRID = np.round(np.linspace(0, 1, 101), 4)[1:]
TMINUS = T_GRID[None, :] * EX[:, None] / DE[:, None]
TAU_C = -T_GRID[None, :, None] * np.maximum(e - EBAR, 0)[None, None, :]
TAU_B = TMINUS[:, :, None] * np.maximum(EBAR - e, 0)[None, None, :]
TAU = np.where(contrib[None, None, :], TAU_C, TAU_B)     # [K, T, N]
MB = M[:, None, :]
BIG = 1e9

# Tolerance in EUR/t on the self-consistency test: an outsider within SLACK of the
# price is still treated as unwilling. Load-bearing at the coalition boundary, so
# rq2.robustness sweeps it rather than leaving it implicit.
SLACK = 0.01


def solve_full(ab, ac, slack=SLACK):
    """Endogenous-coverage operating point for a full report profile (ab, ac)."""
    prefs = np.maximum(0, ab[None, None, :] + ac[None, None, :] * COV[:, None, None]
                       + AT[None, None, :] * TAU)
    price = np.where(MB, prefs, BIG).min(2)          # min member willingness
    nonmax = np.where(~MB, prefs, -BIG).max(2)       # max non-member willingness
    ok = (price > 0) & (nonmax < price + slack)      # self-consistency (+ slack)
    obj = np.where(ok, COV[:, None] * price, -1.0)
    k, t = np.unravel_index(np.argmax(obj), obj.shape)
    if obj[k, t] < 0:
        return None                                       # no self-consistent point
    return dict(p=float(price[k, t]), c=float(COV[k]),
                Tplus=float(T_GRID[t]), Tminus=float(TMINUS[k, t]),
                members=M[k].copy())


def true_willingness(i, c, Tplus, Tminus):
    tau = -Tplus * (e[i] - EBAR) if contrib[i] else Tminus * (EBAR - e[i])
    return max(0.0, AB[i] + AC[i] * c + AT[i] * tau)


def utility(i, out, kind):
    if out is None:
        return -1e6                                       # invalid: never chosen as best
    p, c, Tp, Tm = out["p"], out["c"], out["Tplus"], out["Tminus"]
    if kind == "peaked":
        # Target willingness moves with the realised (c, T) — unlike the control.
        return -abs(p - true_willingness(i, c, Tp, Tm))
    if kind == "sqloss":
        return -(p - true_willingness(i, c, Tp, Tm)) ** 2
    if kind == "transfer":
        if not out["members"][i]:
            return 0.0
        if contrib[i]:
            return -Tp * p * (e[i] - EBAR)
        return Tm * p * (EBAR - e[i])
    raise ValueError(kind)


TRUTHFUL = solve_full(AB, AC)


def best_response(i, kind, sigma0=SIGMA0, budget=BUDGET, seed=SEED,
                  n_restarts=RESTARTS):
    u_truth = utility(i, TRUTHFUL, kind)

    def neg_u(x):
        ab = AB.copy(); ac = AC.copy()
        ab[i] += x[0]; ac[i] += x[1]
        return -utility(i, solve_full(ab, ac), kind)

    best_f, best_x = cma_minimize(
        neg_u, budget, sigma0=sigma0, seed=seed, n_restarts=n_restarts,
        warm_starts=WARM_STARTS,
    )
    u_best = -best_f
    return max(0.0, u_best - u_truth), best_x, u_truth, u_best


if __name__ == "__main__":
    t = TRUTHFUL
    print(f"truthful endogenous point: p = {t['p']:.2f}, c_model = {t['c']:.4f}, "
          f"T+ = {t['Tplus']:.2f}, T- = {t['Tminus']:.4f}")
    print(f"members: {[n for n, m in zip(names, t['members']) if m]}")
    print("(sanity: 26.70 / 0.8687 / 0.24 / 0.3417 with the six joiners)\n")
    for kind in ("peaked", "transfer"):
        print(f"--- utility = {kind} (endogenous coverage)")
        rows = []
        for i, n in enumerate(names):
            regret, x, ut, ub = best_response(i, kind)
            # describe the best deviation's realised outcome
            ab = AB.copy(); ac = AC.copy(); ab[i] += x[0]; ac[i] += x[1]
            o = solve_full(ab, ac)
            desc = (f"p={o['p']:.2f}, c={o['c']:.3f}" if o else "invalid")
            rows.append((n, round(ut, 3), round(ub, 3), round(regret, 3),
                         f"({x[0]:+.1f},{x[1]:+.1f})", desc))
        tb = pd.DataFrame(rows, columns=["actor", "u(truth)", "u(best)", "regret",
                                         "best (db,dc)", "realised outcome"])
        print(tb.to_string(index=False))
        print(f"NashConv = {tb['regret'].sum():.3f}\n")
