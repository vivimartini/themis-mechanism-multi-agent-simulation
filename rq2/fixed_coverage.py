"""RQ2 fixed-coverage control."""
import numpy as np
import pandas as pd
from paths import ACTORS_CSV
from rq2.oracle import cma_minimize, DEFAULT_SEED, DEFAULT_SIGMA0, WARM_STARTS
import rq2.endogenous_coverage as _R

EBAR = 6.6
# The control freezes coverage and rates AT the truthful endogenous point, so
# derive them from that solve rather than transcribing them. Hardcoding let the
# control drift silently from the experiment it is a control for.
T_PLUS = _R.TRUTHFUL["Tplus"]      # truthful contributor rate, fraction of price
T_MINUS = _R.TRUTHFUL["Tminus"]    # truthful beneficiary rate, fraction of price
assert abs(T_PLUS - 0.24) < 5e-3 and abs(T_MINUS - 0.3417) < 5e-3, \
    f"truthful rates moved: T+={T_PLUS}, T-={T_MINUS}"
SEED = DEFAULT_SEED
CMA_SIGMA0 = DEFAULT_SIGMA0
CMA_BUDGET = 900          # objective evaluations per actor, split over restarts

df = pd.read_csv(ACTORS_CSV)
names = df["name"].tolist()
e = df["e"].to_numpy(float)
pop = df["pop_m"].to_numpy(float)
w = pop * e
w = w / w.sum()                                      # emission weights
ab = df["alpha_base"].to_numpy(float)
ac = df["alpha_cov"].to_numpy(float)
at = df["alpha_trf"].to_numpy(float)
contrib = e > EBAR

# Exact weight of the six joiners — don't hardcode 0.8687; that rounds the
# quantile onto the wrong side of the US pivot.
TRUTHFUL_MEMBERS = ["CHINA", "UNITED STATES", "EUROPEAN UNION", "INDIA",
                    "ADV. CARBON-PRICED CONDITIONAL JOINERS", "LOW-CARBON FRONTIER"]
C_STAR = float(w[[names.index(n) for n in TRUTHFUL_MEMBERS]].sum())
assert abs(C_STAR - _R.TRUTHFUL["c"]) < 1e-12, \
    f"control coverage {C_STAR} != endogenous truthful coverage {_R.TRUTHFUL['c']}"

tau = np.where(contrib, -T_PLUS * (e - EBAR), T_MINUS * (EBAR - e))  # fixed-T transfer arg


def scalar_report(i, db=0.0, dc=0.0):
    """Actor i's willingness at (c*, T*) under a (delta_base, delta_cov) perturbation."""
    return max(0.0, (ab[i] + db) + (ac[i] + dc) * C_STAR + at[i] * tau[i])


TRUTH = np.array([scalar_report(i) for i in range(len(names))])


def quantile_price(reports):
    """Lower weighted (1-c*) quantile; tie-safe at the coverage boundary."""
    order = np.argsort(-reports, kind="stable")
    cum = np.cumsum(w[order])
    k = np.searchsorted(cum, C_STAR - 1e-12)
    return reports[order[k]]


def outcome(reports):
    p = quantile_price(reports)
    members = reports >= p - 1e-12
    return p, members


def utility(i, reports, kind):
    p, members = outcome(reports)
    if kind == "peaked":                      # public outcome, single-peaked at truth
        return -abs(p - TRUTH[i])
    if kind == "transfer":                    # mechanism accounting, EUR per capita
        if not members[i]:
            return 0.0
        if contrib[i]:
            return -T_PLUS * p * (e[i] - EBAR)
        return T_MINUS * p * (EBAR - e[i])
    raise ValueError(kind)


def best_response(i, kind, sigma0=CMA_SIGMA0, budget=CMA_BUDGET, seed=SEED,
                  n_restarts=3):
    """CMA over (delta_base, delta_cov). Returns (regret, delta, best report).

    Truthful reports often sit on a flat plateau (anything above the pivot
    leaves p unchanged), so we track best-ever ourselves rather than trusting
    cma.optimize()'s early stop.
    """
    reports = TRUTH.copy()
    u_truth = utility(i, reports, kind)

    def neg_u(x):
        r = reports.copy()
        r[i] = scalar_report(i, x[0], x[1])   # only actor i deviates; others fixed
        return -utility(i, r, kind)

    best_f, best_x = cma_minimize(
        neg_u, budget, sigma0=sigma0, seed=seed, n_restarts=n_restarts,
        warm_starts=WARM_STARTS,
    )
    u_best = -best_f
    regret = max(0.0, u_best - u_truth)
    r_best = scalar_report(i, best_x[0], best_x[1])
    return regret, best_x, r_best, u_truth, u_best


if __name__ == "__main__":
    p0, m0 = outcome(TRUTH)
    print(f"truthful fixed-coverage point: p = {p0:.2f}, members = "
          f"{[n for n, m in zip(names, m0) if m]}")
    print(f"(sanity: should equal the operating price 26.70 with the six joiners)\n")
    for kind in ("peaked", "transfer"):
        print(f"--- utility = {kind}")
        rows = []
        for i, n in enumerate(names):
            regret, x, r, ut, ub = best_response(i, kind)
            rows.append((n, TRUTH[i], r, ut, ub, regret))
        t = pd.DataFrame(rows, columns=["actor", "truthful report", "best report",
                                        "u(truth)", "u(best)", "regret"])
        print(t.round(3).to_string(index=False))
        print(f"NashConv = {t['regret'].sum():.4f}\n")
