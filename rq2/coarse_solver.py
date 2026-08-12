"""Coarse-grid RQ2 solver."""
import numpy as np
import cma
import rq2.endogenous_coverage as R
from rq2.oracle import WARM_STARTS as PORTFOLIO
T_GRID = np.round(np.linspace(0, 1, 26), 4)[1:]   # drop T=0 (no contributor transfers)
TMINUS = T_GRID[None, :] * R.EX[:, None] / R.DE[:, None]
TAU_C = -T_GRID[None, :, None] * np.maximum(R.e - R.EBAR, 0)[None, None, :]
TAU_B = TMINUS[:, :, None] * np.maximum(R.EBAR - R.e, 0)[None, None, :]
ATTAU = R.AT[None, None, :] * np.where(R.contrib[None, None, :], TAU_C, TAU_B)
COVC = R.COV[:, None, None]
BIG = 1e9


def solve(ab, ac, tminus_cap=None, cov_floor=None, pool_cap=None, slack=R.SLACK):
    """Argmax c*p over coalitions and T, subject to self-consistency and guardrails."""
    prefs = ab[None, None, :] + ac[None, None, :] * COVC + ATTAU
    np.maximum(prefs, 0, out=prefs)
    # Price = minimum member willingness; non-members must sit strictly below it.
    price = np.where(R.MB, prefs, BIG).min(2)
    nonmax = np.where(~R.MB, prefs, -BIG).max(2)
    ok = (price > 0) & (nonmax < price + slack)
    if tminus_cap is not None:
        ok &= TMINUS <= tminus_cap
    if cov_floor is not None:
        ok &= (R.COV >= cov_floor)[:, None]   # floor on realised coalition coverage
    if pool_cap is not None:
        ok &= T_GRID[None, :] * price * R.EX[:, None] <= pool_cap
    obj = np.where(ok, R.COV[:, None] * price, -1.0)
    k, t = np.unravel_index(np.argmax(obj), obj.shape)
    if obj[k, t] < 0:
        return None
    return dict(p=float(price[k, t]), c=float(R.COV[k]),
                Tplus=float(T_GRID[t]), Tminus=float(TMINUS[k, t]),
                members=R.M[k].copy())


def true_will(i, out, ab, ac):
    tau = (-out["Tplus"] * (R.e[i] - R.EBAR) if R.contrib[i]
           else out["Tminus"] * (R.EBAR - R.e[i]))
    return max(0.0, ab[i] + ac[i] * out["c"] + R.AT[i] * tau)


def u_peaked(i, out, ab, ac):
    # Peaked utility: prefer the price closest to i's true willingness at the outcome.
    return -1e6 if out is None else -abs(out["p"] - true_will(i, out, ab, ac))


def oracle(i, ab, ac, kw, cma_budget=40, n_rand=40, sigma0=8.0, seed=42):
    """Hybrid best-response: portfolio warm starts, uniform random sampling, then an
    ALWAYS-run CMA refinement from the best point found. The earlier profit-gated
    variant (CMA only if the portfolio found profit) was beaten by plain random
    search in the optimizer-comparison experiment; the union oracle dominates both.
    """
    tout = solve(ab, ac, **kw)
    u0 = u_peaked(i, tout, ab, ac)

    def u_at(x):
        a1 = ab.copy(); a2 = ac.copy()
        a1[i] += x[0]; a2[i] += x[1]
        return u_peaked(i, solve(a1, a2, **kw), ab, ac)

    best_u, best_x = u0, np.zeros(2)
    rng = np.random.default_rng(seed)
    # Phase 1: portfolio + uniform random warm start (skip (0,0), already evaluated).
    for x in PORTFOLIO[1:] + [rng.uniform(-60, 60, 2) for _ in range(n_rand)]:
        u = u_at(x)
        if u > best_u:
            best_u, best_x = u, np.array(x)
    # Phase 2: CMA-ES from the best warm start (always run, not profit-gated).
    es = cma.CMAEvolutionStrategy(list(best_x), sigma0,
                                  {"seed": seed + 1, "verbose": -9,
                                   "maxfevals": cma_budget})
    while not es.stop():
        xs = es.ask()
        fs = [-u_at(x) for x in xs]
        es.tell(xs, fs)
        j = int(np.argmin(fs))
        if -fs[j] > best_u:
            best_u, best_x = -fs[j], np.array(xs[j])
    return max(0.0, best_u - u0), best_x, tout


def attack_mode(i, best_x, tout, ab, ac, kw):
    """Heuristic label for the profitable deviation found by the oracle."""
    if np.allclose(best_x, 0):
        return "none"
    a1 = ab.copy(); a2 = ac.copy(); a1[i] += best_x[0]; a2[i] += best_x[1]
    out = solve(a1, a2, **kw)
    if out is None:
        return "collapse"
    was_in, now_in = bool(tout["members"][i]), bool(out["members"][i])
    if not was_in and now_in:
        return "entry"
    if (was_in and not now_in) or out["c"] < tout["c"] - 0.03:
        return "exit-shrink"
    # Same coalition/coverage but higher contributor rate -> extortion on the margin.
    if abs(out["c"] - tout["c"]) <= 0.03 and out["Tplus"] > tout["Tplus"] + 0.1:
        return "extortion"
    return "nudge"
