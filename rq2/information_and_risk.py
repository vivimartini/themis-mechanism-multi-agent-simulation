"""How much a deviator must know: prior-only attacks, noise sensitivity, simultaneous response."""
import cma
import numpy as np
import pandas as pd

import rq2.endogenous_coverage as R
import rq2.coarse_solver as F
from rq2.oracle import WARM_STARTS, EXIT_SHRINK_STARTS
from paths import INFORISK_NPZ

N_WORLDS = 60
N_RANDOM = 64
SIGMAS = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
N_NOISE_DRAWS = 8
SEED = 11

# Chapter 3 prior (same ranges as engine.scenario_prior); unlisted actors stay 0.
AB_LO = np.zeros(9); AB_HI = np.zeros(9)
for _n, (_lo, _hi) in {"CHINA": (0, 8.63), "UNITED STATES": (2, 4),
                       "EUROPEAN UNION": (15, 25),
                       "ADV. CARBON-PRICED CONDITIONAL JOINERS": (4, 8),
                       "INDIA": (0, 4)}.items():
    _j = R.names.index(_n)
    AB_LO[_j], AB_HI[_j] = _lo, _hi


def draw_world(rng):
    return rng.uniform(AB_LO, AB_HI), R.AC * rng.uniform(0.5, 1.5) * rng.lognormal(0, 0.2, 9)


def candidates(i, rng):
    """Deviation candidates for actor i: portfolio, point-calibration BR, random."""
    pt = F.oracle(i, R.AB, R.AC, {}, cma_budget=200, sigma0=12.0, seed=42 + i)[1]
    xs = [np.zeros(2), np.asarray(pt, float)]
    xs += [np.asarray(v, float) for v in WARM_STARTS + EXIT_SHRINK_STARTS]
    xs += [rng.uniform(-60, 60, 2) for _ in range(N_RANDOM)]
    return np.array(xs), np.asarray(pt, float)


def payoff_matrix(i, X, worlds):
    """U[x, w] = actor i's peaked utility playing deviation X[x] in world w.

    Actor i's own type is held at the point calibration in every world: it knows
    itself and is uncertain only about the other eight. Utility is scored against
    i's own true willingness at the realised (c, T).
    """
    U = np.empty((len(X), len(worlds)))
    for wi, (ab_w, ac_w) in enumerate(worlds):
        ab_w = ab_w.copy(); ac_w = ac_w.copy()
        ab_w[i] = R.AB[i]; ac_w[i] = R.AC[i]
        for xi, x in enumerate(X):
            ab = ab_w.copy(); ac = ac_w.copy()
            ab[i] += x[0]; ac[i] += x[1]
            U[xi, wi] = F.u_peaked(i, F.solve(ab, ac), R.AB, R.AC)
    return U


# ------------------------------------------------------------------- A1 + B
def information_and_risk():
    rng = np.random.default_rng(SEED)
    worlds = [draw_world(rng) for _ in range(N_WORLDS)]
    rows, risk_rows = [], []
    store = {}
    for i, name in enumerate(R.names):
        X, pt = candidates(i, np.random.default_rng(SEED + 100 + i))
        U = payoff_matrix(i, X, worlds)
        u_truth = U[0]                                   # X[0] is the zero deviation
        # full information: best deviation chosen per world
        reg_full = (U.max(0) - u_truth)
        # prior information only: one deviation, chosen to maximise the prior mean
        xstar = int(np.argmax(U.mean(1)))
        reg_prior = U[xstar] - u_truth
        # the point-calibration attack, played in every world (risk)
        reg_point = U[1] - u_truth
        rows.append((name, round(reg_full.mean(), 2), round(reg_prior.mean(), 2),
                     round(100 * reg_prior.mean() / reg_full.mean(), 0)
                     if reg_full.mean() > 1e-9 else 0.0,
                     "truthful" if xstar == 0 else f"x={X[xstar].round(0)}"))
        risk_rows.append((name, round(reg_point.mean(), 2),
                          round(float(np.percentile(reg_point, 5)), 2),
                          round(reg_point.min(), 2),
                          f"{100*np.mean(reg_point < -1e-6):.0f}%"))
        store[name] = dict(X=X, U=U, pt=pt, reg_full=reg_full,
                           reg_prior=reg_prior, reg_point=reg_point)
    ta = pd.DataFrame(rows, columns=["actor", "E[regret] full info",
                                     "E[regret] prior only", "% of full info",
                                     "prior-optimal report"])
    tb = pd.DataFrame(risk_rows, columns=["actor", "mean dU", "5th pct", "worst",
                                          "P(backfires)"])
    return ta, tb, store, worlds


# ------------------------------------------------------------------------ A2
def noise_sweep(worlds_seed=SEED):
    """How precisely must a deviator know the others before manipulation pays?"""
    rows = []
    for sig in SIGMAS:
        realised = np.zeros(9)
        for i in range(9):
            X, _ = candidates(i, np.random.default_rng(SEED + 100 + i))
            # true-world payoff of every candidate, computed once
            u_true = np.array([_play(i, X[k], R.AB, R.AC) for k in range(len(X))])
            rng = np.random.default_rng(worlds_seed + 7 * i)
            gains = []
            for _ in range(N_NOISE_DRAWS if sig > 0 else 1):
                ab_e = R.AB + rng.normal(0, sig, 9)
                ac_e = R.AC + rng.normal(0, sig, 9)
                ab_e[i] = R.AB[i]; ac_e[i] = R.AC[i]      # knows itself exactly
                u_est = np.array([_play(i, X[k], ab_e, ac_e, truth=(R.AB, R.AC))
                                  for k in range(len(X))])
                k = int(np.argmax(u_est))
                gains.append(u_true[k] - u_true[0])
            realised[i] = float(np.mean(gains))
        rows.append((sig, round(float(realised.sum()), 2),
                     round(float(realised.max()), 2),
                     R.names[int(np.argmax(realised))][:16]))
    # Not NashConv: these are REALISED gains from acting on an estimate, so they
    # can be negative when the estimate misleads the deviator.
    return pd.DataFrame(rows, columns=["noise sd on others' alphas",
                                       "realised total gain", "max realised gain",
                                       "best-placed actor"])


def _play(i, x, ab_env, ac_env, truth=None):
    ab = np.asarray(ab_env, float).copy(); ac = np.asarray(ac_env, float).copy()
    ab[i] += x[0]; ac[i] += x[1]
    t_ab, t_ac = truth if truth is not None else (ab_env, ac_env)
    return F.u_peaked(i, F.solve(ab, ac), t_ab, t_ac)


# ------------------------------------------------------------------------- C
def _br_against(i, ab_prof, ac_prof, budget=200, seed=42):
    """Best further deviation for i given the CURRENT report profile.

    Scored against i's own TRUE type (R.AB, R.AC), not against whatever it is
    currently reporting. F.oracle cannot be reused here: it takes one pair of
    arrays and uses it both as the profile to solve and as the truth to score
    against, which is right when the profile is truthful and wrong once actors
    have already deviated.
    """
    def u_at(x):
        a1 = ab_prof.copy(); a2 = ac_prof.copy()
        a1[i] += x[0]; a2[i] += x[1]
        return F.u_peaked(i, F.solve(a1, a2), R.AB, R.AC)

    best_u, best_x = u_at(np.zeros(2)), np.zeros(2)
    rng = np.random.default_rng(seed)
    for x in [np.asarray(v, float) for v in WARM_STARTS + EXIT_SHRINK_STARTS] \
            + [rng.uniform(-60, 60, 2) for _ in range(40)]:
        u = u_at(x)
        if u > best_u:
            best_u, best_x = u, np.asarray(x, float)
    es = cma.CMAEvolutionStrategy(list(best_x), 12.0,
                                  {"seed": seed + 1, "verbose": -9,
                                   "maxfevals": budget})
    while not es.stop():
        xs = es.ask()
        fs = [-u_at(x) for x in xs]
        es.tell(xs, fs)
        j = int(np.argmin(fs))
        if -fs[j] > best_u:
            best_u, best_x = -fs[j], np.asarray(xs[j], float)
    return best_x


def iterated_best_response(rounds=8, budget=200):
    """Everyone reacts on the same information: simultaneous BR from truth."""
    D = np.zeros((9, 2))
    traj = []
    for r in range(rounds + 1):
        ab = R.AB + D[:, 0]; ac = R.AC + D[:, 1]
        out = F.solve(ab, ac)
        us = np.array([F.u_peaked(i, out, R.AB, R.AC) for i in range(9)])
        trs = np.array([R.utility(i, out, "transfer") if out else 0.0
                        for i in range(9)])
        traj.append(dict(round=r, p=None if out is None else out["p"],
                         c=None if out is None else out["c"],
                         cp=0.0 if out is None else out["c"] * out["p"],
                         u=us.copy(), tr=trs.copy()))
        if r == rounds:
            break
        newD = D.copy()
        for i in range(9):
            x = _br_against(i, ab, ac, budget=budget, seed=42 + i + 1000 * r)
            newD[i] = D[i] + x
        D = newD
    return traj, D


def two_by_two():
    """China x Indonesia, both playing their point-calibration peaked BRs."""
    ic, ii = R.names.index("CHINA"), R.names.index("INDONESIA")
    _, xc, _, _ = R.best_response(ic, "peaked")
    _, xi, _, _ = R.best_response(ii, "peaked")
    cells = {}
    for cn, dc in (("truth", np.zeros(2)), ("BR", xc)):
        for inm, di in (("truth", np.zeros(2)), ("BR", xi)):
            ab = R.AB.copy(); ac = R.AC.copy()
            ab[ic] += dc[0]; ac[ic] += dc[1]
            ab[ii] += di[0]; ac[ii] += di[1]
            o = R.solve_full(ab, ac)
            cells[(cn, inm)] = (o, R.utility(ic, o, "peaked"), R.utility(ii, o, "peaked"))
    return cells, xc, xi


def main():
    print("=" * 78)
    print("A1. HOW MUCH MUST A DEVIATOR KNOW?")
    print("=" * 78)
    ta, tb, store, worlds = information_and_risk()
    print(ta.to_string(index=False))
    full, prior = ta["E[regret] full info"].sum(), ta["E[regret] prior only"].sum()
    print(f"\nNashConv with full information: {full:.2f}")
    print(f"NashConv with prior information only: {prior:.2f} "
          f"({100*prior/full:.0f}% of full)")
    print()

    print("=" * 78)
    print("A2. HOW PRECISELY MUST IT KNOW IT?")
    print("=" * 78)
    tn = noise_sweep()
    print(tn.to_string(index=False))
    print()

    print("=" * 78)
    print("B. DOES THE ATTACK EXPOSE THE DEVIATOR TO RISK?")
    print("=" * 78)
    print(tb.to_string(index=False))
    print()

    print("=" * 78)
    print("C. WHAT IF EVERYONE ACTS ON THE SAME INFORMATION?")
    print("=" * 78)
    traj, D = iterated_best_response()
    tb2 = pd.DataFrame([{"round": t["round"],
                         "p": None if t["p"] is None else round(t["p"], 2),
                         "c": None if t["c"] is None else round(t["c"], 3),
                         "c*p": round(t["cp"], 2),
                         "mean u (peaked)": round(t["u"].mean(), 2),
                         "pool EUR/cap": round(np.abs(t["tr"]).sum(), 1)}
                        for t in traj])
    print(tb2.to_string(index=False))
    u0, uT = traj[0]["u"], traj[-1]["u"]
    t0, tT = traj[0]["tr"], traj[-1]["tr"]
    cmp_ = pd.DataFrame({"actor": [n[:22] for n in R.names],
                         "u_pk truth": u0.round(2), "u_pk final": uT.round(2),
                         "d u_pk": (uT - u0).round(2),
                         "EUR/cap truth": t0.round(1),
                         "EUR/cap final": tT.round(1),
                         "d EUR/cap": (tT - t0).round(1)})
    print()
    print(cmp_.to_string(index=False))
    print()

    cells, _, _ = two_by_two()
    print("China x Indonesia 2x2 (peaked utility; rows = China, cols = Indonesia):")
    print()
    print(f"{'':<14}{'Indo truth':>24}{'Indo BR':>24}")
    for cn in ("truth", "BR"):
        line = f"{'China ' + cn:<14}"
        for inm in ("truth", "BR"):
            o, uc, ui = cells[(cn, inm)]
            cell = "({:+.2f}, {:+.2f}) p={:.1f}".format(uc, ui, o["p"])
            line += f"{cell:>24}"
        print(line)
    print()

    np.savez(INFORISK_NPZ,
             names=np.array(R.names, dtype="U40"),
             reg_full=ta["E[regret] full info"].to_numpy(float),
             reg_prior=ta["E[regret] prior only"].to_numpy(float),
             risk_mean=tb["mean dU"].to_numpy(float),
             risk_p5=tb["5th pct"].to_numpy(float),
             risk_worst=tb["worst"].to_numpy(float),
             sigmas=np.array(SIGMAS),
             noise_nashconv=tn["realised total gain"].to_numpy(float),
             ibr_cp=np.array([t["cp"] for t in traj]),
             ibr_c=np.array([np.nan if t["c"] is None else t["c"] for t in traj]),
             ibr_u=np.array([t["u"] for t in traj]),
             ibr_tr=np.array([t["tr"] for t in traj]))
    print(f"saved {INFORISK_NPZ}")


if __name__ == "__main__":
    main()
