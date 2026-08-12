"""Collect figure headline data."""
import json
import numpy as np

import rq2.endogenous_coverage as R
import rq2.fixed_coverage as FC
from rq2.obstruction_voteselling import obstruction, collusive_surplus
from paths import HEADLINES_JSON

BUDGET = 900


def attack_mode(i, x, tout):
    """Label the deviation found for actor i, on the fine solver."""
    if np.allclose(x, 0):
        return "none"
    ab = R.AB.copy(); ac = R.AC.copy()
    ab[i] += x[0]; ac[i] += x[1]
    out = R.solve_full(ab, ac)
    if out is None:
        return "collapse"
    was, now = bool(tout["members"][i]), bool(out["members"][i])
    if not was and now:
        return "entry"
    if (was and not now) or out["c"] < tout["c"] - 0.03:
        return "exit-shrink"
    if abs(out["c"] - tout["c"]) <= 0.03 and out["Tplus"] > tout["Tplus"] + 0.1:
        return "extortion"
    return "nudge"


def main():
    t = R.TRUTHFUL
    cp0 = t["c"] * t["p"]
    out = {"truthful": {"p": t["p"], "c": t["c"], "Tplus": t["Tplus"],
                        "Tminus": t["Tminus"], "cp": cp0},
           "names": R.names}

    print("fixed-coverage control (peaked, fine grid)...")
    nc_fixed = sum(FC.best_response(i, "peaked", budget=BUDGET)[0] for i in range(9))

    print("endogenous coverage (peaked, fine grid)...")
    regrets, modes, points = [], [], []
    for i in range(9):
        reg, x, _, _ = R.best_response(i, "peaked", budget=BUDGET)
        regrets.append(float(reg))
        modes.append(attack_mode(i, x, t))
        ab = R.AB.copy(); ac = R.AC.copy(); ab[i] += x[0]; ac[i] += x[1]
        o = R.solve_full(ab, ac)
        points.append(None if o is None else {"p": o["p"], "c": o["c"]})
    nc_endog = float(np.sum(regrets))
    out["nashconv"] = {"fixed": float(nc_fixed), "endogenous": nc_endog}
    out["regret"] = {"values": regrets, "modes": modes, "points": points}
    print(f"  NashConv fixed = {nc_fixed:.4f}, endogenous = {nc_endog:.3f}")

    print("obstruction...")
    dmg, dpay = [], []
    for i in range(9):
        f, x, o = obstruction(i)
        dmg.append(float(100 * (cp0 - f) / cp0))
        dpay.append(0.0 if o is None else
                    float(R.utility(i, o, "transfer") - R.utility(i, t, "transfer")))
    out["obstruction"] = {"damage_pct": dmg, "own_transfer_change": dpay}

    print("headline collusion pair (frontier <- China)...")
    a = R.names.index("LOW-CARBON FRONTIER"); b = R.names.index("CHINA")
    s, dA, dB, x, o = collusive_surplus(a, b)
    out["collusion"] = {"pair": "frontier<-China", "percap": float(s),
                        "p": None if o is None else float(o["p"]),
                        "c": None if o is None else float(o["c"])}

    print("same pair under the pool-cap guardrail (regime-map residual)...")
    import rq2.coarse_solver as F
    kw = {"pool_cap": 150000.0}
    t0 = F.solve(R.AB, R.AC, **kw)

    def u_tr(k, oo):
        if oo is None or not oo["members"][k]:
            return 0.0
        return (-oo["Tplus"] * oo["p"] * (R.e[k] - R.EBAR) if R.contrib[k]
                else oo["Tminus"] * oo["p"] * (R.EBAR - R.e[k]))

    uA0, uB0 = u_tr(a, t0), u_tr(b, t0)

    def nj(x):
        a1 = R.AB.copy(); a2 = R.AC.copy()
        a1[b] += x[0]; a2[b] += x[1]
        oo = F.solve(a1, a2, **kw)
        return -((u_tr(a, oo) - uA0) + (u_tr(b, oo) - uB0))

    # Search config must MATCH rq2.exinterim_guardrails.deal() exactly. It is not
    # cma_minimize: it takes the best warm start, then runs a single CMA instance
    # at sigma 40 for 200 evaluations. Using the generic oracle here instead gave
    # 153.06 / EUR 59.52 against the guardrail table's 125.8 / EUR 55.73 — the
    # same deal reported at two different prices, with the regime map plotting
    # the one the guardrail ablation does not.
    import cma
    from rq2.oracle import WARM_STARTS
    starts = WARM_STARTS + [np.array([145.7, 112.6])]
    xg = min(starts, key=nj)
    fg = nj(xg)
    es = cma.CMAEvolutionStrategy([0., 0.], 40.,
                                  {"seed": 42, "verbose": -9, "maxfevals": 200})
    while not es.stop():
        xs = es.ask()
        fs = [nj(x) for x in xs]
        es.tell(xs, fs)
        j = int(np.argmin(fs))
        if fs[j] < fg:
            fg, xg = fs[j], np.asarray(xs[j], float)
    a1 = R.AB.copy(); a2 = R.AC.copy(); a1[b] += xg[0]; a2[b] += xg[1]
    og = F.solve(a1, a2, **kw)
    out["collusion_guarded"] = {"percap": float(max(0.0, -fg)),
                                "p": None if og is None else float(og["p"]),
                                "c": None if og is None else float(og["c"])}

    HEADLINES_JSON.parent.mkdir(parents=True, exist_ok=True)
    HEADLINES_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {HEADLINES_JSON}")


if __name__ == "__main__":
    main()
