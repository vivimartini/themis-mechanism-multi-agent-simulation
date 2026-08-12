"""RQ2 deviation case studies."""
import numpy as np
import pandas as pd

import rq2.endogenous_coverage as R

CASES = ["INDONESIA", "EUROPEAN UNION", "CHINA"]


def state(out):
    """Per-actor state at an outcome, using each actor's OWN true parameters."""
    rows = []
    for i, n in enumerate(R.names):
        member = bool(out["members"][i])
        w = R.true_willingness(i, out["c"], out["Tplus"], out["Tminus"])
        rows.append(dict(actor=n, member=member, w=w,
                         u_peak=R.utility(i, out, "peaked"),
                         u_tr=R.utility(i, out, "transfer")))
    return pd.DataFrame(rows)


def describe(out):
    mem = [n for n, m in zip(R.names, out["members"]) if m]
    return (f"p = {out['p']:6.2f}  c = {out['c']:.4f}  T+ = {out['Tplus']:.2f}  "
            f"T- = {out['Tminus']:.4f}  c*p = {out['c']*out['p']:.2f}\n"
            f"    coalition ({len(mem)}): {', '.join(mem)}")


def ledger(i, kind="peaked"):
    name = R.names[i]
    regret, x, u_truth, u_best = R.best_response(i, kind)
    ab = R.AB.copy(); ac = R.AC.copy()
    ab[i] += x[0]; ac[i] += x[1]
    dev = R.solve_full(ab, ac)
    t = R.TRUTHFUL

    print("=" * 78)
    print(f"CASE: {name}  (utility = {kind}, regret = {regret:.3f} EUR/t)")
    print("=" * 78)
    print(f"report:  truthful (alpha_base, alpha_cov) = ({R.AB[i]:.2f}, {R.AC[i]:.2f})")
    print(f"         strategic                        = ({ab[i]:.2f}, {ac[i]:.2f})"
          f"   [delta = ({x[0]:+.1f}, {x[1]:+.1f})]")
    print()
    print("  TRUTHFUL   " + describe(t))
    print("  STRATEGIC  " + describe(dev))
    print()

    a, b = state(t), state(dev)
    tab = pd.DataFrame({
        "actor": [n[:22] for n in R.names],
        "in/out": [("in " if m else "out") + "->" + ("in " if n_ else "out")
                   for m, n_ in zip(a["member"], b["member"])],
        "w truth": a["w"].round(2), "w strat": b["w"].round(2),
        "u_pk truth": a["u_peak"].round(2), "u_pk strat": b["u_peak"].round(2),
        "d u_pk": (b["u_peak"] - a["u_peak"]).round(2),
        "EUR/cap truth": a["u_tr"].round(1), "EUR/cap strat": b["u_tr"].round(1),
        "d EUR/cap": (b["u_tr"] - a["u_tr"]).round(1),
    })
    mark = ["  <-- deviator" if k == i else "" for k in range(9)]
    tab["  "] = mark
    print(tab.to_string(index=False))

    others = [k for k in range(9) if k != i]
    d_pk = (b["u_peak"] - a["u_peak"]).to_numpy()
    d_tr = (b["u_tr"] - a["u_tr"]).to_numpy()
    losers = [R.names[k] for k in others if d_pk[k] < -1e-6]
    print(f"\n  deviator gains {d_pk[i]:+.2f} peaked / {d_tr[i]:+.1f} EUR per capita.")
    print(f"  others: {len(losers)}/8 worse off on peaked utility; "
          f"sum of others' peaked change {d_pk[others].sum():+.2f}; "
          f"sum of others' transfers {d_tr[others].sum():+.1f} EUR/cap.")
    print(f"  mechanism objective c*p: {t['c']*t['p']:.2f} -> {dev['c']*dev['p']:.2f} "
          f"({100*(dev['c']*dev['p'] - t['c']*t['p'])/(t['c']*t['p']):+.1f}%)")
    if losers:
        print(f"  worst hit: {losers[int(np.argmin(d_pk[[R.names.index(l) for l in losers]]))]}")
    print()
    return dict(name=name, regret=regret, x=x, dev=dev)


def main():
    t = R.TRUTHFUL
    print("Truthful operating point")
    print("  " + describe(t) + "\n")
    print("Each case below is one actor deviating ALONE; everyone else reports")
    print("truthfully. Simultaneous deviation is treated in rq2.information_and_risk.\n")
    out = [ledger(R.names.index(n)) for n in CASES]

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    s = pd.DataFrame([{
        "case": o["name"][:22],
        "regret (EUR/t)": round(o["regret"], 2),
        "p": round(o["dev"]["p"], 2),
        "c": round(o["dev"]["c"], 3),
        "c*p vs truth %": round(100 * (o["dev"]["c"] * o["dev"]["p"]
                                       - t["c"] * t["p"]) / (t["c"] * t["p"]), 1),
    } for o in out])
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
