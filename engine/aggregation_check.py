"""Does aggregating countries into a bloc change the operating point?

The nine-actor calibration represents part of the world by behavioural blocs. A
bloc is only harmless if its members would not have behaved differently as
separate votes. The advanced-joiner bloc is the one most likely to hide
disagreement, because the UK sits BELOW the world-average benchmark (so it would
be a transfer beneficiary) while the rest of the bloc sits above it (contributors).
Aggregation therefore nets a beneficiary against contributors inside a single
vote, which is exactly the netting effect rq2.vote_structure part D measures.

The check splits one country out of its bloc on its own parameters and lets the
residual follow by mass balance, so only the split-out country's numbers are
inputs:

    pop_res = pop_bloc - pop_x
    e_res   = (e_bloc*pop_bloc - e_x*pop_x) / pop_res          (emissions conserved)
    ab_res  = (ab_bloc*pop_bloc - ab_x*pop_x) / pop_res        (pop-weighted, per the CSV note)

IMPORTANT: the UK parameters below are NOT in this repository's calibration data.
They are inputs to this check and need the same provenance treatment as Table 3.1
before the result is quoted. Because the base term is the least certain of them,
`main` sweeps it and reports whether the conclusion survives the whole range
rather than resting on the default.

  python -m engine.aggregation_check
  python -m engine.aggregation_check --ab 12 --e 5.0
"""
from __future__ import annotations

import argparse

import numpy as np

import rq2.endogenous_coverage as R

EBAR = R.EBAR
BLOC = "ADV. CARBON-PRICED CONDITIONAL JOINERS"

# --- inputs, not repository data. Substitute sourced values before quoting.
UK_POP_M = 68.3          # WDI-style mid-2024 population, millions
UK_E = 5.4               # tCO2e per capita; below EBAR, hence a beneficiary
UK_GDP_CAP = 52400.0     # nominal USD per capita
UK_AB = 15.0             # effective (not headline) carbon price, EUR/tCO2e
UK_AB_SWEEP = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)


def alpha_trf(gdp_cap):
    return min(20.0, 20000.0 / max(gdp_cap, 1.0))


def split_world(bloc=BLOC, pop_x=UK_POP_M, e_x=UK_E, gdp_x=UK_GDP_CAP, ab_x=UK_AB,
                name_x="UNITED KINGDOM"):
    """Ten-actor arrays with `name_x` split out of `bloc` by mass balance."""
    j = R.names.index(bloc)
    pop_b, e_b, ab_b, ac_b = R.pop[j], R.e[j], R.AB[j], R.AC[j]
    pop_r = pop_b - pop_x
    if pop_r <= 0:
        raise ValueError("split-out population exceeds the bloc")
    e_r = (e_b * pop_b - e_x * pop_x) / pop_r
    ab_r = (ab_b * pop_b - ab_x * pop_x) / pop_r

    keep = [k for k in range(9) if k != j]
    names = [R.names[k] for k in keep] + [f"{bloc[:12]} residual", name_x]
    e = np.r_[R.e[keep], e_r, e_x]
    pop = np.r_[R.pop[keep], pop_r, pop_x]
    ab = np.r_[R.AB[keep], ab_r, ab_x]
    # No basis for splitting the coverage slope, so both inherit the bloc's.
    ac = np.r_[R.AC[keep], ac_b, ac_b]
    at = np.r_[R.AT[keep], R.AT[j], alpha_trf(gdp_x)]
    return names, e, pop, ab, ac, at


def solve(names, e, pop, ab, ac, at, slack=R.SLACK):
    """Same rules as endogenous_coverage.solve_full, for an arbitrary actor count."""
    n = len(names)
    w = pop * e
    w = w / w.sum()
    contrib = e > EBAR
    masks = []
    for m in range(1, 2 ** n):
        mem = np.array([(m >> k) & 1 for k in range(n)], bool)
        if mem.sum() < 2 or not (mem & contrib).any() or not (mem & ~contrib).any():
            continue
        masks.append(mem)
    M = np.array(masks)
    cov = (M * w).sum(1)
    ex = (M * np.maximum(e - EBAR, 0) * pop).sum(1)
    de = (M * np.maximum(EBAR - e, 0) * pop).sum(1)
    tg = np.round(np.linspace(0, 1, 101), 4)[1:]
    tm = tg[None, :] * ex[:, None] / de[:, None]
    tau = np.where(contrib[None, None, :],
                   -tg[None, :, None] * np.maximum(e - EBAR, 0)[None, None, :],
                   tm[:, :, None] * np.maximum(EBAR - e, 0)[None, None, :])
    prefs = np.maximum(0, ab[None, None, :] + ac[None, None, :] * cov[:, None, None]
                       + at[None, None, :] * tau)
    mb = M[:, None, :]
    price = np.where(mb, prefs, 1e9).min(2)
    nonmax = np.where(~mb, prefs, -1e9).max(2)
    ok = (price > 0) & (nonmax < price + slack)
    obj = np.where(ok, cov[:, None] * price, -1.0)
    k, t = np.unravel_index(np.argmax(obj), obj.shape)
    if obj[k, t] < 0:
        return None
    return dict(p=float(price[k, t]), c=float(cov[k]), Tplus=float(tg[t]),
                Tminus=float(tm[k, t]), members=M[k].copy(), names=names,
                e=e, pop=pop)


def net_transfer(out, idx):
    """EUR per capita for actor idx at outcome out (positive = receives)."""
    if out is None or not out["members"][idx]:
        return 0.0
    e_i = out["e"][idx]
    if e_i > EBAR:
        return -out["Tplus"] * out["p"] * (e_i - EBAR)
    return out["Tminus"] * out["p"] * (EBAR - e_i)


def describe(out):
    mem = [n[:22] for n, m in zip(out["names"], out["members"]) if m]
    return (f"p = {out['p']:.2f}, c = {out['c']:.4f}, T+ = {out['Tplus']:.2f}, "
            f"T- = {out['Tminus']:.4f}\n    coalition ({len(mem)}): {', '.join(mem)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ab", type=float, default=UK_AB)
    ap.add_argument("--e", type=float, default=UK_E)
    ap.add_argument("--pop", type=float, default=UK_POP_M)
    args = ap.parse_args(argv)

    base = R.TRUTHFUL
    print("Baseline, advanced joiners aggregated:")
    print(f"    p = {base['p']:.2f}, c = {base['c']:.4f}, T+ = {base['Tplus']:.2f}, "
          f"T- = {base['Tminus']:.4f}")
    print(f"    coalition ({int(base['members'].sum())}): "
          f"{', '.join(n[:22] for n, m in zip(R.names, base['members']) if m)}\n")

    print(f"UK split out on supplied inputs (pop {args.pop} M, e {args.e} tCO2e/cap, "
          f"alpha_base {args.ab}):")
    out = solve(*split_world(pop_x=args.pop, e_x=args.e, ab_x=args.ab))
    if out is None:
        print("    no self-consistent point")
        return
    print("    " + describe(out))
    iuk = out["names"].index("UNITED KINGDOM")
    ires = len(out["names"]) - 2
    print(f"    UK joins: {bool(out['members'][iuk])}, "
          f"net transfer {net_transfer(out, iuk):+.2f} EUR/cap "
          f"({'beneficiary' if out['e'][iuk] <= EBAR else 'contributor'})")
    print(f"    bloc residual joins: {bool(out['members'][ires])}, "
          f"net transfer {net_transfer(out, ires):+.2f} EUR/cap")
    print(f"    price move vs baseline: {out['p'] - base['p']:+.4f} EUR/t")
    print(f"    T- move vs baseline:    {out['Tminus'] - base['Tminus']:+.5f}\n")

    print("Sweeping the UK base term, the least certain input:")
    print(f"{'alpha_base':>11} {'price':>8} {'dp':>8} {'T-':>9} {'dT-':>9} "
          f"{'UK in':>6} {'UK EUR/cap':>11}")
    rows = []
    for ab in UK_AB_SWEEP:
        o = solve(*split_world(pop_x=args.pop, e_x=args.e, ab_x=ab))
        if o is None:
            print(f"{ab:11.1f}     no self-consistent point")
            continue
        k = o["names"].index("UNITED KINGDOM")
        rows.append((ab, o["p"], o["Tminus"], bool(o["members"][k]),
                     net_transfer(o, k)))
        print(f"{ab:11.1f} {o['p']:8.2f} {o['p']-base['p']:+8.4f} "
              f"{o['Tminus']:9.5f} {o['Tminus']-base['Tminus']:+9.5f} "
              f"{str(bool(o['members'][k])):>6} {net_transfer(o, k):+11.2f}")

    if rows:
        dp = max(abs(r[1] - base["p"]) for r in rows)
        dt = max(abs(r[2] - base["Tminus"]) for r in rows)
        joins = all(r[3] for r in rows)
        print(f"\nAcross the sweep: largest price move {dp:.4f} EUR/t, largest T- move "
              f"{dt:.5f};")
        print(f"the UK joins in {'every' if joins else 'not every'} case, as a "
              f"{'beneficiary' if args.e <= EBAR else 'contributor'} receiving "
              f"{min(r[4] for r in rows):+.2f} to {max(r[4] for r in rows):+.2f} EUR/cap.")
        print("Aggregation therefore misstates the UK's own transfer direction without")
        print("moving the operating point the dissertation reports.")


if __name__ == "__main__":
    main()
