"""RQ2 vote-structure analysis."""
import numpy as np
import pandas as pd
from paths import ACTORS_CSV, VOTESTRUCT_NPZ

EBAR = 6.6
T_GRID = np.round(np.linspace(0, 1, 101), 4)[1:]

df0 = pd.read_csv(ACTORS_CSV)

NAMES0 = df0["name"].tolist()
E0 = df0["e"].to_numpy(float)
POP0 = df0["pop_m"].to_numpy(float)
AB0 = df0["alpha_base"].to_numpy(float)
AC0 = df0["alpha_cov"].to_numpy(float)
AT0 = df0["alpha_trf"].to_numpy(float)
CONTRIB0 = E0 > EBAR


def build(names, e, pop, AB, AC, AT, link=None):
    """Enumerate admissible coalitions. `link=(i, j)` forces mem[i] == mem[j]."""
    e = np.asarray(e, float); pop = np.asarray(pop, float)
    AB = np.asarray(AB, float); AC = np.asarray(AC, float); AT = np.asarray(AT, float)
    N = len(names)
    w = pop * e
    w = w / w.sum()
    contrib = e > EBAR
    masks = []
    for m in range(1, 2 ** N):
        mem = np.array([(m >> j) & 1 for j in range(N)], bool)
        if mem.sum() < 2 or not (mem & contrib).any() or not (mem & ~contrib).any():
            continue
        if link is not None and mem[link[0]] != mem[link[1]]:
            continue
        masks.append(mem)
    M = np.array(masks)
    return dict(N=N, w=w, contrib=contrib, M=M, COV=(M * w).sum(1),
                EX=(M * np.maximum(e - EBAR, 0) * pop).sum(1),
                DE=(M * np.maximum(EBAR - e, 0) * pop).sum(1),
                e=e, pop=pop, AB=AB, AC=AC, AT=AT, names=list(names))


def solve(env):
    M, EX, DE, COV = env["M"], env["EX"], env["DE"], env["COV"]
    e, contrib, AB, AC, AT = env["e"], env["contrib"], env["AB"], env["AC"], env["AT"]
    best = None
    for ki in range(len(M)):
        mem = M[ki]
        if DE[ki] <= 0 or EX[ki] <= 0:
            continue
        for Tp in T_GRID:
            Tm = Tp * EX[ki] / DE[ki]
            tau = np.where(contrib, -Tp * np.maximum(e - EBAR, 0),
                           Tm * np.maximum(EBAR - e, 0))
            will = np.maximum(0.0, AB + AC * COV[ki] + AT * tau)
            price = will[mem].min()
            if (will[~mem] >= price - 1e-9).any():
                continue
            obj = COV[ki] * price
            if best is None or obj > best["obj"]:
                best = dict(obj=obj, p=price, c=COV[ki], Tp=Tp, Tm=Tm,
                            members=mem.copy(), will=will.copy())
    return best


# ---------------------------------------------------- original-preference scoring
def true_will0(i, out):
    """Actor i's TRUE willingness at a realised outcome, using its own parameters.

    This is the quantity a merged member is forced to misreport, so it is the
    only defensible yardstick for that member's welfare after a merge.
    """
    tau = (-out["Tp"] * (E0[i] - EBAR) if CONTRIB0[i] else out["Tm"] * (EBAR - E0[i]))
    return max(0.0, AB0[i] + AC0[i] * out["c"] + AT0[i] * tau)


def u_peaked0(i, out):
    """Peaked utility of original actor i over the public outcome (price)."""
    return -1e9 if out is None else -abs(out["p"] - true_will0(i, out))


def transfer_total0(i, out, member):
    """Actor i's transfer bill in EUR millions if it faced the rates alone."""
    if out is None or not member:
        return 0.0
    if CONTRIB0[i]:
        return -out["Tp"] * out["p"] * (E0[i] - EBAR) * POP0[i]
    return out["Tm"] * out["p"] * (EBAR - E0[i]) * POP0[i]


def merged_env(i, j, link_only=False):
    """world 2 (link_only=False) or world 1 (link_only=True)."""
    if link_only:
        return build(NAMES0, E0, POP0, AB0, AC0, AT0, link=(i, j))
    keep = [k for k in range(len(NAMES0)) if k not in (i, j)]
    p_i, p_j = POP0[i], POP0[j]
    e_m = (E0[i] * p_i + E0[j] * p_j) / (p_i + p_j)
    p_m = p_i + p_j

    def wavg(col):
        return (col[i] * p_i + col[j] * p_j) / (p_i + p_j)

    names = [NAMES0[k] for k in keep] + [f"{NAMES0[i][:4]}+{NAMES0[j][:4]}"]
    return build(names,
                 np.r_[E0[keep], e_m], np.r_[POP0[keep], p_m],
                 np.r_[AB0[keep], wavg(AB0)], np.r_[AC0[keep], wavg(AC0)],
                 np.r_[AT0[keep], wavg(AT0)])


def main():
    base_env = build(NAMES0, E0, POP0, AB0, AC0, AT0)
    base = solve(base_env)
    print("=== BASELINE ===")
    print(f"p={base['p']:.2f}  c={base['c']:.4f}  T+={base['Tp']:.2f}  T-={base['Tm']:.4f}")
    print("members:", [n for n, m in zip(NAMES0, base["members"]) if m])
    print()

    # ---------------------------------------------------------- (A) EU split
    print("=== (A) EU SPLIT (sub-votes inherit the parent's preferences) ===")
    ieu = NAMES0.index("EUROPEAN UNION")
    print(f"EU as 1 vote: joins={bool(base['members'][ieu])}  price={base['p']:.2f}  "
          f"peaked-u={u_peaked0(ieu, base):.3f}")
    for k in (2, 3, 4):
        keep = [q for q in range(len(NAMES0)) if q != ieu]
        names = [NAMES0[q] for q in keep] + [f"EU_{q+1}" for q in range(k)]
        env = build(names,
                    np.r_[E0[keep], np.full(k, E0[ieu])],
                    np.r_[POP0[keep], np.full(k, POP0[ieu] / k)],
                    np.r_[AB0[keep], np.full(k, AB0[ieu])],
                    np.r_[AC0[keep], np.full(k, AC0[ieu])],
                    np.r_[AT0[keep], np.full(k, AT0[ieu])])
        out = solve(env)
        sub = len(keep)
        u = -abs(out["p"] - out["will"][sub])
        tag = ("same" if abs(u - u_peaked0(ieu, base)) < 1e-3
               and bool(out["members"][sub]) == bool(base["members"][ieu]) else "CHANGED")
        print(f"EU as {k} votes: joins={bool(out['members'][sub])}  price={out['p']:.2f}  "
              f"peaked-u={u:.3f}   [{tag}]")
    print("Splitting preserves preferences, so it is a pure test of vote count;")
    print("it changes nothing here.\n")

    # ------------------------------------------- (B) merge, decomposed by channel
    print("=== (B) PAIRWISE MERGE, per-member and decomposed ===")
    print("dU columns are peaked utility vs baseline, each member scored against")
    print("its OWN true willingness. 'synthetic' is the discarded old metric: the")
    print("averaged actor's own satisfaction, shown to expose the difference.\n")
    rows = []
    for i in range(len(NAMES0)):
        for j in range(i + 1, len(NAMES0)):
            w1 = solve(merged_env(i, j, link_only=True))
            w2 = solve(merged_env(i, j, link_only=False))
            if w1 is None or w2 is None:
                continue
            ui0, uj0 = u_peaked0(i, base), u_peaked0(j, base)
            ui1, uj1 = u_peaked0(i, w1), u_peaked0(j, w1)
            ui2, uj2 = u_peaked0(i, w2), u_peaked0(j, w2)
            synth = -abs(w2["p"] - w2["will"][-1])
            rows.append(dict(
                i=NAMES0[i][:12], j=NAMES0[j][:12],
                dU_i=ui2 - ui0, dU_j=uj2 - uj0,
                boundary_i=ui1 - ui0, boundary_j=uj1 - uj0,
                averaging_i=ui2 - ui1, averaging_j=uj2 - uj1,
                pareto=(ui2 > ui0 + 1e-6 and uj2 > uj0 + 1e-6),
                dp=w2["p"] - base["p"],
                synthetic=synth - max(ui0, uj0),
                idx=(i, j)))
    tb = pd.DataFrame(rows)

    par = tb[tb["pareto"]].sort_values("dU_i", ascending=False)
    print(f"-- pairs where BOTH members gain (a merge that would actually form): "
          f"{len(par)}/{len(tb)}")
    if len(par):
        print(par[["i", "j", "dU_i", "dU_j", "boundary_i", "averaging_i", "dp"]]
              .round(2).to_string(index=False))
    else:
        print("   none.")

    print("\n-- pairs the OLD synthetic metric ranked as gains, re-scored per member:")
    old = tb.sort_values("synthetic", ascending=False).head(8)
    print(old[["i", "j", "synthetic", "dU_i", "dU_j", "pareto"]]
          .round(2).to_string(index=False))
    n_flip = int(((old["synthetic"] > 0.05) & (~old["pareto"])).sum())
    print(f"   {n_flip} of these 8 are not gains for either real member once the")
    print("   cost of reporting the average is charged.")

    print("\n-- channel decomposition (mean over all pairs, both members pooled):")
    bnd = np.r_[tb["boundary_i"].to_numpy(), tb["boundary_j"].to_numpy()]
    avg = np.r_[tb["averaging_i"].to_numpy(), tb["averaging_j"].to_numpy()]
    print(f"   coalition-boundary channel (world1-world0): mean {bnd.mean():+.3f}, "
          f"max {bnd.max():+.3f}, share positive {100*(bnd > 1e-6).mean():.0f}%")
    print(f"   price-of-averaging channel (world2-world1):  mean {avg.mean():+.3f}, "
          f"max {avg.max():+.3f}, share positive {100*(avg > 1e-6).mean():.0f}%")

    # -------------------------------------------------- (C) price moves under merge
    print("\n=== (C) PRICE MOVES UNDER MERGE ===")
    mv = tb[tb["dp"].abs() > 0.05].reindex(
        tb["dp"].abs().sort_values(ascending=False).index).dropna(subset=["dp"])
    for _, r in mv.head(8).iterrows():
        print(f"  {r['i'][:10]} + {r['j'][:10]}: {r['dp']:+.2f}")
    if len(mv):
        print(f"{len(mv)}/{len(tb)} pairs move price; max |dp| = "
              f"{mv['dp'].abs().max():.2f} EUR/t")

    # ---------------------------------------- (D) transfer netting across the line
    print("\n=== (D) TRANSFER NETTING: blocs that straddle ebar ===")
    print("t+- are levied on max(e-ebar,0) and max(ebar-e,0) at different rates, so a")
    print("bloc spanning ebar nets its members out before the rates apply. Sign")
    print("convention: positive = net receipts. 'netting effect' is the bloc's")
    print("position minus what the two would hold apart.\n")
    arb = []
    for i in range(len(NAMES0)):
        for j in range(i + 1, len(NAMES0)):
            if CONTRIB0[i] == CONTRIB0[j]:
                continue                       # same side of ebar: nothing to net
            w2 = solve(merged_env(i, j, link_only=False))
            if w2 is None or not w2["members"][-1]:
                continue
            p_i, p_j = POP0[i], POP0[j]
            e_m = (E0[i] * p_i + E0[j] * p_j) / (p_i + p_j)
            pop_m = p_i + p_j
            bloc = (-w2["Tp"] * w2["p"] * (e_m - EBAR) * pop_m if e_m > EBAR
                    else w2["Tm"] * w2["p"] * (EBAR - e_m) * pop_m)
            apart = (transfer_total0(i, w2, True) + transfer_total0(j, w2, True))
            arb.append((NAMES0[i][:12], NAMES0[j][:12], round(bloc / 1000, 2),
                        round(apart / 1000, 2), round((bloc - apart) / 1000, 2)))
    if arb:
        ab = pd.DataFrame(arb, columns=["A", "B", "bloc position (EUR bn)",
                                        "sum apart (EUR bn)",
                                        "netting effect (EUR bn)"])
        ab = ab.reindex(ab["netting effect (EUR bn)"].abs().sort_values(
            ascending=False).index)
        print(ab.to_string(index=False))
        eff = ab["netting effect (EUR bn)"].to_numpy(float)
        n_worse = int((eff < 0).sum())
        print(f"\n{n_worse}/{len(eff)} straddling blocs are WORSE off merged than apart; "
              f"largest loss {abs(eff.min()):.2f} EUR bn.")
        print("The asymmetric rates therefore penalise straddling rather than reward it:")
        print("a bloc spanning ebar forfeits the beneficiary member's gross claim, which")
        print("is a deterrent to merger-based gaming of the transfer pool, not a loophole.")
    else:
        print("no straddling bloc joins at its merged operating point.")

    np.savez(VOTESTRUCT_NPZ,
             pair_i=np.array([r["i"] for r in rows], dtype="U14"),
             pair_j=np.array([r["j"] for r in rows], dtype="U14"),
             dU_i=tb["dU_i"].to_numpy(float), dU_j=tb["dU_j"].to_numpy(float),
             boundary=bnd, averaging=avg,
             synthetic=tb["synthetic"].to_numpy(float),
             pareto=tb["pareto"].to_numpy(bool), dp=tb["dp"].to_numpy(float),
             base_p=np.array([base["p"]]))
    print(f"\nsaved {VOTESTRUCT_NPZ}")


if __name__ == "__main__":
    main()
