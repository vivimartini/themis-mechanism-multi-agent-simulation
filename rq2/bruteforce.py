"""Four-actor world solved on a full 2-unit grid, so no optimiser enters the regret."""
import numpy as np
import pandas as pd
from paths import ACTORS_CSV

EBAR = 6.6
T_GRID = np.round(np.linspace(0, 1, 51), 4)[1:]
df0 = pd.read_csv(ACTORS_CSV)


def build(names, e, pop, AB, AC, AT):
    e = np.asarray(e, float)
    pop = np.asarray(pop, float)
    AB = np.asarray(AB, float)
    AC = np.asarray(AC, float)
    AT = np.asarray(AT, float)
    N = len(names)
    w = pop * e
    w = w / w.sum()
    contrib = e > EBAR
    masks = []
    for m in range(1, 2 ** N):
        mem = np.array([(m >> j) & 1 for j in range(N)], bool)
        if mem.sum() < 2 or not (mem & contrib).any() or not (mem & ~contrib).any():
            continue
        masks.append(mem)
    M = np.array(masks)
    return dict(
        M=M, COV=(M * w).sum(1),
        EX=(M * np.maximum(e - EBAR, 0) * pop).sum(1),
        DE=(M * np.maximum(EBAR - e, 0) * pop).sum(1),
        e=e, contrib=contrib, AB=AB, AC=AC, AT=AT, names=names,
    )


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
                best = dict(obj=obj, p=price, c=COV[ki], will=will.copy(),
                            members=mem.copy())
    return best


if __name__ == "__main__":
    pick = ["CHINA", "UNITED STATES", "INDIA", "INDONESIA"]
    idx = [df0["name"].tolist().index(n) for n in pick]
    sub = df0.iloc[idx].reset_index(drop=True)
    AB0 = sub["alpha_base"].to_numpy(float)
    AC0 = sub["alpha_cov"].to_numpy(float)
    E = sub["e"].to_numpy(float)
    P = sub["pop_m"].to_numpy(float)
    AT0 = sub["alpha_trf"].to_numpy(float)

    base = solve(build(pick, E, P, AB0, AC0, AT0))
    print(f"4-actor world: {pick}")
    print(f"truthful p={base['p']:.2f} c={base['c']:.3f} "
          f"members={[n for n, m in zip(pick, base['members']) if m]}\n")
    print("EXHAUSTIVE regret (2-unit grid, no oracle):")
    for i, n in enumerate(pick):
        u0 = -abs(base["p"] - base["will"][i])
        best = 0.0
        for db in np.arange(-60, 61, 2.0):
            for dc in np.arange(-60, 61, 2.0):
                AB = AB0.copy()
                AC = AC0.copy()
                AB[i] += db
                AC[i] += dc
                out = solve(build(pick, E, P, AB, AC, AT0))
                if out is not None:
                    best = max(best, -abs(out["p"] - out["will"][i]) - u0)
        role = "PIVOT" if n == "UNITED STATES" else ("BOUNDARY" if n == "INDONESIA" else "")
        print(f"  {n:<16} regret={best:6.2f}  {role}")
