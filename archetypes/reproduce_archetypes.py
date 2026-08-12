"""Published-archetype reproduction."""
from itertools import combinations

from archetypes.preferences import (
    ASSUMED_CONTRIB, BENEF, C_HI, C_LO, CONTRIB,
)

# Published archetype table (§4.2): name, pop (M), e (tCO2e/cap), GHG share (%)
ARCH = {
    "A1": ("EU27 bloc",          453,  7.2,  6.2),
    "A2": ("Other high-income",  404, 10.7,  8.3),
    "A3": ("United States",      349, 17.2, 11.6),
    "A4": ("China",             1427, 10.8, 29.6),
    "A5": ("Fossil exporters",   470, 14.6, 13.2),
    "A6": ("India & developing",3723,  3.7, 26.3),
    "A7": ("Less wealthy",      1326,  1.9,  4.8),
}
EBAR = 6.4

PUB_PRICE, PUB_TPLUS, PUB_TMINUS = 20.97, 11.67, 6.09
PUB_COALITION = ("A1", "A2", "A4", "A6", "A7")
PUB_TRANSFERS = {"A1": -9, "A2": -50, "A4": -51, "A6": +17, "A7": +28}
PUB_POOL_BN, PUB_COV_GHG, PUB_COV_POP = 98.0, 0.752, 0.90

# All contributor curves (digitised + assumed)
ALL_CONTRIB = {**CONTRIB, **ASSUMED_CONTRIB}

# Beneficiary rate grid for Stage B (0 … 30)
TGRID = [i / 1000.0 * 30.0 for i in range(1001)]


def interp(lo, hi, c):
    return lo + (c - C_LO) / (C_HI - C_LO) * (hi - lo)


def ex_de(coalition):
    ex = sum(ARCH[a][1] * (ARCH[a][2] - EBAR) for a in coalition if ARCH[a][2] > EBAR)
    de = sum(ARCH[a][1] * (EBAR - ARCH[a][2]) for a in coalition if ARCH[a][2] < EBAR)
    return ex, de


def per_capita_transfer(a, tplus, tminus):
    gap = ARCH[a][2] - EBAR
    return -tplus * gap if gap > 0 else tminus * (-gap)


def coverage(coalition):
    ghg = sum(ARCH[a][3] for a in coalition) / 100.0
    pop = sum(ARCH[a][1] for a in coalition) / sum(v[1] for v in ARCH.values())
    return ghg, pop


def stage_a():
    print("=" * 72)
    print("STAGE A  transfer accounting")
    print("=" * 72)
    ex, de = ex_de(PUB_COALITION)
    print(f"contributor excess mass    {ex:10.1f}")
    print(f"beneficiary deficit mass   {de:10.1f}")
    print(f"balanced t+/t-             {de/ex:10.4f}   ({de/ex:.2f})")
    tp = (PUB_TMINUS + 0.01) * de / ex
    print(f"pair anchored at t-=6.10:  t+ = {tp:.2f}, t- = 6.10")
    print("\nper-capita transfers, EUR/person/year (computed vs published)")
    for a in PUB_COALITION:
        print(f"  {ARCH[a][0]:22s} {per_capita_transfer(a, PUB_TPLUS, PUB_TMINUS):+8.2f}"
              f" {PUB_TRANSFERS[a]:+8d}")
    pool = sum(-per_capita_transfer(a, PUB_TPLUS, PUB_TMINUS) * ARCH[a][1]
               for a in PUB_COALITION if ARCH[a][2] > EBAR) / 1000.0
    payout = sum(per_capita_transfer(a, PUB_TPLUS, PUB_TMINUS) * ARCH[a][1]
                 for a in PUB_COALITION if ARCH[a][2] < EBAR) / 1000.0
    print(f"\ncontributor pool   EUR {pool:5.1f}B   published EUR {PUB_POOL_BN:.0f}B")
    print(f"payout             EUR {payout:5.1f}B   imbalance {abs(pool-payout)/pool*100:.2f}%")
    cg, cp = coverage(PUB_COALITION)
    print(f"emissions coverage {cg*100:5.1f}%   published {PUB_COV_GHG*100:.1f}%")
    print(f"population coverage{cp*100:5.1f}%   published {PUB_COV_POP*100:.0f}%")


def willingness(a, c, tplus, tminus, contrib_tbl=None, benef_tbl=None):
    contrib_tbl = ALL_CONTRIB if contrib_tbl is None else contrib_tbl
    benef_tbl = BENEF if benef_tbl is None else benef_tbl
    if a in contrib_tbl:
        d = contrib_tbl[a]
        xi = interp(d["lo"][0], d["hi"][0], c)
        sl = interp(-d["lo"][0] / d["lo"][1], -d["hi"][0] / d["hi"][1], c)
        return max(0.0, xi + sl * tplus)
    d = benef_tbl[a]
    k = interp(d["lo"], d["hi"], c)
    return max(0.0, tminus / k) if k > 0 else 0.0


def solve_stage_b(contrib_tbl=None, benef_tbl=None):
    """Argmax c*p over coalitions and the beneficiary rate. Returns the best tuple."""
    keys = list(ARCH)
    best = None
    for size in range(1, len(keys) + 1):
        for coal in combinations(keys, size):
            ex, de = ex_de(coal)
            if ex <= 0 or de <= 0:
                continue
            c = sum(ARCH[a][3] for a in coal) / 100.0
            for tm in TGRID:
                tp = tm * de / ex
                w = {a: willingness(a, c, tp, tm, contrib_tbl, benef_tbl)
                     for a in keys}
                p = min(w[a] for a in coal)
                if p <= 0 or any(w[a] >= p for a in keys if a not in coal):
                    continue
                obj = c * p
                if best is None or obj > best[0]:
                    best = (obj, coal, c, p, tp, tm)
    return best


def worked_example(price=50.0, share=0.2, ebar_world=6.6):
    """The universal-membership example in the Themis proposal.

    Everyone joins at a common price, a fixed share of revenue is pooled and paid
    back equally per head. Revenue per capita is price*e_i, so the equal payment
    is share*price*ebar and country i nets share*price*(ebar - e_i). Closed form,
    so this is an exact check on the transfer direction and scale rather than a
    numerical reproduction.
    """
    payment = share * price * ebar_world
    return payment, lambda e_i: payment - share * price * e_i


def stage_zero():
    print("=" * 72)
    print("STAGE 0  published worked example (universal membership)")
    print("=" * 72)
    payment, net = worked_example()
    print(f"equal per-capita payment   EUR {payment:.2f}   published EUR 66")
    for label, e_i, pub in (("South Korea", 12.8, -62), ("India", 3.0, +36)):
        print(f"  {label:<12} at {e_i:4.1f} tCO2e/cap: net {net(e_i):+7.2f}"
              f"   published {pub:+d}")


def sensitivity():
    """How much of the Stage B price gap is digitisation error?

    Perturb the two lines the gap is most plausibly attributable to and re-solve.
    India (A6) is a beneficiary, so its digitised parameter is the rate slope k;
    China (A4) is a contributor, so its parameter is the curve intercept.
    """
    print("\n" + "=" * 72)
    print("STAGE B SENSITIVITY  digitisation check")
    print("=" * 72)
    base = solve_stage_b()
    p0 = base[3]
    print(f"baseline price EUR {p0:.2f}\n")
    rows = []
    for freq, frac in (("India (A6) slope", 0.10), ("China (A4) intercept", 0.05)):
        for sign in (+1, -1):
            cont = {k: {kk: vv for kk, vv in v.items()} for k, v in ALL_CONTRIB.items()}
            ben = {k: dict(v) for k, v in BENEF.items()}
            if "India" in freq:
                for key in ("lo", "hi"):
                    ben["A6"][key] = BENEF["A6"][key] * (1 + sign * frac)
            else:
                for key in ("lo", "hi"):
                    old = ALL_CONTRIB["A4"][key]
                    cont["A4"][key] = (old[0] * (1 + sign * frac), old[1])
            b = solve_stage_b(cont, ben)
            if b is None:
                rows.append((freq, sign * frac, None, None, None))
                continue
            rows.append((freq, sign * frac, b[3], b[3] - p0,
                         b[1] == base[1] and abs(b[2] - base[2]) < 1e-9))
    for freq, d, p, dp, same in rows:
        if p is None:
            print(f"  {freq:<22} {d:+.0%}:  no self-consistent point")
            continue
        print(f"  {freq:<22} {d:+.0%}:  price EUR {p:5.2f}  ({dp:+.2f})"
              f"   coalition+coverage {'unchanged' if same else 'CHANGED'}")
    moves = [abs(r[3]) for r in rows if r[3] is not None]
    if moves:
        print(f"\nlargest price move from a plausible digitisation error: "
              f"EUR {max(moves):.2f}; the Stage B gap is EUR "
              f"{abs(p0 - PUB_PRICE):.2f}.")


def stage_b():
    print("\n" + "=" * 72)
    print("STAGE B  elicitation reproduction")
    print("=" * 72)
    best = solve_stage_b()
    if best is None:
        print("no self-consistent point found")
        return
    _, coal, c, p, tp, tm = best
    print(f"selected coalition   {', '.join(coal)}")
    print(f"operating price      EUR {p:.2f}   published EUR {PUB_PRICE:.2f}"
          f"   gap EUR {abs(p - PUB_PRICE):.2f}")
    print(f"coverage             {c*100:.1f}%   published {PUB_COV_GHG*100:.1f}%")
    print(f"transfer rates       t+ {tp:.2f}  t- {tm:.2f}"
          f"   published {PUB_TPLUS:.2f} / {PUB_TMINUS:.2f}")
    # Quick check that China / India sit near the published price
    w_cn = willingness("A4", PUB_COV_GHG, PUB_TPLUS, PUB_TMINUS)
    w_in = willingness("A6", PUB_COV_GHG, PUB_TPLUS, PUB_TMINUS)
    print(f"\nsanity at published (c,t): China p={w_cn:.2f}, India p={w_in:.2f}"
          f"  (want ~{PUB_PRICE})")


if __name__ == "__main__":
    stage_zero()
    print()
    stage_a()
    stage_b()
    sensitivity()
