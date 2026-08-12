"""Scenario-prior simulation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import rq2.endogenous_coverage as R
from paths import MC_PRICES_NPZ

OUT = MC_PRICES_NPZ

N_DRAWS = 4000
SEED = 0

BASE_RANGES = {
    "CHINA": (0.0, 8.63),
    "UNITED STATES": (2.0, 4.0),
    "EUROPEAN UNION": (15.0, 25.0),
    "ADV. CARBON-PRICED CONDITIONAL JOINERS": (4.0, 8.0),
    "INDIA": (0.0, 4.0),
}


def _base_bounds():
    lo = np.zeros(9)
    hi = np.zeros(9)
    for name, (a, b) in BASE_RANGES.items():
        j = R.names.index(name)
        lo[j], hi[j] = a, b
    return lo, hi


def draw_world(rng: np.random.Generator, ab_lo, ab_hi):
    ab = rng.uniform(ab_lo, ab_hi)  # unlisted actors stay at 0
    ac = R.AC * rng.uniform(0.5, 1.5) * rng.lognormal(0.0, 0.2, 9)
    return ab, ac


def _price_setter(out, ab, ac):
    """Index of the member whose willingness binds the price at this outcome.

    The price is the minimum member willingness, so the binding actor is the
    argmin over members of its willingness under THIS world's parameters.
    """
    tau = np.where(R.contrib, -out["Tplus"] * np.maximum(R.e - R.EBAR, 0.0),
                   out["Tminus"] * np.maximum(R.EBAR - R.e, 0.0))
    will = np.maximum(0.0, ab + ac * out["c"] + R.AT * tau)
    masked = np.where(out["members"], will, np.inf)
    return int(np.argmin(masked))


def run_prior(n: int = N_DRAWS, seed: int = SEED, fix_cov: bool = False,
              fix_base: bool = False):
    """Solve n prior draws.

    Returns price, coverage, per-draw coalition membership and the index of the
    price-setting actor. Membership and the binding actor are what RQ1's
    identifiability claim rests on, so they are recorded rather than recomputed
    by hand. The draw sequence is unchanged, so p and c still bit-match the
    committed npz.

    fix_cov / fix_base hold one channel at its baseline for the variance
    attribution; both default off, which is the reported prior.
    """
    ab_lo, ab_hi = _base_bounds()
    ab_mid = 0.5 * (ab_lo + ab_hi)
    rng = np.random.default_rng(seed)
    p = np.empty(n)
    c = np.empty(n)
    members = np.empty((n, 9), dtype=bool)
    setter = np.empty(n, dtype=int)
    for i in range(n):
        ab, ac = draw_world(rng, ab_lo, ab_hi)   # always drawn: keeps the stream fixed
        if fix_base:
            ab = ab_mid
        if fix_cov:
            ac = R.AC
        out = R.solve_full(ab, ac)
        if out is None:
            raise RuntimeError(f"no self-consistent point at draw {i}")
        p[i] = out["p"]
        c[i] = out["c"]
        members[i] = out["members"]
        setter[i] = _price_setter(out, ab, ac)
    return p, c, members, setter


def membership_stats(members, setter, names=None):
    """Join frequency, modal coalition and price-setter frequency."""
    names = names or R.names
    n = len(members)
    join = members.mean(0)
    keys, counts = np.unique(members, axis=0, return_counts=True)
    top = int(np.argmax(counts))
    setter_freq = np.bincount(setter, minlength=9) / n
    return dict(join=join, modal=keys[top], modal_share=counts[top] / n,
                n_distinct=len(keys), setter=setter_freq)


def summarise(p: np.ndarray) -> dict:
    return {
        "n": int(p.size),
        "median": float(np.median(p)),
        "mean": float(p.mean()),
        "p5": float(np.percentile(p, 5)),
        "p95": float(np.percentile(p, 95)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="compare regenerated draws to existing mc_operating_prices.npz")
    args = ap.parse_args(argv)

    print(f"scenario prior: n={args.n}, seed={args.seed}")
    print(f"baseline alphas (China α_base={R.AB[R.names.index('CHINA')]:.2f})")
    p, c, members, setter = run_prior(n=args.n, seed=args.seed)
    s = summarise(p)
    print(f"median {s['median']:.2f}  mean {s['mean']:.2f}  "
          f"5–95% {s['p5']:.2f}–{s['p95']:.2f}")

    # ---- what RQ1 actually claims: membership is identified, the price is not
    st = membership_stats(members, setter)
    print("\nCoalition membership across the prior:")
    for j, nm in enumerate(R.names):
        print(f"  {nm[:34]:<34} joins {100*st['join'][j]:5.1f}%   "
              f"sets the price {100*st['setter'][j]:5.1f}%")
    modal = [R.names[j][:16] for j in range(9) if st["modal"][j]]
    print(f"\ndistinct coalitions seen: {st['n_distinct']}")
    print(f"modal coalition ({len(modal)} members, {100*st['modal_share']:.1f}% of draws): "
          f"{', '.join(modal)}")
    always = [R.names[j][:16] for j in range(9) if st["join"][j] > 0.999]
    never = [R.names[j][:16] for j in range(9) if st["join"][j] < 0.001]
    print(f"in every draw: {', '.join(always) if always else 'none'}")
    print(f"in no draw:    {', '.join(never) if never else 'none'}")

    if args.check and args.out.exists():
        d = np.load(args.out)
        ok_p = np.allclose(p, d["p"])
        ok_c = np.allclose(c, d["c"])
        print(f"\nbit-match existing {args.out.name}: p={ok_p}, c={ok_c}")
        if not (ok_p and ok_c):
            raise SystemExit(1)
        return

    # ---- variance attribution: which channel drives the price?
    # Each pass re-draws the same stream and freezes one channel at baseline, so
    # the shares are a first-order attribution rather than an exact ANOVA
    # decomposition and are not expected to sum to one.
    print("\nVariance attribution for the operating price:")
    v_full = float(np.var(p))
    p_base, _, _, _ = run_prior(n=args.n, seed=args.seed, fix_cov=True)
    p_cov, _, _, _ = run_prior(n=args.n, seed=args.seed, fix_base=True)
    for lab, arr in (("base terms only (coverage slopes fixed)", p_base),
                     ("coverage slopes only (base terms fixed)", p_cov)):
        v = float(np.var(arr))
        print(f"  {lab:<42} var {v:8.3f}  = {100*v/v_full:5.1f}% of full")
    print(f"  {'full prior':<42} var {v_full:8.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, p=p, c=c, members=members, setter=setter,
             join=st["join"], setter_freq=st["setter"],
             var_full=np.array([v_full]),
             var_base=np.array([float(np.var(p_base))]),
             var_cov=np.array([float(np.var(p_cov))]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
