"""RQ2 smoke validation."""
import sys
from pathlib import Path
import numpy as np
import rq2.endogenous_coverage as R
import rq2.coarse_solver as F
import rq2.fixed_coverage as FC
from rq2.oracle import cma_minimize, DEFAULT_SEED
from paths import (CLIMATE_NPZ, EXINTERIM_NPZ, GUARDRAIL_NPZ, HEADLINES_JSON,
                   INFORISK_NPZ, LOCALITY_NPZ, MC_PRICES_NPZ, PSRO_NPZ,
                   SEMANTICS_NPZ, SLACK_NPZ, TRANSFERPARAM_NPZ, VOTESTRUCT_NPZ)

TOL = 1.0   # regret/NashConv tolerance (budget=300 is a smoke test, not full 900)


def main():
    """Run release smoke checks and return a process exit code."""
    errors = []

    def check(label, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            errors.append(label)

    print("=== validate_rq2 ===\n")

    # --- derived data consumed by rq2.make_figures
    figure_inputs = (
        MC_PRICES_NPZ, EXINTERIM_NPZ, GUARDRAIL_NPZ, PSRO_NPZ, HEADLINES_JSON,
        SEMANTICS_NPZ, INFORISK_NPZ, CLIMATE_NPZ, VOTESTRUCT_NPZ,
        TRANSFERPARAM_NPZ, LOCALITY_NPZ, SLACK_NPZ,
    )
    missing_inputs = [Path(path).name for path in figure_inputs
                      if not Path(path).exists()]
    check("all figure inputs present", not missing_inputs,
          "" if not missing_inputs else f"missing: {', '.join(missing_inputs)}")

    # --- solver agreement at point calibration
    t = R.TRUTHFUL
    tf = F.solve(R.AB, R.AC)
    p0, _ = FC.outcome(FC.TRUTH)
    check("endogenous price ≈ 26.70", abs(t["p"] - 26.70) < 0.05,
          f"p={t['p']:.4f}")
    check("endogenous c ≈ 0.8686", abs(t["c"] - 0.86864) < 0.001,
          f"c={t['c']:.6f}")
    check("fixed vs endogenous price", abs(p0 - t["p"]) < 1e-3)
    check("coarse vs fine grid",
          abs(tf["p"] - t["p"]) < 1e-3
          and abs(tf["c"] - t["c"]) < 1e-3)
    check("T rates match control", abs(t["Tplus"] - FC.T_PLUS) < 0.01
          and abs(t["Tminus"] - FC.T_MINUS) < 0.01)

    # --- DSIC control: zero peaked regret
    regrets_pk = [FC.best_response(i, "peaked", budget=300)[0]
                  for i in range(9)]
    nc_fixed = sum(regrets_pk)
    check("fixed-coverage peaked NashConv = 0", nc_fixed < 0.01,
          f"NashConv={nc_fixed:.4f}")

    # --- endogenous peaked NashConv (reference ~19.6)
    regrets = [R.best_response(i, "peaked", budget=300)[0] for i in range(9)]
    nc_endog = sum(regrets)
    check("endogenous peaked NashConv ≈ 19.6", abs(nc_endog - 19.6) < TOL,
          f"NashConv={nc_endog:.3f}")

    # --- oracle reproducibility (hybrid, coarse grid)
    r_a = [F.oracle(i, R.AB, R.AC, {}, seed=DEFAULT_SEED + i)[0]
           for i in range(9)]
    r_b = [F.oracle(i, R.AB, R.AC, {}, seed=DEFAULT_SEED + i)[0]
           for i in range(9)]
    check("hybrid oracle reproducible", np.allclose(r_a, r_b))

    # --- shared CMA search smoke test
    f1, x1 = cma_minimize(
        lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2,
        200, seed=DEFAULT_SEED,
    )
    check("cma_minimize finds minimum", abs(f1) < 0.5,
          f"f={f1:.4f}, x={x1}")

    # --- obstruction headline (China damage ~69%)
    from rq2.obstruction_voteselling import obstruction
    cp0 = t["c"] * t["p"]
    f_cn, _, _ = obstruction(R.names.index("CHINA"), budget=300)
    dmg_cn = 100 * (cp0 - f_cn) / cp0
    check("China obstruction damage ≈ 69%", abs(dmg_cn - 68.9) < 2.0,
          f"{dmg_cn:.1f}%")

    # --- engine.themis vs the vectorised solver, UNDER MISREPORTS.
    # rq2.robustness runs the full sweep; this is the smoke-test version.
    import pandas as pd
    from engine.themis import run_mechanism_selfconsistent, EngineConfig
    from paths import ACTORS_CSV

    base = pd.read_csv(ACTORS_CSV)
    rng = np.random.default_rng(5)
    cfg = EngineConfig(t_steps=101)
    worst = 0.0
    for _ in range(6):
        i = int(rng.integers(0, 9))
        dx = rng.uniform(-40, 40, 2)
        ab = R.AB.copy()
        ac = R.AC.copy()
        ab[i] += dx[0]
        ac[i] += dx[1]
        fast = R.solve_full(ab, ac)
        df = base.copy()
        df["alpha_base"] = ab
        df["alpha_cov"] = ac
        ref = run_mechanism_selfconsistent(df, config=cfg)
        if fast is not None:
            worst = max(worst, abs(ref["p_star"] - fast["p"]))
    check("engine.themis agrees under misreports", worst < 1e-6,
          f"max |dp| = {worst:.2e}")

    # --- report semantics: the DSIC claim holds only on the peaked domain
    import rq2.report_semantics as RS
    nc_peak = sum(RS.regret_fixed(i, "peak", budget=200)[0]
                  for i in range(9))
    nc_cap = sum(RS.regret_fixed(i, "cap", budget=200)[0]
                 for i in range(9))
    check("fixed coverage: peaked NashConv = 0", nc_peak < 0.01,
          f"{nc_peak:.4f}")
    check("fixed coverage: cap reading IS manipulable", nc_cap > 1.0,
          f"{nc_cap:.2f}")

    # --- slack tolerance is not load-bearing
    t_tight = R.solve_full(R.AB, R.AC, slack=0.0)
    check("truthful point invariant to self-consistency slack",
          abs(t_tight["p"] - t["p"]) < 1e-9
          and t_tight["c"] == t["c"],
          f"p={t_tight['p']:.4f}")

    print()
    if errors:
        print(f"{len(errors)} check(s) failed: {', '.join(errors)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
