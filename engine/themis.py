"""Reference Themis mechanism."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

EBAR_DEFAULT = 6.6  # EDGAR 2025 world average; published seven-archetype table uses 6.4

@dataclass
class EngineConfig:
    ebar: float = EBAR_DEFAULT
    c_min: float = 0.01
    c_max: float = 1.0
    c_steps: int = 100
    t_min: float = 0.0
    t_max: float = 1.0
    t_steps: int = 101
    t_cap: Optional[float] = None


def normalise_actor_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "idx", "e", "pop_m", "gdp_cap", "headline_price", "effective_price",
        "alpha_base", "alpha_cov", "alpha_trf", "weight",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    # Always DERIVE emission weights from the primary data. The CSV also ships a
    # `weight` column, but it is stored to 4 d.p.; trusting it made this engine
    # solve a slightly different model from the RQ2 solvers (coverage 0.8687 vs
    # 0.868644, price 26.705 vs 26.703). rq2.robustness part C now checks the two
    # implementations agree under misreports, which only holds if both derive.
    out["weight"] = out["pop_m"] * out["e"]
    wsum = out["weight"].sum()
    if wsum > 0:
        out["weight"] = out["weight"] / wsum
    out["role"] = np.where(out["e"] > EBAR_DEFAULT, "Contributor", "Beneficiary")
    return out


def arrays(actor_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    df = normalise_actor_df(actor_df)
    return {
        "names": df["name"].astype(str).to_numpy(),
        "e": df["e"].astype(float).to_numpy(),
        "pop": df["pop_m"].astype(float).to_numpy(),
        "gdp_cap": df["gdp_cap"].astype(float).to_numpy(),
        "alpha_base": df["alpha_base"].astype(float).to_numpy(),
        "alpha_cov": df["alpha_cov"].astype(float).to_numpy(),
        "alpha_trf": df["alpha_trf"].astype(float).to_numpy(),
        "weights": df["weight"].astype(float).to_numpy(),
    }


def solve_tminus(e: np.ndarray, pop: np.ndarray, tplus: float, ebar: float = EBAR_DEFAULT,
                 active: Optional[np.ndarray] = None) -> float:
    if active is None:
        active = np.ones_like(e, dtype=bool)
    excess = np.maximum(e - ebar, 0.0) * pop
    deficit = np.maximum(ebar - e, 0.0) * pop
    denom = float(deficit[active].sum())
    if denom <= 1e-12:
        return 0.0
    return float(tplus * excess[active].sum() / denom)


def preference_values(e: np.ndarray, alpha_base: np.ndarray, alpha_cov: np.ndarray, alpha_trf: np.ndarray,
                      c: float, tplus: float, tminus: float, ebar: float = EBAR_DEFAULT) -> np.ndarray:
    transfer_effect = np.where(e > ebar, -tplus * (e - ebar), tminus * (ebar - e))
    vals = alpha_base + alpha_cov * c + alpha_trf * transfer_effect
    return np.maximum(vals, 0.0)


def price_for_coverage(preferences: np.ndarray, weights: np.ndarray, c_target: float) -> float:
    order = np.argsort(-preferences)
    cum = 0.0
    for idx in order:
        cum += float(weights[idx])
        if cum + 1e-12 >= c_target:
            return float(preferences[idx])
    return float(np.min(preferences))


def transfer_accounting(actor_df: pd.DataFrame, p: float, tplus: float, tminus: float,
                        ebar: float = EBAR_DEFAULT, join_flags: Optional[np.ndarray] = None) -> pd.DataFrame:
    df = normalise_actor_df(actor_df).copy()
    if join_flags is None:
        join_flags = np.ones(len(df), dtype=bool)
    df["joins"] = join_flags
    df["gap"] = ebar - df["e"]
    df["status"] = np.where(df["e"] > ebar, "Contributor", "Beneficiary")
    df["collected_per_cap"] = p * df["e"]
    df["sent_per_cap"] = np.where(df["e"] > ebar, tplus * p * (df["e"] - ebar), 0.0)
    df["received_per_cap"] = np.where(df["e"] <= ebar, tminus * p * (ebar - df["e"]), 0.0)
    df["net_transfer_per_cap"] = df["received_per_cap"] - df["sent_per_cap"]
    df["retained_domestic_per_cap"] = p * df["e"] - (np.where(df["e"] > ebar, tplus * p * (df["e"] - ebar), 0.0))
    df["total_sent_mEUR"] = np.where(df["joins"], df["sent_per_cap"] * df["pop_m"], 0.0)
    df["total_received_mEUR"] = np.where(df["joins"], df["received_per_cap"] * df["pop_m"], 0.0)
    df["net_total_mEUR"] = df["total_received_mEUR"] - df["total_sent_mEUR"]
    return df


def run_mechanism_selfconsistent(actor_df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Dict[str, Any]:
    """Run mechanism using coalition enumeration (2^N) to find the best self-consistent coalition.
    Self-consistent means: transfers from THAT coalition justify every member's join decision."""
    if config is None:
        config = EngineConfig()
    df = normalise_actor_df(actor_df)
    arr = arrays(df)
    e, pop, weights, names = arr["e"], arr["pop"], arr["weights"], arr["names"]
    ab, ac, at = arr["alpha_base"], arr["alpha_cov"], arr["alpha_trf"]
    N = len(e)
    ebar = config.ebar
    tmax = config.t_max if config.t_cap is None else min(config.t_max, config.t_cap)
    t_grid = np.round(np.linspace(config.t_min, tmax, config.t_steps), 4)

    best_global: Optional[Dict] = None

    for mask_int in range(1, 2**N):
        members = [i for i in range(N) if mask_int & (1 << i)]
        if len(members) < 2:
            continue
        has_c = any(e[i] > ebar for i in members)
        has_b = any(e[i] <= ebar for i in members)
        if not (has_c and has_b):
            continue
        coverage = float(sum(weights[i] for i in members))

        for tplus in t_grid:
            if tplus <= 0:
                continue
            # t- from THIS coalition only
            j_excess = sum(max(0, e[i]-ebar)*pop[i] for i in members)
            j_deficit = sum(max(0, ebar-e[i])*pop[i] for i in members)
            if j_deficit < 1e-12:
                continue
            tminus = float(tplus * j_excess / j_deficit)
            # Compute all actors' willingness
            tau = np.where(e > ebar, -tplus*(e-ebar), tminus*(ebar-e))
            prefs = np.maximum(0, ab + ac*coverage + at*tau)
            # Price = min willingness among members
            member_prefs = [float(prefs[i]) for i in members]
            price = min(member_prefs)
            if price <= 0:
                continue
            # Self-consistency: no non-member is willing. Member willingness
            # holds by construction because price is their minimum preference.
            non_members = [i for i in range(N) if i not in members]
            all_out = all(prefs[i] < price + 0.01 for i in non_members)
            if not all_out:
                continue
            obj = coverage * price
            if best_global is None or obj > best_global["obj"]:
                best_global = {
                    "obj": obj, "tplus": float(tplus), "tminus": tminus,
                    "c": coverage, "p": price, "members": members,
                    "prefs": prefs.copy(),
                }

    if best_global is None:
        raise RuntimeError(
            "no self-consistent operating point; the ex-ante weighted-quantile "
            "solver is a different model and must not be substituted silently"
        )

    bg = best_global
    join = np.zeros(N, dtype=bool)
    for i in bg["members"]:
        join[i] = True
    actual_coverage = float(weights[join].sum())
    accounting = transfer_accounting(df, bg["p"], bg["tplus"], bg["tminus"], ebar=ebar, join_flags=join)

    # Build curves at selected T+
    c_grid = np.round(np.linspace(config.c_min, config.c_max, max(config.c_steps, 60)), 4)
    curve_rows = []
    for c in c_grid:
        prefs_c = preference_values(e, ab, ac, at, float(c), bg["tplus"], bg["tminus"], ebar=ebar)
        price_c = price_for_coverage(prefs_c, weights, float(c))
        curve_rows.append({"coverage": float(c), "feasible_price": float(price_c), "objective": float(c)*float(price_c)})
    curve_df = pd.DataFrame(curve_rows)

    # Frontier by T+ (reuse ex-ante grid search for the frontier display)
    frontier = []
    for tplus in t_grid:
        tm = solve_tminus(e, pop, float(tplus), ebar=ebar)
        best_obj_t = -1
        best_c_t, best_p_t = 0, 0
        for c in c_grid:
            prefs_c = preference_values(e, ab, ac, at, float(c), float(tplus), tm, ebar=ebar)
            pc = price_for_coverage(prefs_c, weights, float(c))
            ot = float(c) * pc
            if ot > best_obj_t:
                best_obj_t, best_c_t, best_p_t = ot, float(c), pc
        frontier.append({"Tplus": float(tplus), "Tminus_expected": tm, "c": best_c_t, "p": best_p_t, "objective": best_obj_t})

    actor_results = df.copy()
    actor_results["preference_at_solution"] = bg["prefs"]
    actor_results["joins"] = join
    actor_results["join_status"] = np.where(join, "Joiner", "Non-joiner")

    return {
        "p_star": bg["p"], "c_star": bg["c"], "Tplus_star": bg["tplus"],
        "Tminus_expected": bg["tminus"], "Tminus_actual": bg["tminus"],
        "objective": bg["obj"], "actual_coverage": actual_coverage,
        "actor_results": actor_results, "accounting": accounting, "curve": curve_df,
        "frontier_by_Tplus": pd.DataFrame(frontier), "preferences": bg["prefs"],
        "join_flags": join, "config": config,
        "self_consistent": True,
    }


def run_mechanism(actor_df: pd.DataFrame, config: Optional[EngineConfig] = None,
                  params: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    if config is None:
        config = EngineConfig()
    df = normalise_actor_df(actor_df)
    arr = arrays(df)
    e, pop, weights, names = arr["e"], arr["pop"], arr["weights"], arr["names"]
    alpha_base = arr["alpha_base"] if params is None else params.get("alpha_base", arr["alpha_base"])
    alpha_cov = arr["alpha_cov"] if params is None else params.get("alpha_cov", arr["alpha_cov"])
    alpha_trf = arr["alpha_trf"] if params is None else params.get("alpha_trf", arr["alpha_trf"])
    tmax = config.t_max if config.t_cap is None else min(config.t_max, config.t_cap)
    c_grid = np.round(np.linspace(config.c_min, config.c_max, config.c_steps), 4)
    t_grid = np.round(np.linspace(config.t_min, tmax, config.t_steps), 4)
    records: List[Tuple[float,float,float,float,float,np.ndarray]] = []
    best = None
    for tplus in t_grid:
        tminus_expected = solve_tminus(e, pop, float(tplus), ebar=config.ebar)
        for c in c_grid:
            prefs = preference_values(e, alpha_base, alpha_cov, alpha_trf, float(c), float(tplus), tminus_expected, ebar=config.ebar)
            price = price_for_coverage(prefs, weights, float(c))
            objective = float(c) * price
            rec = (objective, float(c), float(price), float(tplus), float(tminus_expected), prefs)
            records.append(rec)
            if best is None or objective > best[0] + 1e-12 or (abs(objective - best[0]) <= 1e-12 and c > best[1]):
                best = rec
    objective, c_star, p_star, tplus_star, tminus_expected, prefs_star = best
    join = prefs_star + 1e-9 >= p_star
    actual_coverage = float(weights[join].sum())
    tminus_actual = solve_tminus(e, pop, tplus_star, ebar=config.ebar, active=join)
    accounting = transfer_accounting(df, p_star, tplus_star, tminus_actual, ebar=config.ebar, join_flags=join)
    # curves for selected T+
    selected_curve_rows = []
    for c in c_grid:
        prefs = preference_values(e, alpha_base, alpha_cov, alpha_trf, float(c), tplus_star, tminus_actual, ebar=config.ebar)
        price = price_for_coverage(prefs, weights, float(c))
        selected_curve_rows.append({"coverage": float(c), "feasible_price": float(price), "objective": float(c)*float(price)})
    curve_df = pd.DataFrame(selected_curve_rows)
    # best frontier by T+
    frontier = []
    for t in sorted(set(r[3] for r in records)):
        subset = [r for r in records if r[3] == t]
        r = max(subset, key=lambda z: z[0])
        frontier.append({"Tplus": r[3], "Tminus_expected": r[4], "c": r[1], "p": r[2], "objective": r[0]})
    actor_results = df.copy()
    actor_results["preference_at_solution"] = prefs_star
    actor_results["joins"] = join
    actor_results["join_status"] = np.where(join, "Joiner", "Non-joiner")
    return {
        "p_star": float(p_star), "c_star": float(c_star), "Tplus_star": float(tplus_star),
        "Tminus_expected": float(tminus_expected), "Tminus_actual": float(tminus_actual),
        "objective": float(objective), "actual_coverage": actual_coverage,
        "actor_results": actor_results, "accounting": accounting, "curve": curve_df,
        "frontier_by_Tplus": pd.DataFrame(frontier), "preferences": prefs_star,
        "join_flags": join, "config": config,
    }


def diagnostics(actor_df: pd.DataFrame, res: Dict[str, Any], tolerance: float = 1e-5) -> pd.DataFrame:
    df = normalise_actor_df(actor_df)
    ar = res["actor_results"]
    acc = res["accounting"]
    join = res["join_flags"]
    p = res["p_star"]
    c = res["c_star"]
    weights = df["weight"].to_numpy(float)
    prefs = ar["preference_at_solution"].to_numpy(float)
    actual_coverage = float(weights[join].sum())
    rows = []
    def add(name, passed, detail): rows.append({"Check": name, "Status": "PASS" if passed else "FAIL", "Detail": detail})
    add("All joiners accept selected price", bool(np.all(prefs[join] + 1e-9 >= p)), f"minimum joiner willingness = {prefs[join].min() if join.any() else np.nan:.2f}, p* = {p:.2f}")
    add("All non-joiners below selected price", bool(np.all(prefs[~join] < p + 1e-8)), f"maximum non-joiner willingness = {prefs[~join].max() if (~join).any() else np.nan:.2f}, p* = {p:.2f}")
    add("Actual coverage meets target", actual_coverage + 1e-9 >= c, f"actual = {actual_coverage:.4f}, target = {c:.4f}")
    balance = float(acc["total_sent_mEUR"].sum() - acc["total_received_mEUR"].sum())
    add("Transfer pool balances", abs(balance) < max(1e-4, tolerance*max(1, abs(float(acc["total_sent_mEUR"].sum())))), f"balance = {balance:.6f} mEUR")
    add("Emissions weights sum to 1", abs(float(df["weight"].sum()) - 1.0) < 1e-6, f"sum = {float(df['weight'].sum()):.6f}")
    add("No missing alpha parameters", not df[["alpha_base", "alpha_cov", "alpha_trf"]].isna().any().any(), "alpha_base, alpha_cov, alpha_trf present")
    add("Fixed world-average benchmark", abs(res["config"].ebar - EBAR_DEFAULT) < 1e-12, f"ē = {res['config'].ebar}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from paths import ACTORS_CSV
    actors = pd.read_csv(ACTORS_CSV)
    res = run_mechanism_selfconsistent(actors)
    sent = float(res["accounting"].loc[res["join_flags"], "total_sent_mEUR"].sum())
    print(f"p* = {res['p_star']:.2f}")
    print(f"c* = {res['c_star']:.4f}  (actual {res['actual_coverage']:.4f})")
    print(f"T+ = {res['Tplus_star']:.2f}   T- = {res['Tminus_actual']:.4f}")
    print(f"pool ≈ EUR {sent/1000:.1f}B")
    print("joiners:", ", ".join(actors.loc[res["join_flags"], "name"].tolist()))
