"""Check that a dissertation release contains coherent, portable artifacts.

This is an inventory and metadata check; it does not rerun experiments.
Run from any directory with the validated Python environment:

    python scripts/check_release.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
REFERENCE = ROOT / "reference_outputs"

NPZ_NAMES = {
    "alpharank_leave_one_out.npz",
    "climate_benefit.npz",
    "exinterim_regrets.npz",
    "guardrail_ablation.npz",
    "indonesia_locality_sweep.npz",
    "information_risk.npz",
    "mc_operating_prices.npz",
    "psro_alpharank.npz",
    "report_semantics.npz",
    "slack_sweep.npz",
    "transfer_parameterisation.npz",
    "vote_structure.npz",
}

FIGURE_STEMS = {
    "fig_alpharank_mass",
    "fig_climate_breakeven",
    "fig_exinterim_exploitability",
    "fig_guardrail_ablation",
    "fig_indonesia_locality",
    "fig_information_risk",
    "fig_mc_operating_price",
    "fig_nashconv_fixed_vs_endogenous",
    "fig_obstruction_damage",
    "fig_optimizer_comparison",
    "fig_regime_map",
    "fig_regret_geography",
    "fig_report_semantics",
    "fig_rq2_pipeline",
    "fig_slack_robustness",
    "fig_transfer_parameterisation",
    "fig_vote_structure",
}

MANIFEST_MODULES = {
    "engine.themis",
    "engine.scenario_prior",
    "engine.transfer_parameterisation",
    "engine.aggregation_check",
    "archetypes.reproduce_archetypes",
    "rq2.validate",
    "rq2.fixed_coverage",
    "rq2.endogenous_coverage",
    "rq2.sqloss_check",
    "rq2.report_semantics",
    "rq2.case_study",
    "rq2.obstruction_voteselling",
    "rq2.climate_benefit",
    "rq2.information_and_risk",
    "rq2.exinterim_guardrails",
    "rq2.psro_lite",
    "rq2.validate_alpharank",
    "rq2.sensitivity_alpharank",
    "rq2.ablation_alpharank",
    "rq2.sensitivity_oracles",
    "rq2.vote_structure",
    "rq2.bruteforce",
    "rq2.robustness",
    "rq2.collect_headlines",
}

TEXT_FILES = (
    ROOT / "README.md",
    ROOT / "REPRODUCIBILITY.md",
    ROOT / "DATA_PROVENANCE.md",
    ROOT / "DISSERTATION_CROSSWALK.md",
    ROOT / "CITATION.cff",
    DATA / "headlines.json",
    REFERENCE / "experiment_manifest.md",
    REFERENCE / "full_run.txt",
    REFERENCE / "run_provenance.txt",
    *sorted(ROOT.glob("*.py")),
    *sorted((ROOT / "engine").glob("*.py")),
    *sorted((ROOT / "archetypes").glob("*.py")),
    *sorted((ROOT / "rq2").glob("*.py")),
    *sorted((ROOT / "scripts").glob("*.py")),
    *sorted(ROOT.glob("*.sh")),
    *sorted((ROOT / "scripts").glob("*.sh")),
    *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
)

SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|private[_-]?key)"
    r"\s*[:=]\s*['\"][^'\"]+"
)


def main() -> int:
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{status}] {label}{suffix}")
        if not condition:
            failures.append(label)

    print("=== dissertation release integrity ===\n")

    present_npz = {path.name for path in DATA.glob("*.npz")}
    missing_npz = sorted(NPZ_NAMES - present_npz)
    check("all numeric datasets present", not missing_npz,
          f"missing: {', '.join(missing_npz)}" if missing_npz else "")

    missing_figures = [
        f"{stem}.{suffix}"
        for stem in sorted(FIGURE_STEMS)
        for suffix in ("pdf", "png")
        if not (FIGURES / f"{stem}.{suffix}").is_file()
    ]
    check("17 PDF/PNG figure pairs present", not missing_figures,
          f"missing: {', '.join(missing_figures)}" if missing_figures else "")

    headlines_path = DATA / "headlines.json"
    try:
        headlines = json.loads(headlines_path.read_text())
        p = float(headlines["truthful"]["p"])
        c = float(headlines["truthful"]["c"])
        fixed = float(headlines["nashconv"]["fixed"])
        endogenous = float(headlines["nashconv"]["endogenous"])
        china_damage = float(headlines["obstruction"]["damage_pct"][0])
        check("headline values match submitted results",
              math.isclose(p, 26.70, abs_tol=0.05)
              and math.isclose(c, 0.86864, abs_tol=0.001)
              and math.isclose(fixed, 0.0, abs_tol=0.01)
              and math.isclose(endogenous, 19.63, abs_tol=0.1)
              and math.isclose(china_damage, 68.9, abs_tol=0.2),
              f"p={p:.3f}, c={c:.5f}, NashConv={fixed:.2f}/{endogenous:.2f}, "
              f"China damage={china_damage:.1f}%")
    except (FileNotFoundError, KeyError, TypeError, ValueError,
            json.JSONDecodeError) as exc:
        check("headline values match submitted results", False, str(exc))

    manifest_path = REFERENCE / "experiment_manifest.md"
    manifest = manifest_path.read_text() if manifest_path.exists() else ""
    missing_modules = sorted(
        module for module in MANIFEST_MODULES
        if f"`{module}`" not in manifest
    )
    check("manifest covers every pipeline module", not missing_modules,
          f"missing: {', '.join(missing_modules)}" if missing_modules else "")

    missing_docs = [str(path.relative_to(ROOT)) for path in TEXT_FILES
                    if not path.is_file()]
    check("release documentation and records present", not missing_docs,
          f"missing: {', '.join(missing_docs)}" if missing_docs else "")

    leaked_paths: list[str] = []
    secret_hits: list[str] = []
    posix_users = "/" + "Users" + "/"
    for path in TEXT_FILES:
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        if posix_users in text or re.search(r"[A-Za-z]:\\Users\\", text):
            leaked_paths.append(str(path.relative_to(ROOT)))
        if SECRET_PATTERN.search(text):
            secret_hits.append(str(path.relative_to(ROOT)))
    check("no local absolute paths in release text", not leaked_paths,
          ", ".join(leaked_paths))
    check("no obvious secrets in release text", not secret_hits,
          ", ".join(secret_hits))

    print()
    if failures:
        print(f"{len(failures)} release check(s) failed: {', '.join(failures)}")
        return 1
    print("All release integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
