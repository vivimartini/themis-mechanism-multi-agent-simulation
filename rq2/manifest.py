"""Generate the experiment manifest."""
from paths import REFERENCE

# module | question | sampling | draws/seed | grid | utility | outputs
ROWS = [
    ("engine.themis",
     "Reference operating point on the nine-actor baseline",
     "deterministic", "—", "fine (101 T x 100 c)", "—",
     "p*, c*, T+-, pool"),
    ("engine.scenario_prior",
     "RQ1: which features of the operating point does the evidence identify? "
     "Price distribution, coalition membership, price-setter frequency, and "
     "variance attribution across the two preference channels",
     "**Monte Carlo**", "n=4000, seed 0 (x3 passes for the attribution)",
     "fine (101 T)", "—",
     "mc_operating_prices.npz, fig_mc_operating_price"),
    ("engine.transfer_parameterisation",
     "Is the operating point well posed under absolute vs ratio transfers?",
     "exhaustive", "11025 designs", "441 coalitions x 25 t+ x 4000 p",
     "—", "fig_transfer_parameterisation"),
    ("engine.aggregation_check",
     "Does representing countries as a bloc change the operating point? "
     "Splits the UK out of the advanced-joiner bloc by mass balance",
     "deterministic", "sweep over the split member's base term",
     "fine (101 T), 2^10 coalitions", "—",
     "price / T- / own-transfer sensitivity"),
    ("archetypes.reproduce_archetypes",
     "Reproduce the published worked example, transfer accounting and "
     "elicitation, and attribute the Stage B price gap to digitisation",
     "deterministic", "—", "published table / digitised lines", "—",
     "Stage 0 + A + B + sensitivity"),
    ("rq2.validate",
     "Smoke validation of reported baseline, solver agreement, and figure inputs",
     "deterministic", "CMA seed 42; 200–300 evals", "mixed", "peaked",
     "pass/fail checks"),
    ("rq2.fixed_coverage",
     "DSIC control: is the price rule manipulable at frozen coverage?",
     "deterministic search", "CMA seed 42, 900 evals", "coverage and T frozen",
     "peaked, transfer", "NashConv = 0 (peaked)"),
    ("rq2.endogenous_coverage",
     "Exploitability once coverage and rates respond to reports",
     "deterministic search", "CMA seed 42, 900 evals", "fine (101 T)",
     "peaked, transfer", "NashConv ~ 19.6"),
    ("rq2.sqloss_check",
     "Does squared loss preserve the set of exploitable actors?",
     "deterministic search", "CMA seed 42, 900 evals", "fine (101 T)",
     "peaked, squared loss", "exploitable-actor comparison"),
    ("rq2.report_semantics",
     "Which reading of a report does the DSIC claim need?",
     "deterministic search", "CMA seed 42, 900 evals", "both regimes",
     "peak, cap, transfer", "2x3 NashConv grid, du/dp table"),
    ("rq2.case_study",
     "What exactly does each headline deviation do, and to whom?",
     "deterministic search", "CMA seed 42, 900 evals", "fine (101 T)",
     "peaked", "per-actor ledgers"),
    ("rq2.obstruction_voteselling",
     "Sabotage damage and pairwise vote-selling surplus",
     "deterministic search", "CMA seed 42/42+i", "fine (101 T)",
     "transfer (TU)", "damage %, joint surplus"),
    ("rq2.climate_benefit",
     "At what climate valuation does obstruction stop paying?",
     "deterministic search", "CMA seed 42+i", "fine + coarse",
     "transfer + B*(c*p)", "break-even B*, fig_climate_breakeven"),
    ("rq2.information_and_risk",
     "What must a deviator know, what does it risk, what if others react?",
     "**Monte Carlo**", "60 worlds seed 11; 8 noise draws",
     "coarse (25 T)", "peaked",
     "information rent, backfire rates, IBR trajectory"),
    ("rq2.exinterim_guardrails",
     "Ex-interim regret and guardrail ablations",
     "**Monte Carlo** (part A only)", "120 worlds, seed 7",
     "coarse (25 T)", "peaked, transfer",
     "exinterim_regrets.npz, guardrail_ablation.npz"),
    ("rq2.psro_lite",
     "Where does the empirical game settle under alpha-Rank / PSRO?",
     "deterministic", "seed 42, M=50", "coarse (25 T)", "peaked",
     "stationary mass, dominant sink"),
    ("rq2.validate_alpharank",
     "Alpha-Rank validation fixtures",
     "deterministic", "RPS, dominance, 5x2", "finite games", "—",
     "transition and stationarity checks"),
    ("rq2.sensitivity_alpharank",
     "Alpha-Rank population-size sensitivity",
     "deterministic", "m=5...200; alpha=0.05,0.5,5", "restricted game", "peaked",
     "truthful and top-profile mass"),
    ("rq2.ablation_alpharank",
     "Leave-one-attack-out Alpha-Rank ablation",
     "deterministic", "m=50; alpha=5", "restricted game", "peaked",
     "truthful mass and top outcome"),
    ("rq2.sensitivity_oracles",
     "Random-search guardrail sensitivity",
     "deterministic", "900 draws, seed 3", "coarse (25 T)", "peaked",
     "NashConv by design"),
    ("rq2.vote_structure",
     "Does splitting or merging a vote pay, and through which channel?",
     "deterministic", "—", "fine (101 T), 3 worlds per pair", "peaked",
     "per-member dU, channel split"),
    ("rq2.bruteforce",
     "Does the story survive in a world small enough to enumerate?",
     "exhaustive", "61x61 grid, 4 actors", "51 T", "peaked",
     "regret by role"),
    ("rq2.robustness",
     "Locality, slack sensitivity, cross-implementation agreement",
     "deterministic", "161-point sweep; 24 profiles seed 5", "fine (101 T)",
     "peaked", "locality + slack npz"),
    ("rq2.collect_headlines",
     "Point-calibration headlines for the figures",
     "deterministic search", "CMA seed 42, 900 evals", "fine (101 T)",
     "peaked, transfer", "headlines.json"),
]

HEADER = """# Experiment manifest

Auto-generated by `python -m rq2.manifest`. Do not edit by hand.

"""

COLS = ["module", "question", "sampling", "draws / seed", "solver grid",
        "utility", "feeds"]

# Numbers the dissertation cites that the current code can no longer produce.
ARCHIVED = [
    dict(value="NashConv 3.60",
         name="profit-gated fast oracle (superseded)",
         what="Endogenous peaked NashConv at the point calibration, coarse T "
              "grid, using a best-response oracle whose CMA-ES refinement ran "
              "ONLY when the warm-start portfolio had already found a "
              "profitable deviation.",
         why="The gate suppressed search precisely where the portfolio was "
             "uninformative, so the oracle reported far less exploitability "
             "than plain uniform random search at the same budget. It was "
             "replaced rather than tuned.",
         successor="rq2/coarse_solver.py:oracle — CMA refinement now ALWAYS "
                   "runs from the best warm start. Current values: hybrid "
                   "20.30, uniform random 22.34, fine-grid reference 19.63.",
         repro="No. The gated variant is not in the tree. Quote 3.60 only as a "
               "superseded measurement; the argument in the optimiser "
               "comparison stands on the 22.34-vs-20.30 result, which is "
               "regenerated by rq2.exinterim_guardrails part C."),
]


def main():
    lines = [HEADER, "| " + " | ".join(COLS) + " |",
             "|" + "|".join(["---"] * len(COLS)) + "|"]
    for r in ROWS:
        lines.append("| " + " | ".join(f"`{r[0]}`" if i == 0 else str(r[i])
                                       for i in range(len(COLS))) + " |")
    lines.append("")
    lines.append("## Archived measurements")
    lines.append("")
    lines.append("Values cited in the dissertation that the current tree CANNOT "
                 "regenerate, because the implementation that produced them was "
                 "removed. Recorded here so each has a citable provenance rather "
                 "than appearing as an unsourced number.")
    lines.append("")
    for r in ARCHIVED:
        lines.append(f"**{r['value']} — {r['name']}**")
        lines.append("")
        lines.append(f"- *What it measured:* {r['what']}")
        lines.append(f"- *Why it is gone:* {r['why']}")
        lines.append(f"- *Successor:* {r['successor']}")
        lines.append(f"- *Reproducible today:* {r['repro']}")
        lines.append("")
    out = REFERENCE / "experiment_manifest.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
