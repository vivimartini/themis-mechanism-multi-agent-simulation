"""Repo paths — single place so scripts don't hardcode filenames."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
REFERENCE = ROOT / "reference_outputs"

ACTORS_CSV = DATA / "actors_baseline.csv"
MC_PRICES_NPZ = DATA / "mc_operating_prices.npz"
EXINTERIM_NPZ = DATA / "exinterim_regrets.npz"

# Results written by the experiments and read back by rq2.make_figures, so that
# no figure carries a hand-transcribed number.
SEMANTICS_NPZ = DATA / "report_semantics.npz"
VOTESTRUCT_NPZ = DATA / "vote_structure.npz"
CLIMATE_NPZ = DATA / "climate_benefit.npz"
INFORISK_NPZ = DATA / "information_risk.npz"
TRANSFERPARAM_NPZ = DATA / "transfer_parameterisation.npz"
LOCALITY_NPZ = DATA / "indonesia_locality_sweep.npz"
SLACK_NPZ = DATA / "slack_sweep.npz"
GUARDRAIL_NPZ = DATA / "guardrail_ablation.npz"
PSRO_NPZ = DATA / "psro_alpharank.npz"
ALPHARANK_ABLATION_NPZ = DATA / "alpharank_leave_one_out.npz"

# Point-calibration headlines (regret geography, guardrails, obstruction,
# optimizer comparison) emitted by rq2.collect_headlines.
HEADLINES_JSON = DATA / "headlines.json"
