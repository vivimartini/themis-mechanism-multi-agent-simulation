"""Build figures from experiment outputs."""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paths import (DATA, FIGURES, MC_PRICES_NPZ, EXINTERIM_NPZ, GUARDRAIL_NPZ,
                   PSRO_NPZ, HEADLINES_JSON, SEMANTICS_NPZ, INFORISK_NPZ,
                   CLIMATE_NPZ, VOTESTRUCT_NPZ, TRANSFERPARAM_NPZ,
                   LOCALITY_NPZ, SLACK_NPZ)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

OUT = Path(os.environ.get("RQ2_FIGURES", FIGURES))
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.environ.get("RQ2_DATA", DATA))

# House palette. Gray is reserved for negligible / no-attack categories, so it is
# a status colour rather than a fifth series.
BLUE, RED, PURPLE, GRAY, INK, TEAL = ("#4a7c9b", "#c0392b", "#8e44ad",
                                      "#7f8c8d", "#2c3e50", "#16a085")
MODE_COLOR = {"entry": "#2a7fba", "exit-shrink": RED, "extortion": PURPLE,
              "nudge": GRAY, "none": "#bdc3c7", "collapse": "#34495e"}
SHORT = ["China", "US", "EU", "India", "Russia", "Indonesia",
         "Adv. joiners", "Frontier", "Rentiers"]

TW = 6.197   # \linewidth = 447.87 TeX pt at 72.27 pt/in
# saved 527pt wide and placed at 0.6\linewidth prints its 9pt labels at 4.6pt.
# Figures are authored at WF * TW so LaTeX does not rescale them: a figure
# bbox_inches="tight" only crops, so scale >= 1 and text never shrinks.
# saved width = printed width  =>  scale 1  =>  authored pt == printed pt.

built, skipped = [], []


def save(fig, name, **kw):
    fig.savefig(OUT / f"{name}.pdf", **kw)
    fig.savefig(OUT / f"{name}.png", dpi=300, **kw)
    plt.close(fig)
    built.append(name)


def load_npz(path):
    p = DATA_DIR / Path(path).name
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def load_json(path):
    p = DATA_DIR / Path(path).name
    return json.loads(p.read_text()) if p.exists() else None


def miss(name, src):
    skipped.append(f"{name} (needs {src})")


# ---------------------------------------------- 01: MC operating-price histogram
d = load_npz(MC_PRICES_NPZ)
H = load_json(HEADLINES_JSON)
if d is None:
    miss("fig_mc_operating_price", "engine.scenario_prior")
else:
    p_point = H["truthful"]["p"] if H else 26.70
    fig, ax = plt.subplots(figsize=(0.72 * TW, 2.9))
    ax.hist(d["p"], bins=60, color=BLUE, alpha=0.85)
    ax.axvline(p_point, color="k", lw=1.4, ls="--")
    ax.text(p_point + 0.7, ax.get_ylim()[1] * 0.72,
            f"point estimate\n€{p_point:.2f}", fontsize=8)
    ax.axvline(20.97, color=RED, lw=1.4, ls=":")
    ax.text(15.0, ax.get_ylim()[1] * 0.72, "published\n€20.97", fontsize=8,
            color=RED, ha="right")
    ax.set_xlabel("operating price $p^*$ (€/tCO$_2$e)")
    ax.set_ylabel("scenario draws")
    fig.tight_layout(); save(fig, "fig_mc_operating_price")

# --------------------------------------------- 02: fixed vs endogenous NashConv
if H is None:
    miss("fig_nashconv_fixed_vs_endogenous", "rq2.collect_headlines")
else:
    nf, ne = H["nashconv"]["fixed"], H["nashconv"]["endogenous"]
    fig, ax = plt.subplots(figsize=(0.55 * TW, 3.0))
    ax.bar(["fixed coverage\n(DSIC control)", "endogenous\ncoverage"], [nf, ne],
           color=[BLUE, RED], width=0.55)
    ax.set_ylabel("NashConv (peaked domain, €/tCO$_2$e)")
    ax.text(0, ne * 0.02, f"{nf:.4f}", ha="center", fontsize=9)
    ax.text(1, ne * 1.02, f"{ne:.2f}", ha="center", fontsize=9)
    ax.text(0.5, -0.22, "fine transfer grid (headline)", ha="center", fontsize=8,
            color="#666", transform=ax.get_xaxis_transform())
    fig.tight_layout(); save(fig, "fig_nashconv_fixed_vs_endogenous")

# ------------------------------------------------------- 03: regret geography
if H is None:
    miss("fig_regret_geography", "rq2.collect_headlines")
else:
    vals = np.array(H["regret"]["values"], float)
    modes = H["regret"]["modes"]
    order = np.argsort(-vals)
    fig, ax = plt.subplots(figsize=(0.72 * TW, 3.3))
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals[order], color=[MODE_COLOR[modes[i]] for i in order])
    ax.set_yticks(y); ax.set_yticklabels([SHORT[i] for i in order])
    ax.set_xlabel("best-response regret (peaked, €/tCO$_2$e)")
    # Only legend a mode that has a bar you can actually see. "none" is always
    # zero-length and "nudge" here is 0.04 against a 9.21 maximum, so both would
    # appear as swatches with no visible referent. Their values are printed on
    # the rows regardless.
    # Legend via Patch handles, NOT zero-size bars: ax.bar(0, 0, ...) draws a
    # real artist at index 0, which on a barh chart is a flat line across the
    # bottom row and reads as a strikethrough through that row's value label.
    vis = max(vals) * 0.01
    handles = [Patch(facecolor=MODE_COLOR[m], label=m)
               for m in ("entry", "exit-shrink", "extortion", "nudge", "none")
               if any(modes[i] == m and vals[i] >= vis for i in order)]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    # Label every actor, including the zero-regret ones: an unlabelled empty row
    # reads as missing data rather than as "no profitable deviation exists".
    for yi, i in zip(y, order):
        ax.text(vals[i] + max(vals) * 0.015, yi, f"{vals[i]:.2f}", va="center",
                fontsize=8, color="#333" if vals[i] > 0.02 else "#999")
    ax.margins(x=0.10)
    ax.set_xlim(left=0)
    fig.tight_layout(); save(fig, "fig_regret_geography")

# -------------------------------------------------- 04: ex-interim exploitability
A = load_npz(EXINTERIM_NPZ)
if A is None:
    miss("fig_exinterim_exploitability", "rq2.exinterim_guardrails")
else:
    REG, MODE = A["REG"], A["MODE"]
    fig, axes = plt.subplots(1, 2, figsize=(0.98 * TW, 3.2),
                             gridspec_kw={"width_ratios": [1.0, 1.5]})
    nc = REG.sum(1)
    axes[0].hist(nc, bins=24, color=BLUE, alpha=0.85)
    if H:
        axes[0].axvline(H["nashconv"]["endogenous"], color="k", ls="--", lw=1.2)
        axes[0].text(H["nashconv"]["endogenous"] + 0.2,
                     axes[0].get_ylim()[1] * 0.88, "point\ncalibration", fontsize=8)
    axes[0].set_xlabel("NashConv per world"); axes[0].set_ylabel("worlds")
    modes = ["entry", "exit-shrink", "extortion", "nudge"]
    share = np.zeros((9, len(modes)))
    for i in range(9):
        pos = REG[:, i] > 0.01
        for k, m in enumerate(modes):
            share[i, k] = (100 * np.mean([mm == m for mm in MODE[pos, i]])
                           * pos.mean() if pos.any() else 0.0)
    left = np.zeros(9); yy = np.arange(9)[::-1]
    for k, m in enumerate(modes):
        # Label only modes with mass: "extortion" is 0% in every world here, so
        # legending it puts a purple swatch against no purple segment.
        lbl = m if share[:, k].max() > 0.05 else None
        axes[1].barh(yy, share[:, k], left=left, color=MODE_COLOR[m], label=lbl)
        left += share[:, k]
    axes[1].set_yticks(yy); axes[1].set_yticklabels(SHORT, fontsize=8)
    axes[1].set_xlabel("% of worlds attacked, by mode")
    axes[1].xaxis.grid(True, color="#eee", lw=0.6, zorder=0)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); save(fig, "fig_exinterim_exploitability", bbox_inches="tight")

# ------------------------------------------------------- 05: guardrail ablation
G = load_npz(GUARDRAIL_NPZ)
if G is None:
    miss("fig_guardrail_ablation", "rq2.exinterim_guardrails")
else:
    labs = [str(s) for s in G["designs"]]
    pretty = {"baseline": "baseline", "T- <= 1": "T\u207b \u2264 1",
              "c >= 0.5": "c \u2265 0.5", "pool <= 150bn": "pool cap",
              "all three": "all three"}
    labs = [pretty.get(l, l) for l in labs]
    panels = [(G["nashconv"], "endog. NashConv", BLUE, "{:.1f}"),
              (G["obstruction_pct"], "max obstruction (%)", RED, "{:.0f}"),
              (G["frontier_china"], "collusion surplus (€/cap)", PURPLE, "{:.0f}")]
    fig, axes = plt.subplots(1, 3, figsize=(0.98 * TW, 3.0))
    for ax, (v, t, c, fmt) in zip(axes, panels):
        bars = ax.bar(range(len(v)), v, color=c, width=0.62)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=8)
        ax.set_title(t, fontsize=8.5); ax.margins(y=0.16)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val, fmt.format(val),
                    ha="center", va="bottom", fontsize=8)
    axes[1].axhline(G["obstruction_pct"][0], color="#999", lw=0.7, ls=":")
    fig.tight_layout(); save(fig, "fig_guardrail_ablation", bbox_inches="tight")

# ------------------------------------------------- 05b: oracle validation
# Regret is a maximum, so a HIGHER bar is a stronger adversary. At equal budget
# uniform random search is not dominated by the hybrid: it weakly dominates it on
# every actor. The figure has to show that per actor, because the aggregate alone
# invites the reading that the two are close.
if G is None or H is None:
    miss("fig_optimizer_comparison", "rq2.exinterim_guardrails + collect_headlines")
else:
    hyb, rnd = G["opt_hybrid"].astype(float), G["opt_random"].astype(float)
    order = np.argsort(-rnd)
    fig, axes = plt.subplots(1, 2, figsize=(0.98 * TW, 4.0),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    yy = np.arange(9)[::-1]
    h = 0.34
    ax.barh(yy + h / 2, rnd[order], height=h, color=RED, label="random search")
    ax.barh(yy - h / 2, hyb[order], height=h, color=BLUE, label="hybrid oracle")
    # One label per row at the end of the longer bar. At 8pt the two stacked
    # labels collide on rows where the values are close (9.20 vs 9.15), and a
    # combined "random / hybrid" label is both shorter and easier to compare.
    pad = max(rnd) * 0.02
    for k, i in enumerate(order):
        txt = (f"{rnd[i]:.2f}" if abs(rnd[i] - hyb[i]) < 5e-3
               else f"{rnd[i]:.2f} / {hyb[i]:.2f}")
        ax.text(max(rnd[i], hyb[i]) + pad, yy[k], txt, va="center",
                fontsize=8, color="#333")
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[i] for i in order], fontsize=8)
    ax.set_xlabel("best-response regret found (€/tCO$_2$e)")
    ax.set_title("per actor, equal budget", fontsize=8.5)
    ax.margins(x=0.16)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    tot = [float(rnd.sum()), float(hyb.sum()), H["nashconv"]["endogenous"]]
    labs = ["random\nsearch", "hybrid\noracle", "fine-grid\nreference"]
    bars = ax.bar(range(3), tot, color=[RED, BLUE, INK], width=0.6)
    for b, v in zip(bars, tot):
        ax.text(b.get_x() + b.get_width() / 2, v + max(tot) * 0.02, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("NashConv (peaked, €/tCO$_2$e)")
    ax.set_ylim(0, max(tot) * 1.22)
    ax.set_title("aggregate", fontsize=8.5)
    # The reference bar is lower because it uses the FINE transfer grid, not
    # because it searches worse; the coarse grid slightly overstates regret.
    ax.text(0.5, -0.30, "first two: coarse T grid (25)\nreference: fine T grid (101)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            color="#666")
    n_beat = int((rnd >= hyb - 1e-9).sum())
    fig.suptitle(f"Random search is competitive, not dominated: it matches or beats "
                 f"the hybrid on {n_beat}/9 actors",
                 fontsize=8, color="#555", y=1.02)
    fig.tight_layout(); save(fig, "fig_optimizer_comparison", bbox_inches="tight")

# ------------------------------------------------------------- 06: regime map
P = load_npz(PSRO_NPZ)
if H is None:
    miss("fig_regime_map", "rq2.collect_headlines")
else:
    pts = [("truthful", H["truthful"]["c"], H["truthful"]["p"], INK, "o", (8, 10))]
    for nm, key, col, mk, off in [("Indonesia entry", "INDONESIA",
                                   MODE_COLOR["entry"], "^", (-12, -18)),
                                  ("China exit", "CHINA",
                                   MODE_COLOR["exit-shrink"], "v", (8, -4))]:
        j = H["names"].index(key)
        q = H["regret"]["points"][j]
        if q:
            pts.append((nm, q["c"], q["p"], col, mk, off))
    if P is not None and np.isfinite(P["sink_c"][0]):
        # Label up-and-LEFT: offset to the right put the text beside the China
        # exit marker, 0.18 further along c, so it read as labelling that point.
        pts.append(("Alpha-Rank sink", float(P["sink_c"][0]), float(P["sink_p"][0]),
                    RED, "s", (-10, 12)))
    if H.get("collusion", {}).get("c"):
        pts.append(("frontier-China", H["collusion"]["c"], H["collusion"]["p"],
                    MODE_COLOR["extortion"], "D", (10, -2)))
    if H.get("collusion_guarded", {}).get("c"):
        pts.append(("guarded residual", H["collusion_guarded"]["c"],
                    H["collusion_guarded"]["p"], TEAL, "D", (10, 8)))
    ymax = max(p for _, _, p, _, _, _ in pts) * 1.16
    fig, ax = plt.subplots(figsize=(0.95 * TW, 3.7))
    cc = np.linspace(0.05, 1.0, 300)
    # Label each iso-objective curve near the TOP of the axes rather than at the
    # right edge. At the right edge the low levels (5 and 10) sit only 5 EUR/t
    # apart on a 165 EUR/t axis, so they collide with each other, with the
    # Indonesia marker and with the axis itself. Along the top they separate
    # horizontally by construction, since x = level / y.
    y_lab = ymax * 0.955
    for lev in [5, 10, round(H["truthful"]["cp"], 1), 40, 59.8]:
        ax.plot(cc, lev / cc, color="#e2e2e2", lw=0.8, zorder=0)
        x_lab = lev / y_lab
        if 0.075 <= x_lab <= 0.62:                  # keep clear of the floor note
            ax.text(x_lab, y_lab, f"c·p={lev:g}", fontsize=8, color="#b0b0b0",
                    ha="center", va="top", zorder=1)
        else:                                        # too flat to reach the top
            ax.text(0.985, lev / 0.985, f"c·p={lev:g}", fontsize=8,
                    color="#b0b0b0", ha="right", va="bottom", zorder=1)
    for lab, c, p, col, mk, off in pts:
        ax.scatter([c], [p], color=col, marker=mk, s=46, zorder=3)
        ax.annotate(lab, (c, p), textcoords="offset points", xytext=off,
                    fontsize=8, color=col, zorder=4,
                    ha="right" if off[0] < 0 else "left",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5,
                                    shrinkA=0, shrinkB=2))
    ax.axvline(0.5, color=PURPLE, lw=0.9, ls="--")
    ax.text(0.515, ymax * 0.97, "coverage floor $c\\geq 0.5$ (guarded designs)",
            fontsize=8, color=PURPLE, ha="left", va="top")
    ax.set_xlabel("modelled coverage c"); ax.set_ylabel("price p (€/tCO$_2$e)")
    ax.set_xlim(0.05, 1.0); ax.set_ylim(0, ymax)
    fig.tight_layout(); save(fig, "fig_regime_map", bbox_inches="tight")

# ------------------------------------------------------ 07: alpha-Rank mass
if P is None:
    miss("fig_alpharank_mass", "rq2.psro_lite")
else:
    panels, labels, masses = P["panels"], P["labels"], P["masses"]
    fig, axes = plt.subplots(1, len(panels), figsize=(0.98 * TW, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, pan, lab, ms in zip(axes, panels, labels, masses):
        keep = [k for k in range(len(ms)) if ms[k] > 1e-4 or "TRUTH" in lab[k]]
        # One deviator per line, abbreviated: at 8pt each column is only ~0.76in
        # wide (two panels share the text width), and "+ Adv. joiners" overruns
        # into the neighbouring column.
        ABBR = {"Adv. joiners": "Adv.", "Indonesia": "Indo.", "Frontier": "Front.",
                "Rentiers": "Rent.", "ALL-TRUTHFUL": "all\ntruthful"}
        L = ["\n+ ".join(ABBR.get(t, t) for t in str(lab[k]).split(" + "))
             for k in keep]
        V = [ms[k] for k in keep]
        ax.bar(range(len(V)), V,
               color=[BLUE if "TRUTH" in lab[k] else RED for k in keep])
        ax.set_xticks(range(len(V))); ax.set_xticklabels(L, fontsize=8)
        # The panel label is persisted by psro_lite in ASCII; render the relation
        # with the same glyph the guardrail figure uses.
        ax.set_title(str(pan).replace("<=", "≤").replace(">=", "≥"),
                     fontsize=9)
        ax.margins(y=0.18)
        for k, v in enumerate(V):
            ax.text(k, max(v, 0.02) + 0.01,
                    "< 0.001" if v < 1e-3 else f"{v:.2f}",
                    ha="center", fontsize=8)
    axes[0].set_ylabel("$\\alpha$-Rank stationary mass")
    fig.tight_layout(); save(fig, "fig_alpharank_mass", bbox_inches="tight")

# --------------------------------------------------------- 08: pipeline diagram
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Sized to the A4 text width (21.0cm - 2x2.6cm margins = 15.8cm = 6.22in) so that
# \includegraphics[width=\linewidth] prints it 1:1 and the point sizes below are
# the sizes the examiner actually reads. The previous 8.8in single-row layout was
# scaled to 0.71 on the page, which turned 5.7pt labels into ~4.0pt. Six boxes
# cannot hold legible text across 6.22in, so the flow wraps onto two rows and the
# phase bands become box colours plus a legend.
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

PHASE = {"env": ("#d9e8f0", "Environment"),
         "adv": ("#f5d5c8", "Adversary"),
         "test": ("#e4e2d4", "Empirical game / design tests")}

fig, ax = plt.subplots(figsize=(0.98 * TW, 3.5)); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

BW, BH = 0.285, 0.26
COLS = (0.02, 0.3575, 0.695)
ROWS = (0.66, 0.20)                       # top row, bottom row (box bottoms)
stages = [
    ("Country data +\ncalibrated parameters", "env"),
    ("Preference layer\n$p_i(c,T)$", "env"),
    ("Themis solver $\\mathcal{M}(r)$:\n441-coalition,\nself-consistent", "adv"),
    ("Best-response oracle:\nportfolio + random\n+ CMA-ES", "adv"),
    ("Regret / NashConv\n+ attack labels", "test"),
    ("$\\alpha$-Rank\nmeta-solver", "test"),
]
pos = []
for k, (lab, ph) in enumerate(stages):
    x0, y0 = COLS[k % 3], ROWS[k // 3]
    ax.add_patch(FancyBboxPatch((x0, y0), BW, BH,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor=PHASE[ph][0], edgecolor="#666",
                                linewidth=0.9, zorder=2, clip_on=False))
    ax.text(x0 + BW / 2, y0 + BH / 2, lab, ha="center", va="center",
            fontsize=8.5, linespacing=1.35, zorder=3)
    pos.append((x0, y0))

def arrow(p0, p1, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 color="#444", lw=1.3, zorder=1,
                                 shrinkA=0, shrinkB=0, **kw))

for r in (0, 1):                                   # within-row arrows
    for k in (0, 1):
        i = 3 * r + k
        arrow((pos[i][0] + BW + 0.004, pos[i][1] + BH / 2),
              (pos[i + 1][0] - 0.004, pos[i + 1][1] + BH / 2))
# wrap from the end of the top row to the start of the bottom row
# Wrap from the end of the top row to the start of the bottom row, routed through
# the gap between rows. A single arc bulges far enough left to look as though it
# leaves the middle box, so the path is drawn explicitly.
x_from = pos[2][0] + BW / 2
x_to = pos[3][0] + BW / 2
y_mid = (pos[0][1] + pos[3][1] + BH) / 2
ax.plot([x_from, x_from, x_to], [pos[2][1] - 0.004, y_mid, y_mid],
        color="#444", lw=1.3, solid_capstyle="round", zorder=1)
arrow((x_to, y_mid), (x_to, pos[3][1] + BH + 0.004))

ax.legend(handles=[Patch(facecolor=c, edgecolor="#666", label=l)
                   for c, l in PHASE.values()],
          loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3,
          frameon=False, fontsize=8, handlelength=1.4, columnspacing=1.4)
ax.text(0.5, 0.075, "guardrail ablations rerun this loop under design constraints",
        ha="center", fontsize=8, color="#666")
save(fig, "fig_rq2_pipeline", bbox_inches="tight")

# ------------------------------------------------------ 09: obstruction damage
if H is None:
    miss("fig_obstruction_damage", "rq2.collect_headlines")
else:
    dmg = np.array(H["obstruction"]["damage_pct"], float)
    dpay = np.array(H["obstruction"]["own_transfer_change"], float)
    keep = np.argsort(-dmg)[:6]
    y = np.arange(len(keep))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(0.90 * TW, 3.3), sharey=True)
    axes[0].barh(y, dmg[keep], color=RED, height=0.55)
    for yi, v in zip(y, dmg[keep]):
        axes[0].text(v + 1.2, yi, f"{v:.1f}", va="center", fontsize=8, color="#333")
    axes[0].set_xlabel("damage to $c\\,p$ (%)")
    axes[0].set_title("mechanism damage", fontsize=8.5)
    axes[0].set_xlim(0, max(dmg) * 1.2)
    axes[1].barh(y, dpay[keep], color=BLUE, height=0.55)
    axes[1].axvline(0, color="k", lw=0.8)
    for yi, v in zip(y, dpay[keep]):
        off = 1.8 if v >= 0 else -1.8
        axes[1].text(v + off, yi, f"{v:+.1f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=8, color="#333")
    axes[1].set_xlabel("own transfer payoff change (€/cap)")
    axes[1].set_title("private payoff", fontsize=8.5)
    lim = max(abs(dpay[keep])) * 1.35
    axes[1].set_xlim(-lim, lim)
    axes[0].set_yticks(y); axes[0].set_yticklabels([SHORT[i] for i in keep])
    fig.suptitle("Contributor sabotage can be privately profitable; beneficiary "
                 "sabotage is self-defeating", fontsize=8, color="#555", y=1.02)
    fig.tight_layout(); save(fig, "fig_obstruction_damage", bbox_inches="tight")

# ------------------------------------------------------- 11: report semantics
S = load_npz(SEMANTICS_NPZ)
if S is None:
    miss("fig_report_semantics", "rq2.report_semantics")
else:
    sem = [str(s) for s in S["semantics"]]
    nc = S["nashconv"]                      # [regime, semantics]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    w = 0.26
    xs = np.arange(2)
    for k, s in enumerate(sem):
        col = {"peak": BLUE, "cap": RED, "transfer": PURPLE}[s]
        b = ax.bar(xs + (k - 1) * w, nc[:, k], width=w, color=col, label=s)
        for bb, v in zip(b, nc[:, k]):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 4, f"{v:.0f}",
                    ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["fixed coverage\n(DSIC control)", "endogenous\ncoverage"],
                       fontsize=8)
    ax.set_ylabel("NashConv (€/tCO$_2$e)")
    ax.set_title("what a report is taken to mean", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, title="report semantics",
              title_fontsize=8)
    ax.margins(y=0.22)
    ax.annotate("strategy-proof\nonly here", xy=(-w, nc[0, 0] + 16),
                xytext=(-w, 100), fontsize=8, color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7,
                                shrinkA=2, shrinkB=1))

    ax = axes[1]
    dd = S["dudp"]
    order = np.argsort(dd)
    cols = [RED if dd[i] < -1e-9 else (BLUE if dd[i] > 1e-9 else "#bdc3c7")
            for i in order]
    yy = np.arange(9)[::-1]
    ax.barh(yy, dd[order], color=cols, height=0.6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[i] for i in order], fontsize=8)
    ax.set_xlabel("$\\partial u_i/\\partial p$  (€/cap per €/t)")
    ax.set_title("which way each member wants the price to move", fontsize=8.5)
    ax.margins(x=0.14)
    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in
                       ((RED, "contributor: wants lower $p$"),
                        (BLUE, "beneficiary: wants higher $p$"),
                        ("#bdc3c7", "non-member: indifferent"))],
              frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); save(fig, "fig_report_semantics", bbox_inches="tight")

# --------------------------------------------------- 12: information and risk
I = load_npz(INFORISK_NPZ)
if I is None:
    miss("fig_information_risk", "rq2.information_and_risk")
else:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9))
    ax = axes[0]
    rf, rp = I["reg_full"], I["reg_prior"]
    order = np.argsort(-rf)
    yy = np.arange(9)[::-1]
    h = 0.36
    ax.barh(yy + h / 2, rf[order], height=h, color=RED, label="full information")
    ax.barh(yy - h / 2, rp[order], height=h, color=BLUE, label="prior only")
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[i] for i in order], fontsize=8)
    ax.set_xlabel("E[regret] (€/tCO$_2$e)")
    ax.set_title("what the deviator knows", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    ax.plot(I["sigmas"], I["noise_nashconv"], "-o", color=BLUE, lw=2, ms=5)
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.set_xlabel("estimation error on others' $\\alpha$ (sd)")
    ax.set_ylabel("realised total gain (€/t)")
    ax.set_title("how precisely it must know it", fontsize=8.5)
    ax.annotate("exact knowledge", xy=(I["sigmas"][0], I["noise_nashconv"][0]),
                xytext=(6, -6), textcoords="offset points", fontsize=8,
                color="#555")

    ax = axes[2]
    mean, p5 = I["risk_mean"], I["risk_p5"]
    sel = np.argsort(-mean)[:6]
    yy2 = np.arange(len(sel))[::-1]
    ax.barh(yy2, mean[sel], color=BLUE, height=0.5, label="mean")
    ax.scatter(p5[sel], yy2, color=RED, marker="|", s=90, zorder=3,
               label="5th percentile")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(yy2); ax.set_yticklabels([SHORT[i] for i in sel], fontsize=8)
    ax.set_xlabel("$\\Delta u$ from the point attack (€/t)")
    ax.set_title("what it risks", fontsize=8.5)
    ax.margins(x=0.10, y=0.14)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); save(fig, "fig_information_risk", bbox_inches="tight")

# ------------------------------------------------------ 13: climate break-even
C = load_npz(CLIMATE_NPZ)
if C is None:
    miss("fig_climate_breakeven", "rq2.climate_benefit")
else:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    b = C["bstar"]
    ok = np.isfinite(b) & (b > 0)
    idx = np.argsort(-np.where(ok, b, -1))[:int(ok.sum())]
    yy = np.arange(len(idx))[::-1]
    ax.barh(yy, b[idx], color=RED, height=0.55)
    for yi, v in zip(yy, b[idx]):
        ax.text(v + max(b[idx]) * 0.02, yi, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[i] for i in idx], fontsize=8)
    ax.set_xlabel("break-even $B^*$ (€/cap per unit $c\\,p$)")
    ax.set_title("obstruction pays only below this\nclimate valuation", fontsize=8.5)
    ax.margins(x=0.16)
    n_never = int((~ok).sum())
    ax.text(0.98, 0.04, f"{n_never} actors never obstruct\n(it costs them transfers too)",
            transform=ax.transAxes, ha="right", fontsize=8, color="#555")

    ax = axes[1]
    ax.plot(C["B_grid"], C["n_attack"], "-o", color=GRAY, lw=1.8, ms=4.5,
            label="any profitable deviation")
    ax.plot(C["B_grid"], C["n_damaging"], "-o", color=RED, lw=2, ms=5,
            label="deviation that damages $c\\,p$")
    ax.set_xlabel("climate valuation $B$ (€/cap per unit $c\\,p$)")
    ax.set_ylabel("actors")
    ax.set_ylim(0, 9.5)
    ax.set_title("attacks persist; they stop being damaging", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); save(fig, "fig_climate_breakeven", bbox_inches="tight")

# --------------------------------------------------------- 14: vote structure
V = load_npz(VOTESTRUCT_NPZ)
if V is None:
    miss("fig_vote_structure", "rq2.vote_structure")
else:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    syn, di, dj = V["synthetic"], V["dU_i"], V["dU_j"]
    worst = np.minimum(di, dj)
    ax.scatter(syn, worst, s=26, color=RED, alpha=0.8, edgecolor="white", lw=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8, ls=":")
    ax.set_xlabel("old metric: averaged actor's gain")
    ax.set_ylabel("worse-off real member's $\\Delta u$")
    ax.set_title("the discarded merge metric", fontsize=8.5)
    q = ((syn > 0.05) & (worst < 0)).sum()
    ax.text(0.03, 0.06, f"{q} pairs look like gains\nbut hurt a real member",
            transform=ax.transAxes, fontsize=8, color="#555")

    ax = axes[1]
    ax.hist([V["boundary"], V["averaging"]], bins=18, color=[BLUE, RED],
            label=["coalition boundary", "price of averaging"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("$\\Delta u$ contribution (€/tCO$_2$e)")
    ax.set_ylabel("member-pairs")
    ax.set_title("merging, decomposed by channel", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "fig_vote_structure", bbox_inches="tight")

# --------------------------------------------- 15: transfer parameterisation
TP = load_npz(TRANSFERPARAM_NPZ)
if TP is None:
    miss("fig_transfer_parameterisation", "engine.transfer_parameterisation")
else:
    n = int(TP["n_designs"][0])
    absh = TP["abs_hist"].astype(float) / n * 100
    ratio = np.array([0.0, 100.0, 0.0])
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    cats = ["no operating price", "exactly one", "two or more"]
    cols = [RED, BLUE, PURPLE]
    left = np.zeros(2)
    for k, (cat, col) in enumerate(zip(cats, cols)):
        v = np.array([absh[k], ratio[k]])
        ax.barh([1, 0], v, left=left, color=col, height=0.5, label=cat)
        for yi, (vv, ll) in zip([1, 0], zip(v, left)):
            if vv > 6:
                ax.text(ll + vv / 2, yi, f"{vv:.0f}%", ha="center", va="center",
                        fontsize=8, color="white")
        left = left + v
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["absolute $t^{\\pm}$",
                        "ratio $T^{\\pm}\\!=\\!t^{\\pm}/p$"], fontsize=9)
    ax.set_xlabel(f"% of {n} (coalition, transfer-level) designs")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.24))
    ax.set_title("well-posedness of the operating point", fontsize=9)
    fig.tight_layout(); save(fig, "fig_transfer_parameterisation", bbox_inches="tight")

# ---------------------------------------------------- 16: Indonesia locality
L = load_npz(LOCALITY_NPZ)
if L is None:
    miss("fig_indonesia_locality", "rq2.robustness")
else:
    db, pp, cc2, nn = L["db"], L["p"], L["c"], L["n"]
    jump = np.where(np.diff(nn) != 0)[0]
    # Two stacked panels sharing x rather than a twin y-axis: price and coverage
    # are different measures and must not share a scale.
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 3.6), sharex=True,
                             layout="constrained")
    # One entity (the operating point) shown two ways, so one colour in both panels.
    axes[0].step(db, pp, where="post", color=BLUE, lw=1.8)
    axes[0].set_ylabel("price $p$ (€/t)")
    axes[1].step(db, cc2, where="post", color=BLUE, lw=1.8)
    axes[1].set_ylabel("coverage $c$")
    axes[1].set_xlabel("Indonesia's reported $\\alpha_{base}$ deviation (€/t)")
    for ax in axes:
        for k in jump:
            ax.axvline(db[k + 1], color=RED, lw=1.0, ls="--")
    if len(jump):
        axes[0].annotate(f"Indonesia enters at\n$\\Delta\\alpha_{{base}}$ = "
                         f"{db[jump[0]+1]:+.1f}",
                         xy=(db[jump[0] + 1], pp.min()), xytext=(-10, 6),
                         textcoords="offset points", fontsize=8, color=RED,
                         ha="right", va="bottom")
    axes[0].set_title("the operating point is piecewise constant in the report",
                      fontsize=8.5)
    save(fig, "fig_indonesia_locality", bbox_inches="tight")

# ------------------------------------------------------- 17: slack robustness
SL = load_npz(SLACK_NPZ)
if SL is None:
    miss("fig_slack_robustness", "rq2.robustness")
else:
    s = SL["slacks"].astype(float).copy()
    s[s == 0] = 1e-5                    # place the exact-zero case on the log axis
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    # INK vs entry-blue: validated pair (normal-vision dE 23.3, deutan 22.3).
    # The house blue #4a7c9b against #2a7fba fails the normal-vision floor at 5.1.
    ax.plot(s, SL["nashconv"], "-o", color=INK, lw=2, ms=5, label="NashConv (all actors)")
    ax.plot(s, SL["indonesia"], "-s", color=MODE_COLOR["entry"], lw=2, ms=5,
            label="Indonesia regret")
    ax.set_xscale("log")
    ax.set_xlabel("self-consistency slack (€/tCO$_2$e, log scale)")
    ax.set_ylabel("€/tCO$_2$e")
    ax.set_ylim(0, max(SL["nashconv"]) * 1.25)
    ax.axvline(0.01, color="#999", lw=0.9, ls="--")
    ax.text(0.011, max(SL["nashconv"]) * 1.16, "reported setting", fontsize=8,
            color="#555")
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1])
    ax.set_xticklabels(["0", "$10^{-4}$", "$10^{-3}$", "0.01", "0.1", "0.5"])
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("the tie-break tolerance does not create the exploitability",
                 fontsize=8.5)
    fig.tight_layout(); save(fig, "fig_slack_robustness", bbox_inches="tight")


print(f"wrote {len(built)} figures to {OUT}")
for s in skipped:
    print(f"  SKIPPED {s}")
