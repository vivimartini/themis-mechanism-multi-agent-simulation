"""Measure the font sizes a figure will actually PRINT at.

A saved figure's point sizes are not the printed point sizes. `bbox_inches="tight"`
crops the canvas to its content, so the saved width bears no fixed relation to the
width the document asks for, and LaTeX rescales by

    scale = width_factor * linewidth / saved_width

Every text run is multiplied by that. A figure authored at 9pt and placed at
0.6\\linewidth while saving 527pt wide prints at 4.6pt.

This script reads the Tf operators out of each PDF's content streams and reports
the distribution of run sizes, scaled to the page. Dominant size matters more than
minimum: one small annotation is a blemish, but if the dominant run is 4.6pt then
every tick and label in the figure is illegible.

  python scripts/check_figure_fonts.py            # uses the dissertation mapping
  python scripts/check_figure_fonts.py --floor 8
"""
from __future__ import annotations

import argparse
import collections
import re
import zlib
from pathlib import Path

# A4 (21.0cm) less 2x2.6cm margins, in TeX points (72.27 pt/in).
LINEWIDTH_PT = (21.0 - 2 * 2.6) / 2.54 * 72.27

# repository filename -> width factor used by the corresponding \includegraphics
WIDTHS = {
    "fig_mc_operating_price.pdf": 1.00,
    "fig_rq2_pipeline.pdf": 1.00,
    "fig_optimizer_comparison.pdf": 0.98,
    "fig_nashconv_fixed_vs_endogenous.pdf": 0.55,
    "fig_regret_geography.pdf": 0.72,
    "fig_exinterim_exploitability.pdf": 0.98,
    "fig_regime_map.pdf": 0.95,
    "fig_obstruction_damage.pdf": 0.90,
    "fig_guardrail_ablation.pdf": 0.98,
    "fig_alpharank_mass.pdf": 0.98,
}

TF = re.compile(rb"/[A-Za-z0-9_+-]+\s+([0-9]*\.?[0-9]+)\s+Tf")
MEDIABOX = re.compile(rb"MediaBox\s*\[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")


def streams(raw: bytes):
    """Yield decompressed content streams (FlateDecode, else raw)."""
    for m in re.finditer(rb"stream\r?\n", raw):
        start = m.end()
        end = raw.find(b"endstream", start)
        if end < 0:
            continue
        chunk = raw[start:end]
        try:
            yield zlib.decompress(chunk)
        except zlib.error:
            yield chunk


def font_runs(path: Path):
    raw = path.read_bytes()
    sizes = []
    for s in streams(raw):
        sizes += [round(float(v), 2) for v in TF.findall(s)]
    box = MEDIABOX.search(raw)
    width = float(box.group(3)) - float(box.group(1)) if box else float("nan")
    return sizes, width


SUB_RATIO = 0.7          # matplotlib mathtext renders sub/superscripts at 0.7x


def split_sub(sizes):
    """Separate mathtext sub/superscript runs from base text runs.

    A subscript in "tCO$_2$e" is legitimately smaller than the label it sits in;
    applying an 8pt floor to it would fail every figure in the document for a
    reason no examiner would recognise as a defect. A run counts as a subscript
    when its size is 0.7x some other size present in the same figure.
    """
    uniq = sorted(set(sizes))
    subs = {s for s in uniq
            if any(abs(s - SUB_RATIO * t) < 0.05 for t in uniq if t > s)}
    return [s for s in sizes if s not in subs], [s for s in sizes if s in subs]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "figures")
    ap.add_argument("--floor", type=float, default=8.0)
    args = ap.parse_args(argv)

    print(f"linewidth = {LINEWIDTH_PT:.2f} pt   floor = {args.floor:g} pt "
          f"(base text; sub/superscripts follow at {SUB_RATIO:g}x)\n")
    hdr = (f"{'figure':<28}{'w':>5}{'saved':>8}{'scale':>7}"
           f"{'domin.':>8}{'runs':>6}{'min base':>10}{'min sub':>9}  verdict")
    print(hdr); print("-" * len(hdr))
    bad = []
    for name, wf in sorted(WIDTHS.items()):
        p = args.dir / name
        if not p.exists():
            print(f"{name:<28}  MISSING"); bad.append(name); continue
        sizes, saved = font_runs(p)
        if not sizes:
            print(f"{name:<28}  no text runs found"); continue
        scale = wf * LINEWIDTH_PT / saved
        base, subs = split_sub(sizes)
        cnt = collections.Counter(base)
        dom_sz, dom_n = cnt.most_common(1)[0]
        dom_p = dom_sz * scale
        min_base = min(base) * scale
        min_sub = min(subs) * scale if subs else float("nan")
        ok = min_base >= args.floor - 1e-6
        if not ok:
            bad.append(name)
        sub_txt = "     —" if subs == [] else f"{min_sub:>9.1f}"
        print(f"{name:<28}{wf:>5.2f}{saved:>8.0f}{scale:>7.3f}"
              f"{dom_p:>8.1f}{dom_n:>6d}{min_base:>10.1f}{sub_txt}"
              f"  {'ok' if ok else 'FAILS'}")
    print()
    if bad:
        print(f"{len(bad)} figure(s) with base text below {args.floor:g}pt: "
              + ", ".join(sorted(bad)))
        return 1
    print(f"all {len(WIDTHS)} figures: base text at or above {args.floor:g}pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
