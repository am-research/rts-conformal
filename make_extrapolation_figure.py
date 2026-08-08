"""
make_extrapolation_figure.py
----------------------------
Merges the per-DGP outputs of exp_r1_extrapolation.py and renders the figure
used in the revised manuscript (new Experiment 8).

The panel per data-generating process shows, against extrapolation depth
(alpha on a log scale), the probability that each method's bound lies at or
above the true quantile -- i.e. the realised confidence level of an upper
limit whose nominal level is 1 - delta = 95%.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIG_DIR = "figures"

ALPHAS = [1e-2, 1e-3, 1e-4, 1e-5]

SERIES = [
    ("Tolerance", "Conformal tolerance bound", "tab:blue", "-", "o"),
    ("EVT-gof-profile", "EVT (seq. GoF) profile UCL", "tab:green", "--", "s"),
    ("EVT-cv-profile", "EVT (MBPTA-CV) profile UCL", "tab:red", "--", "^"),
    ("EVT-cv-point", "EVT (MBPTA-CV) point est.", "tab:orange", ":", "v"),
]


def load():
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "r1_extrapolation_*.json"))):
        with open(p) as f:
            out.extend(json.load(f))
    merged = os.path.join(RESULTS_DIR, "r1_extrapolation.json")
    with open(merged, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Merged {len(out)} datasets -> {merged}")
    return out


def main():
    data = load()
    if not data:
        raise SystemExit("no results found; run exp_r1_extrapolation.py first")

    order = {"gpd": 0, "lognormal": 1, "mixture": 2}
    data.sort(key=lambda d: order.get(d["dgp"], 99))

    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 4.2),
                             sharey=True)
    if len(data) == 1:
        axes = [axes]

    for ax, d in zip(axes, data):
        for key, label, color, ls, mk in SERIES:
            xs, ys = [], []
            for a in ALPHAS:
                c = d["cells"].get(f"{key}|{a:g}")
                if c is None or c["inf_rate"] > 0.99:
                    continue
                xs.append(a)
                ys.append(c["q_cov"] * 100)
            if xs:
                ax.plot(xs, ys, marker=mk, ls=ls, color=color, label=label,
                        lw=2.2, ms=7)

        ax.axhline(95, color="k", ls="-.", lw=1.0, alpha=0.7)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_ylim(-4, 104)
        ax.set_xlabel(r"target exceedance level $\alpha$")
        ax.set_title(d["label"], fontsize=14)
        ax.grid(alpha=0.3)

        # Mark where the distribution-free bound stops being available
        finite = [a for a in ALPHAS
                  if d["cells"].get(f"Tolerance|{a:g}", {}).get("inf_rate", 1) <= 0.99]
        if finite:
            ax.axvspan(min(finite), min(ALPHAS) / 2, color="grey", alpha=0.12)

    axes[0].set_ylabel("P(bound $\\geq$ true quantile)  [%]")
    axes[0].annotate("nominal 95%", xy=(0.03, 0.90), xycoords="axes fraction",
                     fontsize=12, color="k")
    axes[-1].legend(fontsize=12, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"extrapolation_validity.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p)

    # Console summary table for the manuscript text
    print("\nRealised confidence level (%) of a nominal 95% upper limit:")
    for d in data:
        print(f"\n  {d['label']}   (n_cal={d['n_cal']}, {d['n_trials']} trials)")
        hdr = "".join(f"{a:>11g}" for a in ALPHAS)
        print(f"    {'true quantile':22s}" +
              "".join(f"{d['q_true'][f'{a:g}']:11.1f}" for a in ALPHAS))
        print(f"    {'method':22s}{hdr}")
        for key, label, *_ in SERIES:
            row = ""
            for a in ALPHAS:
                c = d["cells"].get(f"{key}|{a:g}")
                row += ("     --    " if (c is None or c["inf_rate"] > 0.99)
                        else f"{c['q_cov']*100:10.1f}%")
            print(f"    {label:22s}{row}")


if __name__ == "__main__":
    main()
