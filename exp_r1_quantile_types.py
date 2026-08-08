"""
exp_r1_quantile_types.py
------------------------
This experiment compares the CP-pWCET bound against all nine sample-quantile
definitions of Hyndman & Fan (1996), plus the distribution-free tolerance bound,
on exchangeable draws from several execution-time distributions.

The purpose is not to show that CP is a better estimator -- it is one of the
nine, evaluated at a particular index -- but to quantify what that index choice
buys: every interpolating type (4-9), including the NumPy/R default (type 7)
that a practitioner would reach for by default, undercovers by a small but
systematic margin, and the margin grows as n falls.

Usage:  python exp_r1_quantile_types.py [--trials 2000]
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

from quantile_estimators import hyndman_fan, HF_LABELS, tolerance_bound
from cp_methods import cp_pwcet

warnings.filterwarnings("ignore")

MU = 200.0
OUT_DIR = "results"


def make_dgp(kind):
    if kind == "gpd_light":
        xi, sig = -0.1, 50.0
    elif kind == "gpd_heavy":
        xi, sig = 0.2, 50.0
    elif kind == "gpd_vheavy":
        xi, sig = 0.4, 50.0
    else:
        raise ValueError(kind)

    def sampler(n, rng):
        u = rng.uniform(size=n)
        return MU + (sig / xi) * ((1 - u) ** (-xi) - 1)

    def tq(a):
        return MU + (sig / xi) * (a ** (-xi) - 1)

    return sampler, tq, f"GPD xi={xi}"


def run(kind, n_cal, alpha, n_trials, n_test, delta, seed):
    sampler, tq, label = make_dgp(kind)
    q_true = tq(alpha)
    rng = np.random.default_rng(seed)

    names = [HF_LABELS[k] for k in range(1, 10)] + ["CP-pWCET", "Tolerance"]
    acc = {nm: {"cov": [], "eff": []} for nm in names}

    for _ in range(n_trials):
        cal = sampler(n_cal, rng)
        test = sampler(n_test, rng)
        bounds = {HF_LABELS[k]: hyndman_fan(cal, 1 - alpha, k)
                  for k in range(1, 10)}
        bounds["CP-pWCET"] = cp_pwcet(cal, alpha)
        bounds["Tolerance"] = tolerance_bound(cal, alpha, delta)
        for nm, b in bounds.items():
            acc[nm]["cov"].append(1.0 if not np.isfinite(b)
                                  else float(np.mean(test <= b)))
            if np.isfinite(b):
                acc[nm]["eff"].append(b / q_true)

    res = {"dataset": label, "n_cal": n_cal, "alpha": alpha,
           "n_trials": n_trials, "q_true": q_true, "methods": {}}
    for nm in names:
        cov = np.asarray(acc[nm]["cov"])
        eff = np.asarray(acc[nm]["eff"]) if acc[nm]["eff"] else np.array([np.nan])
        res["methods"][nm] = {
            "cov_mean": float(np.mean(cov)),
            "cov_ci": float(1.96 * np.std(cov) / np.sqrt(len(cov))),
            "shortfall_pp": float(max(0.0, (1 - alpha) - np.mean(cov)) * 100),
            "eff_mean": float(np.nanmean(eff)),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--n-test", type=int, default=20_000)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "r1_quantile_types.json"))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for kind in ("gpd_light", "gpd_heavy", "gpd_vheavy"):
        for n_cal in (50, 200, 1000):
            r = run(kind, n_cal, args.alpha, args.trials,
                    args.n_test, args.delta, seed=hash((kind, n_cal)) % 2**31)
            results.append(r)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    for r in results:
        print(f"\n=== {r['dataset']}, n={r['n_cal']}, alpha={r['alpha']}, "
              f"{r['n_trials']} trials (nominal coverage "
              f"{(1-r['alpha'])*100:.0f}%) ===")
        print(f"    {'estimator':30s} {'coverage':>10s} {'shortfall':>10s} {'eff':>7s}")
        for nm, m in r["methods"].items():
            flag = "  <-- undercovers" if m["shortfall_pp"] > 0.05 else ""
            print(f"    {nm:30s} {m['cov_mean']*100:9.2f}% "
                  f"{m['shortfall_pp']:9.2f}pp {m['eff_mean']:7.3f}{flag}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
