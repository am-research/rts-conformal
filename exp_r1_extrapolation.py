"""
exp_r1_extrapolation.py
-----------------------

This experiment supplies that example. EVT is given every advantage the reviewer
asked for:

  * automatic threshold selection (MBPTA-CV and sequential AD goodness-of-fit),
    not a fixed 90th percentile
  * a profile-likelihood confidence limit on the quantile, which includes the
    binomial uncertainty in the exceedance rate and is well calibrated at
    realistic exceedance counts

and it is then evaluated in the regime that actually matters for certification:
extrapolation to alpha well below 1/n.

Three data-generating processes:

  gpd        - correctly specified; EVT should and does succeed
  lognormal  - in the Gumbel domain of attraction but slowly converging
  mixture    - a light GPD body with a low-density heavier component that is
               invisible at any threshold a selector would choose, but that
               dominates the quantile for alpha <~ 1e-3 (cf. the low-density
               tail mixtures of Blau Manau 2025)

The diagnostic under test: CP and EVT are compared at the largest alpha where CP
still returns a finite bound. A disagreement there predicts EVT failure at the
smaller alpha values where CP is silent.

Usage:  python exp_r1_extrapolation.py [--trials 300] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
from scipy import optimize as opt

import evt_improved as E
from cp_methods import cp_pwcet
from quantile_estimators import tolerance_bound, min_n_for_marginal

warnings.filterwarnings("ignore")

MU = 200.0
OUT_DIR = "results"


# =============================================================================
# Data-generating processes
# =============================================================================

class DGP:
    def __init__(self, name, sampler, survival, label):
        self.name = name
        self.sampler = sampler
        self.survival = survival
        self.label = label

    def quantile(self, alpha: float) -> float:
        return float(opt.brentq(lambda t: self.survival(t) - alpha,
                                MU + 1e-9, MU + 1e9, xtol=1e-8))


def dgp_gpd(xi=0.2, sigma=50.0):
    def sampler(n, rng):
        u = rng.uniform(size=n)
        return MU + (sigma / xi) * ((1 - u) ** (-xi) - 1)

    def surv(t):
        return (1 + xi * (t - MU) / sigma) ** (-1 / xi) if t > MU else 1.0

    return DGP("gpd", sampler, surv, f"GPD xi={xi} (well specified)")


def dgp_lognormal(s=0.6, scale=50.0):
    from scipy import stats as st

    def sampler(n, rng):
        return MU + np.exp(rng.normal(0.0, s, size=n)) * scale

    def surv(t):
        if t <= MU:
            return 1.0
        return float(st.norm.sf(np.log((t - MU) / scale) / s))

    return DGP("lognormal", sampler, surv, f"Lognormal s={s} (slow convergence)")


def dgp_mixture(xi=0.05, sigma=50.0, w=0.002, shift=500.0,
                xi2=0.45, sigma2=150.0):
    """Light GPD body with a low-density heavy component far out in the tail.

    Calibrated so the mixture is within ~3% of the pure-GPD body at alpha=1e-2
    (hence invisible to any threshold selector) but a factor of ~2 above it at
    alpha=1e-4.
    """
    def sampler(n, rng):
        m = rng.uniform(size=n) < w
        u = rng.uniform(size=n)
        base = MU + (sigma / xi) * ((1 - u) ** (-xi) - 1)
        heavy = MU + shift + (sigma2 / xi2) * ((1 - u) ** (-xi2) - 1)
        return np.where(m, heavy, base)

    def surv(t):
        a = (1 + xi * (t - MU) / sigma) ** (-1 / xi) if t > MU else 1.0
        b = ((1 + xi2 * (t - MU - shift) / sigma2) ** (-1 / xi2)
             if t > MU + shift else 1.0)
        return (1 - w) * min(a, 1.0) + w * min(b, 1.0)

    return DGP("mixture", sampler, surv,
               f"Mixture w={w} (low-density tail component)")


DGPS = [dgp_gpd(), dgp_lognormal(), dgp_mixture()]


# =============================================================================
# One trial
# =============================================================================

def trial_bounds(cal, alphas, delta):
    """Compute every method's bound at every alpha for one calibration set."""
    out = {}
    for a in alphas:
        out[("CP", a)] = cp_pwcet(cal, a)
        out[("Tolerance", a)] = tolerance_bound(cal, a, delta)

    for tm, tag in (("mbpta_cv", "cv"), ("seq_gof", "gof")):
        fit = E.fit_evt(cal, threshold_method=tm)
        for a in alphas:
            if fit is None:
                out[(f"EVT-{tag}-point", a)] = np.inf
                out[(f"EVT-{tag}-profile", a)] = np.inf
            else:
                out[(f"EVT-{tag}-point", a)] = E.gpd_quantile(fit, a)
                out[(f"EVT-{tag}-profile", a)] = E.gpd_quantile_profile_ci(
                    cal, fit, a, delta)
    return out


METHODS = ["CP", "Tolerance", "EVT-cv-point", "EVT-cv-profile",
           "EVT-gof-point", "EVT-gof-profile"]


def run_dgp(dgp, n_cal, alphas, delta, n_trials, seed):
    q_true = {a: dgp.quantile(a) for a in alphas}
    rng = np.random.default_rng(seed)

    acc = {(m, a): {"cov": [], "ratio": []} for m in METHODS for a in alphas}
    for _ in range(n_trials):
        cal = dgp.sampler(n_cal, rng)
        b = trial_bounds(cal, alphas, delta)
        for (m, a), v in b.items():
            rec = acc[(m, a)]
            rec["cov"].append(float(v >= q_true[a]))
            rec["ratio"].append(float(v / q_true[a]) if np.isfinite(v) else np.inf)

    res = {"dgp": dgp.name, "label": dgp.label, "n_cal": n_cal,
           "delta": delta, "n_trials": n_trials,
           "q_true": {f"{a:g}": q_true[a] for a in alphas},
           "cp_min_n": {f"{a:g}": min_n_for_marginal(a) for a in alphas},
           "cells": {}}
    for (m, a), rec in acc.items():
        ratios = np.asarray(rec["ratio"])
        finite = ratios[np.isfinite(ratios)]
        res["cells"][f"{m}|{a:g}"] = {
            "q_cov": float(np.mean(rec["cov"])),
            "median_ratio": float(np.median(finite)) if len(finite) else float("inf"),
            "inf_rate": float(np.mean(~np.isfinite(ratios))),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-cal", type=int, default=10_000)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--dgp", type=int, default=-1,
                    help="run only this DGP index (0=gpd,1=lognormal,2=mixture)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_trials = 30 if args.quick else args.trials
    alphas = [1e-2, 1e-3, 1e-4, 1e-5]

    todo = list(enumerate(DGPS))
    if args.dgp >= 0:
        todo = [(args.dgp, DGPS[args.dgp])]
    out_path = args.out or os.path.join(
        OUT_DIR, f"r1_extrapolation{'' if args.dgp < 0 else '_' + str(args.dgp)}.json")

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for i, dgp in todo:
        print(f"\n=== {dgp.label}  (n_cal={args.n_cal}, {n_trials} trials) ===",
              flush=True)
        r = run_dgp(dgp, args.n_cal, alphas, args.delta, n_trials, seed=500 + i)
        results.append(r)

        hdr = "  ".join(f"{a:>9g}" for a in alphas)
        print(f"    true q:        {'  '.join(f'{r[chr(113)+chr(95)+chr(116)+chr(114)+chr(117)+chr(101)][f'{a:g}']:9.1f}' for a in alphas)}")
        print(f"    {'method':16s} {hdr}")
        for m in METHODS:
            cells = [r["cells"][f"{m}|{a:g}"] for a in alphas]
            cov = "  ".join(
                ("  n/a(inf)" if c["inf_rate"] > 0.99 else f"{c['q_cov']*100:8.1f}%")
                for c in cells)
            print(f"    {m:16s} {cov}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
