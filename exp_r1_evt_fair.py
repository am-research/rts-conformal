"""
exp_r1_evt_fair.py
------------------

The original submission compared a conformal interval against an EVT point
estimate obtained at a fixed 90th-percentile threshold. This experiment removes
both handicaps:

  * threshold selection is automatic (MBPTA-CV and sequential AD goodness-of-fit)
  * EVT is evaluated both as a point estimate and as a one-sided upper
    confidence limit (delta method and profile likelihood)

Two distinct notions of coverage are reported, because conflating them is the
source of the apparent unfairness:

  observation coverage : P(T_new <= C_hat), the pWCET quantity of interest.
                         This is what Theorem 1 (CP) controls.
  quantile coverage    : P(q_{1-alpha} <= C_hat), the probability that the bound
                         sits above the *true* quantile. This is what a
                         confidence interval on the quantile controls, at its
                         own level 1-delta.

Usage:  python exp_r1_evt_fair.py [--trials 500] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

import evt_improved as E
from cp_methods import cp_pwcet, empirical_quantile

warnings.filterwarnings("ignore")

OUT_DIR = "results"


# =============================================================================
# Data-generating processes, including deliberately misspecified tails
# =============================================================================

def make_dgp(kind: str, **kw):
    """Return (sampler(n, rng), true_quantile(alpha), label)."""
    mu = kw.get("mu", 200.0)

    if kind == "gpd":
        xi, sig = kw.get("xi", 0.2), kw.get("sigma", 50.0)

        def sampler(n, rng):
            u = rng.uniform(size=n)
            if abs(xi) < 1e-9:
                return mu - sig * np.log(1 - u)
            return mu + (sig / xi) * ((1 - u) ** (-xi) - 1)

        def tq(alpha):
            if abs(xi) < 1e-9:
                return mu - sig * np.log(alpha)
            return mu + (sig / xi) * (alpha ** (-xi) - 1)

        return sampler, tq, f"GPD xi={xi}"

    if kind == "lognormal":
        # Sub-exponential but *not* GPD at moderate thresholds: the classic
        # slowly-converging domain-of-attraction case (xi -> 0 very slowly).
        s = kw.get("s", 0.6)

        def sampler(n, rng):
            return mu + np.exp(rng.normal(0.0, s, size=n)) * 50.0

        def tq(alpha):
            from scipy import stats as st
            return mu + np.exp(st.norm.ppf(1 - alpha) * s) * 50.0

        return sampler, tq, f"Lognormal s={s}"

    if kind == "mixture_bump":
        # Low-density mixture in the high-quantile tail (cf. Blau Manau 2025):
        # a GPD body with a small, heavier second component that only becomes
        # visible beyond the quantile at which the threshold is chosen.
        xi, sig = kw.get("xi", 0.1), kw.get("sigma", 50.0)
        w = kw.get("w", 0.01)
        shift = kw.get("shift", 400.0)
        xi2, sig2 = kw.get("xi2", 0.35), kw.get("sigma2", 120.0)

        def sampler(n, rng):
            m = rng.uniform(size=n) < w
            u = rng.uniform(size=n)
            base = mu + (sig / xi) * ((1 - u) ** (-xi) - 1)
            heavy = mu + shift + (sig2 / xi2) * ((1 - u) ** (-xi2) - 1)
            return np.where(m, heavy, base)

        def tq(alpha):
            # Numerical inversion of the mixture survival function
            from scipy import optimize as opt

            def surv(t):
                a = (1 + xi * (t - mu) / sig) ** (-1 / xi) if t > mu else 1.0
                b = ((1 + xi2 * (t - mu - shift) / sig2) ** (-1 / xi2)
                     if t > mu + shift else 1.0)
                return (1 - w) * min(a, 1.0) + w * min(b, 1.0)

            return float(opt.brentq(lambda t: surv(t) - alpha,
                                    mu, mu + 1e7, xtol=1e-6))

        return sampler, tq, f"Mixture bump w={w}"

    raise ValueError(kind)


# =============================================================================
# Method set
# =============================================================================

def evaluate_methods(cal, alpha, delta):
    """Return {method_name: bound} for one calibration draw."""
    out = {}
    out["CP"] = cp_pwcet(cal, alpha)
    out["EQ"] = empirical_quantile(cal, alpha)

    for tm, tag in (("fixed90", "fix90"), ("mbpta_cv", "cv"), ("seq_gof", "gof")):
        fit = E.fit_evt(cal, threshold_method=tm)
        if fit is None:
            out[f"EVT-{tag}-point"] = np.inf
            if tm != "fixed90":
                out[f"EVT-{tag}-delta"] = np.inf
            continue
        out[f"EVT-{tag}-point"] = E.gpd_quantile(fit, alpha)
        if tm != "fixed90":
            out[f"EVT-{tag}-delta"] = E.gpd_quantile_delta_ci(fit, alpha, delta)[2]
        if tm == "mbpta_cv":
            out["EVT-cv-profile"] = E.gpd_quantile_profile_ci(cal, fit, alpha, delta)
    return out


# =============================================================================
# Monte-Carlo driver
# =============================================================================

def run(dgp_kind, dgp_kw, n_cal, alpha, delta, n_trials, n_test, seed):
    sampler, tq, label = make_dgp(dgp_kind, **dgp_kw)
    q_true = tq(alpha)
    rng = np.random.default_rng(seed)

    acc = {}
    for t in range(n_trials):
        cal = sampler(n_cal, rng)
        test = sampler(n_test, rng)
        bounds = evaluate_methods(cal, alpha, delta)
        for name, b in bounds.items():
            rec = acc.setdefault(name, {"obs_cov": [], "q_cov": [],
                                        "eff": [], "inf": 0})
            if not np.isfinite(b):
                rec["inf"] += 1
                rec["obs_cov"].append(1.0)
                rec["q_cov"].append(1.0)
                continue
            rec["obs_cov"].append(float(np.mean(test <= b)))
            rec["q_cov"].append(float(b >= q_true))
            rec["eff"].append(float(b / q_true))

    summary = {"dataset": label, "n_cal": n_cal, "alpha": alpha,
               "delta": delta, "n_trials": n_trials, "q_true": q_true,
               "methods": {}}
    for name, rec in acc.items():
        oc = np.asarray(rec["obs_cov"])
        qc = np.asarray(rec["q_cov"])
        ef = np.asarray(rec["eff"]) if rec["eff"] else np.array([np.nan])
        summary["methods"][name] = {
            "obs_cov_mean": float(np.mean(oc)),
            "obs_cov_ci": float(1.96 * np.std(oc) / np.sqrt(len(oc))),
            "obs_shortfall_pp": float(max(0.0, (1 - alpha) - np.mean(oc)) * 100),
            "q_cov_rate": float(np.mean(qc)),
            "eff_median": float(np.nanmedian(ef)),
            "inf_rate": rec["inf"] / n_trials,
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-cal", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "r1_evt_fair.json"))
    args = ap.parse_args()

    n_trials = 50 if args.quick else args.trials
    n_test = 5_000 if args.quick else 20_000

    configs = [
        ("gpd", {"xi": 0.2}),
        ("gpd", {"xi": 0.4}),
        ("lognormal", {"s": 0.6}),
        ("mixture_bump", {"w": 0.01}),
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for i, (kind, kw) in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {kind} {kw} ...", flush=True)
        s = run(kind, kw, args.n_cal, args.alpha, args.delta,
                n_trials, n_test, seed=1000 + i)
        results.append(s)
        print(f"    {s['dataset']}: true q = {s['q_true']:.1f}")
        for m, r in sorted(s["methods"].items()):
            print(f"      {m:20s} obscov={r['obs_cov_mean']*100:6.2f}% "
                  f"short={r['obs_shortfall_pp']:5.2f}pp "
                  f"qcov={r['q_cov_rate']*100:6.1f}% "
                  f"eff={r['eff_median']:.3f} inf={r['inf_rate']*100:.0f}%")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
