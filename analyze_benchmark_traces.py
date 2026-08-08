"""
analyze_benchmark_traces.py
---------------------------
Ingests execution-time traces measured on a real target by the harness in
bench_harness/ and runs the full analysis needed for the revised Section 5:
CP, tolerance bound, EVT with automatic threshold selection and profile
confidence limits, plus the exchangeability diagnostics.

Input: one CSV per benchmark, as emitted by pwcet_dump_csv(), i.e. optional
'#key,value' header lines followed by one integer sample per line.

    python analyze_benchmark_traces.py traces/*.csv --out results/bench.json

Every number this produces is measured. Nothing here simulates a target; if no
traces are supplied it exits rather than inventing data.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np

import evt_improved as E
from cp_methods import cp_pwcet, empirical_quantile
from quantile_estimators import (tolerance_bound, min_n_for_marginal,
                                 min_n_for_tolerance)

warnings.filterwarnings("ignore")


# =============================================================================
# Trace loading
# =============================================================================

def load_trace(path: str) -> tuple[np.ndarray, dict]:
    meta, vals = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].split(",", 1)
                if len(parts) == 2:
                    meta[parts[0].strip()] = parts[1].strip()
                continue
            try:
                vals.append(float(line.split(",")[0]))
            except ValueError:
                continue
    x = np.asarray(vals, dtype=float)
    meta.setdefault("task", os.path.splitext(os.path.basename(path))[0])
    meta.setdefault("unit", "cycles")
    return x, meta


# =============================================================================
# Diagnostics
# =============================================================================

def diagnostics(x: np.ndarray, n_lags: int = 20) -> dict:
    """Exchangeability diagnostics: serial correlation, stationarity, drift."""
    out = {}
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        xl = x if len(x) <= 50000 else x[:50000]
        lb = acorr_ljungbox(xl, lags=list(range(1, n_lags + 1)), return_df=True)
        out["ljungbox_min_p"] = float(lb["lb_pvalue"].min())
        out["ljungbox_n_sig"] = int((lb["lb_pvalue"] < 0.05).sum())
    except Exception as exc:
        out["ljungbox_error"] = str(exc)

    try:
        from statsmodels.tsa.stattools import adfuller
        # autolag="AIC" on a 200k-point series builds very large design
        # matrices; a fixed modest lag on a capped prefix is ample here and
        # keeps the test within memory.
        xs = x if len(x) <= 20000 else x[:20000]
        out["adf_p"] = float(adfuller(xs, maxlag=20, autolag=None)[1])
    except Exception as exc:
        out["adf_error"] = str(exc)

    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    out["acf_lag1"] = float(np.sum(xc[:-1] * xc[1:]) / denom) if denom > 0 else 0.0

    # Relative drift: mean of last decile vs first decile
    d = max(len(x) // 10, 1)
    first, last = float(np.mean(x[:d])), float(np.mean(x[-d:]))
    out["drift_rel"] = (last - first) / first if first > 0 else 0.0

    out["n"] = int(len(x))
    out["min"] = float(np.min(x))
    out["max"] = float(np.max(x))
    out["mean"] = float(np.mean(x))
    out["cv"] = float(np.std(x) / np.mean(x)) if np.mean(x) > 0 else 0.0
    # Observed-WCET ratio: how far the max sits above the median
    out["max_over_median"] = float(np.max(x) / np.median(x))
    return out


# =============================================================================
# Coverage evaluation by repeated sub-sampling
# =============================================================================

def subsample_evaluation(x: np.ndarray, n_cal: int, alpha: float,
                         delta: float, n_trials: int, seed: int) -> dict:
    """Split the measured trace into calibration/test repeatedly.

    Sub-sampling is done by random permutation, which is the correct protocol
    under the exchangeability hypothesis being tested. When the trace is
    autocorrelated this protocol destroys the serial structure, so results are
    reported alongside the diagnostics rather than in place of them.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < n_cal + 200:
        return {"error": f"trace too short: {n} < {n_cal + 200}"}

    methods = ["CP", "Tolerance", "EQ", "EVT-cv-point", "EVT-cv-profile",
               "EVT-gof-point", "EVT-gof-profile"]
    acc = {m: {"cov": [], "bound": []} for m in methods}

    for _ in range(n_trials):
        perm = rng.permutation(n)
        cal, test = x[perm[:n_cal]], x[perm[n_cal:]]

        b = {"CP": cp_pwcet(cal, alpha),
             "Tolerance": tolerance_bound(cal, alpha, delta),
             "EQ": empirical_quantile(cal, alpha)}
        for tm, tag in (("mbpta_cv", "cv"), ("seq_gof", "gof")):
            fit = E.fit_evt(cal, threshold_method=tm)
            if fit is None:
                b[f"EVT-{tag}-point"] = np.inf
                b[f"EVT-{tag}-profile"] = np.inf
            else:
                b[f"EVT-{tag}-point"] = E.gpd_quantile(fit, alpha)
                b[f"EVT-{tag}-profile"] = E.gpd_quantile_profile_ci(
                    cal, fit, alpha, delta)

        for m, v in b.items():
            acc[m]["cov"].append(float(np.mean(test <= v)))
            acc[m]["bound"].append(float(v) if np.isfinite(v) else float("inf"))

    res = {}
    for m in methods:
        cov = np.asarray(acc[m]["cov"])
        bd = np.asarray(acc[m]["bound"])
        finite = bd[np.isfinite(bd)]
        res[m] = {
            "cov_mean": float(np.mean(cov)),
            "cov_ci95": float(1.96 * np.std(cov) / np.sqrt(len(cov))),
            "shortfall_pp": float(max(0.0, (1 - alpha) - np.mean(cov)) * 100),
            "bound_median": float(np.median(finite)) if len(finite) else float("inf"),
            "inf_rate": float(np.mean(~np.isfinite(bd))),
        }
    return res


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", help="CSV trace files or globs")
    ap.add_argument("--n-cal", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", default="results/benchmark_analysis.json")
    args = ap.parse_args()

    paths = []
    for p in args.traces:
        paths.extend(sorted(glob.glob(p)))
    if not paths:
        raise SystemExit("no trace files matched; nothing to analyse")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = []

    print(f"Calibration size {args.n_cal}, alpha={args.alpha}, delta={args.delta}")
    print(f"Minimum n for a marginal bound at this alpha: "
          f"{min_n_for_marginal(args.alpha)}")
    print(f"Minimum n for a (1-delta) tolerance bound:    "
          f"{min_n_for_tolerance(args.alpha, args.delta)}\n")

    for path in paths:
        x, meta = load_trace(path)
        if len(x) < 100:
            print(f"  skip {path}: only {len(x)} samples")
            continue

        diag = diagnostics(x)
        cov = subsample_evaluation(x, args.n_cal, args.alpha, args.delta,
                                   args.trials, seed=abs(hash(path)) % 2**31)
        fit = E.fit_evt(x, threshold_method="seq_gof")

        rec = {"file": path, "meta": meta, "diagnostics": diag,
               "coverage": cov,
               "full_trace_gpd_fit": (None if fit is None else
                                      {"u": fit.u, "xi": fit.xi,
                                       "sigma": fit.sigma, "k": fit.k,
                                       "n": fit.n})}
        results.append(rec)

        print(f"=== {meta['task']}  ({diag['n']} samples, {meta['unit']}) ===")
        print(f"    mean={diag['mean']:.1f}  CV={diag['cv']:.3f}  "
              f"max/median={diag['max_over_median']:.2f}")
        print(f"    ACF(1)={diag['acf_lag1']:+.3f}  "
              f"drift={diag['drift_rel']*100:+.2f}%  "
              f"LB sig lags={diag.get('ljungbox_n_sig','?')}/20  "
              f"ADF p={diag.get('adf_p', float('nan')):.4f}")
        if fit is not None:
            print(f"    GPD fit (seq-GoF): xi={fit.xi:+.3f} sigma={fit.sigma:.1f} "
                  f"u={fit.u:.1f} k={fit.k}")
        if "error" in cov:
            print(f"    coverage: {cov['error']}")
        else:
            print(f"    {'method':18s} {'coverage':>10s} {'shortfall':>10s} {'bound':>12s}")
            for m, r in cov.items():
                print(f"    {m:18s} {r['cov_mean']*100:9.2f}% "
                      f"{r['shortfall_pp']:9.2f}pp {r['bound_median']:12.1f}")
        print()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
