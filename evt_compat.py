"""
evt_compat.py
-------------
Drop-in replacement for ``cp_methods.evt_gpd`` that routes to the strengthened
EVT baseline of ``evt_improved.py``.

The point of this module is that re-running Experiments 1-4 and 7 with the
strengthened baseline requires changing exactly one import line in
``run_experiments.py``:

    # before
    from cp_methods import cp_pwcet, evt_gpd, empirical_quantile
    # after
    from cp_methods import cp_pwcet, empirical_quantile
    from evt_compat import evt_gpd

The signature is unchanged, so every existing call site keeps working. The
``threshold_method`` and ``threshold_pct`` arguments are accepted and ignored
except when ``threshold_method='percentile_legacy'``, which reproduces the
original fixed-percentile behaviour for ablation.

Configuration
-------------
Set the module-level defaults, or the environment variables, before running:

    EVT_THRESHOLD   mbpta_cv (default) | seq_gof | fixed90
    EVT_MODE        point (default) | profile | delta
    EVT_DELTA       confidence level for interval modes (default 0.05)

``point`` yields the MLE quantile estimate and is the correct setting for
experiments that compare point estimates. ``profile`` yields a one-sided
(1-delta) profile-likelihood upper limit and is the correct setting for
experiments that compare intervals.

Note on cost: ``mbpta_cv`` adds roughly 0.04 s per call over the legacy rule,
which is negligible. ``seq_gof`` adds roughly 0.6 s per call and will dominate
the runtime of a 500-trial sweep; use it for the headline comparison rather
than for every experiment. ``profile`` adds roughly 0.1 s per call.

The sequential goodness-of-fit rule needs ``ad_null_table.npz``, which ships
with this workspace. If it is absent it is regenerated automatically (~155 s,
one off) and cached.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

import evt_improved as _E
import cp_methods as _legacy

# Capture the original implementation at import time. run_experiments_v2.py
# rebinds cp_methods.evt_gpd to this module's evt_gpd so that
# cp_methods.run_coverage_trial picks up the strengthened baseline; without
# this snapshot, the fallback below would resolve back to this module and
# recurse until the stack is exhausted.
_LEGACY_EVT_GPD = _legacy.evt_gpd

warnings.filterwarnings("ignore")

# --- configuration -----------------------------------------------------------

THRESHOLD = os.environ.get("EVT_THRESHOLD", "mbpta_cv")
MODE = os.environ.get("EVT_MODE", "point")
DELTA = float(os.environ.get("EVT_DELTA", "0.05"))

_VALID_THRESHOLDS = {"mbpta_cv", "seq_gof", "fixed90"}
_VALID_MODES = {"point", "profile", "delta"}


def configure(threshold: str | None = None,
              mode: str | None = None,
              delta: float | None = None) -> None:
    """Set the baseline configuration programmatically."""
    global THRESHOLD, MODE, DELTA
    if threshold is not None:
        if threshold not in _VALID_THRESHOLDS:
            raise ValueError(f"threshold must be one of {_VALID_THRESHOLDS}")
        THRESHOLD = threshold
    if mode is not None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}")
        MODE = mode
    if delta is not None:
        DELTA = float(delta)


def describe() -> str:
    return f"EVT baseline: threshold={THRESHOLD}, mode={MODE}, delta={DELTA}"


# --- drop-in replacement -----------------------------------------------------

def evt_gpd(calibration: np.ndarray,
            alpha: float,
            threshold_method: str = "percentile",
            threshold_pct: float = 0.90) -> float:
    """Strengthened EVT-GPD bound with the legacy call signature.

    ``threshold_method='percentile_legacy'`` falls through to the original
    implementation, so an ablation against the fixed-threshold baseline can be
    run from the same code path.
    """
    if threshold_method == "percentile_legacy":
        return _LEGACY_EVT_GPD(calibration, alpha, "percentile", threshold_pct)

    cal = np.asarray(calibration, dtype=float)
    if len(cal) < 20:
        # Too few points for any threshold rule to be meaningful; fall back to
        # the legacy estimator rather than silently returning +inf, so the
        # small-n stress test still exercises EVT.
        return _LEGACY_EVT_GPD(cal, alpha, "percentile", threshold_pct)

    fit = _E.fit_evt(cal, threshold_method=THRESHOLD)
    if fit is None:
        return float("inf")

    if MODE == "point":
        return _E.gpd_quantile(fit, alpha)
    if MODE == "delta":
        return _E.gpd_quantile_delta_ci(fit, alpha, DELTA)[2]
    return _E.gpd_quantile_profile_ci(cal, fit, alpha, DELTA)


def evt_gpd_batch(calibration: np.ndarray, alphas,
                  threshold_method: str = "percentile",
                  threshold_pct: float = 0.90) -> np.ndarray:
    """Vectorised form. Fits once and evaluates at every alpha, which is both
    faster and more faithful than refitting per level."""
    cal = np.asarray(calibration, dtype=float)
    if threshold_method == "percentile_legacy" or len(cal) < 20:
        return np.array([evt_gpd(cal, a, threshold_method, threshold_pct)
                         for a in alphas])  # safe: uses the snapshot below

    fit = _E.fit_evt(cal, threshold_method=THRESHOLD)
    if fit is None:
        return np.full(len(alphas), float("inf"))

    if MODE == "point":
        return np.array([_E.gpd_quantile(fit, a) for a in alphas])
    if MODE == "delta":
        return np.array([_E.gpd_quantile_delta_ci(fit, a, DELTA)[2]
                         for a in alphas])
    return np.array([_E.gpd_quantile_profile_ci(cal, fit, a, DELTA)
                     for a in alphas])


def evt_gpd_params(calibration: np.ndarray, *args, **kwargs):
    """Fitted (xi, sigma, u, k) under the strengthened baseline."""
    fit = _E.fit_evt(np.asarray(calibration, dtype=float),
                     threshold_method=THRESHOLD)
    if fit is None:
        return None
    return {"xi": fit.xi, "sigma": fit.sigma, "u": fit.u,
            "k": fit.k, "n": fit.n}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    mu, xi, sig = 200.0, 0.2, 50.0
    cal = mu + (sig / xi) * ((1 - rng.uniform(size=2000)) ** (-xi) - 1)
    true_q = mu + (sig / xi) * (0.05 ** (-xi) - 1)
    print(describe())
    print(f"true q(0.95) = {true_q:.2f}")
    for th in ("fixed90", "mbpta_cv", "seq_gof"):
        configure(threshold=th, mode="point")
        print(f"  {th:9s} point   = {evt_gpd(cal, 0.05):8.2f}")
    configure(threshold="mbpta_cv", mode="profile")
    print(f"  mbpta_cv  profile = {evt_gpd(cal, 0.05):8.2f}")
    print(f"  legacy    fixed   = "
          f"{evt_gpd(cal, 0.05, 'percentile_legacy'):8.2f}")
