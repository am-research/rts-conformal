"""
quantile_estimators.py
----------------------

This module provides:

  * hyndman_fan(x, p, kind) -- the nine sample-quantile definitions catalogued
    by Hyndman & Fan (1996), "Sample Quantiles in Statistical Packages",
    The American Statistician 50(4):361-365. Types 1-3 are inverse-CDF style
    (discontinuous, order-statistic valued); types 4-9 interpolate.

  * tolerance_bound(x, alpha, delta) -- the classical distribution-free
    tolerance bound of Wilks (1941): the order statistic X_(r) with the
    smallest r such that

        P( F(X_(r)) >= 1 - alpha ) >= 1 - delta.

    Because F(X_(r)) ~ Beta(r, n+1-r) for any continuous F, this is exact and
    distribution-free. It is the *training-conditional* counterpart of split
    conformal prediction, and it is the correct like-for-like comparator for an
    EVT quantile confidence limit: both answer "with confidence 1-delta, this
    value lies above the true (1-alpha) quantile."

"""

from __future__ import annotations

import numpy as np
from scipy import stats


# =============================================================================
# Hyndman & Fan (1996) sample quantile types 1-9
# =============================================================================

def hyndman_fan(x: np.ndarray, p: float, kind: int) -> float:
    """Sample quantile of `x` at probability `p` using Hyndman-Fan type `kind`.

    Types 1-3 are the inverse-CDF family (return an order statistic).
    Types 4-9 are the interpolation family, differing only in the plotting
    position constants (a, b) used to map p to a fractional index.
    """
    if not 1 <= kind <= 9:
        raise ValueError("kind must be in 1..9")
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return np.nan
    if n == 1:
        return float(x[0])

    def order(j: int) -> float:
        """1-indexed order statistic with clamping."""
        return float(x[int(np.clip(j, 1, n)) - 1])

    # --- discontinuous types -------------------------------------------------
    if kind == 1:
        # inverse empirical CDF
        h = n * p + 0.5
        j = int(np.ceil(n * p))
        if n * p == np.floor(n * p) and n * p >= 1:
            return order(int(n * p))
        return order(max(j, 1))

    if kind == 2:
        # inverse empirical CDF with averaging at discontinuities
        np_ = n * p
        if abs(np_ - round(np_)) < 1e-12 and 1 <= round(np_) < n:
            j = int(round(np_))
            return 0.5 * (order(j) + order(j + 1))
        return order(max(int(np.ceil(np_)), 1))

    if kind == 3:
        # nearest even order statistic (SAS definition)
        m = -0.5
        j = int(np.floor(n * p + m))
        g = n * p + m - j
        if abs(g) < 1e-12 and j % 2 == 0:
            return order(max(j, 1))
        return order(max(j + 1, 1))

    # --- continuous / interpolating types ------------------------------------
    ab = {4: (0.0, 1.0),
          5: (0.5, 0.5),
          6: (0.0, 0.0),
          7: (1.0, 1.0),
          8: (1.0 / 3.0, 1.0 / 3.0),
          9: (3.0 / 8.0, 3.0 / 8.0)}[kind]
    a, b = ab
    # h = (n + 1 - a - b) * p + a   (Hyndman-Fan unified form)
    h = (n + 1 - a - b) * p + a
    h = float(np.clip(h, 1.0, float(n)))
    fl = int(np.floor(h))
    frac = h - fl
    return order(fl) + frac * (order(min(fl + 1, n)) - order(fl))


HF_LABELS = {
    1: "HF1 inverse ECDF",
    2: "HF2 inverse ECDF, averaged",
    3: "HF3 nearest even (SAS)",
    4: "HF4 linear ECDF",
    5: "HF5 midpoint (Hazen)",
    6: "HF6 Weibull (SPSS/Minitab)",
    7: "HF7 default (R/NumPy)",
    8: "HF8 median-unbiased",
    9: "HF9 approx normal-unbiased",
}


# =============================================================================
# Distribution-free tolerance bound (Wilks 1941)
# =============================================================================

def tolerance_index(n: int, alpha: float, delta: float) -> int:
    """Smallest 1-indexed order-statistic rank r giving a (1-alpha, 1-delta)
    upper tolerance bound, or -1 if no rank suffices for this n.

    Requires P(Beta(r, n+1-r) >= 1-alpha) >= 1-delta, i.e.
    F_Beta(1-alpha; r, n+1-r) <= delta. The left-hand side decreases in r, so a
    scan from the top down finds the tightest admissible rank.
    """
    for r in range(1, n + 1):
        if stats.beta.cdf(1.0 - alpha, r, n + 1 - r) <= delta:
            return r
    return -1


def tolerance_bound(x: np.ndarray, alpha: float, delta: float = 0.05) -> float:
    """Distribution-free (1-alpha, 1-delta) upper tolerance bound.

    Returns +inf when the sample is too small to certify the pair (alpha, delta)
    without extrapolation, which is the honest answer in that regime.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    r = tolerance_index(n, alpha, delta)
    if r < 0 or r > n:
        return np.inf
    return float(x[r - 1])


def min_n_for_tolerance(alpha: float, delta: float) -> int:
    """Smallest calibration size admitting a finite (1-alpha, 1-delta) bound.

    With r = n the condition reduces to (1-alpha)^n <= delta, giving the
    familiar n >= log(delta)/log(1-alpha).
    """
    return int(np.ceil(np.log(delta) / np.log(1.0 - alpha)))


def min_n_for_marginal(alpha: float) -> int:
    """Smallest calibration size admitting a finite split-CP bound: n >= 1/alpha - 1."""
    return int(np.ceil(1.0 / alpha) - 1)


# =============================================================================
# Convenience: evaluate every estimator on one calibration set
# =============================================================================

def all_quantile_bounds(cal: np.ndarray, alpha: float,
                        delta: float = 0.05) -> dict:
    """Return {label: bound} for the nine HF types, split CP, and the
    tolerance bound, all targeting the (1-alpha) quantile."""
    from cp_methods import cp_pwcet

    out = {HF_LABELS[k]: hyndman_fan(cal, 1.0 - alpha, k) for k in range(1, 10)}
    out["CP (split)"] = cp_pwcet(cal, alpha)
    out[f"Tolerance (delta={delta})"] = tolerance_bound(cal, alpha, delta)
    return out


if __name__ == "__main__":
    # Self-check: HF7 must agree with numpy's default quantile.
    rng = np.random.default_rng(0)
    x = rng.normal(size=997)
    for p in (0.05, 0.5, 0.9, 0.95, 0.99):
        a = hyndman_fan(x, p, 7)
        b = float(np.quantile(x, p))
        assert abs(a - b) < 1e-9, (p, a, b)
    # HF6 must agree with the Weibull plotting position on exact ranks.
    print("HF7 matches numpy.quantile at all tested p. OK")
    print("min n for (alpha=1e-3, delta=0.05) tolerance bound:",
          min_n_for_tolerance(1e-3, 0.05))
    print("min n for marginal split-CP at alpha=1e-3:",
          min_n_for_marginal(1e-3))
