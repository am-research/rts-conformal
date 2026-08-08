"""
evt_improved.py
---------------
A strengthened EVT-GPD baseline for pWCET estimation, built to address
Reviewer #1's objections that the original EVT baseline was (a) handicapped by
a fixed 90th-percentile threshold and (b) compared as a point estimate against
a conformal *interval*.

Provides:

  Threshold selection
    - fixed_percentile      : the original (retained as an ablation baseline)
    - mbpta_cv              : coefficient-of-variation stability rule
                              (Abella et al., MBPTA-CV, RTSS 2017)
    - sequential_gof        : sequential Anderson-Darling GoF tests over a
                              candidate ladder with the ForwardStop rule
                              (Bader, Yan & Zhang, Ann. Appl. Stat. 2018)

  Quantile estimation with uncertainty
    - gpd_quantile          : point estimate (Eq. 2 of the paper)
    - gpd_quantile_delta_ci : delta-method standard error and one-sided upper
                              confidence limit on the quantile
                              (Smith 1987; Coles 2001, Ch. 4)
    - gpd_quantile_profile_ci : profile-likelihood upper confidence limit,
                              which is better calibrated than the delta method
                              in the tail

All functions operate on a 1-D array of execution-time measurements.
Times are in microseconds unless otherwise noted.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats, optimize


# =============================================================================
# GPD log-likelihood and MLE
# =============================================================================

def gpd_nll(params: np.ndarray, excesses: np.ndarray) -> float:
    """Negative log-likelihood of the GPD for a vector of positive excesses.

    Parameterisation: G(y) = 1 - (1 + xi*y/sigma)^(-1/xi), sigma > 0.
    """
    xi, sigma = params
    if sigma <= 0:
        return np.inf
    z = 1.0 + xi * excesses / sigma
    if xi < 0:
        # Support is bounded above by sigma / |xi|
        if np.any(z <= 0):
            return np.inf
    else:
        if np.any(z <= 0):
            return np.inf
    if abs(xi) < 1e-10:
        return len(excesses) * np.log(sigma) + np.sum(excesses) / sigma
    return (len(excesses) * np.log(sigma)
            + (1.0 + 1.0 / xi) * np.sum(np.log(z)))


def fit_gpd(excesses: np.ndarray) -> Optional[Tuple[float, float]]:
    """MLE fit of (xi, sigma). Returns None if the fit fails.

    scipy's genpareto.fit is used as the primary route with a Nelder-Mead
    fallback, because scipy occasionally fails to converge on small samples of
    heavy-tailed excesses.
    """
    excesses = np.asarray(excesses, dtype=float)
    excesses = excesses[excesses > 0]
    if len(excesses) < 10:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            xi, _loc, sigma = stats.genpareto.fit(excesses, floc=0)
            if sigma > 0 and np.isfinite(xi) and np.isfinite(sigma):
                # Sanity check against the direct likelihood
                if np.isfinite(gpd_nll(np.array([xi, sigma]), excesses)):
                    return float(xi), float(sigma)
        except Exception:
            pass

        # Fallback: method-of-moments start + Nelder-Mead
        m, v = np.mean(excesses), np.var(excesses)
        xi0 = 0.5 * (1.0 - m ** 2 / v) if v > 0 else 0.1
        xi0 = float(np.clip(xi0, -0.45, 0.45))
        sig0 = max(m * (1.0 - xi0), 1e-8)
        try:
            res = optimize.minimize(gpd_nll, x0=np.array([xi0, sig0]),
                                    args=(excesses,), method="Nelder-Mead",
                                    options={"maxiter": 2000, "xatol": 1e-8,
                                             "fatol": 1e-8})
            if res.success and res.x[1] > 0:
                return float(res.x[0]), float(res.x[1])
        except Exception:
            pass
    return None


# =============================================================================
# Threshold selection
# =============================================================================

def threshold_fixed_percentile(data: np.ndarray, pct: float = 0.90) -> float:
    """The original fixed-percentile rule, retained as an ablation baseline."""
    return float(np.quantile(data, pct))


def threshold_mbpta_cv(data: np.ndarray,
                       pct_lo: float = 0.50,
                       pct_hi: float = 0.99,
                       n_grid: int = 40,
                       min_exceedances: int = 25,
                       z: float = 1.96,
                       stability_window: int = 5) -> float:
    """Coefficient-of-variation threshold selection (MBPTA-CV, Abella 2017).

    For excesses that are genuinely GPD(xi, sigma) with xi < 1/2, the
    coefficient of variation of the excesses is

        CV = sd / mean = 1 / sqrt(1 - 2*xi),

    which does *not* depend on the threshold. The rule therefore scans a ladder
    of candidate thresholds and returns the lowest one from which the CV curve
    stays inside the confidence band of its own value, i.e. the point at which
    CV stops drifting and becomes flat.

    The band uses the asymptotic standard error of the CV of an i.i.d. sample,

        se(CV) ~= CV * sqrt((1 + 2*CV^2) / (4*k)),

    which is the standard delta-method expression and avoids the cost of a
    bootstrap inside a Monte-Carlo loop.

    Falls back to the 90th percentile if no stable region is found.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    pcts = np.linspace(pct_lo, pct_hi, n_grid)
    cands = np.quantile(data, pcts)

    us, cvs, ses = [], [], []
    for u in cands:
        exc = data[data > u] - u
        k = len(exc)
        if k < min_exceedances:
            continue
        m = np.mean(exc)
        if m <= 0:
            continue
        cv = np.std(exc, ddof=1) / m
        us.append(u)
        cvs.append(cv)
        ses.append(cv * np.sqrt((1.0 + 2.0 * cv ** 2) / (4.0 * k)))

    if len(us) < stability_window + 1:
        return threshold_fixed_percentile(data, 0.90)

    us = np.asarray(us)
    cvs = np.asarray(cvs)
    ses = np.asarray(ses)

    # Lowest threshold whose band contains every CV value in the window above it
    for i in range(len(us) - stability_window):
        lo = cvs[i] - z * ses[i]
        hi = cvs[i] + z * ses[i]
        window = cvs[i + 1: i + 1 + stability_window]
        if np.all((window >= lo) & (window <= hi)):
            return float(us[i])

    return threshold_fixed_percentile(data, 0.90)


# --- Anderson-Darling goodness-of-fit for the GPD -----------------------------

def _ad_statistic_gpd(excesses: np.ndarray, xi: float, sigma: float) -> float:
    """Anderson-Darling A^2 statistic for a GPD fit."""
    k = len(excesses)
    y = np.sort(excesses)
    if abs(xi) < 1e-10:
        z = 1.0 - np.exp(-y / sigma)
    else:
        base = 1.0 + xi * y / sigma
        if np.any(base <= 0):
            return np.inf
        z = 1.0 - base ** (-1.0 / xi)
    eps = 1e-12
    z = np.clip(z, eps, 1.0 - eps)
    i = np.arange(1, k + 1)
    a2 = -k - np.sum((2 * i - 1) / k * (np.log(z) + np.log(1.0 - z[::-1])))
    return float(a2)


class ADNullTable:
    """Cached null distribution of the GPD Anderson-Darling statistic.

    The A^2 statistic with *estimated* parameters does not have a distribution-
    free null, so its critical values depend on xi and (weakly) on the number of
    exceedances k. Rather than relying on published tables, this class simulates
    the null once over a grid of (xi, k) and interpolates, which keeps the
    experiment self-contained and reproducible.
    """

    def __init__(self,
                 xi_grid=(-0.4, -0.2, 0.0, 0.2, 0.4),
                 k_grid=(25, 50, 100, 200, 400),
                 n_sim: int = 600,
                 seed: int = 20240,
                 quantile_grid=None):
        self.xi_grid = np.asarray(xi_grid, dtype=float)
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.q_grid = (np.asarray(quantile_grid, dtype=float)
                       if quantile_grid is not None
                       else np.linspace(0.005, 0.995, 199))
        rng = np.random.default_rng(seed)
        # table[i, j, :] = quantiles of A^2 under GPD(xi_i) with k_j exceedances
        self.table = np.zeros((len(self.xi_grid), len(self.k_grid),
                               len(self.q_grid)))
        for i, xi in enumerate(self.xi_grid):
            for j, k in enumerate(self.k_grid):
                stats_sim = np.empty(n_sim)
                stats_sim[:] = np.nan
                for s in range(n_sim):
                    u = rng.uniform(size=int(k))
                    if abs(xi) < 1e-10:
                        y = -np.log(1.0 - u)
                    else:
                        y = ((1.0 - u) ** (-xi) - 1.0) / xi
                    fit = fit_gpd(y)
                    if fit is None:
                        continue
                    stats_sim[s] = _ad_statistic_gpd(y, fit[0], fit[1])
                good = stats_sim[np.isfinite(stats_sim)]
                if len(good) < 20:
                    good = np.array([0.5, 1.0, 2.0])
                self.table[i, j, :] = np.quantile(good, self.q_grid)

    def pvalue(self, a2: float, xi: float, k: int) -> float:
        """Approximate p-value by bilinear interpolation over (xi, k)."""
        if not np.isfinite(a2):
            return 0.0
        xi_c = float(np.clip(xi, self.xi_grid[0], self.xi_grid[-1]))
        k_c = float(np.clip(k, self.k_grid[0], self.k_grid[-1]))

        i = int(np.searchsorted(self.xi_grid, xi_c, side="right") - 1)
        i = int(np.clip(i, 0, len(self.xi_grid) - 2))
        j = int(np.searchsorted(self.k_grid, k_c, side="right") - 1)
        j = int(np.clip(j, 0, len(self.k_grid) - 2))

        wx = ((xi_c - self.xi_grid[i])
              / (self.xi_grid[i + 1] - self.xi_grid[i]))
        wk = ((k_c - self.k_grid[j]) / (self.k_grid[j + 1] - self.k_grid[j]))

        curve = ((1 - wx) * (1 - wk) * self.table[i, j]
                 + wx * (1 - wk) * self.table[i + 1, j]
                 + (1 - wx) * wk * self.table[i, j + 1]
                 + wx * wk * self.table[i + 1, j + 1])

        # p = P(A2_null >= a2) = 1 - F(a2)
        frac = float(np.interp(a2, curve, self.q_grid, left=0.0, right=1.0))
        return float(np.clip(1.0 - frac, 1e-6, 1.0))


_AD_TABLE: Optional[ADNullTable] = None
_AD_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "ad_null_table.npz")


def get_ad_table() -> ADNullTable:
    """Lazily build and cache the module-level AD null table.

    The table is expensive to simulate (~15k GPD fits) but identical across
    runs, so it is memoised in-process and persisted to disk next to this file.
    """
    global _AD_TABLE
    if _AD_TABLE is not None:
        return _AD_TABLE

    if os.path.exists(_AD_CACHE_PATH):
        try:
            z = np.load(_AD_CACHE_PATH)
            t = ADNullTable.__new__(ADNullTable)
            t.xi_grid = z["xi_grid"]
            t.k_grid = z["k_grid"]
            t.q_grid = z["q_grid"]
            t.table = z["table"]
            _AD_TABLE = t
            return _AD_TABLE
        except Exception:
            pass

    _AD_TABLE = ADNullTable()
    try:
        np.savez_compressed(_AD_CACHE_PATH,
                            xi_grid=_AD_TABLE.xi_grid,
                            k_grid=_AD_TABLE.k_grid,
                            q_grid=_AD_TABLE.q_grid,
                            table=_AD_TABLE.table)
    except Exception:
        pass
    return _AD_TABLE


def forward_stop(pvalues: np.ndarray, gamma: float = 0.10) -> int:
    """ForwardStop rule of G'Sell et al. (2016), as used by Bader et al. (2018).

    Returns the index of the chosen threshold in the candidate ladder, i.e. the
    smallest threshold for which the GPD fit is not rejected in an ordered,
    FDR-controlled sense. Returns -1 if every candidate is rejected.
    """
    p = np.clip(np.asarray(pvalues, dtype=float), 1e-12, 1.0 - 1e-12)
    if len(p) == 0:
        return -1
    cum = np.cumsum(-np.log(1.0 - p)) / np.arange(1, len(p) + 1)
    ok = np.where(cum <= gamma)[0]
    if len(ok) == 0:
        return 0  # no evidence to reject the very first candidate ordering
    khat = int(ok[-1])
    return min(khat + 1, len(p) - 1)


def threshold_sequential_gof(data: np.ndarray,
                             pct_lo: float = 0.50,
                             pct_hi: float = 0.98,
                             n_grid: int = 20,
                             min_exceedances: int = 25,
                             gamma: float = 0.10) -> float:
    """Automated threshold selection by sequential AD goodness-of-fit testing.

    Tests GPD adequacy at an increasing ladder of candidate thresholds and
    applies ForwardStop to pick the lowest threshold at which the GPD model is
    tenable. This is the "stronger statistical proof" alternative to visual
    mean-excess / Hill inspection.
    """
    data = np.asarray(data, dtype=float)
    pcts = np.linspace(pct_lo, pct_hi, n_grid)
    cands = np.quantile(data, pcts)

    table = get_ad_table()
    us, pvals = [], []
    for u in cands:
        exc = data[data > u] - u
        if len(exc) < min_exceedances:
            continue
        fit = fit_gpd(exc)
        if fit is None:
            continue
        xi, sigma = fit
        a2 = _ad_statistic_gpd(exc, xi, sigma)
        us.append(u)
        pvals.append(table.pvalue(a2, xi, len(exc)))

    if len(us) == 0:
        return threshold_fixed_percentile(data, 0.90)

    idx = forward_stop(np.asarray(pvals), gamma=gamma)
    if idx < 0:
        idx = len(us) - 1
    return float(us[idx])


THRESHOLD_METHODS = {
    "fixed90": lambda d: threshold_fixed_percentile(d, 0.90),
    "mbpta_cv": threshold_mbpta_cv,
    "seq_gof": threshold_sequential_gof,
}


# =============================================================================
# Quantile estimation with uncertainty
# =============================================================================

@dataclass
class EVTFit:
    u: float
    xi: float
    sigma: float
    k: int
    n: int

    @property
    def zeta_u(self) -> float:
        """Empirical exceedance rate P(T > u)."""
        return self.k / self.n


def fit_evt(data: np.ndarray, threshold_method: str = "mbpta_cv") -> Optional[EVTFit]:
    """Select a threshold, fit the GPD above it, and return the fit object."""
    data = np.asarray(data, dtype=float)
    sel = THRESHOLD_METHODS.get(threshold_method)
    if sel is None:
        raise ValueError(f"unknown threshold method: {threshold_method}")
    u = sel(data)
    exc = data[data > u] - u
    fit = fit_gpd(exc)
    if fit is None:
        return None
    return EVTFit(u=float(u), xi=fit[0], sigma=fit[1],
                  k=int(len(exc)), n=int(len(data)))


def gpd_quantile(fit: EVTFit, alpha: float) -> float:
    """Point estimate of the (1-alpha) quantile from a PoT fit."""
    y = fit.zeta_u / alpha
    if y <= 0:
        return np.inf
    if abs(fit.xi) < 1e-8:
        return fit.u + fit.sigma * np.log(y)
    try:
        return float(fit.u + (fit.sigma / fit.xi) * (y ** fit.xi - 1.0))
    except (OverflowError, FloatingPointError):
        return np.inf


def _quantile_gradient(fit: EVTFit, alpha: float) -> np.ndarray:
    """Gradient of the quantile w.r.t. (zeta_u, sigma, xi)."""
    xi, sigma, zeta = fit.xi, fit.sigma, fit.zeta_u
    y = zeta / alpha
    if abs(xi) < 1e-8:
        d_zeta = sigma / zeta
        d_sigma = np.log(y)
        d_xi = 0.5 * sigma * np.log(y) ** 2
    else:
        d_zeta = sigma * (y ** (xi - 1.0)) / alpha
        d_sigma = (y ** xi - 1.0) / xi
        d_xi = (-sigma / xi ** 2) * (y ** xi - 1.0) + (sigma / xi) * (y ** xi) * np.log(y)
    return np.array([d_zeta, d_sigma, d_xi], dtype=float)


def _gpd_param_cov(fit: EVTFit) -> np.ndarray:
    """Asymptotic covariance of (sigma_hat, xi_hat), Smith (1987) / Coles (2001).

        V = (1/k) * [[2*sigma^2*(1+xi),  -sigma*(1+xi)],
                     [-sigma*(1+xi),      (1+xi)^2   ]]

    Valid for xi > -1/2 (the regular case).
    """
    xi, sigma, k = fit.xi, fit.sigma, fit.k
    c = 1.0 + xi
    return (1.0 / k) * np.array([[2.0 * sigma ** 2 * c, -sigma * c],
                                 [-sigma * c, c ** 2]], dtype=float)


def gpd_quantile_delta_ci(fit: EVTFit, alpha: float,
                          delta: float = 0.05) -> Tuple[float, float, float]:
    """Delta-method quantile estimate, standard error, and upper confidence limit.

    Returns (point_estimate, standard_error, upper_confidence_limit) where the
    UCL is the one-sided (1 - delta) limit, i.e. point + z_{1-delta} * se.

    The full covariance includes the binomial uncertainty in the exceedance rate
    zeta_u, treated as independent of (sigma_hat, xi_hat) in the standard
    treatment (Coles 2001, Sec. 4.3.3).
    """
    q = gpd_quantile(fit, alpha)
    if not np.isfinite(q):
        return q, np.inf, np.inf
    if fit.xi <= -0.5:
        # MLE is not asymptotically normal in this regime
        return q, np.nan, np.inf

    grad = _quantile_gradient(fit, alpha)  # order: (zeta, sigma, xi)
    v_par = _gpd_param_cov(fit)            # order: (sigma, xi)
    v_zeta = fit.zeta_u * (1.0 - fit.zeta_u) / fit.n

    cov = np.zeros((3, 3))
    cov[0, 0] = v_zeta
    cov[1:, 1:] = v_par

    var = float(grad @ cov @ grad)
    if not np.isfinite(var) or var < 0:
        return q, np.inf, np.inf
    se = float(np.sqrt(var))
    z = float(stats.norm.ppf(1.0 - delta))
    return q, se, float(q + z * se)

def _profile_nll_pinned(xq: float, fit: EVTFit, exc, alpha: float) -> float:
    """Minimised negative log-likelihood with the target quantile pinned at xq.

    The likelihood is the product of a binomial term for the exceedance rate
    zeta_u and the GPD likelihood of the excesses:

        -logL(zeta, xi, sigma) = -[k log zeta + (n-k) log(1-zeta)]
                                 + gpd_nll(xi, sigma)

    Pinning x_alpha = u + (sigma/xi)[(zeta/alpha)^xi - 1] lets sigma be solved
    out, leaving a two-parameter profile over (xi, zeta). The binomial term must
    be retained: the quantile depends on zeta_u as well as on (sigma, xi), and
    profiling only over the GPD parameters would treat the exceedance rate as
    known and understate the interval width. The delta method accounts for that
    term, so omitting it here would make the two EVT intervals non-comparable.
    """
    k, n, u = fit.k, fit.n, fit.u

    def obj(p):
        xi, zeta = float(p[0]), float(p[1])
        if not (1e-9 < zeta < 1.0 - 1e-9) or not (-0.49 < xi < 0.95):
            return np.inf
        y = zeta / alpha
        if y <= 0:
            return np.inf
        if abs(xi) < 1e-8:
            if y <= 1.0:
                return np.inf
            sigma = (xq - u) / np.log(y)
        else:
            denom = y ** xi - 1.0
            if abs(denom) < 1e-12:
                return np.inf
            sigma = xi * (xq - u) / denom
        if sigma <= 0 or not np.isfinite(sigma):
            return np.inf
        binom = -(k * np.log(zeta) + (n - k) * np.log(1.0 - zeta))
        val = binom + gpd_nll(np.array([xi, sigma]), exc)
        return val if np.isfinite(val) else np.inf

    best = np.inf
    zeta_hat = fit.zeta_u
    starts = [(fit.xi, 1.0), (max(fit.xi - 0.25, -0.45), 0.9),
              (min(fit.xi + 0.25, 0.9), 1.1)]
    for xi0, zf in starts:
        z0 = float(np.clip(zeta_hat * zf, 1e-6, 0.999))
        if not np.isfinite(obj(np.array([xi0, z0]))):
            continue
        try:
            r = optimize.minimize(obj, x0=np.array([xi0, z0]),
                                  method="Nelder-Mead",
                                  options={"maxiter": 250, "xatol": 1e-6,
                                           "fatol": 1e-8})
            if np.isfinite(r.fun):
                best = min(best, float(r.fun))
        except Exception:
            continue
    return best


def gpd_quantile_profile_ci(data: np.ndarray, fit: EVTFit, alpha: float,
                            delta: float = 0.05,
                            max_expand: float = 40.0,
                            tol_rel: float = 1e-3) -> float:
    """One-sided profile-likelihood upper confidence limit on the quantile.

    Inverts the likelihood-ratio statistic in the target quantile. The profile
    deviance is monotone increasing above the MLE, so the crossing point is
    located by expansion followed by bisection rather than a grid scan.

    Returns +inf if the deviance never reaches the critical value within the
    search range, i.e. the data place no usable upper constraint on the tail.
    """
    exc = np.asarray(data)[np.asarray(data) > fit.u] - fit.u
    if len(exc) < 10:
        return np.inf

    q_hat, se, _ = gpd_quantile_delta_ci(fit, alpha, delta)
    if not np.isfinite(q_hat) or not np.isfinite(se) or se <= 0:
        return np.inf

    nll_hat = _profile_nll_pinned(q_hat, fit, exc, alpha)
    if not np.isfinite(nll_hat):
        return np.inf

    # One-sided (1-delta) limit corresponds to the upper end of the two-sided
    # (1-2*delta) likelihood-ratio interval.
    crit = 0.5 * stats.chi2.ppf(1.0 - 2.0 * delta, df=1)

    def dev(xq: float) -> float:
        return _profile_nll_pinned(xq, fit, exc, alpha) - nll_hat

    # Expand until the deviance exceeds the critical value
    lo, hi = q_hat, q_hat + 2.0 * se
    steps = 0
    while dev(hi) <= crit:
        hi = q_hat + (hi - q_hat) * 2.0
        steps += 1
        if (hi - q_hat) > max_expand * se or steps > 12:
            return np.inf

    # Bisection on the monotone deviance
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if dev(mid) > crit:
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= tol_rel * max(abs(q_hat), 1e-9):
            break
    return float(hi)


# =============================================================================
# Convenience wrappers matching the interface of cp_methods.evt_gpd
# =============================================================================

def evt_bound(data: np.ndarray, alpha: float,
              threshold_method: str = "mbpta_cv",
              mode: str = "point",
              delta: float = 0.05) -> float:
    """Unified EVT bound.

    mode='point' : the MLE point estimate of the (1-alpha) quantile.
    mode='delta' : the one-sided (1-delta) delta-method upper confidence limit.
    mode='profile': the one-sided (1-delta) profile-likelihood upper limit.
    """
    fit = fit_evt(data, threshold_method=threshold_method)
    if fit is None:
        return np.inf
    if mode == "point":
        return gpd_quantile(fit, alpha)
    if mode == "delta":
        return gpd_quantile_delta_ci(fit, alpha, delta)[2]
    if mode == "profile":
        return gpd_quantile_profile_ci(data, fit, alpha, delta)
    raise ValueError(f"unknown mode: {mode}")
