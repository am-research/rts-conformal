"""
cp_methods.py
-------------
Core implementations of:
  - Split conformal prediction (CP-pWCET)
  - Weighted conformal prediction
  - Conformalized quantile regression (CQR)
  - Adaptive conformal inference (ACI)
  - EVT-GPD baseline
  - Empirical quantile baseline

All methods return an upper bound C such that P(T > C) <= alpha.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
import warnings


# =============================================================================
# Split Conformal Prediction (CP-pWCET)
# =============================================================================

def cp_pwcet(calibration: np.ndarray, alpha: float) -> float:
    """
    Split conformal prediction upper bound on execution time.

    Parameters
    ----------
    calibration : array of shape (n,)
        Execution time measurements used for calibration.
    alpha : float
        Target miscoverage level. The bound satisfies P(T > C) <= alpha.

    Returns
    -------
    float
        The (1-alpha)-CP bound. Returns +inf if n < ceil(1/alpha) - 1,
        meaning the calibration set is too small for a non-trivial bound
        at this alpha level.

    Notes
    -----
    Implements Eq. (3) of the paper. Coverage guarantee (Theorem 1):
        1 - alpha <= P(T_{n+1} <= C) <= 1 - alpha + 1/(n+1)
    """
    n = len(calibration)
    level = np.ceil((n + 1) * (1 - alpha)) / n

    if level > 1.0:
        return np.inf

    # The (ceil((n+1)(1-alpha)))-th order statistic of calibration + {+inf}
    # Equivalent to the ceil((n+1)(1-alpha))/n quantile of the empirical
    # distribution, clipped at the maximum observed value for alpha > 1/(n+1).
    sorted_cal = np.sort(calibration)
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1  # 0-indexed

    if idx >= n:
        return np.inf
    return float(sorted_cal[idx])


def cp_pwcet_batch(calibration: np.ndarray, alphas) -> np.ndarray:
    """Vectorised version: returns bounds for multiple alpha levels."""
    return np.array([cp_pwcet(calibration, a) for a in alphas])


# =============================================================================
# Empirical Quantile (naive baseline, no finite-sample correction)
# =============================================================================

def empirical_quantile(calibration: np.ndarray, alpha: float) -> float:
    """
    Naive (1-alpha) empirical quantile. No finite-sample guarantee.
    Included as a baseline to show the effect of the CP correction.
    """
    return float(np.quantile(calibration, 1 - alpha))


# =============================================================================
# EVT-GPD (Peak over Threshold)
# =============================================================================

def _select_threshold_percentile(data: np.ndarray, pct: float = 0.90) -> float:
    """Select threshold as the pct-th percentile of the data."""
    return float(np.quantile(data, pct))


def _select_threshold_mean_excess(data: np.ndarray,
                                   pct_range=(0.80, 0.98),
                                   n_grid: int = 50) -> float:
    """
    Select threshold by the mean excess plot heuristic:
    find the region where the mean excess function is approximately linear,
    choosing the lowest threshold in that region.

    Falls back to the 90th percentile if the heuristic fails.
    """
    sorted_data = np.sort(data)
    pcts = np.linspace(pct_range[0], pct_range[1], n_grid)
    thresholds = np.quantile(data, pcts)
    mean_excesses = np.array([
        np.mean(sorted_data[sorted_data > u] - u) for u in thresholds
    ])

    # Fit a line to mean excess vs threshold; choose the threshold where
    # linearity is best (minimum residual in a sliding window).
    best_u = thresholds[len(thresholds) // 2]
    best_r2 = -np.inf
    window = max(5, n_grid // 5)
    for i in range(n_grid - window):
        x = thresholds[i:i+window]
        y = mean_excesses[i:i+window]
        if np.std(x) < 1e-10:
            continue
        slope, intercept, r, *_ = stats.linregress(x, y)
        if r**2 > best_r2:
            best_r2 = r**2
            best_u = thresholds[i]

    return float(best_u)


def evt_gpd(calibration: np.ndarray, alpha: float,
            threshold_method: str = 'percentile',
            threshold_pct: float = 0.90) -> float:
    """
    EVT-GPD pWCET estimator (Peak over Threshold).

    Fits a Generalised Pareto Distribution to threshold exceedances via MLE
    and returns the (1-alpha) quantile estimate.

    Parameters
    ----------
    calibration : array of shape (n,)
    alpha : float
        Target miscoverage level.
    threshold_method : 'percentile' or 'mean_excess'
    threshold_pct : float
        Percentile for threshold selection (used if threshold_method='percentile').

    Returns
    -------
    float
        EVT-GPD upper bound on execution time at level 1-alpha.
    """
    n = len(calibration)
    if threshold_method == 'percentile':
        u = _select_threshold_percentile(calibration, threshold_pct)
    else:
        u = _select_threshold_mean_excess(calibration)

    exceedances = calibration[calibration > u] - u
    k = len(exceedances)

    if k < 10:
        warnings.warn(f"Only {k} exceedances above threshold {u:.2f}. "
                      "EVT estimate will be unreliable.")
        return np.inf

    # Fit GPD via MLE using scipy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # scipy's genpareto: loc=0, scale=sigma, c=xi (shape)
            xi, loc, sigma = stats.genpareto.fit(exceedances, floc=0)
        except Exception:
            return np.inf

    # Recover tail quantile (Eq. 2 in paper)
    # P(T > t) = (k/n) * (1 + xi*(t-u)/sigma)^{-1/xi}  for xi != 0
    # Set this equal to alpha and solve for t:
    # t = u + sigma/xi * ((n/k * 1/alpha)^xi - 1)
    zeta_u = k / n  # P(T > u) estimated empirically

    if abs(xi) < 1e-8:
        # Exponential tail (xi -> 0 limit)
        bound = u + sigma * np.log(zeta_u / alpha)
    else:
        try:
            bound = u + (sigma / xi) * ((zeta_u / alpha) ** xi - 1)
        except (OverflowError, FloatingPointError):
            bound = np.inf

    return float(bound)


def evt_gpd_batch(calibration: np.ndarray, alphas,
                  threshold_method: str = 'percentile',
                  threshold_pct: float = 0.90) -> np.ndarray:
    """Vectorised EVT-GPD over multiple alpha levels."""
    return np.array([
        evt_gpd(calibration, a, threshold_method, threshold_pct) for a in alphas
    ])


def evt_gpd_params(calibration: np.ndarray,
                   threshold_pct: float = 0.90):
    """
    Return the fitted GPD parameters (xi, sigma) and number of exceedances.
    Useful for diagnostics.
    """
    u = _select_threshold_percentile(calibration, threshold_pct)
    exceedances = calibration[calibration > u] - u
    k = len(exceedances)
    if k < 10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            xi, _, sigma = stats.genpareto.fit(exceedances, floc=0)
            return {'xi': xi, 'sigma': sigma, 'u': u, 'k': k,
                    'n': len(calibration)}
        except Exception:
            return None


# =============================================================================
# Weighted Conformal Prediction (for covariate shift)
# =============================================================================

def weighted_cp(calibration_scores: np.ndarray,
                calibration_weights: np.ndarray,
                test_weight: float,
                alpha: float) -> float:
    """
    Weighted conformal prediction for covariate shift.

    Parameters
    ----------
    calibration_scores : array (n,)
        Nonconformity scores on calibration set (e.g. execution times T_i).
    calibration_weights : array (n,)
        Importance weights w_i = dQ/dP(T_i) for each calibration point.
    test_weight : float
        Weight for the test point: w_{n+1} = dQ/dP(T_{n+1}).
        In practice, estimated from deployment context covariates.
    alpha : float

    Returns
    -------
    float
        Weighted CP upper bound.
    """
    n = len(calibration_scores)
    all_weights = np.append(calibration_weights, test_weight)
    normalised = all_weights / all_weights.sum()

    # Sort scores and compute weighted CDF
    sort_idx = np.argsort(calibration_scores)
    sorted_scores = np.append(calibration_scores[sort_idx], np.inf)
    # Weights for sorted calibration points + infinity
    sorted_weights = np.append(normalised[sort_idx], normalised[-1])
    # Actually need weights for calibration points only, then +inf gets test weight
    cal_weights_sorted = calibration_weights[sort_idx]
    cal_weights_normalised = np.append(
        cal_weights_sorted / all_weights.sum(),
        test_weight / all_weights.sum()
    )

    # Find smallest score where cumulative weighted mass >= 1-alpha
    cum_mass = np.cumsum(cal_weights_normalised)
    idx = np.searchsorted(cum_mass, 1 - alpha)
    if idx >= len(sorted_scores):
        return np.inf
    return float(sorted_scores[min(idx, len(calibration_scores) - 1)])


# =============================================================================
# Conformalized Quantile Regression (CQR)
# =============================================================================

def cqr_calibrate(calibration_times: np.ndarray,
                  calibration_lower_preds: np.ndarray,
                  calibration_upper_preds: np.ndarray,
                  alpha: float) -> float:
    """
    Calibrate CQR: compute the conformity correction term q_hat.

    The nonconformity score for CQR (Romano et al., 2019) is:
        s_i = max(q_lo(x_i) - T_i, T_i - q_hi(x_i))
    For one-sided upper bounds (WCET context), we use:
        s_i = T_i - q_{1-alpha}(x_i)

    Parameters
    ----------
    calibration_times : array (n,)
    calibration_lower_preds : array (n,)  -- q_alpha/2(x_i) predictions
    calibration_upper_preds : array (n,)  -- q_{1-alpha/2}(x_i) predictions
        For one-sided use, pass the (1-alpha) quantile predictions as upper.
    alpha : float

    Returns
    -------
    float
        The correction q_hat to add to the base quantile predictor.
    """
    # One-sided nonconformity score for upper bound
    scores = calibration_times - calibration_upper_preds
    return cp_pwcet(scores, alpha)  # Same CP formula applied to residuals


def cqr_predict(x_test_upper_pred: np.ndarray,
                correction: float) -> np.ndarray:
    """
    Apply CQR correction to get upper bounds.

    Parameters
    ----------
    x_test_upper_pred : array (m,)
        Base quantile regressor predictions at test points.
    correction : float
        The calibrated correction from cqr_calibrate.

    Returns
    -------
    array (m,)
        Input-conditional upper bounds on execution time.
    """
    return x_test_upper_pred + correction


# =============================================================================
# Adaptive Conformal Inference (ACI) — Gibbs & Candès, NeurIPS 2021
# =============================================================================

class AdaptiveCI:
    """
    Online adaptive conformal inference for non-stationary sequences.

    Maintains a running effective level alpha_t that adjusts based on
    recent coverage:
        alpha_{t+1} = alpha_t + gamma * (alpha - 1[T_t > C_t])

    Parameters
    ----------
    alpha : float
        Target long-run miscoverage rate.
    gamma : float
        Step size. Larger values adapt faster but produce more volatile bounds.
    alpha_init : float or None
        Initial alpha level. Defaults to alpha.
    alpha_min : float
        Minimum alpha (prevents degenerate bounds). Default 1e-6.
    alpha_max : float
        Maximum alpha. Default 0.5.
    """

    def __init__(self, alpha: float, gamma: float = 0.05,
                 alpha_init: float = None,
                 alpha_min: float = 1e-6,
                 alpha_max: float = 0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha_init if alpha_init is not None else alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.history_scores = []  # running calibration window
        self.history_alpha = [self.alpha_t]
        self.history_bounds = []
        self.history_coverage = []
        self.window_size = None  # None = use all history; set for sliding window

    def update(self, new_score: float, observed_time: float = None) -> float:
        """
        Process a new observation, update alpha_t, and return the next bound.

        Parameters
        ----------
        new_score : float
            New execution time observation (nonconformity score = T_i).
        observed_time : float or None
            If provided, compute coverage flag for this step.

        Returns
        -------
        float
            The CP bound for the NEXT time step using the updated alpha_t.
        """
        # Compute bound using current history and current alpha_t
        if len(self.history_scores) < 2:
            bound = np.inf
        else:
            scores = np.array(self.history_scores)
            if self.window_size is not None:
                scores = scores[-self.window_size:]
            bound = cp_pwcet(scores, self.alpha_t)

        self.history_bounds.append(bound)

        # Coverage check
        if observed_time is not None and bound < np.inf:
            covered = int(observed_time <= bound)
        else:
            covered = 1  # count inf bound as covered
        self.history_coverage.append(covered)

        # Update alpha_t (Gibbs-Candès update rule)
        err_indicator = 1 - covered  # 1 if T_t > C_t (missed), 0 otherwise
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err_indicator)
        self.alpha_t = np.clip(self.alpha_t, self.alpha_min, self.alpha_max)
        self.history_alpha.append(self.alpha_t)

        # Add score to history
        self.history_scores.append(new_score)

        # Return current bound (for the step just processed)
        return bound

    def run_on_sequence(self, sequence: np.ndarray) -> dict:
        """
        Run ACI on a complete sequence and return results.

        Parameters
        ----------
        sequence : array (T,)
            Complete execution time trace.

        Returns
        -------
        dict with keys:
            'bounds'    : array (T,) of predicted bounds
            'alpha_t'   : array (T+1,) of effective alpha levels
            'coverage'  : array (T,) of 0/1 coverage indicators
            'long_run_coverage' : float, mean coverage over sequence
        """
        self.__init__(self.alpha, self.gamma)  # reset
        for t_val in sequence:
            self.update(t_val, observed_time=t_val)

        return {
            'bounds': np.array(self.history_bounds),
            'alpha_t': np.array(self.history_alpha),
            'coverage': np.array(self.history_coverage),
            'long_run_coverage': np.mean(self.history_coverage)
        }


# =============================================================================
# Coverage evaluation utilities
# =============================================================================

def empirical_coverage(bound: float, test_times: np.ndarray) -> float:
    """Fraction of test times <= bound."""
    return float(np.mean(test_times <= bound))


def coverage_shortfall(bound: float, test_times: np.ndarray,
                       alpha: float) -> float:
    """max(0, nominal_coverage - empirical_coverage). Positive = undercoverage."""
    return max(0.0, (1 - alpha) - empirical_coverage(bound, test_times))


def bound_efficiency(bound: float, true_quantile: float) -> float:
    """Ratio of estimated bound to true quantile. >1 means conservative."""
    if np.isinf(bound) or true_quantile <= 0:
        return np.inf
    return bound / true_quantile


def run_coverage_trial(calibration: np.ndarray, test: np.ndarray,
                       alpha: float, true_quantile: float) -> dict:
    """
    Run a single coverage trial for all methods and return metrics.

    Returns
    -------
    dict
        Keys: 'cp_bound', 'evt_bound', 'evt_adj_bound', 'eq_bound',
              'cp_cov', 'evt_cov', 'evt_adj_cov', 'eq_cov',
              'cp_short', 'evt_short', 'evt_adj_short', 'eq_short',
              'cp_eff', 'evt_eff', 'evt_adj_eff', 'eq_eff'
    """
    cp_b = cp_pwcet(calibration, alpha)
    evt_b = evt_gpd(calibration, alpha, threshold_method='percentile')
    evt_adj_b = evt_gpd(calibration, alpha, threshold_method='mean_excess')
    eq_b = empirical_quantile(calibration, alpha)

    results = {}
    for name, b in [('cp', cp_b), ('evt', evt_b),
                    ('evt_adj', evt_adj_b), ('eq', eq_b)]:
        results[f'{name}_bound'] = b
        results[f'{name}_cov'] = empirical_coverage(b, test)
        results[f'{name}_short'] = coverage_shortfall(b, test, alpha)
        results[f'{name}_eff'] = bound_efficiency(b, true_quantile)
    return results
