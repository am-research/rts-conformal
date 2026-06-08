"""
data_generator.py
-----------------
Synthetic execution-time dataset generator.

Four dataset classes matching the paper's Section 4.1:
  1. GPD-tail (light tail, xi < 0)
  2. GPD-tail (heavy tail, xi > 0)
  3. Mixed Gaussian + GPD  (bimodal cache hit/miss)
  4. Non-stationary (thermal drift)

All times are in microseconds (us).
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration for a single synthetic dataset."""
    name: str
    kind: str               # 'gpd_light', 'gpd_heavy', 'mixed', 'nonstationary'
    mu: float = 200.0       # nominal mean execution time (us)
    sigma: float = 30.0     # base spread
    xi: float = 0.0         # GPD shape parameter
    sigma_gpd: float = 50.0 # GPD scale
    mix_weight: float = 0.05  # fraction of GPD component in mixed dataset
    drift_fraction: float = 0.05  # drift as fraction of mu (non-stationary)
    seed: int = 42


# Canonical configurations used in the paper
PAPER_DATASETS = [
    DatasetConfig('GPD xi=-0.1 (light)', 'gpd_light', xi=-0.1, sigma_gpd=50),
    DatasetConfig('GPD xi=-0.2 (light)', 'gpd_light', xi=-0.2, sigma_gpd=50),
    DatasetConfig('GPD xi=0.0 (Exp)',    'gpd_heavy', xi=0.0,  sigma_gpd=50),
    DatasetConfig('GPD xi=0.1 (heavy)',  'gpd_heavy', xi=0.1,  sigma_gpd=50),
    DatasetConfig('GPD xi=0.2 (heavy)',  'gpd_heavy', xi=0.2,  sigma_gpd=50),
    DatasetConfig('GPD xi=0.3 (heavy)',  'gpd_heavy', xi=0.3,  sigma_gpd=50),
    DatasetConfig('Mixed b=0.05',        'mixed',     xi=0.2,  sigma_gpd=60,
                  mix_weight=0.05),
    DatasetConfig('Mixed b=0.10',        'mixed',     xi=0.2,  sigma_gpd=60,
                  mix_weight=0.10),
    DatasetConfig('Drift d=0.01',        'nonstationary', drift_fraction=0.01),
    DatasetConfig('Drift d=0.05',        'nonstationary', drift_fraction=0.05),
    DatasetConfig('Drift d=0.10',        'nonstationary', drift_fraction=0.10),
]


def _gpd_samples(n: int, xi: float, sigma: float, mu: float,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Generate n samples from a GPD-based execution time distribution.
    T = mu + GPD(xi, sigma) where GPD is sampled via inverse CDF.
    """
    u = rng.uniform(0, 1, n)
    if abs(xi) < 1e-8:
        # Exponential limit
        samples = -sigma * np.log(1 - u)
    else:
        samples = sigma / xi * ((1 - u) ** (-xi) - 1)
    return mu + samples


def generate_dataset(cfg: DatasetConfig, N: int = 20_000,
                     seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a full pool of N execution time samples for a given config.

    Parameters
    ----------
    cfg : DatasetConfig
    N : int
        Total number of samples to generate.
    seed : int or None
        Random seed. If None, uses cfg.seed.

    Returns
    -------
    np.ndarray of shape (N,), execution times in microseconds.
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)

    if cfg.kind in ('gpd_light', 'gpd_heavy'):
        samples = _gpd_samples(N, cfg.xi, cfg.sigma_gpd, cfg.mu, rng)

    elif cfg.kind == 'mixed':
        # Bimodal: fast Gaussian body + heavy-tailed GPD tail
        n_gaussian = int(N * (1 - cfg.mix_weight))
        n_gpd = N - n_gaussian
        body = rng.normal(cfg.mu, cfg.sigma, n_gaussian)
        tail = _gpd_samples(n_gpd, cfg.xi, cfg.sigma_gpd,
                             cfg.mu + 2 * cfg.sigma, rng)
        samples = np.concatenate([body, tail])
        rng.shuffle(samples)

    elif cfg.kind == 'nonstationary':
        # Linear drift: mean increases by drift_fraction * mu over the sequence
        drift_total = cfg.drift_fraction * cfg.mu
        means = cfg.mu + np.linspace(0, drift_total, N)
        samples = rng.normal(means, cfg.sigma)

    else:
        raise ValueError(f"Unknown dataset kind: {cfg.kind}")

    # Clip to realistic minimum (task always takes at least mu/10 us)
    samples = np.clip(samples, cfg.mu / 10, None)
    return samples.astype(np.float64)


def get_true_quantile(cfg: DatasetConfig, alpha: float) -> float:
    """
    Compute the true (1-alpha) quantile of the data-generating distribution.
    Used for efficiency evaluation.

    For non-stationary datasets, returns the quantile of the *initial*
    distribution (drift = 0) since the true quantile is time-varying.
    """
    if cfg.kind in ('gpd_light', 'gpd_heavy'):
        if abs(cfg.xi) < 1e-8:
            # Exponential: quantile = mu - sigma * log(alpha)
            return cfg.mu + (-cfg.sigma_gpd * np.log(alpha))
        else:
            return cfg.mu + cfg.sigma_gpd / cfg.xi * (alpha ** (-cfg.xi) - 1)

    elif cfg.kind == 'mixed':
        # No closed form; estimate from a large sample
        rng = np.random.default_rng(999)
        large_sample = generate_dataset(cfg, N=500_000, seed=999)
        return float(np.quantile(large_sample, 1 - alpha))

    elif cfg.kind == 'nonstationary':
        # Initial distribution quantile
        return float(stats.norm.ppf(1 - alpha, loc=cfg.mu, scale=cfg.sigma))

    else:
        raise ValueError(f"Unknown kind: {cfg.kind}")


def split_calibration_test(data: np.ndarray, n_cal: int,
                            n_test: int = 10_000,
                            seed: int = 0) -> tuple:
    """
    Split data into a calibration set of size n_cal and a test set.

    Parameters
    ----------
    data : array (N,)
    n_cal : int
    n_test : int
    seed : int

    Returns
    -------
    calibration : array (n_cal,)
    test        : array (n_test,)
    """
    rng = np.random.default_rng(seed)
    N = len(data)
    assert n_cal + n_test <= N, \
        f"Requested {n_cal} + {n_test} = {n_cal+n_test} > N={N}"
    idx = rng.permutation(N)
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:n_cal + n_test]
    return data[cal_idx], data[test_idx]


# ---------------------------------------------------------------------------
# Covariate-dependent dataset for CQR experiment
# ---------------------------------------------------------------------------

def generate_cqr_dataset(N: int = 15_000,
                          xi: float = 0.2,
                          sigma_gpd: float = 50.0,
                          seed: int = 42) -> tuple:
    """
    Generate a covariate-dependent execution time dataset for the CQR
    experiment (Section 4.4 of paper).

    Model: T_i = mu(x_i) + epsilon_i
      x_i ~ Uniform(0, 1)   (input complexity, e.g. cache-cold fraction)
      mu(x) = 100 + 500*x   (linear mean latency)
      epsilon_i ~ GPD(xi, sigma_gpd)

    Returns
    -------
    X : array (N,) of covariate values
    T : array (N,) of execution times
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, N)
    mu_x = 100 + 500 * X
    if abs(xi) < 1e-8:
        noise = rng.exponential(sigma_gpd, N)
    else:
        u = rng.uniform(0, 1, N)
        noise = sigma_gpd / xi * ((1 - u) ** (-xi) - 1)
    T = mu_x + noise
    T = np.clip(T, 10, None)
    return X, T.astype(np.float64)


def get_true_conditional_quantile(x: np.ndarray, alpha: float,
                                   xi: float = 0.2,
                                   sigma_gpd: float = 50.0) -> np.ndarray:
    """True conditional (1-alpha)-quantile of T|X=x for the CQR dataset."""
    mu_x = 100 + 500 * x
    if abs(xi) < 1e-8:
        q_noise = -sigma_gpd * np.log(alpha)
    else:
        q_noise = sigma_gpd / xi * (alpha ** (-xi) - 1)
    return mu_x + q_noise
