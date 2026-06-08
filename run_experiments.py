r"""
run_experiments.py
------------------
Runs all five experiments from the paper and saves:
  - JSON results files  (for paper table filling)
  - PDF figures         (for paper figures)

Usage:
    python run_experiments.py
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy import stats
from sklearn.model_selection import train_test_split

from cp_methods import (
    cp_pwcet, evt_gpd, empirical_quantile,
    empirical_coverage, coverage_shortfall, bound_efficiency,
    run_coverage_trial, AdaptiveCI, cqr_calibrate, cqr_predict
)
from data_generator import (
    DatasetConfig, PAPER_DATASETS, generate_dataset,
    get_true_quantile, split_calibration_test,
    generate_cqr_dataset, get_true_conditional_quantile
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

ALPHAS_MAIN = [0.01, 0.05, 0.10]
N_CALS = [50, 100, 200, 500, 1000, 2000, 5000]
N_TRIALS = 500       # independent calibration draws per (n, alpha, dataset)
N_POOL = 20_000      # total pool size per dataset
N_TEST = 10_000      # held-out test set size

# For the paper, we focus on four representative datasets
FOCUS_DATASETS = [
    DatasetConfig('GPD xi=-0.1', 'gpd_light', xi=-0.1, sigma_gpd=50),
    DatasetConfig('GPD xi=0.2',  'gpd_heavy', xi=0.2,  sigma_gpd=50),
    DatasetConfig('Mixed b=0.05','mixed',     xi=0.2,  sigma_gpd=60, mix_weight=0.05),
    DatasetConfig('Drift d=0.05','nonstationary', drift_fraction=0.05),
]

# Results accumulator
ALL_RESULTS = {}

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
COLORS = {
    'cp':      '#2166ac',
    'evt':     '#d6604d',
    'evt_adj': '#f4a582',
    'eq':      '#999999',
    'aci':     '#4dac26',
    'nominal': '#000000',
}


# =============================================================================
# EXPERIMENT 1 & 2: Coverage validity and shape sensitivity
# =============================================================================

def run_exp1_exp2():
    """
    Experiment 1: Coverage vs calibration size for all four focus datasets.
    Experiment 2: Shape sensitivity of EVT-GPD.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1 & 2: Coverage validity")
    print("="*60)

    exp1_results = {}
    # Structure: {dataset_name: {alpha: {n: {method: [coverages]}}}}

    for cfg in FOCUS_DATASETS:
        print(f"\n  Dataset: {cfg.name}")
        data_pool = generate_dataset(cfg, N=N_POOL)
        exp1_results[cfg.name] = {}

        for alpha in ALPHAS_MAIN:
            exp1_results[cfg.name][alpha] = {}

            for n_cal in tqdm(N_CALS, desc=f"    alpha={alpha}", leave=False):
                trial_results = {m: [] for m in
                                 ['cp_cov', 'evt_cov', 'evt_adj_cov', 'eq_cov',
                                  'cp_eff', 'evt_eff', 'cp_short', 'evt_short']}
                true_q = get_true_quantile(cfg, alpha)

                for trial in range(N_TRIALS):
                    cal, test = split_calibration_test(data_pool, n_cal,
                                                        N_TEST, seed=trial)
                    r = run_coverage_trial(cal, test, alpha, true_q)
                    for m in trial_results:
                        trial_results[m].append(r[m])

                exp1_results[cfg.name][alpha][n_cal] = {
                    m: {
                        'mean': float(np.mean(v)),
                        'std':  float(np.std(v)),
                        'q05':  float(np.quantile(v, 0.05)),
                        'q95':  float(np.quantile(v, 0.95)),
                    }
                    for m, v in trial_results.items()
                }

    ALL_RESULTS['exp1'] = exp1_results

    # -------------------------------------------------------------------------
    # Figure 1: Coverage vs n  (4 datasets x 3 alphas grid)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(9, 9), sharey='row', sharex='col')

    for row, cfg in enumerate(FOCUS_DATASETS):
        for col, alpha in enumerate(ALPHAS_MAIN):
            ax = axes[row][col]
            res = exp1_results[cfg.name][alpha]
            n_vals = N_CALS

            for method, color, label, ls in [
                ('cp_cov',      COLORS['cp'],  'CP-pWCET',  '-'),
                ('evt_cov',     COLORS['evt'], 'EVT-GPD',   '--'),
                ('eq_cov',      COLORS['eq'],  'Emp. quantile', ':'),
            ]:
                means = [res[n][method]['mean'] for n in n_vals]
                stds  = [res[n][method]['std']  for n in n_vals]
                ax.plot(n_vals, means, color=color, linestyle=ls, label=label)
                ax.fill_between(n_vals,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                color=color, alpha=0.15)

            ax.axhline(1 - alpha, color=COLORS['nominal'],
                       linewidth=0.8, linestyle='--', label='Nominal')
            ax.set_xscale('log')
            ax.set_ylim(0.80, 1.02)
            ax.set_xlim(N_CALS[0], N_CALS[-1])

            if row == 0:
                ax.set_title(f'$\\alpha = {alpha}$')
            if col == 0:
                short_name = cfg.name.replace('GPD ', '').replace('Mixed ', 'Mix ')
                ax.set_ylabel(f'{short_name}\nCoverage')
            if row == 3:
                ax.set_xlabel('Calibration size $n$')
            if row == 0 and col == 2:
                ax.legend(loc='lower right', frameon=False)

    fig.tight_layout()
    fig.savefig('figures/coverage_vs_n.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved figures/coverage_vs_n.pdf")

    # -------------------------------------------------------------------------
    # Figure 2: Shape parameter sensitivity
    # -------------------------------------------------------------------------
    xi_values = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
    alpha_sens = 0.05
    n_vals_sens = [100, 500, 1000]

    shape_results = {n: [] for n in n_vals_sens}

    for xi in tqdm(xi_values, desc='  Shape sensitivity'):
        cfg_s = DatasetConfig(f'xi={xi}', 'gpd_heavy' if xi >= 0 else 'gpd_light',
                               xi=xi, sigma_gpd=50)
        data_pool = generate_dataset(cfg_s, N=N_POOL, seed=7)
        true_q = get_true_quantile(cfg_s, alpha_sens)

        for n_cal in n_vals_sens:
            covs = []
            for trial in range(N_TRIALS):
                cal, test = split_calibration_test(data_pool, n_cal, N_TEST,
                                                    seed=trial)
                b_evt = evt_gpd(cal, alpha_sens)
                if np.isfinite(b_evt):
                    covs.append(empirical_coverage(b_evt, test))
            # From-mean shortfall: max(0, nominal - mean_coverage)
            # Consistent with the theorem-based definition used throughout.
            sf_mean = max(0.0, (1 - alpha_sens) - np.mean(covs)) if covs else 0.0
            shape_results[n_cal].append({
                'xi': xi,
                'evt_shortfall_mean': float(sf_mean),
                'evt_shortfall_std':  0.0,   # std not meaningful for from-mean metric
                'n_valid': len(covs),
            })

    ALL_RESULTS['exp2'] = shape_results

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for n_cal, ls, mk in zip(n_vals_sens, ['-', '--', ':'], ['o', 's', '^']):
        xis = [r['xi'] for r in shape_results[n_cal]]
        means = [r['evt_shortfall_mean'] for r in shape_results[n_cal]]
        stds  = [r['evt_shortfall_std']  for r in shape_results[n_cal]]
        ax.plot(xis, means, color=COLORS['evt'], linestyle=ls,
                marker=mk, markersize=4, label=f'EVT-GPD $n={n_cal}$')
        ax.fill_between(xis, [m-s for m,s in zip(means,stds)],
                             [m+s for m,s in zip(means,stds)],
                             color=COLORS['evt'], alpha=0.1)

    ax.axhline(0, color=COLORS['cp'], linewidth=1.2,
               label='CP-pWCET (always 0)')
    ax.axvline(0, color='#aaaaaa', linewidth=0.6, linestyle=':')
    ax.set_xlabel('True shape parameter $\\xi$')
    ax.set_ylabel('Coverage shortfall')
    ax.set_title(f'EVT shape sensitivity ($\\alpha={alpha_sens}$)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig('figures/shape_sensitivity.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved figures/shape_sensitivity.pdf")


# =============================================================================
# EXPERIMENT 3: Efficiency and EVT–CP complementarity
# =============================================================================

def run_exp3():
    """
    Figure: Bound efficiency vs alpha for CP and EVT.
    Shows complementary ranges of validity.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Bound efficiency and complementarity")
    print("="*60)

    cfg = DatasetConfig('GPD xi=0.2', 'gpd_heavy', xi=0.2, sigma_gpd=50)
    data_pool = generate_dataset(cfg, N=N_POOL)
    alpha_range = np.logspace(-1, -6, 60)
    n_eff = 1000

    cp_bounds_all = []
    evt_bounds_all = []

    for trial in tqdm(range(200), desc='  Efficiency trials'):
        cal, _ = split_calibration_test(data_pool, n_eff, N_TEST, seed=trial)
        cp_b  = np.array([cp_pwcet(cal, a) for a in alpha_range])
        evt_b = np.array([evt_gpd(cal, a) for a in alpha_range])
        cp_bounds_all.append(cp_b)
        evt_bounds_all.append(evt_b)

    cp_bounds_all  = np.array(cp_bounds_all)
    evt_bounds_all = np.array(evt_bounds_all)
    true_quantiles = np.array([get_true_quantile(cfg, a) for a in alpha_range])

    cp_eff_mean  = np.nanmean(cp_bounds_all / true_quantiles, axis=0)
    evt_eff_mean = np.nanmean(evt_bounds_all / true_quantiles, axis=0)

    ALL_RESULTS['exp3'] = {
        'alpha_range': alpha_range.tolist(),
        'cp_eff_mean':  [x if np.isfinite(x) else None for x in cp_eff_mean.tolist()],
        'evt_eff_mean': [x if np.isfinite(x) else None for x in evt_eff_mean.tolist()],
    }

    # CP is inf for alpha < 1/n; mark that boundary
    cp_valid_mask = alpha_range >= 1 / (n_eff + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Main plot
    ax = axes[0]
    ax.plot(alpha_range[cp_valid_mask], cp_eff_mean[cp_valid_mask],
            color=COLORS['cp'], label='CP-pWCET')
    ax.plot(alpha_range, evt_eff_mean, color=COLORS['evt'],
            linestyle='--', label='EVT-GPD')
    ax.axhline(1.0, color='k', linewidth=0.6, linestyle=':')
    ax.axvline(1 / (n_eff + 1), color=COLORS['cp'], linewidth=0.6,
               linestyle=':', alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Target miscoverage $\\alpha$')
    ax.set_ylabel('Bound efficiency ($\\hat{C} / q^*_{1-\\alpha}$)')
    ax.set_title(f'Bound efficiency ($n={n_eff}$, GPD $\\xi=0.2$)')
    ax.legend(frameon=False)
    ax.invert_xaxis()

    # Inset: jointly informative range
    ax2 = axes[1]
    joint_mask = (alpha_range >= 5e-5) & (alpha_range <= 5e-3) & cp_valid_mask
    ax2.plot(alpha_range[joint_mask], cp_eff_mean[joint_mask],
             color=COLORS['cp'], label='CP-pWCET')
    ax2.plot(alpha_range[joint_mask], evt_eff_mean[joint_mask],
             color=COLORS['evt'], linestyle='--', label='EVT-GPD')
    ax2.axhline(1.0, color='k', linewidth=0.6, linestyle=':')
    ax2.set_xscale('log')
    ax2.set_xlabel('$\\alpha$')
    ax2.set_ylabel('Efficiency')
    ax2.set_title('Jointly informative range')
    ax2.legend(frameon=False)
    ax2.invert_xaxis()

    fig.tight_layout()
    fig.savefig('figures/efficiency_vs_alpha.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved figures/efficiency_vs_alpha.pdf")


# =============================================================================
# EXPERIMENT 4: CQR conditional bounds
# =============================================================================

def run_exp4():
    """
    CQR conditional bounds for covariate-dependent execution time.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: CQR conditional bounds")
    print("="*60)

    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        print("  lightgbm not available; using gradient boosted regressor from sklearn")
        HAS_LGB = False
        from sklearn.ensemble import GradientBoostingRegressor

    alpha = 0.05
    n_train = 5_000
    n_cal = 2_000
    n_test = 5_000
    N_total = n_train + n_cal + n_test

    X, T = generate_cqr_dataset(N=N_total)

    X_train, X_rest, T_train, T_rest = train_test_split(
        X, T, test_size=(n_cal + n_test), random_state=42)
    X_cal, X_test, T_cal, T_test = train_test_split(
        X_rest, T_rest, test_size=n_test, random_state=42)

    # Train quantile regressor
    if HAS_LGB:
        model = lgb.LGBMRegressor(
            objective='quantile', alpha=1 - alpha,
            n_estimators=200, learning_rate=0.05,
            num_leaves=31, verbose=-1)
        model.fit(X_train.reshape(-1, 1), T_train)
        q_cal_preds  = model.predict(X_cal.reshape(-1, 1))
        q_test_preds = model.predict(X_test.reshape(-1, 1))
    else:
        model = GradientBoostingRegressor(
            loss='quantile', alpha=1 - alpha,
            n_estimators=200, learning_rate=0.05, max_depth=4)
        model.fit(X_train.reshape(-1, 1), T_train)
        q_cal_preds  = model.predict(X_cal.reshape(-1, 1))
        q_test_preds = model.predict(X_test.reshape(-1, 1))

    # CQR calibration
    correction = cqr_calibrate(T_cal, q_cal_preds, q_cal_preds, alpha)
    cqr_bounds_test = cqr_predict(q_test_preds, correction)

    # Unconditional CP
    cp_bound_unconditional = cp_pwcet(T_cal, alpha)

    # True conditional quantile
    x_grid = np.linspace(0, 1, 200)
    true_q_grid = get_true_conditional_quantile(x_grid, alpha)
    q_model_grid = model.predict(x_grid.reshape(-1, 1))
    cqr_grid = q_model_grid + correction

    # Coverage by x-bin
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_cov_cp  = []
    bin_cov_cqr = []
    bin_centres = []
    for i in range(n_bins):
        mask = (X_test >= bin_edges[i]) & (X_test < bin_edges[i+1])
        if mask.sum() < 5:
            continue
        bin_centres.append(0.5 * (bin_edges[i] + bin_edges[i+1]))
        bin_cov_cp.append(empirical_coverage(cp_bound_unconditional, T_test[mask]))
        bin_cov_cqr.append(empirical_coverage(
            float(np.mean(cqr_bounds_test[mask])), T_test[mask]))

    # Overall coverages
    overall_cp_cov  = empirical_coverage(cp_bound_unconditional, T_test)
    overall_cqr_cov = float(np.mean(T_test <= cqr_bounds_test))

    print(f"  Unconditional CP bound: {cp_bound_unconditional:.1f} us, "
          f"coverage: {overall_cp_cov:.4f}")
    print(f"  CQR marginal coverage:  {overall_cqr_cov:.4f}")

    ALL_RESULTS['exp4'] = {
        'cp_bound': float(cp_bound_unconditional),
        'overall_cp_cov': overall_cp_cov,
        'overall_cqr_cov': overall_cqr_cov,
        'correction': float(correction),
        'bin_centres': bin_centres,
        'bin_cov_cp': bin_cov_cp,
        'bin_cov_cqr': bin_cov_cqr,
    }

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    ax.scatter(X_test[:500], T_test[:500], s=3, alpha=0.3, color='#aaaaaa',
               label='Test data (sample)')
    ax.plot(x_grid, true_q_grid, color='k', linewidth=1.2,
            linestyle='--', label=f'True $q_{{1-\\alpha}}(x)$')
    ax.axhline(cp_bound_unconditional, color=COLORS['cp'],
               linewidth=1.5, label=f'CP bound ({cp_bound_unconditional:.0f} µs)')
    ax.plot(x_grid, cqr_grid, color=COLORS['aci'],
            linewidth=1.5, label='CQR bound')
    ax.set_xlabel('Input complexity $x$')
    ax.set_ylabel('Execution time (µs)')
    ax.set_title('Conditional bounds (CQR vs CP)')
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(bin_centres, bin_cov_cp, color=COLORS['cp'],
            marker='o', markersize=4, label='CP (unconditional)')
    ax.plot(bin_centres, bin_cov_cqr, color=COLORS['aci'],
            marker='s', markersize=4, label='CQR')
    ax.axhline(1 - alpha, color='k', linewidth=0.8, linestyle='--',
               label='Nominal 95%')
    ax.set_xlabel('Input complexity $x$')
    ax.set_ylabel('Empirical coverage (per bin)')
    ax.set_title('Conditional coverage by input bin')
    ax.set_ylim(0.80, 1.02)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig('figures/cqr_conditional.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved figures/cqr_conditional.pdf")


# =============================================================================
# EXPERIMENT 5: ACI under drift
# =============================================================================

def run_exp5():
    """
    Adaptive conformal inference under distribution shift.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: ACI under drift")
    print("="*60)

    alpha = 0.05
    gammas = [0.01, 0.05, 0.10]
    drift_cfg = DatasetConfig('Drift d=0.05', 'nonstationary', drift_fraction=0.05)
    T_seq = generate_dataset(drift_cfg, N=10_000)
    N_seq = len(T_seq)
    window = 500

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- Static CP: refit from scratch on each window (or use fixed initial cal)
    # For a fair baseline, use the first 500 as calibration, then eval on rest
    n_init = 500
    cal_init = T_seq[:n_init]
    cp_bound_static = cp_pwcet(cal_init, alpha)

    # Roll window to compute rolling CP coverage
    all_aci_results = {}
    for gamma in gammas:
        aci = AdaptiveCI(alpha=alpha, gamma=gamma)
        aci_res = aci.run_on_sequence(T_seq)
        all_aci_results[gamma] = aci_res

    # Rolling coverage for static CP
    static_covered = (T_seq <= cp_bound_static).astype(float)

    def rolling_mean(arr, w):
        ret = np.cumsum(arr, dtype=float)
        ret[w:] = ret[w:] - ret[:-w]
        result = ret[w - 1:] / w
        return result

    ax = axes[0]
    x_range = np.arange(window - 1, N_seq)

    # Static CP rolling coverage
    rc_static = rolling_mean(static_covered, window)
    ax.plot(x_range, rc_static, color=COLORS['cp'], linestyle='-',
            label='Static CP', linewidth=1.5)

    for gamma, ls in zip(gammas, ['--', '-', ':']):
        aci_cov = all_aci_results[gamma]['coverage'].astype(float)
        rc_aci = rolling_mean(aci_cov, window)
        ax.plot(x_range, rc_aci, color=COLORS['aci'], linestyle=ls,
                label=f'ACI $\\gamma={gamma}$', linewidth=1.5)

    ax.axhline(1 - alpha, color='k', linewidth=0.8, linestyle='--',
               label=f'Nominal {int((1-alpha)*100)}%')
    ax.set_ylabel(f'Rolling coverage\n(window={window})')
    ax.set_title(f'Coverage under distribution shift (drift $d=0.05\\mu$)')
    ax.legend(frameon=False, loc='lower left')
    ax.set_ylim(0.75, 1.02)

    # Effective alpha_t trajectory for best gamma
    ax2 = axes[1]
    best_gamma = 0.05
    alpha_t = all_aci_results[best_gamma]['alpha_t']
    bounds_aci = all_aci_results[best_gamma]['bounds']

    ax2.plot(np.arange(len(alpha_t)), alpha_t, color=COLORS['aci'],
             linewidth=1.2, label=f'$\\alpha_t$ (ACI $\\gamma={best_gamma}$)')
    ax2.axhline(alpha, color='k', linewidth=0.8, linestyle='--',
                label=f'Target $\\alpha={alpha}$')
    ax2.set_xlabel('Time step $t$')
    ax2.set_ylabel('Effective $\\alpha_t$')
    ax2.set_title('ACI effective miscoverage level')
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig('figures/aci_drift.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved figures/aci_drift.pdf")

    # Compute stats for paper
    n_transient = 200  # look-ahead for "recovery"
    for gamma in gammas:
        cov = all_aci_results[gamma]['coverage']
        print(f"  ACI gamma={gamma}: "
              f"mean coverage={np.mean(cov):.4f}, "
              f"std={np.std(rolling_mean(cov.astype(float), window)):.4f}")

    print(f"  Static CP: init coverage={np.mean(static_covered[:1000]):.4f}, "
          f"final coverage={np.mean(static_covered[-1000:]):.4f}")

    # Save results
    ALL_RESULTS['exp5'] = {
        'static_cp_init_cov': float(np.mean(static_covered[:1000])),
        'static_cp_final_cov': float(np.mean(static_covered[-1000:])),
        'aci_results': {
            str(g): {
                'mean_cov': float(np.mean(all_aci_results[g]['coverage'])),
                'std_rolling': float(np.std(
                    rolling_mean(all_aci_results[g]['coverage'].astype(float), window)
                )),
            }
            for g in gammas
        }
    }


# =============================================================================
# EXPERIMENT 3: Small-n stress test (was missing from original script)
# =============================================================================

def run_exp_smalln():
    """
    Experiment 3: EQ false-precision at small calibration sizes.

    Tests n in {10, 15, 20, 30, 50, 75, 100} with 400 trials.
    Uses the GPD xi=-0.1 dataset (light tail) as the representative case,
    averaged over all four focus datasets for robustness.

    Shortfall metric: max(0, nominal - mean_coverage)  [from-mean, theorem-consistent]
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Small-n stress test")
    print("="*60)

    alpha    = 0.05
    N_TRIALS = 400      # intentional exception from global N_TRIALS=500
    n_vals   = [10, 15, 20, 30, 50, 75, 100]

    results = {}
    for cfg in FOCUS_DATASETS:
        data_pool = generate_dataset(cfg, N=N_POOL)
        results[cfg.name] = {}
        for n_cal in n_vals:
            cp_covs, evt_covs, eq_covs = [], [], []
            cp_inf, evt_inf = 0, 0
            true_q = get_true_quantile(cfg, alpha)
            for trial in range(N_TRIALS):
                cal, test = split_calibration_test(data_pool, n_cal,
                                                   N_TEST, seed=trial)
                b_cp  = cp_pwcet(cal, alpha)
                b_evt = evt_gpd(cal, alpha)
                b_eq  = empirical_quantile(cal, alpha)
                cov_eq = empirical_coverage(b_eq, test)
                eq_covs.append(cov_eq)
                if np.isfinite(b_cp):
                    cp_covs.append(empirical_coverage(b_cp, test))
                else:
                    cp_inf += 1
                if np.isfinite(b_evt):
                    evt_covs.append(empirical_coverage(b_evt, test))
                else:
                    evt_inf += 1

            results[cfg.name][n_cal] = {
                'cp_mean_cov':  float(np.mean(cp_covs))  if cp_covs  else None,
                'evt_mean_cov': float(np.mean(evt_covs)) if evt_covs else None,
                'eq_mean_cov':  float(np.mean(eq_covs)),
                'eq_shortfall': float(max(0, (1-alpha) - np.mean(eq_covs))),
                'cp_inf_frac':  cp_inf / N_TRIALS,
                'evt_inf_frac': evt_inf / N_TRIALS,
            }
            r = results[cfg.name][n_cal]
            cp_str  = 'inf' if r['cp_mean_cov']  is None else f"{r['cp_mean_cov']*100:.2f}%"
            evt_str = 'inf' if r['evt_mean_cov'] is None else f"{r['evt_mean_cov']*100:.2f}%"
            print(f"  {cfg.name:20s} n={n_cal:3d}  "
                  f"CP={cp_str:>8s}  "
                  f"EVT={evt_str:>8s}  "
                  f"EQ={r['eq_mean_cov']*100:.2f}%  "
                  f"EQ_sf={r['eq_shortfall']*100:.2f}pp")

    ALL_RESULTS['exp_smalln'] = results

    # Figure: coverage and shortfall vs n, averaged across datasets
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    x = n_vals

    for cfg in FOCUS_DATASETS:
        cp_means  = [results[cfg.name][n]['cp_mean_cov']  for n in x]
        evt_means = [results[cfg.name][n]['evt_mean_cov'] for n in x]
        eq_means  = [results[cfg.name][n]['eq_mean_cov']  for n in x]
        eq_sfs    = [results[cfg.name][n]['eq_shortfall'] for n in x]

        # Replace None (inf bound) with NaN for plotting
        cp_plot  = [v if v is not None else float('nan') for v in cp_means]
        evt_plot = [v if v is not None else float('nan') for v in evt_means]

        axes[0].plot(x, [v*100 if not np.isnan(v) else float('nan')
                         for v in cp_plot],
                     color=COLORS['cp'], linewidth=1.0, alpha=0.4)
        axes[0].plot(x, [v*100 for v in eq_means],
                     color=COLORS['eq'], linewidth=1.0, alpha=0.4)
        axes[1].plot(x, [v*100 for v in eq_sfs],
                     color=COLORS['eq'], linewidth=1.0, alpha=0.4)

    # Average over datasets
    cp_avg  = [np.nanmean([results[c.name][n]['cp_mean_cov']  or float('nan')
                           for c in FOCUS_DATASETS]) for n in x]
    eq_avg  = [np.mean([results[c.name][n]['eq_mean_cov']
                        for c in FOCUS_DATASETS]) for n in x]
    eq_sf_avg = [np.mean([results[c.name][n]['eq_shortfall']
                          for c in FOCUS_DATASETS]) for n in x]

    axes[0].plot(x, [v*100 if not np.isnan(v) else float('nan')
                     for v in cp_avg],
                 color=COLORS['cp'], linewidth=2.0, label='CP-pWCET (mean)')
    axes[0].plot(x, [v*100 for v in eq_avg],
                 color=COLORS['eq'], linewidth=2.0, label='EQ (mean)')
    axes[0].axhline(95.0, color='k', linewidth=0.8, linestyle='--',
                    label='Nominal 95%')
    axes[0].set_xlabel('Calibration size $n$')
    axes[0].set_ylabel('Coverage (%)')
    axes[0].set_title(f'Small-$n$ coverage ($\\alpha={alpha}$, {N_TRIALS} trials)')
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_ylim(80, 102)

    axes[1].plot(x, [v*100 for v in eq_sf_avg],
                 color=COLORS['eq'], linewidth=2.0, label='EQ shortfall (mean)')
    axes[1].axhline(0, color=COLORS['cp'], linewidth=1.2,
                    label='CP-pWCET (0 by theorem)')
    axes[1].set_xlabel('Calibration size $n$')
    axes[1].set_ylabel('Shortfall $\\max(0, 0.95 - \\bar{\\mathrm{cov}})$ (pp)')
    axes[1].set_title('Shortfall vs calibration size')
    axes[1].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig('figures/small_n_stress.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved figures/small_n_stress.pdf")


# =============================================================================
# SUMMARY TABLE (Table 2 in paper)
# =============================================================================

def run_summary_table():
    """
    Generate Table 2: Coverage and efficiency for all datasets at n=500, alpha=0.05.
    """
    print("\n" + "="*60)
    print("SUMMARY TABLE (n=500, alpha=0.05)")
    print("="*60)

    alpha = 0.05
    n_cal = 500
    table_data = []

    for cfg in FOCUS_DATASETS:
        data_pool = generate_dataset(cfg, N=N_POOL)
        true_q = get_true_quantile(cfg, alpha)
        results_per_method = {m: {'cov': [], 'short': [], 'eff': []}
                               for m in ['cp', 'evt', 'eq']}

        for trial in tqdm(range(N_TRIALS), desc=f'  {cfg.name}', leave=False):
            cal, test = split_calibration_test(data_pool, n_cal, N_TEST, seed=trial)
            r = run_coverage_trial(cal, test, alpha, true_q)
            for m in ['cp', 'evt', 'eq']:
                results_per_method[m]['cov'].append(r[f'{m}_cov'])
                results_per_method[m]['short'].append(r[f'{m}_short'])
                results_per_method[m]['eff'].append(r[f'{m}_eff'])

        for method in ['cp', 'evt', 'eq']:
            d = results_per_method[method]
            entry = {
                'dataset': cfg.name,
                'method': method,
                'coverage_mean': float(np.mean(d['cov'])),
                'coverage_ci': float(1.96 * np.std(d['cov']) / np.sqrt(N_TRIALS)),
                'shortfall_mean': float(np.mean(d['short'])),
                'efficiency_mean': float(np.nanmean([x for x in d['eff']
                                                      if np.isfinite(x)])),
            }
            table_data.append(entry)
            print(f"  {cfg.name:20s} | {method:7s} | "
                  f"cov={entry['coverage_mean']:.4f}±{entry['coverage_ci']:.4f} | "
                  f"short={entry['shortfall_mean']:.4f} | "
                  f"eff={entry['efficiency_mean']:.4f}")

    ALL_RESULTS['table2'] = table_data


# =============================================================================
# PRINT RESULTS SUMMARY for paper
# =============================================================================

def print_paper_results():
    r"""Print a summary of all \RESULT{...} values to fill into main.tex."""
    print("\n" + "="*60)
    print("PAPER RESULTS SUMMARY  (copy into main.tex)")
    print("="*60)

    r = ALL_RESULTS

    # Exp 1: EVT threshold for adequate coverage
    if 'exp1' in r:
        # Find n where EVT first reaches nominal coverage (within 0.5%) on mixed
        for cfg_name in r['exp1']:
            if 'Mixed' in cfg_name:
                data_05 = r['exp1'][cfg_name].get(0.05, {})
                for n_cal in N_CALS:
                    evt_cov = data_05.get(n_cal, {}).get('evt_cov', {}).get('mean', 0)
                    if evt_cov >= 0.945:
                        print(f"\\RESULT{{E1:EVT_n_threshold}} = {n_cal}")
                        break

        # CP and EVT coverage at n=100, alpha=0.05 on mixed
        for cfg_name in r['exp1']:
            if 'Mixed' in cfg_name:
                n100 = r['exp1'][cfg_name].get(0.05, {}).get(100, {})
                cp_cov  = n100.get('cp_cov', {}).get('mean', '?')
                evt_cov = n100.get('evt_cov', {}).get('mean', '?')
                print(f"\\RESULT{{E1:CP_cov_n100}}  = {cp_cov*100:.1f}\\%")
                print(f"\\RESULT{{E1:EVT_cov_n100}} = {evt_cov*100:.1f}\\%")

        # Efficiency at n=500, alpha=0.05 on GPD heavy
        for cfg_name in r['exp1']:
            if 'xi=0.2' in cfg_name:
                n500 = r['exp1'][cfg_name].get(0.05, {}).get(500, {})
                cp_eff  = n500.get('cp_eff', {}).get('mean', '?')
                evt_eff = n500.get('evt_eff', {}).get('mean', '?')
                print(f"\\RESULT{{E1:CP_eff}}  = {cp_eff:.3f}")
                print(f"\\RESULT{{E1:EVT_eff}} = {evt_eff:.3f}")

    # Exp 2: worst EVT shortfall
    if 'exp2' in r:
        all_s = [entry['evt_shortfall_mean']
                 for entries in r['exp2'].values()
                 for entry in entries]
        worst = max(all_s) if all_s else '?'
        print(f"\\RESULT{{E2:EVT_worst_shortfall}} = {worst*100:.1f}\\%")

    # Exp 4
    if 'exp4' in r:
        print(f"\\RESULT{{E4:CP_bound}}       = {r['exp4']['cp_bound']:.0f}")
        cqr_worst = min(r['exp4']['bin_cov_cqr']) if r['exp4']['bin_cov_cqr'] else '?'
        print(f"\\RESULT{{E4:CQR_worst_cov}}  = {(1-cqr_worst)*100:.1f}\\%")

    # Exp 5
    if 'exp5' in r:
        print(f"\\RESULT{{E5:CP_init_cov}}  = {r['exp5']['static_cp_init_cov']*100:.1f}\\%")
        print(f"\\RESULT{{E5:CP_final_cov}} = {r['exp5']['static_cp_final_cov']*100:.1f}\\%")
        best = r['exp5']['aci_results']['0.05']
        print(f"\\RESULT{{E5:ACI_mean_cov}} = {best['mean_cov']*100:.1f}\\%")
        print(f"\\RESULT{{E5:ACI_std_cov}}  = {best['std_rolling']*100:.2f}\\%")

    # Table 2
    if 'table2' in r:
        print("\nTable 2 entries:")
        for entry in r['table2']:
            tag = f"T:{entry['dataset'].replace(' ','').replace('=','').replace('.','')}" \
                  f":{entry['method'].upper()}"
            print(f"  {tag}:cov  = {entry['coverage_mean']*100:.2f} "
                  f"({entry['coverage_ci']*100:.2f})")
            print(f"  {tag}:short= {entry['shortfall_mean']*100:.2f}")
            print(f"  {tag}:eff  = {entry['efficiency_mean']:.3f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("CP-pWCET Experiment Runner")
    print(f"  N_trials={N_TRIALS}, N_pool={N_POOL}, N_test={N_TEST}")

    run_exp1_exp2()
    run_exp3()
    run_exp_smalln()
    run_exp4()
    run_exp5()
    run_summary_table()

    # Save all numeric results
    with open('results/results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2, default=lambda x:
                  x.tolist() if hasattr(x, 'tolist') else x)
    print("\nSaved results/results.json")

    print_paper_results()

    print("\nDone. Figures are in ./figures/")    
