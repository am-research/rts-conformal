"""
generate_figures.py
-------------------
Produces all five paper figures.
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from cp_methods import (
    cp_pwcet, evt_gpd, empirical_quantile,
    empirical_coverage, coverage_shortfall, bound_efficiency,
    run_coverage_trial, AdaptiveCI, cqr_calibrate, cqr_predict
)
from data_generator import (
    DatasetConfig, generate_dataset, get_true_quantile,
    split_calibration_test, generate_cqr_dataset,
    get_true_conditional_quantile
)

OUT = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

# Reduce trials for speed; increase for final camera-ready
N_TRIALS = 100
N_POOL   = 20_000
N_TEST   = 8_000
N_CALS   = [50, 100, 200, 500, 1000, 2000, 5000]
ALPHAS   = [0.01, 0.05, 0.10]

FOCUS = [
    DatasetConfig('GPD $\\xi=-0.1$',  'gpd_light', xi=-0.1, sigma_gpd=50),
    DatasetConfig('GPD $\\xi=0.2$',   'gpd_heavy', xi=0.2,  sigma_gpd=50),
    DatasetConfig('Mixed $b=0.05$',   'mixed',     xi=0.2,  sigma_gpd=60, mix_weight=0.05),
    DatasetConfig('Drift $d=0.05\\mu$','nonstationary', drift_fraction=0.05),
]

COLORS = {
    'cp':   '#2166ac',
    'evt':  '#d6604d',
    'eq':   '#888780',
    'aci':  '#4dac26',
    'nom':  '#000000',
}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 200,
    'lines.linewidth': 1.4,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# =============================================================================
# FIGURE 1: coverage_vs_n.pdf
# =============================================================================
print("Figure 1: coverage_vs_n …")
fig, axes = plt.subplots(4, 3, figsize=(8.5, 9),
                          sharey='row', sharex='col',
                          gridspec_kw={'hspace': 0.45, 'wspace': 0.12})

for row, cfg in enumerate(tqdm(FOCUS, desc='  datasets')):
    data_pool = generate_dataset(cfg, N=N_POOL)
    for col, alpha in enumerate(ALPHAS):
        ax = axes[row][col]
        true_q = get_true_quantile(cfg, alpha)

        cp_means, cp_stds   = [], []
        evt_means, evt_stds = [], []
        eq_means, eq_stds   = [], []

        for n_cal in N_CALS:
            cp_covs, evt_covs, eq_covs = [], [], []
            for t in range(N_TRIALS):
                cal, test = split_calibration_test(data_pool, n_cal, N_TEST, seed=t)
                r = run_coverage_trial(cal, test, alpha, true_q)
                cp_covs.append(r['cp_cov'])
                evt_covs.append(r['evt_cov'])
                eq_covs.append(r['eq_cov'])
            cp_means.append(np.mean(cp_covs));  cp_stds.append(np.std(cp_covs))
            evt_means.append(np.mean(evt_covs)); evt_stds.append(np.std(evt_covs))
            eq_means.append(np.mean(eq_covs));  eq_stds.append(np.std(eq_covs))

        for means, stds, color, ls, lbl in [
            (cp_means,  cp_stds,  COLORS['cp'],  '-',  'CP-pWCET'),
            (evt_means, evt_stds, COLORS['evt'], '--', 'EVT-GPD'),
            (eq_means,  eq_stds,  COLORS['eq'],  ':',  'EQ'),
        ]:
            ax.plot(N_CALS, means, color=color, linestyle=ls, label=lbl, lw=1.4)
            ax.fill_between(N_CALS,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=color, alpha=0.12)

        ax.axhline(1 - alpha, color=COLORS['nom'], lw=0.7,
                   linestyle='--', label='Nominal')
        ax.set_xscale('log')
        ax.set_ylim(0.82, 1.005)
        ax.set_xlim(N_CALS[0], N_CALS[-1])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

        if row == 0:
            ax.set_title(f'$\\alpha = {alpha}$', pad=4)
        if col == 0:
            ax.set_ylabel(cfg.name + '\nCoverage', labelpad=3)
        if row == 3:
            ax.set_xlabel('Calibration $n$')
        if row == 0 and col == 2:
            ax.legend(loc='lower right', frameon=False, fontsize=7.5)

fig.savefig(os.path.join(OUT, 'coverage_vs_n.pdf'), bbox_inches='tight')
plt.close(fig)
print("  → coverage_vs_n.pdf")

# =============================================================================
# FIGURE 2: shape_sensitivity.pdf
# =============================================================================
print("Figure 2: shape_sensitivity …")
xi_vals = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
alpha_s  = 0.05
n_sens   = [100, 500, 1000]
shape_res = {n: [] for n in n_sens}

for xi in tqdm(xi_vals, desc='  xi values'):
    kind = 'gpd_heavy' if xi >= 0 else 'gpd_light'
    cfg_s = DatasetConfig(f'xi={xi}', kind, xi=xi, sigma_gpd=50)
    data_pool = generate_dataset(cfg_s, N=N_POOL, seed=7)
    true_q = get_true_quantile(cfg_s, alpha_s)
    for n_cal in n_sens:
        sfalls = []
        for t in range(N_TRIALS):
            cal, test = split_calibration_test(data_pool, n_cal, N_TEST, seed=t)
            b = evt_gpd(cal, alpha_s)
            sfalls.append(coverage_shortfall(b, test, alpha_s))
        shape_res[n_cal].append({
            'xi': xi,
            'mean': float(np.mean(sfalls)),
            'std':  float(np.std(sfalls)),
        })

fig, ax = plt.subplots(figsize=(5.5, 3.2))
markers = ['o', 's', '^']
lstyles = ['-', '--', ':']
for n_cal, mk, ls in zip(n_sens, markers, lstyles):
    xis   = [r['xi']  for r in shape_res[n_cal]]
    means = [r['mean'] for r in shape_res[n_cal]]
    stds  = [r['std']  for r in shape_res[n_cal]]
    ax.plot(xis, means, color=COLORS['evt'], marker=mk, markersize=4,
            linestyle=ls, label=f'EVT-GPD $n={n_cal}$')
    ax.fill_between(xis, [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                    color=COLORS['evt'], alpha=0.09)

ax.axhline(0, color=COLORS['cp'], lw=1.2, label='CP-pWCET (always 0)')
ax.axvline(0, color='#aaaaaa', lw=0.6, linestyle=':')
ax.set_xlabel('True shape parameter $\\xi$')
ax.set_ylabel('Coverage shortfall (pp)')
ax.set_title(f'EVT shape sensitivity ($\\alpha = {alpha_s}$)')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'shape_sensitivity.pdf'), bbox_inches='tight')
plt.close(fig)
print("  → shape_sensitivity.pdf")

# =============================================================================
# FIGURE 3: efficiency_vs_alpha.pdf
# =============================================================================
print("Figure 3: efficiency_vs_alpha …")
cfg_e    = DatasetConfig('GPD xi=0.2', 'gpd_heavy', xi=0.2, sigma_gpd=50)
data_e   = generate_dataset(cfg_e, N=N_POOL)
alpha_rng = np.logspace(-1, -5.5, 50)
n_eff     = 1000

cp_eff_all, evt_eff_all = [], []
for t in tqdm(range(100), desc='  efficiency trials'):
    cal, _ = split_calibration_test(data_e, n_eff, N_TEST, seed=t)
    cp_b  = np.array([cp_pwcet(cal, a) for a in alpha_rng])
    evt_b = np.array([evt_gpd(cal, a) for a in alpha_rng])
    cp_eff_all.append(cp_b);  evt_eff_all.append(evt_b)

cp_all  = np.array(cp_eff_all)
evt_all = np.array(evt_eff_all)
true_qs = np.array([get_true_quantile(cfg_e, a) for a in alpha_rng])

cp_eff_mean  = np.nanmean(cp_all / true_qs,  axis=0)
evt_eff_mean = np.nanmean(evt_all / true_qs, axis=0)
cp_eff_std   = np.nanstd(cp_all / true_qs,   axis=0)
evt_eff_std  = np.nanstd(evt_all / true_qs,  axis=0)

valid = alpha_rng >= 1.0 / (n_eff + 1)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

ax = axes[0]
ax.plot(alpha_rng[valid], cp_eff_mean[valid],
        color=COLORS['cp'], label='CP-pWCET')
ax.fill_between(alpha_rng[valid],
                (cp_eff_mean - cp_eff_std)[valid],
                (cp_eff_mean + cp_eff_std)[valid],
                color=COLORS['cp'], alpha=0.12)
ax.plot(alpha_rng, evt_eff_mean, color=COLORS['evt'],
        linestyle='--', label='EVT-GPD')
ax.fill_between(alpha_rng,
                evt_eff_mean - evt_eff_std,
                evt_eff_mean + evt_eff_std,
                color=COLORS['evt'], alpha=0.12)
ax.axhline(1.0, color='k', lw=0.6, linestyle=':')
ax.axvline(1.0 / (n_eff + 1), color=COLORS['cp'], lw=0.6,
           linestyle=':', alpha=0.7, label='$1/(n+1)$ limit')
ax.set_xscale('log');  ax.invert_xaxis()
ax.set_xlabel('Target miscoverage $\\alpha$')
ax.set_ylabel('Bound efficiency ($\\hat{C}/q^*_{1-\\alpha}$)')
ax.set_title(f'Efficiency ($n={n_eff}$, GPD $\\xi=0.2$)')
ax.legend(frameon=False, fontsize=8)

# Inset: jointly informative range
ax2 = axes[1]
jmask = (alpha_rng >= 8e-5) & (alpha_rng <= 3e-3) & valid
if jmask.sum() > 2:
    ax2.plot(alpha_rng[jmask], cp_eff_mean[jmask],
             color=COLORS['cp'], label='CP-pWCET')
    ax2.plot(alpha_rng[jmask], evt_eff_mean[jmask],
             color=COLORS['evt'], linestyle='--', label='EVT-GPD')
    ax2.fill_between(alpha_rng[jmask],
                     (cp_eff_mean - cp_eff_std)[jmask],
                     (cp_eff_mean + cp_eff_std)[jmask],
                     color=COLORS['cp'], alpha=0.12)
    ax2.fill_between(alpha_rng[jmask],
                     evt_eff_mean[jmask] - evt_eff_std[jmask],
                     evt_eff_mean[jmask] + evt_eff_std[jmask],
                     color=COLORS['evt'], alpha=0.12)
ax2.axhline(1.0, color='k', lw=0.6, linestyle=':')
ax2.set_xscale('log'); ax2.invert_xaxis()
ax2.set_xlabel('$\\alpha$')
ax2.set_ylabel('Efficiency')
ax2.set_title('Jointly informative range')
ax2.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'efficiency_vs_alpha.pdf'), bbox_inches='tight')
plt.close(fig)
print("  → efficiency_vs_alpha.pdf")

# =============================================================================
# FIGURE 4: cqr_conditional.pdf
# =============================================================================
print("Figure 4: cqr_conditional …")
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_LGB = False

from sklearn.model_selection import train_test_split as sk_split

alpha_cqr = 0.05
X_all, T_all = generate_cqr_dataset(N=15_000)
X_tr, X_rest, T_tr, T_rest = sk_split(X_all, T_all, test_size=7000, random_state=0)
X_cal, X_te, T_cal, T_te = sk_split(X_rest, T_rest, test_size=4000, random_state=1)

if HAS_LGB:
    mdl = lgb.LGBMRegressor(objective='quantile', alpha=1-alpha_cqr,
                              n_estimators=200, learning_rate=0.05,
                              num_leaves=31, verbose=-1)
else:
    mdl = GradientBoostingRegressor(loss='quantile', alpha=1-alpha_cqr,
                                     n_estimators=200, max_depth=4)
mdl.fit(X_tr.reshape(-1, 1), T_tr)
q_cal = mdl.predict(X_cal.reshape(-1, 1))
q_te  = mdl.predict(X_te.reshape(-1, 1))

corr        = cqr_calibrate(T_cal, q_cal, q_cal, alpha_cqr)
cqr_bounds  = cqr_predict(q_te, corr)
cp_unc      = cp_pwcet(T_cal, alpha_cqr)

x_grid = np.linspace(0, 1, 300)
true_q_g = get_true_conditional_quantile(x_grid, alpha_cqr)
cqr_g    = mdl.predict(x_grid.reshape(-1, 1)) + corr

# Per-bin conditional coverage
n_bins = 10
edges = np.linspace(0, 1, n_bins + 1)
bc, cp_c, cqr_c = [], [], []
for i in range(n_bins):
    mask = (X_te >= edges[i]) & (X_te < edges[i+1])
    if mask.sum() < 5: continue
    bc.append(0.5*(edges[i]+edges[i+1]))
    cp_c.append(empirical_coverage(cp_unc, T_te[mask]))
    cqr_c.append(np.mean(T_te[mask] <= cqr_bounds[mask]))

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
})
fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.2))

ax = axes[0]
samp = min(600, len(X_te))
idx  = np.random.default_rng(0).choice(len(X_te), samp, replace=False)
ax.scatter(X_te[idx], T_te[idx], s=3, alpha=0.25, color='#aaaaaa',
           rasterized=True, label='Test samples')
ax.plot(x_grid, true_q_g, 'k--', lw=1.2, label='True $q_{0.95}(x)$')
ax.axhline(cp_unc, color=COLORS['cp'], lw=1.6,
           label=f'CP bound ({cp_unc:.0f}\\,µs)')
ax.plot(x_grid, cqr_g, color=COLORS['aci'], lw=2.2, label='CQR bound')
ax.set_xlabel('Input complexity $x$')
ax.set_ylabel('Execution time (µs)')
ax.set_title('Unconditional CP vs. CQR bounds')
ax.legend(frameon=False, fontsize=11)

ax = axes[1]
ax.plot(bc, cp_c,  color=COLORS['cp'],  marker='o', ms=7,
        label='CP (unconditional)')
ax.plot(bc, cqr_c, color=COLORS['aci'], marker='s', ms=7,
        label='CQR')
ax.axhline(1 - alpha_cqr, color='k', lw=1.2, linestyle='--',
           label='Nominal 95%')
ax.set_xlabel('Input complexity $x$')
ax.set_ylabel('Empirical coverage (per bin)')
ax.set_title('Conditional coverage by input bin')
ax.set_ylim(0.82, 1.02)
ax.legend(frameon=False, fontsize=11)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'cqr_conditional.pdf'), bbox_inches='tight')
plt.close(fig)
print("  → cqr_conditional.pdf")

# =============================================================================
# FIGURE 5: aci_drift.pdf
# =============================================================================
print("Figure 5: aci_drift …")
drift_cfg = DatasetConfig('Drift', 'nonstationary', drift_fraction=0.05)
T_seq     = generate_dataset(drift_cfg, N=10_000)
alpha_aci = 0.05
gammas    = [0.01, 0.05, 0.10]
window    = 500

# Static CP: calibrate on first window, eval on rest
cal_static = T_seq[:window]
cp_static  = cp_pwcet(cal_static, alpha_aci)
static_cov = (T_seq <= cp_static).astype(float)

aci_results = {}
for g in gammas:
    aci = AdaptiveCI(alpha=alpha_aci, gamma=g)
    aci_results[g] = aci.run_on_sequence(T_seq)

def roll(arr, w):
    ret = np.cumsum(arr.astype(float))
    ret[w:] = ret[w:] - ret[:-w]
    return ret[w-1:] / w

x_ax = np.arange(window - 1, len(T_seq))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                 gridspec_kw={'hspace': 0.18})

# Top: rolling coverage
ax1.plot(x_ax, roll(static_cov, window), color=COLORS['cp'],
         lw=2.2, label='Static CP', zorder=3)
lss = ['-', '--', ':']
for g, ls in zip(gammas, lss):
    cov = aci_results[g]['coverage'].astype(float)
    ax1.plot(x_ax, roll(cov, window), color=COLORS['aci'],
             lw=2.0, linestyle=ls, label=f'ACI $\\gamma={g}$')
ax1.axhline(1 - alpha_aci, color='k', lw=0.8, linestyle='--',
            label='Nominal 95%')
ax1.set_ylabel(f'Rolling coverage\n(window = {window})')
ax1.set_ylim(0.78, 1.01)
ax1.set_title('Coverage under distribution shift (drift $d = 0.05\\mu$)')
ax1.legend(frameon=False, fontsize=8, ncol=2)

# Annotate the degradation
ax1.annotate('Static CP\ndegrades', xy=(8500, np.mean(static_cov[8000:9000])),
             xytext=(7000, 0.83),
             arrowprops=dict(arrowstyle='->', color=COLORS['cp'], lw=0.8),
             fontsize=7.5, color=COLORS['cp'])

# Bottom: effective alpha_t for best gamma (0.01)
best_g = 0.01
alpha_t = aci_results[best_g]['alpha_t']
ax2.plot(np.arange(len(alpha_t)), alpha_t, color=COLORS['aci'],
         lw=2.0, label=f'$\\alpha_t$ ($\\gamma={best_g}$)')
ax2.axhline(alpha_aci, color='k', lw=0.8, linestyle='--',
            label=f'Target $\\alpha = {alpha_aci}$')
ax2.set_xlabel('Time step $t$')
ax2.set_ylabel('Effective $\\alpha_t$')
ax2.set_title(f'ACI effective miscoverage level ($\\gamma = {best_g}$)')
ax2.legend(frameon=False, fontsize=11)

fig.savefig(os.path.join(OUT, 'aci_drift.pdf'), bbox_inches='tight')
plt.close(fig)
print("  → aci_drift.pdf")

print(f"\nAll figures written to {os.path.abspath(OUT)}")
print("\nFigure file sizes:")
for f in ['coverage_vs_n.pdf','shape_sensitivity.pdf',
          'efficiency_vs_alpha.pdf','cqr_conditional.pdf','aci_drift.pdf']:
    path = os.path.join(OUT, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {f}: {size/1024:.1f} KB")
    else:
        print(f"  {f}: MISSING")
