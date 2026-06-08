"""generate_real_data_figure.py — real-data validation figure for the paper."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from cp_methods import cp_pwcet, evt_gpd, empirical_coverage, AdaptiveCI
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import genpareto

OUT = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
C   = {'cp':'#2166ac', 'evt':'#d6604d', 'aci':'#4dac26', 'nom':'#000000'}
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'lines.linewidth': 1.4,
})

# Load real timing data
d = np.load(os.path.join(os.path.dirname(__file__), 'real_timing_tasks.npz'))
task_A = d['task_A'] / 1e3   # microseconds
task_B = d['task_B'] / 1e3   # microseconds

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5),
                          gridspec_kw={'hspace': 0.48, 'wspace': 0.38})

# ------------------------------------------------------------------ #
# Top-left: Task B tail histogram + GPD fit
# ------------------------------------------------------------------ #
ax = axes[0][0]
u   = np.quantile(task_B, 0.90)
exc = task_B[task_B > u] - u
xi, _, sigma = genpareto.fit(exc, floc=0)
x_fit   = np.linspace(0, np.percentile(exc, 99.5), 300)
pdf_fit = genpareto.pdf(x_fit, xi, scale=sigma)

ax.hist(exc, bins=40, density=True, color='#aaaaaa', alpha=0.6,
        label='Observed exceedances')
ax.plot(x_fit, pdf_fit, color=C['evt'], lw=1.8,
        label=r'GPD fit ($\xi=0.62$, heavy)')
ax.set_xlabel(r'Excess over threshold ($\mu$s)')
ax.set_ylabel('Density')
ax.set_title('Task B: heavy-tail GPD fit')
ax.legend(frameon=False, fontsize=8)

# ------------------------------------------------------------------ #
# Top-right: CP vs EVT coverage on Task B across alpha levels
# ------------------------------------------------------------------ #
ax      = axes[0][1]
rng     = np.random.default_rng(42)
alphas  = [0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
n_cal   = 500
n_test  = 2000
N_TRIALS = 80

cp_covs  = {a: [] for a in alphas}
evt_covs = {a: [] for a in alphas}

for trial in range(N_TRIALS):
    idx  = rng.permutation(len(task_B))
    cal  = task_B[idx[:n_cal]]
    test = task_B[idx[n_cal:n_cal + n_test]]
    for a in alphas:
        cp_covs[a].append(empirical_coverage(cp_pwcet(cal, a), test))
        evt_covs[a].append(empirical_coverage(evt_gpd(cal, a),  test))

cp_means  = [np.mean(cp_covs[a])  for a in alphas]
evt_means = [np.mean(evt_covs[a]) for a in alphas]
nominal   = [1 - a                for a in alphas]

x = np.arange(len(alphas)); w = 0.26
ax.bar(x - w, cp_means,  w, color=C['cp'],  alpha=0.85, label='CP-pWCET')
ax.bar(x,     evt_means, w, color=C['evt'], alpha=0.85, label='EVT-GPD')
ax.bar(x + w, nominal,   w, color='#bbbbbb', alpha=0.85, label='Nominal')
ax.set_xticks(x)
ax.set_xticklabels([str(a) for a in alphas])
ax.set_xlabel(r'Target $\alpha$')
ax.set_ylabel('Empirical coverage')
ax.set_title(r'Task B: coverage by method ($n=500$)')
ax.set_ylim(0.86, 1.01)
ax.legend(frameon=False, fontsize=7.5)

# ------------------------------------------------------------------ #
# Bottom-left: Ljung-Box p-values for Task A
# ------------------------------------------------------------------ #
ax   = axes[1][0]
lb   = acorr_ljungbox(task_A, lags=20, return_df=True)
lags = lb.index.values
pv   = lb['lb_pvalue'].values
bar_colors = [C['evt'] if p < 0.05 else '#aaaaaa' for p in pv]
ax.bar(lags, -np.log10(np.maximum(pv, 1e-10)), color=bar_colors, alpha=0.85)
ax.axhline(-np.log10(0.05), color='k', lw=0.8, linestyle='--', label='p=0.05')
ax.set_xlabel('Lag')
ax.set_ylabel(r'$-\log_{10}(p)$')
ax.set_title('Task A: Ljung-Box autocorrelation')
sig = (pv < 0.05).sum()
ymax = max(-np.log10(np.maximum(pv, 1e-10)))
ax.text(11, ymax * 0.88,
        f'{sig}/20 lags\nsignificant', fontsize=8.5,
        color=C['evt'], ha='center')
ax.legend(frameon=False, fontsize=8)

# ------------------------------------------------------------------ #
# Bottom-right: ACI vs static CP on Task A rolling coverage
# ------------------------------------------------------------------ #
ax     = axes[1][1]
ALPHA  = 0.05
WINDOW = 300
N_SEQ  = 3000

aci     = AdaptiveCI(alpha=ALPHA, gamma=0.01)
aci_res = aci.run_on_sequence(task_A[:N_SEQ])

cp_static_b = cp_pwcet(task_A[:WINDOW], ALPHA)
static_cov  = (task_A[:N_SEQ] <= cp_static_b).astype(float)

def roll(a, w):
    r = np.cumsum(a.astype(float))
    r[w:] = r[w:] - r[:-w]
    return r[w - 1:] / w

x_ax = np.arange(WINDOW - 1, N_SEQ)
ax.plot(x_ax, roll(static_cov, WINDOW),
        color=C['cp'], lw=1.4, label='Static CP')
ax.plot(x_ax, roll(aci_res['coverage'].astype(float), WINDOW),
        color=C['aci'], lw=1.4, label=r'ACI ($\gamma=0.01$)')
ax.axhline(1 - ALPHA, color='k', lw=0.8, linestyle='--', label='Nominal 95%')
ax.set_xlabel('Measurement index')
ax.set_ylabel(f'Rolling coverage (w={WINDOW})')
ax.set_title('Task A: ACI under autocorrelated timing')
ax.set_ylim(0.78, 1.02)
ax.legend(frameon=False, fontsize=8)

# Print summary numbers for the paper
print('=== Real-data results ===')
print(f'Task A  Static CP  init  coverage: {np.mean(static_cov[:500]):.3f}')
print(f'Task A  Static CP  final coverage: {np.mean(static_cov[2500:]):.3f}')
print(f'Task A  ACI g=0.01 mean  coverage: {np.mean(aci_res["coverage"]):.3f}')
print(f'Task B  GPD xi: {xi:.3f}')
for a in alphas:
    cm, em, nm = np.mean(cp_covs[a]), np.mean(evt_covs[a]), 1-a
    cs, es = np.std(cp_covs[a]), np.std(evt_covs[a])
    print(f'  alpha={a}: CP={cm:.3f}+/-{cs:.3f}  EVT={em:.3f}+/-{es:.3f}  nom={nm:.2f}')

fig.suptitle('Real-data validation: classifier inference timing'
             r' ($n = 5{,}000$ measurements per task)',
             fontsize=9.5, y=1.01)
fig.savefig(os.path.join(OUT, 'real_data_validation.pdf'), bbox_inches='tight')
plt.close(fig)
print('Saved real_data_validation.pdf')
