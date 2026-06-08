"""
collect_timing_data.py
----------------------
Collects real wall-clock inference timing measurements for:
  Task A: Random Forest classifier (100 trees, 20 features)
  Task B: Gradient Boosted Machine regressor (100 estimators)

Saves real_timing_tasks.npz in the same directory, ready for
generate_real_data_figure.py.

Usage:
    pip install scikit-learn numpy
    python collect_timing_data.py

"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification, make_regression
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_WARMUP      = 200    # warmup calls discarded (JIT, cache warm-up)
N_MEASURE     = 5000   # timing measurements to keep per task
BATCH         = 1      # inference calls per timing measurement (keep at 1)
N_TRAIN       = 2000   # training set size
N_FEATURES_A  = 20     # Task A: RF classifier features
N_FEATURES_B  = 15     # Task B: GBM regressor features
RF_TREES      = 100    # Task A: number of trees
GBM_ESTIMATORS = 100   # Task B: number of boosting stages
SEED          = 42

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'real_timing_tasks.npz')

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Task A: Random Forest classifier
# ---------------------------------------------------------------------------
print("=" * 60)
print("Task A: Random Forest classifier")
print(f"  {RF_TREES} trees, {N_FEATURES_A} features, single-sample inference")
print("=" * 60)

X_train_A, y_train_A = make_classification(
    n_samples=N_TRAIN, n_features=N_FEATURES_A,
    n_informative=10, n_redundant=5,
    random_state=SEED)

rf = RandomForestClassifier(
    n_estimators=RF_TREES,
    max_depth=None,
    random_state=SEED,
    n_jobs=1)          # single-threaded for deterministic timing
rf.fit(X_train_A, y_train_A)

# Generate test samples
X_test_A = rng.standard_normal((N_WARMUP + N_MEASURE, N_FEATURES_A))

print(f"  Warming up ({N_WARMUP} calls)...", end='', flush=True)
for i in range(N_WARMUP):
    _ = rf.predict_proba(X_test_A[i:i+1])
print(" done")

print(f"  Measuring ({N_MEASURE} calls)...", end='', flush=True)
task_A_ns = np.empty(N_MEASURE, dtype=np.float64)
for i in range(N_MEASURE):
    t0 = time.perf_counter_ns()
    _ = rf.predict_proba(X_test_A[N_WARMUP + i : N_WARMUP + i + 1])
    task_A_ns[i] = time.perf_counter_ns() - t0
    if (i + 1) % 1000 == 0:
        print(f" {i+1}", end='', flush=True)
print(" done")

mean_A = task_A_ns.mean() / 1e3
std_A  = task_A_ns.std()  / 1e3
print(f"  Mean: {mean_A:.1f} µs   Std: {std_A:.1f} µs   "
      f"Min: {task_A_ns.min()/1e3:.1f} µs   Max: {task_A_ns.max()/1e3:.1f} µs")

# ---------------------------------------------------------------------------
# Task B: Gradient Boosted Machine regressor
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Task B: Gradient Boosted Machine regressor")
print(f"  {GBM_ESTIMATORS} estimators, {N_FEATURES_B} features, "
      "single-sample inference")
print("=" * 60)

X_train_B, y_train_B = make_regression(
    n_samples=N_TRAIN, n_features=N_FEATURES_B,
    n_informative=10, noise=10.0,
    random_state=SEED)

gbm = GradientBoostingRegressor(
    n_estimators=GBM_ESTIMATORS,
    max_depth=4,
    learning_rate=0.05,
    random_state=SEED)
gbm.fit(X_train_B, y_train_B)

X_test_B = rng.standard_normal((N_WARMUP + N_MEASURE, N_FEATURES_B))

# Vary active feature count to simulate real-world input variability
# (some features sparse/zero, mimicking cache-cold fraction)
n_active = rng.integers(1, N_FEATURES_B + 1, N_WARMUP + N_MEASURE)

print(f"  Warming up ({N_WARMUP} calls)...", end='', flush=True)
for i in range(N_WARMUP):
    x = X_test_B[i:i+1].copy()
    x[0, n_active[i]:] = 0.0   # zero out inactive features
    _ = gbm.predict(x)
print(" done")

print(f"  Measuring ({N_MEASURE} calls)...", end='', flush=True)
task_B_ns = np.empty(N_MEASURE, dtype=np.float64)
for i in range(N_MEASURE):
    x = X_test_B[N_WARMUP + i : N_WARMUP + i + 1].copy()
    x[0, n_active[N_WARMUP + i]:] = 0.0
    t0 = time.perf_counter_ns()
    _ = gbm.predict(x)
    task_B_ns[i] = time.perf_counter_ns() - t0
    if (i + 1) % 1000 == 0:
        print(f" {i+1}", end='', flush=True)
print(" done")

mean_B = task_B_ns.mean() / 1e3
std_B  = task_B_ns.std()  / 1e3
print(f"  Mean: {mean_B:.1f} µs   Std: {std_B:.1f} µs   "
      f"Min: {task_B_ns.min()/1e3:.1f} µs   Max: {task_B_ns.max()/1e3:.1f} µs")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
n_active_kept = n_active[N_WARMUP:].astype(np.int64)

np.savez(OUT,
         task_A    = task_A_ns,
         task_B    = task_B_ns,
         n_active  = n_active_kept)

print()
print("=" * 60)
print(f"Saved {N_MEASURE} measurements per task to:")
print(f"  {OUT}")
print()
print("Summary (values in nanoseconds as stored):")
print(f"  task_A:   mean={task_A_ns.mean():.0f} ns  "
      f"std={task_A_ns.std():.0f} ns")
print(f"  task_B:   mean={task_B_ns.mean():.0f} ns  "
      f"std={task_B_ns.std():.0f} ns")
print(f"  n_active: min={n_active_kept.min()}  max={n_active_kept.max()}")
print()
print("Next step: python generate_real_data_figure.py")
print("=" * 60)