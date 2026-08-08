# Conformal Prediction for Probabilistic Worst-Case Execution Time

Reproducibility package for *Conformal Prediction for Probabilistic Worst-Case
Execution Time: Distribution-Free Coverage Guarantees* (revised version, Journal
of Systems Architecture).

This repository contains the code, data, and firmware needed to reproduce every
experiment in the paper, including the material added during the major revision:
the strengthened Extreme Value Theory (EVT) baseline, the Hyndman--Fan
quantile-estimator comparison, the extrapolation/misspecification study, and the
STM32F746 Cortex-M7 hardware campaign.

## Layout

```
.
├── cp_methods.py               Core conformal-prediction methods (split CP, weighted CP, CQR, ACI)
├── data_generator.py           Synthetic execution-time generators (GPD, mixture, drift, lognormal)
├── run_experiments.py          Original synthetic experiment suite (Experiments 1–7)
├── run_experiments_v2.py       Revision driver: reruns the suite under the strengthened EVT baseline
├── generate_figures.py         Figures 1–6 (coverage, sensitivity, small-n, efficiency, CQR, ACI)
├── generate_real_data_figure.py  Figure 7 (real-data validation)
├── real_timing_tasks.npz       Measured inference-latency traces used in Experiment 7
│
├── evt_improved.py             Strengthened EVT: MBPTA-CV + sequential Anderson–Darling (ForwardStop)
│                               threshold selection; delta-method and profile-likelihood CIs
├── evt_compat.py               Drop-in replacement wiring the strengthened EVT into cp_methods
├── quantile_estimators.py      Hyndman–Fan types 1–9, Wilks tolerance bound, calibration-size formulas
│
├── exp_r1_evt_fair.py          Experiment 2 recast: fixed-90th vs MBPTA-CV threshold ablation
├── exp_r1_quantile_types.py    Experiment 8: comparison against Hyndman–Fan sample quantiles
├── exp_r1_extrapolation.py     Experiment 9: validity under extrapolation and misspecification
├── make_extrapolation_figure.py  Figure 8 (extrapolation validity)
│
├── analyze_benchmark_traces.py Coverage analysis of the measured Cortex-M7 traces
├── make_hw_figure.py           Figure 9 (hardware benchmark distributions and tails)
│
└── hardware/                   STM32F746ZG Cortex-M7 measurement campaign (Experiment 10)
    ├── firmware/               Bare-metal measurement firmware (arm-none-eabi-gcc)
    │   ├── meas_main11.c        Three-kernel campaign (bsort, insertsort, ludcmp), 921600 baud
    │   ├── meas_main12.c        Quicksort-only campaign
    │   ├── startup_fault.c      Startup + fault handler (reports #FAULT rather than hanging)
    │   ├── link.ld              Linker script
    │   ├── k/                   TACLeBench kernels 
    │   ├── measure11.bin        Prebuilt three-kernel binary
    │   └── measure12.bin        Prebuilt quicksort binary
    ├── capture/
    │   └── capture_v4.ps1       Serial capture (921600 baud, transmit-complete paced)
    ├── traces/                  Measured cycle-count traces (6 configurations)
    │   ├── bsort_cache_{on,off}.csv
    │   ├── insertsort_cache_{on,off}.csv
    │   └── ludcmp_cache_{on,off}.csv
    └── results/
        └── bench_3kernel.json  Coverage results for all six traces
```

## Reproducing the synthetic experiments

```bash
pip install -r requirements.txt

# Experiments 1–7 under the strengthened EVT baseline
python run_experiments_v2.py

# Experiment 2 threshold ablation (fixed-90th vs MBPTA-CV)
python exp_r1_evt_fair.py

# Experiment 8 (Hyndman–Fan quantile comparison)
python exp_r1_quantile_types.py

# Experiment 9 (extrapolation / misspecification)
python exp_r1_extrapolation.py

# Figures
python generate_figures.py
python make_extrapolation_figure.py
python generate_real_data_figure.py
```

## Reproducing the hardware experiment (Experiment 10)

Requires an STMicroelectronics NUCLEO-F746ZG board (Cortex-M7) and
`arm-none-eabi-gcc`.

```bash
# Build (or use the prebuilt .bin files)
cd hardware/firmware
arm-none-eabi-gcc -mcpu=cortex-m7 -mthumb -mfloat-abi=hard \
  -mfpu=fpv5-sp-d16 -O2 -ffreestanding -Ik -nostdlib -T link.ld \
  startup_fault.c meas_main11.c k/*.o -lgcc -o measure11.elf
arm-none-eabi-objcopy -O binary measure11.elf measure11.bin
```

1. Drag `measure11.bin` onto the `NODE_F746ZG` drive.
2. Capture the serial stream:
   ```powershell
   powershell -ExecutionPolicy Bypass -File ../capture/capture_v4.ps1 COM5 traces
   ```
3. Analyse:
   ```bash
   python analyze_benchmark_traces.py "hardware/traces/*.csv" \
     --n-cal 10000 --alpha 0.05 --trials 25 --out hardware/results/bench_3kernel.json
   python make_hw_figure.py
   ```

### Notes on the hardware setup

- The FPU is single-precision only (`fpv5-sp-d16`); double-precision math is
  emulated in software, which is why `ludcmp` with caches disabled shows a
  heavy tail (ξ̂ ≈ +0.26) while the other traces are bounded.
- The `ludcmp` cache-off trace ends at 161,502 activations (an ST-LINK virtual
  COM-port transfer terminated the stream early); the retained samples are
  unaffected.
- 921600 baud with a transmit-complete wait after each sample is required to
  prevent ST-LINK buffer overruns during sustained capture.

## Data availability

The measured traces (`hardware/traces/`) and inference-latency data
(`real_timing_tasks.npz`) are included in this repository.
