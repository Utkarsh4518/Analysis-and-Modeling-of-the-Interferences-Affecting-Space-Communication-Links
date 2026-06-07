# Analysis and Modeling of the Interferences Affecting Space Communication Links

A Python simulation framework for quantifying Radio Frequency Interference (RFI) on satellite communication links, built on ITU-R recommendation models.

> **Research Project** — Hamburg University of Technology (TUHH), WinSem 2025-26

---

## Overview

This project models how interference from external transmitters degrades the signal quality of a victim GEO satellite link. It implements a complete link-budget chain — from free-space propagation and antenna patterns to aggregate interference, SNR degradation, and EPFD — across five frequency bands (S, X, Ku, K, Ka).

A frequency-sweep analysis mode enables continuous evaluation of interference impact across an entire band, producing SNR-loss curves that reveal frequency-dependent vulnerability.

---

## ITU-R Standards Implemented

| Standard | Purpose |
|----------|---------|
| **ITU-R P.525** | Free-space path loss |
| **ITU-R S.1528** | Receive antenna off-axis gain pattern (main lobe + side lobe) |
| **ITU-R S.1325** | Aggregate interference methodology |
| **EPFD** | Equivalent Power Flux Density for regulatory compliance |

---

## Key Features

- **Multi-band analysis** across S (3 GHz), X (8.4 GHz), Ku (14 GHz), K (22 GHz), and Ka (32 GHz) bands
- **Frequency-sweep mode** for continuous SNR-loss evaluation within a band
- **HPBW-based pointing margin** — automatic 30% half-power beamwidth offset applied to off-axis angle
- **Statistical interference modeling** — log-normal samples with configurable duty cycle
- **Distance × Frequency heatmap** generation for two-dimensional vulnerability mapping
- **Metrics computed:**
  - Baseline and degraded SNR
  - Carrier-to-Interference ratio (C/I)
  - Equivalent Power Flux Density (EPFD)
  - Time-fraction exceedance probabilities (P > 1 dB, P > 3 dB)

---

## Project Structure

The workflow flows: **`Project_Equations.tex`** → **`rfi/equations_itu.py`** → **`rfi/scenario.py`** → **`notebooks/`**

```
rfi-model/
├── Project_Equations.tex            # LaTeX source of all model equations
├── rfi/
│   ├── __init__.py                  # Package exports
│   ├── equations_itu.py             # ITU-R equation implementations
│   └── scenario.py                  # Scenario engine, band configs, sweep runner
├── notebooks/
│   └── RFI-FreqSweep Analysis.ipynb # Frequency-sweep analysis and plots
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Utkarsh4518/Analysis-and-Modeling-of-the-Interferences-Affecting-Space-Communication-Links.git
cd Analysis-and-Modeling-of-the-Interferences-Affecting-Space-Communication-Links
pip install -r requirements.txt
```

---

## Usage

### Run a single-band scenario

```python
from rfi.scenario import run_multi_entry_rfi_scenario, VICTIM_BANDS, INTERFERENCE_SCENARIOS

result = run_multi_entry_rfi_scenario(
    band_params=VICTIM_BANDS["Ka-band"],
    interferer_list=INTERFERENCE_SCENARIOS["Moderate"],
    time_sim_samples=5000,
)

print(f"Baseline SNR : {result['Baseline SNR (dB)']:.2f} dB")
print(f"SNR Loss      : {result['SNR Loss (dB)']:.2f} dB")
print(f"P(loss > 1 dB): {result['P(SNR Loss > 1 dB) (%)']:.1f}%")
```

### Run a frequency sweep

```python
from rfi.scenario import run_frequency_sweep_rfi_scenario, VICTIM_BANDS, INTERFERENCE_SCENARIOS, FREQUENCY_SWEEP_POINTS
import numpy as np

result = run_frequency_sweep_rfi_scenario(
    base_params=VICTIM_BANDS["Ka-band"],
    interferer_list=INTERFERENCE_SCENARIOS["Strong"],
    freq_values_ghz=FREQUENCY_SWEEP_POINTS["Ka-band"],
)

print(result["SNR Loss (dB)"])
```

---

## Interference Scenarios

Three pre-configured scenarios model increasing threat levels:

| Scenario | EIRP (dBW) | Description |
|----------|-----------|-------------|
| **Weak** | 10 | Low-power interferer |
| **Moderate** | 20 | Medium-power interferer |
| **Strong** | 30 | High-power interferer |

All scenarios use GEO distance (36 000 km), 0.5° off-axis angle, 6 dB log-normal spread, and 100% duty cycle.

---

## Equations Reference

All equations are formally documented in [`Project_Equations.tex`](Project_Equations.tex) with derivations and variable definitions. The LaTeX document covers:

1. Free-space path loss (P.525)
2. Received carrier power (link budget)
3. Off-axis antenna gain (S.1528)
4. Single-entry and aggregate interference
5. Thermal noise and SNR (baseline + degraded)
6. SNR loss due to RFI
7. Equivalent Power Flux Density (EPFD)

---

## Author

**Utkarsh Maurya**
Hamburg University of Technology (TUHH)
