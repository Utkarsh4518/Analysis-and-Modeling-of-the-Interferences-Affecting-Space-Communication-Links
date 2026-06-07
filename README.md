# Analysis and Modeling of Radio-Frequency Interference for Space Communication Links

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rfi-space-link-simulator.streamlit.app/)

A Python simulation framework for quantifying Radio Frequency Interference (RFI) on satellite communication links, built on ITU-R recommendation models.

> **Research Project** — Hamburg University of Technology (TUHH), WinSem 2025-26  
> **Author:** Utkarsh Maurya ([utkarsh.maurya@tuhh.de](mailto:utkarsh.maurya@tuhh.de))

---

## Introduction

The increasing density of satellite constellations and terrestrial wireless systems has elevated the risk of radio-frequency interference (RFI) to space communication links. The deployment of large Non-Geostationary Satellite Orbit (NGSO) constellations such as Starlink has intensified aggregate interference scenarios affecting a wide range of space communication services.

Existing RFI studies are often limited to individual frequency bands or specific system configurations. They do not offer a consistent basis for comparing how interference translates into link margin attenuation across multiple space communication bands. This project addresses this gap by developing a unified, ITU-R-aligned modeling framework applicable to the **S-, X-, Ku-, K-, and Ka-bands**, enabling direct comparison under identical interference assumptions.

These bands exhibit distinct physical characteristics: higher-frequency bands support larger bandwidths and higher data rates, while lower-frequency bands offer greater robustness against atmospheric attenuation, rain fading, and pointing errors. A unified framework therefore enables evaluation of how interference impacts systems with fundamentally different propagation and implementation trade-offs.

| Band | Frequency Range |
|------|----------------|
| S-band | 2 – 4 GHz |
| X-band | 8 – 12 GHz |
| Ku-band | 12 – 18 GHz |
| K-band | 18 – 27 GHz |
| Ka-band | 27 – 40 GHz |

---

## ITU-R Standards Implemented

| Standard | Purpose |
|----------|---------|
| **ITU-R P.525** | Free-space path loss |
| **ITU-R S.1528** | Receive antenna off-axis gain pattern (main lobe + side lobe) |
| **ITU-R S.1325** | Aggregate interference / simulation methodologies |
| **ITU-R SA.609** | Protection criteria for space research (near-earth) |
| **ITU-R SA.1157** | Protection criteria for space research (deep space) |

---

## Key Features

- **Multi-band analysis** across S (3 GHz), X (8.4 GHz), Ku (14 GHz), K (22 GHz), and Ka (32 GHz) bands
- **Frequency-sweep mode** for continuous SNR-loss evaluation within a band
- **Distance × Frequency heatmap** generation for two-dimensional vulnerability mapping
- **Interplanetary scenarios** — GEO (36 000 km), Moon (~384 400 km), and Mars (~56 million km)
- **HPBW-based pointing margin** — automatic 30% half-power beamwidth offset applied to the off-axis angle
- **Statistical interference modeling** — log-normal samples with configurable duty cycle
- **Metrics computed:**
  - Baseline and degraded SNR
  - Carrier-to-Interference ratio (C/I)
  - Equivalent Power Flux Density (EPFD)
  - Time-fraction exceedance probabilities (P > 1 dB, P > 3 dB)

---

## Key Results

### Baseline Link Performance

The baseline SNR remains nearly constant across the full 2–40 GHz frequency range. This occurs because the increase in antenna gain with frequency is balanced by the corresponding increase in free-space path loss, leaving the received carrier power approximately unchanged.

### Interference Sensitivity vs. Frequency

Across all three interference scenarios (weak, moderate, strong), SNR degradation **decreases as frequency increases**. This is primarily driven by antenna directivity: for a fixed antenna diameter, higher frequencies produce narrower beams, which suppress off-axis interference more effectively. The frequency-dependent pointing margin reinforces this effect. The trend follows an approximately exponential decay:

```
ΔSNR(f) ≈ a · e^(−b·f)
```

| Scenario | a | b |
|----------|------|-------|
| Weak | 0.336 | 0.218 |
| Moderate | 2.627 | 0.196 |
| Strong | 10.370 | 0.143 |

### Distance-Dependent Behaviour

Interference impact decreases rapidly with increasing propagation distance due to the growth of FSPL:
- **GEO (36 000 km):** Significant SNR degradation under moderate and strong interference, especially at lower frequencies
- **Moon (~384 400 km):** Noticeable degradation under strong interference (losses exceeding 1 dB at lower frequencies); antenna directivity provides moderate protection
- **Mars (~56 million km):** Negligible SNR loss across all frequencies and interference levels — path loss alone reduces interference to insignificant levels, restoring noise-limited behaviour

---

## Conclusions

1. **Interference sensitivity is shaped by antenna directivity and propagation loss.** Higher-frequency systems are better protected due to narrower beamwidths.
2. **SNR degradation decreases smoothly with frequency**, well-approximated by an exponential decay function.
3. **Increased distance further suppresses interference** through free-space path loss. At Martian distances, even strong interference becomes negligible.
4. **The interference-to-noise ratio** (Iagg − N) is the key quantity governing SNR degradation, influenced by both frequency and distance through FSPL and antenna discrimination.

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

| Parameter | Weak | Moderate | Strong |
|-----------|------|----------|--------|
| Interferer EIRP (dBW) | 10 | 20 | 30 |
| Separation distance (km) | 36 000 | 36 000 | 36 000 |
| Nominal off-axis angle (deg) | 0.5 | 0.5 | 0.5 |
| Log-normal std. dev. σ (dB) | 6 | 6 | 6 |
| Duty cycle | 1.0 | 1.0 | 1.0 |

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

## References

1. L. Zhang et al., "Evaluating the influence of satellite systems on terrestrial networks: Analyzing S-band interference," ICSI, 2025.
2. A. Karaki et al., "Investigating Ku-band up-link interference from terrestrial microwaves on satellite reception," IEEE MENACOMM, 2025.
3. E. Polo et al., "Investigating NGSO constellation-induced interference on GSO system ground stations," ESA TTC, 2025.
4. A. Bouleux, "Interference induced by NGSO constellations on GSO systems," M.S. thesis, Politecnico di Milano, 2022.

---

## Author

**Utkarsh Maurya**  
Hamburg University of Technology (TUHH)  
Hamburg, Germany  
[utkarsh.maurya@tuhh.de](mailto:utkarsh.maurya@tuhh.de)
