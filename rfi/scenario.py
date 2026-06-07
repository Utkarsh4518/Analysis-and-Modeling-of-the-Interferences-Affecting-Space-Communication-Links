import numpy as np
from typing import Dict, Any, List

from rfi.equations_itu import (
    compute_thermal_noise_dbw,
    free_space_path_loss_db,
    compute_interference_power_dbw,
    compute_aggregate_interference_dbw,
    compute_carrier_to_interference_db,
    compute_snr_with_interference_db,
    compute_off_axis_gain_s1528_db,
    generate_log_normal_interference_samples_dbw,
    compute_time_fraction_exceeded,
)

# --------------------------------Constants--------------------------------

DEFAULT_BW_HZ = 1e6
DEFAULT_L_OTHER_DB = 0.0
DKM_GEO = 36000.0

# --------------------------------Common System--------------------------------

COMMON_SYSTEM = {
    "EIRP_dbw": 40.0,
    "T_sys_k": 300.0,
    "D_m": 1.2,
    "eta": 0.65,
    "BW_Hz": DEFAULT_BW_HZ,
}

# --------------------------------Antenna--------------------------------

def _compute_antenna_params(f_ghz: float, D_m: float, eta: float = 0.65):
    """
    Compute antenna gain (dB) and 3 dB beamwidth (degrees)
    from frequency, diameter and efficiency.
    """
    c = 2.998e8
    wavelength = c / (f_ghz * 1e9)

    G_lin = eta * (np.pi * D_m / wavelength) ** 2
    G_db = 10.0 * np.log10(G_lin)

    # Approximate 3 dB beamwidth (degrees)
    theta_3db = 70.0 * wavelength / D_m   #HPBW implementation

    return G_db, theta_3db

# --------------------------------Validation--------------------------------

def _validate_band_params(band_params: Dict[str, Any]) -> None:
    required = ("f_ghz", "d_km", "T_sys_k", "EIRP_dbw", "D_m")
    missing = [k for k in required if k not in band_params]
    if missing:
        raise ValueError(f"band_params missing required keys: {missing}")

def _validate_interferer(i_params: Dict[str, Any]) -> None:
    required = ("EIRP_int_dbw", "d_km", "theta_off_axis_deg")
    missing = [k for k in required if k not in i_params]
    if missing:
        raise ValueError(f"interferer_list entry missing required keys: {missing}")

# --------------------------------Core Scenario--------------------------------

def run_multi_entry_rfi_scenario(
    band_params: Dict[str, Any],
    interferer_list: List[Dict[str, Any]],
    time_sim_samples: int = 1000,
) -> Dict[str, Any]:
    """
    Core RFI scenario:
    - Computes baseline SNR,
    - Aggregate interference power,
    - C/I,
    - SNR with interference,
    - SNR loss and probabilities of exceeding given loss thresholds.

    Now includes a *deterministic pointing margin* on the receive side:
    theta_used = theta_off_axis_deg + pointing_margin_deg.
    """

    _validate_band_params(band_params)
    for ip in interferer_list:
        _validate_interferer(ip)

    f_ghz = band_params["f_ghz"]
    d_km = band_params["d_km"]
    D_m = band_params["D_m"]
    eta = band_params.get("eta", 0.65)

    T_sys_k = band_params["T_sys_k"]
    BW_hz = band_params.get("BW_Hz", DEFAULT_BW_HZ)

    # Antenna
    G_rx_db, theta_3db = _compute_antenna_params(f_ghz, D_m, eta)
    
    # NEW: band-dependent pointing margin as fraction of HPBW
    hpbw_fraction = 0.3  # 30% of HPBW
    pointing_margin_deg = hpbw_fraction * theta_3db  # Available globally now
    theta_nom_default = 0.5  # fallback nominal angle
    
    # Noise
    N_dbw = compute_thermal_noise_dbw(T_sys_k, BW_hz)

    # Path loss
    L_fs_db = free_space_path_loss_db(f_ghz, d_km)

    # Atmospheric loss
    L_atm_db = 0.0

    # Carrier (received carrier power)
    C_dbw = (
        band_params["EIRP_dbw"]
        + G_rx_db
        - L_fs_db
        - L_atm_db
        - DEFAULT_L_OTHER_DB
    )

    baseline_snr_db = C_dbw - N_dbw

    # ---------------- Interference ----------------

    I_single_powers_dbw = []

    for i_params in interferer_list:

        L_fs_int_db = free_space_path_loss_db(f_ghz, i_params["d_km"])

        # Use interferer nominal angle or default
        theta_nom = i_params.get("theta_off_axis_deg", theta_nom_default)
        
        # Effective angle = nominal + pointing margin (computed globally) #HPBW implementation
        theta_used = theta_nom + pointing_margin_deg

        g_rx_off_axis_db = compute_off_axis_gain_s1528_db(
            g_max=G_rx_db,
            theta_deg=theta_used,
            theta_3db=theta_3db,
        )

        I_single_dbw = compute_interference_power_dbw(
            eirp_int_dbw=i_params["EIRP_int_dbw"],
            l_fs_int_db=L_fs_int_db,
            l_atm_db=L_atm_db,
            g_rx_off_axis_db=g_rx_off_axis_db,
            l_misc_db=DEFAULT_L_OTHER_DB,
        )

        I_single_powers_dbw.append(I_single_dbw)

    I_aggregate_dbw = compute_aggregate_interference_dbw(I_single_powers_dbw)

    C_I_db = compute_carrier_to_interference_db(C_dbw, I_aggregate_dbw)

    SNR_with_I_db = compute_snr_with_interference_db(
        C_dbw, N_dbw, I_aggregate_dbw
    )

    SNR_loss_db = baseline_snr_db - SNR_with_I_db

    if abs(baseline_snr_db) < 1e-6:
        SNR_loss_pct = 0.0
    else:
        SNR_loss_pct = (SNR_loss_db / baseline_snr_db) * 100.0

    # ---------------- Statistical ----------------

    if interferer_list:
        sigma_db = np.mean([i.get("sigma_db", 6.0) for i in interferer_list])
        duty_cycle = np.mean([i.get("duty_cycle", 1.0) for i in interferer_list])
    else:
        sigma_db = 6.0
        duty_cycle = 1.0

    i_samples_dbw = generate_log_normal_interference_samples_dbw(
        mean_dbw=I_aggregate_dbw,
        std_dev_db=sigma_db,
        num_samples=time_sim_samples,
        duty_cycle=duty_cycle,
    )

    snr_with_i_samples_db = compute_snr_with_interference_db(
        C_dbw, N_dbw, i_samples_dbw
    )

    snr_loss_samples_db = baseline_snr_db - snr_with_i_samples_db

    return {
        "Baseline SNR (dB)": baseline_snr_db,
        "I_Aggregate (dBW)": I_aggregate_dbw,
        "C/I_Aggregate (dB)": C_I_db,
        "SNR with I_Agg (dB)": SNR_with_I_db,
        "SNR Loss (dB)": SNR_loss_db,
        "SNR Loss (%)": SNR_loss_pct,
        "P(SNR Loss > 1 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 1.0
        ),
        "P(SNR Loss > 3 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 3.0
        ),
        "SNR Loss Samples": snr_loss_samples_db,
        "Pointing Margin (deg)": pointing_margin_deg,
        "HPBW 3dB (deg)": theta_3db,
        "Effective Theta (deg)": theta_used if 'theta_used' in locals() else np.nan,
    }

# --------------------------------Frequency Sweep--------------------------------

def run_frequency_sweep_rfi_scenario(
    base_params: Dict[str, Any],
    interferer_list: List[Dict[str, Any]],
    freq_values_ghz: np.ndarray,
):

    results = []

    for f in freq_values_ghz:
        params = base_params.copy()
        params["f_ghz"] = f

        res = run_multi_entry_rfi_scenario(params, interferer_list)
        results.append(res)

    return {
        "Baseline SNR (dB)": np.array([r["Baseline SNR (dB)"] for r in results]),
        "SNR Loss (dB)": np.array([r["SNR Loss (dB)"] for r in results]),
    }

# --------------------------------Heatmap--------------------------------

def run_distance_frequency_heatmap(
    base_params: Dict[str, Any],
    interferer_list: List[Dict[str, Any]],
    freq_grid: np.ndarray,
    distance_grid: np.ndarray,
):

    Z = np.zeros((len(distance_grid), len(freq_grid)))

    for i, d in enumerate(distance_grid):
        for j, f in enumerate(freq_grid):

            params = base_params.copy()
            params["f_ghz"] = f
            params["d_km"] = d

            res = run_multi_entry_rfi_scenario(params, interferer_list)

            Z[i, j] = res["SNR Loss (%)"]

    return Z

# --------------------------------Bands--------------------------------

VICTIM_BANDS = {
    "S-band": {"f_ghz": 3.0, "d_km": DKM_GEO, **COMMON_SYSTEM},
    "X-band": {"f_ghz": 8.4, "d_km": DKM_GEO, **COMMON_SYSTEM},
    "Ku-band": {"f_ghz": 14.0, "d_km": DKM_GEO, **COMMON_SYSTEM},
    "K-band": {"f_ghz": 22.0, "d_km": DKM_GEO, **COMMON_SYSTEM},
    "Ka-band": {"f_ghz": 32.0, "d_km": DKM_GEO, **COMMON_SYSTEM},
}

# --------------------------------Interference Scenarios--------------------------------


INTERFERENCE_SCENARIOS = {
    "Weak": [{
        "EIRP_int_dbw": 10.0,
        "d_km": DKM_GEO,
        "theta_off_axis_deg": 0.5,
        "sigma_db": 6.0,
        "duty_cycle": 1.0,
    }],
    "Moderate": [{
        "EIRP_int_dbw": 20.0,
        "d_km": DKM_GEO,
        "theta_off_axis_deg": 0.5,
        "sigma_db": 6.0,
        "duty_cycle": 1.0,
    }],
    "Strong": [{
        "EIRP_int_dbw": 30.0,
        "d_km": DKM_GEO,
        "theta_off_axis_deg": 0.5,
        "sigma_db": 6.0,
        "duty_cycle": 1.0,
    }],
}

# --------------------------------Frequency Points--------------------------------

FREQUENCY_SWEEP_POINTS = {
    "S-band": np.array([2.0, 4.0]),
    "X-band": np.array([8.0, 10.0]),
    "Ku-band": np.array([12.0, 14.0, 16.0]),
    "K-band": np.array([18.0, 20.0, 22.0, 24.0]),
    "Ka-band": np.array([26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0]),
}