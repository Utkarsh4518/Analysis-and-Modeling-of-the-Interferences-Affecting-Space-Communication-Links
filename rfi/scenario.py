import numpy as np
from typing import Dict, Any, List

# Import equation-level models (pure physics, no scenarios)
from rfi.equations_itu import (
    compute_thermal_noise_dbw,
    free_space_path_loss_db,
    compute_interference_power_dbw,
    compute_aggregate_interference_dbw,
    compute_epfd_dbw_m2_mhz,
    compute_carrier_to_interference_db,
    compute_snr_with_interference_db,
    compute_off_axis_gain_s1528_db,
    generate_log_normal_interference_samples_dbw,
    compute_time_fraction_exceeded,
)

# Default constants (fixed parameters)

DEFAULT_BW_HZ = 1e6          # 1 MHz receiver bandwidth (all bands)
DEFAULT_L_ATM_DB = 0.0       # Atmospheric attenuation fixed to 0 dB
DEFAULT_L_OTHER_DB = 0.0     # Miscellaneous / polarization losses fixed to 0 dB

# GEO reference distance (victim link and interferer geometry)
DKM_GEO = 36000.0  # km


# Main scenario engine

def run_multi_entry_rfi_scenario(
    band_params: Dict[str, Any],
    interferer_list: List[Dict[str, Any]],
    time_sim_samples: int = 1000,
) -> Dict[str, Any]:

    # 1. Baseline carrier and noise computation

    f_ghz = band_params["f_ghz"]
    d_km = band_params["d_km"]

    T_sys_k = band_params["T_sys_k"]
    BW_hz = band_params.get("BW_Hz", DEFAULT_BW_HZ) #Use the band’s bandwidth if defined, otherwise default to 1 MHz

    # Thermal noise
    N_dbw = compute_thermal_noise_dbw(T_sys_k, BW_hz)

    # Free-space path loss (victim link)
    L_fs_db = free_space_path_loss_db(f_ghz, d_km)

    # Received carrier power
    C_dbw = (
        band_params["EIRP_dbw"]
        + band_params["G_rx_db"]
        - L_fs_db
        - DEFAULT_L_ATM_DB
        - DEFAULT_L_OTHER_DB
    )

    baseline_snr_db = C_dbw - N_dbw

    # 2. Deterministic aggregate interference

    I_single_powers_dbw = []
    epfd_single_values_db = []

    for i_params in interferer_list:

	# free-space path loss from interferer to victim
        L_fs_int_db = free_space_path_loss_db(
            f_ghz, i_params["d_km"]
        )
	
	# how much the victim antenna hears the interferer
        g_rx_off_axis_db = compute_off_axis_gain_s1528_db(
            g_max=band_params["G_rx_db"],
            theta_deg=i_params["theta_off_axis_deg"],
            theta_3db=band_params["theta_3db"],
        )

	# EIRP Computation
        I_single_dbw = compute_interference_power_dbw(
            eirp_int_dbw=i_params["EIRP_int_dbw"],
            l_fs_int_db=L_fs_int_db,
            l_atm_db=DEFAULT_L_ATM_DB,
            g_rx_off_axis_db=g_rx_off_axis_db,
            l_misc_db=DEFAULT_L_OTHER_DB,
        )

	# Add it to the list
        I_single_powers_dbw.append(I_single_dbw)
	
	# equivalent power flux density computation
        epfd_single_db = compute_epfd_dbw_m2_mhz(
            eirp_int_dbw=i_params["EIRP_int_dbw"],
            g_rx_off_axis_db=g_rx_off_axis_db,
            l_fs_int_db=L_fs_int_db,
            bandwidth_mhz=BW_hz / 1e6,
        )

        epfd_single_values_db.append(epfd_single_db)
    
    # total interference power (dbW)
    I_aggregate_dbw = compute_aggregate_interference_dbw(
        I_single_powers_dbw
    )

    epfd_aggregate_db = compute_aggregate_interference_dbw(
        epfd_single_values_db
    )

    C_I_db = compute_carrier_to_interference_db(
        C_dbw, I_aggregate_dbw
    )

    # computation of SNR with Interference 
    SNR_with_I_db = compute_snr_with_interference_db(
        C_dbw, N_dbw, I_aggregate_dbw
    )

    #degraded SNR
    SNR_loss_db = baseline_snr_db - SNR_with_I_db 

    # =============================================================
    # 3. Statistical characterization
    # =============================================================

    # standard deviation(dB) for interference fluctutation
    sigma_db = (
        np.mean([i.get("sigma_db", 4.0) for i in interferer_list])
        if interferer_list else 4.0
    )
    duty_cycle = (
        np.mean([i.get("duty_cycle", 1.0) for i in interferer_list])
        if interferer_list else 1.0
    )

    i_samples_dbw = generate_log_normal_interference_samples_dbw(
        mean_dbw=I_aggregate_dbw,
        std_dev_db=sigma_db,
        num_samples=time_sim_samples,
        duty_cycle=duty_cycle,
    )

    snr_with_i_samples_db = np.array([
        compute_snr_with_interference_db(C_dbw, N_dbw, i_dbw)
        for i_dbw in i_samples_dbw
    ])

    snr_loss_samples_db = baseline_snr_db - snr_with_i_samples_db

    # 4. Results

    return {
        "Baseline SNR (dB)": baseline_snr_db,
        "I_Aggregate (dBW)": I_aggregate_dbw,
        "C/I_Aggregate (dB)": C_I_db,
        "SNR with I_Agg (dB)": SNR_with_I_db,
        "SNR Loss (dB)": SNR_loss_db,
        "epfd_Aggregate (dBW/m2/MHz)": epfd_aggregate_db,

        "P(SNR Loss > 1 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 1.0
        ),
        "P(SNR Loss > 3 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 3.0
        ),
        "P(SNR Loss > 6 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 6.0
        ),

        "SNR_Loss_Samples_dB": snr_loss_samples_db,
    }


# Victim band scenario definitions (Assumptions)

VICTIM_BANDS = {
    "S-band": {
        "f_ghz": 3.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 26.0,
        "G_rx_db": 32.0,
        "theta_3db": 2.5,
        "T_sys_k": 250.0,
        "BW_Hz": DEFAULT_BW_HZ,
    },
    "X-band": {
        "f_ghz": 8.4,
        "d_km": DKM_GEO,
        "EIRP_dbw": 30.0,
        "G_rx_db": 38.0,
        "theta_3db": 1.5,
        "T_sys_k": 300.0,
        "BW_Hz": DEFAULT_BW_HZ,
    },
    "Ku-band": {
        "f_ghz": 14.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 45.0,
        "G_rx_db": 42.0,
        "theta_3db": 1.2,
        "T_sys_k": 350.0,
        "BW_Hz": DEFAULT_BW_HZ,
    },
    "K-band": {
        "f_ghz": 22.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 50.0,
        "G_rx_db": 45.0,
        "theta_3db": 1.0,
        "T_sys_k": 400.0,
        "BW_Hz": DEFAULT_BW_HZ,
    },
    "Ka-band": {
        "f_ghz": 32.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 55.0,
        "G_rx_db": 48.0,
        "theta_3db": 0.8,
        "T_sys_k": 450.0,
        "BW_Hz": DEFAULT_BW_HZ,
    },
}



# Interference strength scenarios

INTERFERENCE_SCENARIOS = {
    "Weak": [
        {
            "EIRP_int_dbw": 10.0, #10 Watts
            "d_km": DKM_GEO,
            "theta_off_axis_deg": 10.0,
            "sigma_db": 6.0,
            "duty_cycle": 1,
        }
    ],
    "Moderate": [
        {
            "EIRP_int_dbw": 20.0, #100 Watts
            "d_km": DKM_GEO,
            "theta_off_axis_deg": 10,
            "sigma_db": 6.0,
            "duty_cycle": 1,
        }
    ],
    "Strong": [
        {
            "EIRP_int_dbw": 30.0, #1000 Watts
            "d_km": DKM_GEO,
            "theta_off_axis_deg": 10,
            "sigma_db": 6.0,
            "duty_cycle": 1.0,
        }
    ],
}

ALL_SCENARIOS = {
    "bands": VICTIM_BANDS,
    "interference": INTERFERENCE_SCENARIOS,
}
