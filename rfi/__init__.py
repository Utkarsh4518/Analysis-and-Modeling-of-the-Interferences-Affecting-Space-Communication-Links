"""RFI simulation package — ITU-based interference modeling."""

__all__ = [
    # equations_itu
    "compute_thermal_noise_dbw",
    "free_space_path_loss_db",
    "compute_off_axis_gain_s1528_db",
    "compute_interference_power_dbw",
    "compute_aggregate_interference_dbw",
    "compute_carrier_to_interference_db",
    "compute_snr_with_interference_db",
    "compute_epfd_dbw_m2_mhz",
    "compute_time_fraction_exceeded",
    "generate_log_normal_interference_samples_dbw",
    "generate_geometric_sweep",
    # scenario
    "run_multi_entry_rfi_scenario",
    "run_frequency_sweep_rfi_scenario",
    "VICTIM_BANDS",
    "INTERFERENCE_SCENARIOS",
    "FREQUENCY_SWEEP_POINTS",
    #"ALL_SCENARIOS",
]

from rfi.equations_itu import (
    compute_thermal_noise_dbw,
    free_space_path_loss_db,
    compute_off_axis_gain_s1528_db,
    compute_interference_power_dbw,
    compute_aggregate_interference_dbw,
    compute_carrier_to_interference_db,
    compute_snr_with_interference_db,
    compute_epfd_dbw_m2_mhz,
    compute_time_fraction_exceeded,
    generate_log_normal_interference_samples_dbw,
    generate_geometric_sweep,
)

from rfi.scenario import (
    run_multi_entry_rfi_scenario,
    run_frequency_sweep_rfi_scenario,
    VICTIM_BANDS,
    INTERFERENCE_SCENARIOS,
    FREQUENCY_SWEEP_POINTS,
    #ALL_SCENARIOS,
)
