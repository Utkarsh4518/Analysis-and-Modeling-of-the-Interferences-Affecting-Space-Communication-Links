import numpy as np


# 1) noise ------------------------------------------------------------

def compute_thermal_noise_dbw(T_sys_k, BW_hz):
    """Thermal noise power (ITU-R, kTB). Returns dBW."""
    if T_sys_k is None or BW_hz is None:
        raise ValueError("T_sys_k and BW_hz must not be None")
    if T_sys_k <= 0 or BW_hz <= 0:
        raise ValueError("T_sys_k and BW_hz must be positive")
    kb = 1.380649e-23  # J/K
    n = kb * T_sys_k * BW_hz
    if n <= 0:
        return -300
    with np.errstate(divide='ignore'):
        x = 10 * np.log10(n)
    return x


def free_space_path_loss_db(f_ghz, d_km):
    """Free-space path loss (ITU-R P.525). f in GHz, d in km. Returns dB."""
    if f_ghz is None or d_km is None:
        raise ValueError("f_ghz and d_km must not be None")
    if f_ghz <= 0 or d_km <= 0:
        raise ValueError("f_ghz and d_km must be positive")
    f_mhz = f_ghz * 1e3
    fspl = 32.45 + 20 * np.log10(f_mhz) + 20 * np.log10(d_km)
    return fspl


# 2) antenna pattern ---------------------------------------------------

def compute_off_axis_gain_s1528_db(g_max, theta_deg, theta_3db):
    """Off-axis antenna gain (ITU-R S.1528 simplified). Returns dBi."""
    if theta_3db == 0:
        return g_max

    theta_edge = 2.5 * theta_3db

    if theta_deg < theta_edge:
        # main lobe
        tmp = (theta_deg / theta_3db) ** 2
        g = g_max - 12.0 * tmp
    else:
        # side lobe floor
        g = g_max - 30.0

    return g


# 3) interference stuff ------------------------------------------------

def compute_interference_power_dbw(
        eirp_int_dbw,
        l_fs_int_db,
        l_atm_db,
        g_rx_off_axis_db,
        l_misc_db=0.0):
    """Single-entry interference power. I = EIRP - Lfs - Latm + G_rx - Lmisc. Returns dBW."""
    i = eirp_int_dbw - l_fs_int_db - l_atm_db + g_rx_off_axis_db - l_misc_db
    return i


def compute_aggregate_interference_dbw(i_powers_dbw):
    """Aggregate (sum) multiple interference powers in linear domain. Returns dBW."""
    if i_powers_dbw is None or len(i_powers_dbw) == 0:
        return -300.0

    # Vectorized linear sum
    arr = np.asarray(i_powers_dbw, dtype=np.float64)
    lin = np.power(10.0, arr / 10.0)
    s = float(np.sum(lin))

    if s <= 1e-30:
        return -300.0

    return 10.0 * np.log10(s)


def compute_carrier_to_interference_db(c_dbw, i_dbw):
    """C/I ratio in dB."""
    return c_dbw - i_dbw


def compute_snr_with_interference_db(c_dbw, n_dbw, i_dbw):
    """SNRI = 10*log10(C / (N + I)). Accepts scalar or array i_dbw."""
    c_lin = 10 ** (c_dbw / 10.0)
    n_lin = 10 ** (n_dbw / 10.0)
    i_arr = np.asarray(i_dbw, dtype=np.float64)
    i_lin = np.power(10.0, i_arr / 10.0)

    denom = n_lin + i_lin
    # Guard against zero denominator
    with np.errstate(divide='ignore', invalid='ignore'):
        snr_lin = np.where(denom <= 0, 10 ** (999.0 / 10.0), c_lin / denom)
        snr_db = 10.0 * np.log10(snr_lin)

    # Return scalar if input was scalar
    if np.ndim(i_dbw) == 0:
        return float(snr_db)
    return snr_db


# 4) EPFD --------------------------------------------------------------

def compute_epfd_dbw_m2_mhz(
        eirp_int_dbw,
        g_rx_off_axis_db,
        l_fs_int_db,
        bandwidth_mhz):
    """Equivalent Power Flux Density (EPFD). Returns dBW/m²/MHz."""
    if bandwidth_mhz <= 0:
        return -999.0

    epfd = eirp_int_dbw - l_fs_int_db + g_rx_off_axis_db \
           - 10.0 * np.log10(bandwidth_mhz)
    return epfd


# 5) stats / time-fraction ---------------------------------------------

def compute_time_fraction_exceeded(data_samples_db, threshold_db):
    """Fraction (%) of samples exceeding a threshold."""
    arr = np.array(data_samples_db)
    if arr.size == 0:
        return 0.0
    frac = np.mean(arr > threshold_db) * 100.0
    return frac


def generate_log_normal_interference_samples_dbw(
        mean_dbw,
        std_dev_db,
        num_samples,
        duty_cycle=1.0):
    """Log-normal interference samples (Gaussian in dB domain). Returns dBW array."""
    if num_samples <= 0:
        return np.array([])

    s = np.random.normal(mean_dbw, std_dev_db, int(num_samples))

    if duty_cycle < 1.0:
        # crude duty-cycle model
        m = np.random.rand(s.size) < duty_cycle
        s = np.where(m, s, -300.0)

    return s


# 6) geometric sweep ---------------------------------------------------

def generate_geometric_sweep(max_theta_deg, min_theta_deg, num_steps=100):
    """Cosine-shaped angular sweep. Returns angle array in degrees."""
    if num_steps <= 1:
        return np.array([min_theta_deg])

    t = np.linspace(-1.0, 1.0, num_steps)
    sweep = 0.5 * (1.0 + np.cos(np.pi * t))
    out = min_theta_deg + sweep * (max_theta_deg - min_theta_deg)
    return out