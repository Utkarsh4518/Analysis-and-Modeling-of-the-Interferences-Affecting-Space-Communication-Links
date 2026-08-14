"""
RFI Simulator — Streamlit front-end for the rfi package.

All physics comes from rfi.equations_itu and rfi.scenario;
this file only handles UI, layout, and plotting.
"""

import sys, os, numpy as np, streamlit as st, plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Ensure the rfi package is importable regardless of working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfi.scenario import (
    run_multi_entry_rfi_scenario,
    COMMON_SYSTEM,
    INTERFERENCE_SCENARIOS,
    _compute_antenna_params,
    DEFAULT_BW_HZ,
)
from rfi.equations_itu import (
    free_space_path_loss_db,
    compute_off_axis_gain_s1528_db,
    compute_epfd_dbw_m2_mhz,
)

# ---------------------------------------------------------------------------
# Page configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RFI Simulator",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
    }

    .header-container {
        background: linear-gradient(135deg, rgba(99, 110, 250, 0.1) 0%, rgba(0, 204, 150, 0.05) 100%);
        border: 1px solid rgba(99, 110, 250, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }

    .header-title {
        font-size: 32px;
        font-weight: 700;
        margin: 0 0 8px 0;
        background: linear-gradient(90deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .header-subtitle {
        font-size: 14px;
        color: #9ca3af;
        margin: 0;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 32px;
    }

    .kpi-card {
        background: rgba(22, 27, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 110, 250, 0.5);
        box-shadow: 0 12px 30px rgba(99, 110, 250, 0.15);
    }

    .kpi-label {
        font-size: 13px;
        font-weight: 500;
        color: #9ca3af;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.01em;
    }

    .loss-card {
        border-color: rgba(239, 85, 59, 0.2);
    }
    .loss-card:hover {
        border-color: rgba(239, 85, 59, 0.6);
        box-shadow: 0 12px 30px rgba(239, 85, 59, 0.15);
    }
    .loss-card .kpi-value {
        color: #ff6b6b;
    }

    .epfd-card {
        border-color: rgba(0, 204, 150, 0.2);
    }
    .epfd-card:hover {
        border-color: rgba(0, 204, 150, 0.6);
        box-shadow: 0 12px 30px rgba(0, 204, 150, 0.15);
    }
    .epfd-card .kpi-value {
        color: #2ecc71;
    }

    .sidebar-info {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }

    .sidebar-info-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .sidebar-info-row:last-child {
        margin-bottom: 0;
    }
    .sidebar-info-label {
        color: #9ca3af;
    }
    .sidebar-info-value {
        color: #ffffff;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: transparent;
        border: none;
        color: #9ca3af;
        font-weight: 500;
        font-size: 15px;
        transition: all 0.2s ease;
        padding: 0 16px;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.03);
    }

    .stTabs [aria-selected="true"] {
        color: #636EFA !important;
        background-color: rgba(99, 110, 250, 0.08) !important;
        font-weight: 600;
        border-bottom: 2px solid #636EFA !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Session State Initialization (Phase 4 Presets)
# ---------------------------------------------------------------------------
if "distance_km" not in st.session_state:
    st.session_state["distance_km"] = 36000.0

def set_preset_distance(dist_val):
    st.session_state["distance_km"] = float(dist_val)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📡 Simulation Parameters")
    st.markdown("---")

    f_ghz = st.slider(
        "Frequency (GHz)",
        min_value=2.0,
        max_value=40.0,
        value=14.0,
        step=0.5,
        help="Operating frequency of the victim link.",
    )

    st.markdown("#### Mission Presets")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("🛰 GEO", on_click=set_preset_distance, args=(36000.0,), use_container_width=True)
    with col2:
        st.button("🌙 Moon", on_click=set_preset_distance, args=(384400.0,), use_container_width=True)
    with col3:
        st.button("🔴 Mars", on_click=set_preset_distance, args=(56000000.0,), use_container_width=True)

    d_km = st.slider(
        "Link Distance (km)",
        min_value=1_000.0,
        max_value=60_000_000.0,
        step=1_000.0,
        format="%.0f",
        key="distance_km",
        help="Propagation distance to the victim receiver.",
    )

    scenario_name = st.selectbox(
        "Interference Scenario",
        options=list(INTERFERENCE_SCENARIOS.keys()),
        index=1,
        help="Pre-defined interferer power level.",
    )

    st.markdown("---")
    enable_edu = st.checkbox("🎓 Enable Educational Explanations", value=True, help="Show a plain-language physics explanation beneath each graph.")

    st.markdown("---")
    st.caption("🛰️ Fixed Victim-Link Parameters")
    st.markdown(
        f'''
        <div class="sidebar-info">
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Tx EIRP</span>
                <span class="sidebar-info-value">{COMMON_SYSTEM['EIRP_dbw']:.0f} dBW</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Rx Dish Diameter</span>
                <span class="sidebar-info-value">{COMMON_SYSTEM['D_m']} m</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">System Temp (T_sys)</span>
                <span class="sidebar-info-value">{COMMON_SYSTEM['T_sys_k']:.0f} K</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Bandwidth</span>
                <span class="sidebar-info-value">{COMMON_SYSTEM.get('BW_Hz', DEFAULT_BW_HZ) / 1e6:.0f} MHz</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("📋 Selected Scenario Parameters")
    sel = INTERFERENCE_SCENARIOS[scenario_name][0]
    st.markdown(
        f'''
        <div class="sidebar-info">
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">EIRP</span>
                <span class="sidebar-info-value">{sel['EIRP_int_dbw']:.0f} dBW</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Interferer Distance</span>
                <span class="sidebar-info-value">{sel['d_km']:,.0f} km</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Off-axis Angle</span>
                <span class="sidebar-info-value">{sel['theta_off_axis_deg']}°</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Std Dev (σ)</span>
                <span class="sidebar-info-value">{sel['sigma_db']} dB</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Duty Cycle</span>
                <span class="sidebar-info-value">{sel['duty_cycle']}</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Build band_params
# ---------------------------------------------------------------------------
band_params = {
    "f_ghz": f_ghz,
    "d_km": d_km,
    **COMMON_SYSTEM,
}

interferer_list = INTERFERENCE_SCENARIOS[scenario_name]

# ---------------------------------------------------------------------------
# Run core simulation
# ---------------------------------------------------------------------------
result = run_multi_entry_rfi_scenario(
    band_params=band_params,
    interferer_list=interferer_list,
    time_sim_samples=5_000,
)

# ---------------------------------------------------------------------------
# Compute EPFD
# ---------------------------------------------------------------------------
i_params = interferer_list[0]
G_rx_db, theta_3db = _compute_antenna_params(
    f_ghz, COMMON_SYSTEM["D_m"], COMMON_SYSTEM.get("eta", 0.65)
)
pointing_margin_deg = 0.3 * theta_3db
theta_eff = i_params["theta_off_axis_deg"] + pointing_margin_deg
g_rx_off = compute_off_axis_gain_s1528_db(G_rx_db, theta_eff, theta_3db)
l_fs_int = free_space_path_loss_db(f_ghz, i_params["d_km"])
bandwidth_mhz = COMMON_SYSTEM.get("BW_Hz", DEFAULT_BW_HZ) / 1e6

epfd = compute_epfd_dbw_m2_mhz(
    eirp_int_dbw=i_params["EIRP_int_dbw"],
    g_rx_off_axis_db=g_rx_off,
    l_fs_int_db=l_fs_int,
    bandwidth_mhz=bandwidth_mhz,
)

# ---------------------------------------------------------------------------
# Custom Header
# ---------------------------------------------------------------------------
st.markdown(
    f'''
    <div class="header-container">
        <h1 class="header-title">📡 RFI Link-Budget Simulator</h1>
        <p class="header-subtitle">
            <strong>{scenario_name}</strong> scenario  ·  
            f = {f_ghz:.1f} GHz  ·  
            d = {d_km:,.0f} km
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)

with st.expander("ℹ️ About this simulator — read me first", expanded=True):
    st.markdown(
        """
Satellite links share the spectrum with other transmitters — other satellites, terrestrial
microwave links, even other beams of the same constellation. When one of those signals leaks
into a receiver's passband, it's **Radio-Frequency Interference (RFI)**: it raises the effective
noise floor and eats into the link's Signal-to-Noise Ratio (SNR), even though the wanted signal
itself hasn't changed.

**What this tool does:** it models one "victim" downlink (a fixed ground-station dish receiving
from a satellite) and one interfering transmitter, using ITU-R propagation and antenna models
(**P.525** free-space path loss, **S.1528** off-axis antenna gain). You choose the victim's
frequency, link distance, and how strong the interferer is — the app computes how much SNR the
victim link loses as a result.

**How to use it:**
1. In the sidebar, set the **frequency** and **link distance** (or pick a mission preset — GEO, Moon, Mars).
2. Pick an **interference scenario** (Weak / Moderate / Strong).
3. Explore the same result from five angles in the tabs below — a frequency sweep, a
   distance × frequency heatmap, the antenna's directional pattern, a Monte Carlo statistical
   spread, and the link geometry.

The victim system itself (dish size, transmit power, receiver noise temperature) is fixed — see
**Fixed Victim-Link Parameters** in the sidebar — so that every scenario is directly comparable.
Full derivations, assumptions, and research findings are in the
[project README](https://github.com/Utkarsh4518/Analysis-and-Modeling-of-the-Interferences-Affecting-Space-Communication-Links#readme).
        """
    )

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
kpi_data = [
    ("Baseline SNR", f"{result['Baseline SNR (dB)']:.2f} dB", ""),
    ("Degraded SNR", f"{result['SNR with I_Agg (dB)']:.2f} dB", ""),
    ("SNR Loss", f"{result['SNR Loss (dB)']:.2f} dB", "loss-card"),
    ("C / I", f"{result['C/I_Aggregate (dB)']:.2f} dB", ""),
    ("EPFD", f"{epfd:.2f} dBW/m²/MHz", "epfd-card"),
]

kpi_html = '<div class="kpi-grid">\n'
for label, value, css_class in kpi_data:
    kpi_html += f'<div class="kpi-card {css_class}">\n'
    kpi_html += f'<div class="kpi-label">{label}</div>\n'
    kpi_html += f'<div class="kpi-value">{value}</div>\n'
    kpi_html += '</div>\n'
kpi_html += '</div>'
st.markdown(kpi_html, unsafe_allow_html=True)

with st.expander("❓ What do these metrics mean?"):
    st.markdown(
        """
| Metric | Meaning |
|---|---|
| **Baseline SNR** | Signal-to-Noise Ratio of the victim link with *no* interference present — noise-only degradation. |
| **Degraded SNR** | SNR once the interferer's power is added on top of the thermal noise. |
| **SNR Loss** | `Baseline SNR − Degraded SNR` — how much margin the link gives up to interference. Values under ~1 dB are usually negligible; several dB can threaten the link budget. |
| **C / I** | Carrier-to-Interference ratio — the wanted signal's power relative to the interferer's power alone (ignores thermal noise). |
| **EPFD** | Equivalent Power Flux Density (dBW/m²/MHz) — the interference power density arriving at the receiver, normalized to a 1 MHz bandwidth. Used by ITU-R to set regulatory interference limits. |
        """
    )

# ---------------------------------------------------------------------------
# Caching Functions
# ---------------------------------------------------------------------------
@st.cache_data
def run_frequency_sweep_cached(d, scenario):
    interferers = INTERFERENCE_SCENARIOS[scenario]
    freq_sweep = np.linspace(2.0, 40.0, 80)
    sweep_params = {"d_km": d, **COMMON_SYSTEM}
    
    results = []
    for f in freq_sweep:
        params = sweep_params.copy()
        params["f_ghz"] = f
        # Skip statistical noise generation for pure deterministic plots
        res = run_multi_entry_rfi_scenario(params, interferers, time_sim_samples=0)
        results.append(res)
        
    return freq_sweep, {
        "Baseline SNR (dB)": np.array([r["Baseline SNR (dB)"] for r in results]),
        "SNR Loss (dB)": np.array([r["SNR Loss (dB)"] for r in results]),
    }

@st.cache_data
def run_heatmap_cached(d, scenario):
    interferers = INTERFERENCE_SCENARIOS[scenario]
    freq_grid = np.linspace(2.0, 40.0, 40)
    d_lo = max(1_000.0, d * 0.5)
    d_hi = d * 1.5
    dist_grid = np.linspace(d_lo, d_hi, 30)
    
    heatmap_params = {**COMMON_SYSTEM}
    Z = np.zeros((len(dist_grid), len(freq_grid)))
    for i, dist_val in enumerate(dist_grid):
        for j, f_val in enumerate(freq_grid):
            params = {**heatmap_params, "f_ghz": f_val, "d_km": dist_val}
            res = run_multi_entry_rfi_scenario(params, interferers, time_sim_samples=0)
            Z[i, j] = res["SNR Loss (dB)"]
            
    return freq_grid, dist_grid, Z

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "SNR Loss vs Frequency",
    "Distance vs Frequency Heatmap",
    "Antenna Directivity Explorer",
    "Monte Carlo Analysis",
    "Geometry & Topology",
])

# ---------------------------------------------------------------------------
# Tab 1 — SNR Loss vs Frequency
# ---------------------------------------------------------------------------
with tab1:
    st.caption("How much SNR the victim link loses to interference at every frequency between 2–40 GHz, at the currently selected distance and scenario. The red dot marks your current frequency slider position.")
    freq_sweep, sweep_data = run_frequency_sweep_cached(d_km, scenario_name)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freq_sweep,
            y=sweep_data["SNR Loss (dB)"],
            mode="lines",
            line=dict(color="#636EFA", width=3),
            name="SNR Loss",
            hovertemplate="f = %{x:.1f} GHz<br>ΔSNR = %{y:.4f} dB<extra></extra>",
        )
    )

    current_idx = np.argmin(np.abs(freq_sweep - f_ghz))
    fig.add_trace(
        go.Scatter(
            x=[f_ghz],
            y=[sweep_data["SNR Loss (dB)"][current_idx]],
            mode="markers",
            marker=dict(color="#EF553B", size=12, symbol="circle", line=dict(color="white", width=2)),
            name=f"Selected ({f_ghz} GHz)",
            hovertemplate="f = %{x:.1f} GHz<br>ΔSNR = %{y:.4f} dB<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"SNR Loss vs Frequency — {scenario_name} Interference",
        xaxis_title="Frequency (GHz)",
        yaxis_title="SNR Loss (dB)",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f4f6", family="Outfit"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    )

    st.plotly_chart(fig, width="stretch")

    if enable_edu:
        st.info("💡 **Physics Insight:** As frequency increases, the receiver's antenna beamwidth becomes narrower. This higher directivity allows the antenna to suppress off-axis interference much more effectively, resulting in a dramatic decrease in SNR loss.")

# ---------------------------------------------------------------------------
# Tab 2 — Distance vs Frequency Heatmap
# ---------------------------------------------------------------------------
with tab2:
    st.caption("SNR loss across a grid of frequency and distance, so you can see at a glance where the victim link is most vulnerable. Darker regions (per the colorbar) mean more SNR loss.")
    freq_grid, dist_grid, Z = run_heatmap_cached(d_km, scenario_name)

    fig_hm = go.Figure(
        data=go.Heatmap(
            x=np.round(freq_grid, 1),
            y=np.round(dist_grid, 0),
            z=Z,
            colorscale="Viridis",
            reversescale=True,
            colorbar=dict(title="SNR Loss (dB)"),
            hovertemplate=(
                "Frequency: %{x:.1f} GHz<br>"
                "Distance: %{y:,.0f} km<br>"
                "SNR Loss: %{z:.4f} dB"
                "<extra></extra>"
            ),
        )
    )

    fig_hm.update_layout(
        title=f"SNR Loss vs Frequency & Distance — {scenario_name} Interference",
        xaxis_title="Frequency (GHz)",
        yaxis_title="Distance (km)",
        height=540,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f4f6", family="Outfit"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    )

    st.plotly_chart(fig_hm, width="stretch")

    if enable_edu:
        st.info("💡 **Physics Insight:** The heatmap illustrates the dual protection mechanisms of space communications. Moving horizontally (higher frequency) narrows the antenna beam to reject interference spatially. Moving vertically (greater distance) exponentially weakens the interference power via Free-Space Path Loss (FSPL).")

# ---------------------------------------------------------------------------
# Tab 3 — Antenna Directivity Explorer
# ---------------------------------------------------------------------------
with tab3:
    st.caption("The receiver antenna's gain as a function of off-axis angle (ITU-R S.1528). A narrower beam (higher frequency / larger dish) rejects off-axis interference more strongly — independent of the sidebar scenario.")
    col_left, col_right = st.columns([1, 2.5])

    with col_left:
        st.markdown("<h3 style='margin-bottom: 24px;'>🎛️ Parameters</h3>", unsafe_allow_html=True)
        ant_f_ghz = st.slider(
            "Frequency (GHz)",
            min_value=2.0,
            max_value=40.0,
            value=14.0,
            step=0.5,
            key="ant_freq",
        )
        ant_D_m = st.slider(
            "Antenna Diameter (m)",
            min_value=0.3,
            max_value=5.0,
            value=1.2,
            step=0.1,
            key="ant_diam",
        )

        G_max, theta_3db = _compute_antenna_params(ant_f_ghz, ant_D_m, eta=0.65)
        theta_edge = 2.5 * theta_3db
        suppression = G_max - (G_max - 30.0)

        st.markdown("<h3 style='margin-top: 32px; margin-bottom: 16px;'>📊 Key Metrics</h3>", unsafe_allow_html=True)
        st.markdown(
            f'''
            <div class="kpi-card" style="margin-bottom: 12px; padding: 12px 16px;">
                <div class="kpi-label" style="font-size: 11px;">Peak Gain</div>
                <div class="kpi-value" style="font-size: 20px;">{G_max:.2f} dBi</div>
            </div>
            <div class="kpi-card" style="margin-bottom: 12px; padding: 12px 16px; border-color: rgba(239, 85, 59, 0.3);">
                <div class="kpi-label" style="font-size: 11px;">HPBW (θ₃dB)</div>
                <div class="kpi-value" style="font-size: 20px; color: #ff6b6b;">{theta_3db:.2f}°</div>
            </div>
            <div class="kpi-card" style="padding: 12px 16px; border-color: rgba(0, 204, 150, 0.3);">
                <div class="kpi-label" style="font-size: 11px;">Off-axis Suppression</div>
                <div class="kpi-value" style="font-size: 20px; color: #2ecc71;">{suppression:.1f} dB</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col_right:
        theta_max = max(10.0, 5.0 * theta_3db)
        angles = np.linspace(0.0, theta_max, 500)
        gains = np.array([
            compute_off_axis_gain_s1528_db(G_max, th, theta_3db)
            for th in angles
        ])

        fig_ant = go.Figure()
        
        fig_ant.add_trace(
            go.Scatter(
                x=angles,
                y=gains,
                mode="lines",
                line=dict(color="#636EFA", width=3),
                name="Antenna Gain",
                hovertemplate="θ = %{x:.2f}°<br>G = %{y:.2f} dBi<extra></extra>",
            )
        )

        g_3db = G_max - 3.0
        fig_ant.add_trace(
            go.Scatter(
                x=[theta_3db / 2, -theta_3db / 2] if theta_3db > 0 else [],
                y=[g_3db, g_3db],
                mode="markers",
                marker=dict(color="#EF553B", size=10, symbol="diamond"),
                name=f"−3 dB (HPBW = {theta_3db:.2f}°)",
                hovertemplate="Half-power point<br>θ = %{x:.2f}°<br>G = %{y:.2f} dBi<extra></extra>",
            )
        )

        fig_ant.add_hline(
            y=g_3db, line_dash="dash", line_color="#EF553B",
            annotation_text=f"−3 dB ({g_3db:.1f} dBi)", annotation_position="top right",
            annotation_font=dict(color="#f3f4f6")
        )
        fig_ant.add_hline(
            y=G_max - 30.0, line_dash="dot", line_color="#00CC96",
            annotation_text=f"Side-lobe floor ({G_max - 30.0:.1f} dBi)", annotation_position="bottom right",
            annotation_font=dict(color="#f3f4f6")
        )
        fig_ant.add_vline(
            x=theta_edge, line_dash="dash", line_color="#AB63FA",
            annotation_text=f"Edge ({theta_edge:.2f}°)", annotation_position="top right",
            annotation_font=dict(color="#f3f4f6")
        )

        fig_ant.update_layout(
            title=f"Antenna Gain Pattern — {ant_f_ghz} GHz, D = {ant_D_m} m",
            xaxis_title="Off-axis Angle θ (degrees)",
            yaxis_title="Gain (dBi)",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f3f4f6", family="Outfit"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_ant, width="stretch")
        
        if enable_edu:
            st.info("💡 **Physics Insight:** This pattern visualizes the ITU-R S.1528 model. Increasing frequency or antenna diameter physically shrinks the Half-Power Beamwidth (HPBW). A narrower beam forces off-axis interference to fall outside the main lobe and into the highly suppressed side-lobe floor, protecting the link.")

# ---------------------------------------------------------------------------
# Tab 4 — Monte Carlo Analysis
# ---------------------------------------------------------------------------
with tab4:
    st.caption("Interference isn't constant — it fluctuates with weather, pointing errors, and transmitter behavior. This runs 5,000 log-normal samples around the aggregate interference level to show the real spread of SNR loss, not just its average.")
    col_left, col_right = st.columns([1, 2.5])

    with col_left:
        st.markdown("<h3 style='margin-bottom: 24px;'>📊 Statistical Metrics</h3>", unsafe_allow_html=True)
        samples = result.get("SNR Loss Samples", np.array([]))
        
        if len(samples) > 0:
            mean_val = np.mean(samples)
            median_val = np.median(samples)
            p95_val = np.percentile(samples, 95)
        else:
            mean_val = median_val = p95_val = 0.0
            
        p1_val = result["P(SNR Loss > 1 dB) (%)"]
        p3_val = result["P(SNR Loss > 3 dB) (%)"]

        st.markdown(
            f'''
            <div class="kpi-card" style="margin-bottom: 12px; padding: 12px 16px;">
                <div class="kpi-label" style="font-size: 11px;">Mean SNR Loss</div>
                <div class="kpi-value" style="font-size: 20px;">{mean_val:.4f} dB</div>
            </div>
            <div class="kpi-card" style="margin-bottom: 12px; padding: 12px 16px;">
                <div class="kpi-label" style="font-size: 11px;">Median SNR Loss</div>
                <div class="kpi-value" style="font-size: 20px;">{median_val:.4f} dB</div>
            </div>
            <div class="kpi-card" style="margin-bottom: 12px; padding: 12px 16px; border-color: rgba(239, 85, 59, 0.3);">
                <div class="kpi-label" style="font-size: 11px;">95th Percentile</div>
                <div class="kpi-value" style="font-size: 20px; color: #ff6b6b;">{p95_val:.4f} dB</div>
            </div>
            <div class="kpi-card" style="margin-bottom: 12px; padding: 12px 16px;">
                <div class="kpi-label" style="font-size: 11px;">P(loss > 1 dB)</div>
                <div class="kpi-value" style="font-size: 20px;">{p1_val:.2f}%</div>
            </div>
            <div class="kpi-card" style="padding: 12px 16px;">
                <div class="kpi-label" style="font-size: 11px;">P(loss > 3 dB)</div>
                <div class="kpi-value" style="font-size: 20px;">{p3_val:.2f}%</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col_right:
        fig_mc = go.Figure()
        
        if len(samples) > 0:
            fig_mc.add_trace(
                go.Histogram(
                    x=samples,
                    nbinsx=60,
                    marker_color="#636EFA",
                    opacity=0.8,
                    name="SNR Loss Distribution",
                    hovertemplate="SNR Loss: %{x} dB<br>Count: %{y}<extra></extra>"
                )
            )
            
            fig_mc.add_vline(
                x=p95_val, line_dash="dash", line_color="#EF553B",
                annotation_text=f"95th Pct ({p95_val:.3f} dB)", annotation_position="top right",
                annotation_font=dict(color="#f3f4f6")
            )
        
        fig_mc.update_layout(
            title="Monte Carlo Distribution of SNR Loss (5000 samples)",
            xaxis_title="SNR Loss (dB)",
            yaxis_title="Frequency",
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f3f4f6", family="Outfit"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
            bargap=0.05
        )
        st.plotly_chart(fig_mc, width="stretch")
        
        if enable_edu:
            st.info("💡 **Physics Insight:** Real-world interference fluctuates over time due to weather, pointing errors, and moving transmitters. By running thousands of Monte Carlo simulations (log-normal sampling), we uncover the true statistical distribution of the SNR loss, revealing the worst-case risks (95th percentile) that simple averages hide.")

# ---------------------------------------------------------------------------
# Tab 5 — Geometry Visualization
# ---------------------------------------------------------------------------
with tab5:
    st.caption("A 2D top-down sketch of the ground station, the target satellite, and the interferer, drawn to the selected distance and off-axis angle — a sanity check on the geometry behind the numbers above.")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("<h3 style='margin-bottom: 24px;'>📐 Spatial Configuration</h3>", unsafe_allow_html=True)
        if interferer_list:
            interferer = interferer_list[0]
            d_int = interferer.get("d_km", 36000.0)
            theta_deg = interferer.get("theta_off_axis_deg", 0.0)
        else:
            d_int = d_km
            theta_deg = 0.0
            
        st.markdown(
            f'''
            <div class="kpi-card" style="margin-bottom: 12px; padding: 12px 16px;">
                <div class="kpi-label" style="font-size: 11px;">Target Satellite Distance</div>
                <div class="kpi-value" style="font-size: 20px;">{d_km:,.0f} km</div>
            </div>
            <div class="kpi-card" style="margin-bottom: 12px; padding: 12px 16px; border-color: rgba(239, 85, 59, 0.3);">
                <div class="kpi-label" style="font-size: 11px;">Interferer Distance</div>
                <div class="kpi-value" style="font-size: 20px; color: #ff6b6b;">{d_int:,.0f} km</div>
            </div>
            <div class="kpi-card" style="margin-bottom: 12px; padding: 12px 16px;">
                <div class="kpi-label" style="font-size: 11px;">Off-Axis Angle (θ)</div>
                <div class="kpi-value" style="font-size: 20px;">{theta_deg:.2f}°</div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        
    with col_right:
        theta_rad = np.radians(theta_deg)
        x_int = d_int * np.sin(theta_rad)
        y_int = d_int * np.cos(theta_rad)
        
        fig_geom = go.Figure()
        
        # Main Link Line
        fig_geom.add_trace(go.Scatter(
            x=[0, 0], y=[0, d_km],
            mode="lines",
            name="Main Link",
            line=dict(color="#636EFA", width=3, dash="solid")
        ))
        
        # Interferer Link Line
        fig_geom.add_trace(go.Scatter(
            x=[0, x_int], y=[0, y_int],
            mode="lines",
            name="Interference Link",
            line=dict(color="#EF553B", width=2, dash="dash")
        ))
        
        # Markers
        fig_geom.add_trace(go.Scatter(
            x=[0], y=[0],
            mode="markers+text",
            name="Ground Station",
            marker=dict(symbol="square", size=16, color="#00CC96"),
            text=["Ground Station"], textposition="bottom center",
            textfont=dict(color="#00CC96")
        ))
        
        fig_geom.add_trace(go.Scatter(
            x=[0], y=[d_km],
            mode="markers+text",
            name="Target Satellite",
            marker=dict(symbol="star", size=20, color="#636EFA"),
            text=["Satellite"], textposition="top center",
            textfont=dict(color="#636EFA")
        ))
        
        fig_geom.add_trace(go.Scatter(
            x=[x_int], y=[y_int],
            mode="markers+text",
            name="Interferer",
            marker=dict(symbol="cross", size=16, color="#EF553B"),
            text=["Interferer"], textposition="top center",
            textfont=dict(color="#EF553B")
        ))
        
        max_dist = max(d_km, d_int)
        x_margin = max(abs(x_int) * 1.5, max_dist * 0.1)
        
        fig_geom.update_layout(
            title="Link Geometry Visualization (2D Projection)",
            showlegend=False,
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f3f4f6", family="Outfit"),
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)",
                range=[-x_margin, x_margin], showticklabels=False
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)",
                range=[-max_dist * 0.1, max_dist * 1.2], showticklabels=False
            ),
            hovermode="closest"
        )
        st.plotly_chart(fig_geom, width="stretch")
        
        if enable_edu:
            st.info("💡 **Physics Insight:** This spatial projection visualizes the geometric relationship between the Ground Station, the Target Satellite, and the Interferer. Even a small off-axis angle (θ) can mean thousands of kilometers of physical separation at orbital altitudes, which is why highly directional antennas are effective at rejecting noise.")

# ---------------------------------------------------------------------------
# Educational Reference Expander
# ---------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📖 Physical Principles & ITU-R Modeling Reference"):
    st.markdown(
        """
        ### 1. Free-Space Path Loss (FSPL) — ITU-R P.525
        The propagation loss between the transmitter and receiver is given by:
        $$L_{FS} = 32.45 + 20 \\log_{10}(f_{MHz}) + 20 \\log_{10}(d_{km})$$
        As frequency ($f$) increases, the path loss increases quadratically. However, for a directional antenna of fixed physical aperture area, the gain also increases quadratically, which balances the path loss.

        ### 2. Antenna Gain Pattern — ITU-R S.1528
        The off-axis gain pattern of the receiver antenna determines its ability to reject interference from other directions. For a peak gain $G_{max}$ and half-power beamwidth $\\theta_{3dB}$, the gain at off-axis angle $\\theta$ is:
        - **Main Lobe** ($\\theta < 2.5 \\theta_{3dB}$):
          $$G(\\theta) = G_{max} - 12 \\left(\\frac{\\theta}{\\theta_{3dB}}\\right)^2$$
        - **Side Lobe Floor** ($\\theta \\ge 2.5 \\theta_{3dB}$):
          $$G(\\theta) = G_{max} - 30$$
        
        Where the Half-Power Beamwidth (HPBW) is approximated as:
        $$\\theta_{3dB} \\approx 70 \\frac{\\lambda}{D_m}$$
        
        ### 3. Key Insight: High Frequency Robustness
        Because the antenna beamwidth is inversely proportional to frequency ($\\theta_{3dB} \\propto 1/f$), higher-frequency antennas have narrower beams. This narrow beamwidth results in:
        1. A smaller main lobe.
        2. Faster roll-off of the main-lobe gain.
        3. A smaller pointing margin angle.
        
        Consequently, the receiver provides **greater spatial discrimination** against off-axis interferers, resulting in **significantly lower SNR loss** at higher frequencies (such as Ka-band) compared to lower frequencies (such as S-band).
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; padding: 20px 0; border-top: 1px solid rgba(255,255,255,0.08); color:#9ca3af; font-size:13px;">
        Analysis and Modeling of RFI for Space Communication Links · Research Project, Hamburg University of Technology (TUHH), WinSem 2025-26<br>
        <strong>Utkarsh Maurya</strong> ·
        <a href="mailto:utkarsh.maurya@tuhh.de" style="color:#a5b4fc;">utkarsh.maurya@tuhh.de</a> ·
        <a href="https://github.com/Utkarsh4518/Analysis-and-Modeling-of-the-Interferences-Affecting-Space-Communication-Links" style="color:#a5b4fc;" target="_blank">GitHub Repository</a>
    </div>
    """,
    unsafe_allow_html=True
)
