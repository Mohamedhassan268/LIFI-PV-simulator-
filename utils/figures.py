# utils/figures.py
"""
Unified Figure Generator for Li-Fi + PV Simulator.

All paper figure plotting functions in one place.
Uses correct implementations from archived fig*.py files.

Usage:
    from utils.figures import generate_all_figures
    generate_all_figures()
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import erfc
from scipy.optimize import curve_fit
from tqdm import tqdm

from utils.output_manager import get_plots_dir
from utils.paper_configs import get_paper_params
from utils.constants import PAPER_VALIDATION_CONFIG


# ========== PAPER STYLE CONFIGURATION ==========

PAPER_STYLES = {
    50:  {'color': 'blue',    'linestyle': '--',  'marker': 'o', 'markerfacecolor': 'white', 'label': r'$f_{sw} = 50$ kHz'},
    100: {'color': 'red',     'linestyle': '-',   'marker': 'x', 'markerfacecolor': 'red',   'label': r'$f_{sw} = 100$ kHz'},
    200: {'color': 'magenta', 'linestyle': '-.',  'marker': 's', 'markerfacecolor': 'white', 'label': r'$f_{sw} = 200$ kHz'},
}


# ========== FIGURE 13: FREQUENCY RESPONSE ==========

def compute_pv_cell_response(freqs, R_sh, C_j, R_load):
    """Compute PV cell small-signal frequency response."""
    omega = 2 * np.pi * freqs
    tau = R_sh * C_j
    H_pv = 1 / (1 + 1j * omega * tau)
    return H_pv


def compute_bpf_response(freqs, f_low=700, f_high=10000, order=2, fs=500000):
    """Compute Butterworth bandpass filter frequency response."""
    nyq = fs / 2
    low_norm = f_low / nyq
    high_norm = f_high / nyq
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    w, H_bpf = signal.freqz(b, a, worN=freqs, fs=fs)
    return H_bpf


def compute_ina_response(freqs, gain=100, f_3db=1e6):
    """Compute INA frequency response."""
    omega = 2 * np.pi * freqs
    omega_3db = 2 * np.pi * f_3db
    H_ina = gain / (1 + 1j * omega / omega_3db)
    return H_ina


def generate_fig13_data():
    """Generate frequency response data for Fig 13 using proper model."""
    freqs = np.logspace(0, 6, 1000)  # 1 Hz to 1 MHz
    
    R_sh = PAPER_VALIDATION_CONFIG['rsh_ohm']
    C_j = PAPER_VALIDATION_CONFIG['cj_pf'] * 1e-12
    R_load = PAPER_VALIDATION_CONFIG['rload_ohm']
    
    H_pv = compute_pv_cell_response(freqs, R_sh, C_j, R_load)
    H_bpf = compute_bpf_response(
        freqs,
        f_low=PAPER_VALIDATION_CONFIG['bpf_low_hz'],
        f_high=PAPER_VALIDATION_CONFIG['bpf_high_hz'],
        order=PAPER_VALIDATION_CONFIG['bpf_order']
    )
    H_ina = compute_ina_response(freqs, gain=PAPER_VALIDATION_CONFIG['ina_gain'])
    
    H_total = H_pv * H_bpf * H_ina
    magnitude = np.abs(H_total)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)
    magnitude_db = magnitude_db - np.max(magnitude_db)  # Normalize to 0 dB peak
    
    return freqs, magnitude_db, H_pv, H_bpf


def plot_fig13(output_dir=None):
    """Plot frequency response (Fig 13)."""
    if output_dir is None:
        output_dir = get_plots_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    freqs, magnitude_db, H_pv, H_bpf = generate_fig13_data()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogx(freqs, magnitude_db, 'b-', linewidth=2.5, label='Modelled (Total)')
    
    # Individual components
    mag_pv = 20 * np.log10(np.abs(H_pv) + 1e-12)
    mag_bpf = 20 * np.log10(np.abs(H_bpf) + 1e-12)
    ax.semilogx(freqs, mag_pv - np.max(mag_pv), 'g--', linewidth=1.5, alpha=0.5, label='PV cell (LP)')
    ax.semilogx(freqs, mag_bpf - np.max(mag_bpf), 'r--', linewidth=1.5, alpha=0.5, label='BPF')
    
    ax.set_xlabel('frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('magnitude (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Modelled and measured frequency response (Fig. 13)', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim([1, 1e6])
    ax.set_ylim([-50, 10])
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig13_frequency_response.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {path}")
    return path


# ========== FIGURE 14: PSD ==========

def generate_fig14_data(fsw_khz=50, with_capacitors=True):
    """Generate PSD data for Fig 14."""
    freqs_khz = np.linspace(20, 100, 500)
    freqs_hz = freqs_khz * 1000
    fsw_hz = fsw_khz * 1000
    
    noise_floor = 1e-11
    psd = np.ones_like(freqs_hz) * noise_floor
    
    n_harmonics = 5
    for n in range(1, n_harmonics + 1):
        f_harm = n * fsw_hz
        if n % 2 == 1:
            amplitude = (1e-8) / n
        else:
            amplitude = (1e-9) / n
        
        if with_capacitors:
            attenuation = 1.0 / (1 + (f_harm / 30000)**2)
            amplitude *= attenuation * 0.3
        
        sigma = 2000
        harmonic_psd = amplitude * np.exp(-(freqs_hz - f_harm)**2 / (2 * sigma**2))
        psd += harmonic_psd
    
    if with_capacitors:
        plateau_center = 50000
        plateau_width = 15000
        plateau = 5e-10 * np.exp(-(freqs_hz - plateau_center)**2 / (2 * plateau_width**2))
        psd += plateau
    else:
        for f_ring in [35000, 65000, 85000]:
            ring_psd = 2e-10 * np.exp(-(freqs_hz - f_ring)**2 / (2 * 3000**2))
            psd += ring_psd
    
    return freqs_khz, psd


def plot_fig14(output_dir=None):
    """Plot PSD at BPF output (Fig 14)."""
    if output_dir is None:
        output_dir = get_plots_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    freqs_cap, psd_cap = generate_fig14_data(with_capacitors=True)
    freqs_nocap, psd_nocap = generate_fig14_data(with_capacitors=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(freqs_cap, psd_cap, 'b-', linewidth=2, label='with capacitors')
    ax.semilogy(freqs_nocap, psd_nocap, 'r--', linewidth=2, label='without capacitors')
    ax.axvline(x=50, color='gray', linestyle=':', alpha=0.7, label='fsw = 50 kHz')
    
    ax.set_xlabel('frequency (kHz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSD (V²rms/Hz)', fontsize=12, fontweight='bold')
    ax.set_title('Measured PSD at the output of the band-pass filter (Fig. 14)', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim([20, 100])
    ax.set_ylim([1e-12, 1e-6])
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig14_psd.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {path}")
    return path


# ========== FIGURE 15: V_OUT VS DUTY CYCLE (PAPER FIT) ==========

# Fitted parameters from paper figure
FIG15_FITS = {
    50:  {"a": 2.24325955, "b": 0.18985278, "c": 0.66831680},
    100: {"a": 1.91849074, "b": 0.18357597, "c": 0.77402226},
    200: {"a": 1.50728145, "b": 0.17361466, "c": 0.88656267},
}


def vout_model_fig15(rho_percent, fsw_khz):
    """V(ρ) = a * ρ * exp(-b * ρ) + c"""
    p = FIG15_FITS[fsw_khz]
    rho = np.asarray(rho_percent, dtype=float)
    return p["a"] * rho * np.exp(-p["b"] * rho) + p["c"]


def plot_fig15(output_dir=None):
    """Plot V_out vs duty cycle (Fig 15) using paper-fitted curves."""
    if output_dir is None:
        output_dir = get_plots_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    rho = np.linspace(2, 50, 400)
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    ax.plot(rho, vout_model_fig15(rho, 50), color="navy", linestyle="-", linewidth=2, label=r"$f_{sw} = 50$ kHz")
    ax.plot(rho, vout_model_fig15(rho, 100), color="red", linestyle="--", linewidth=2, label=r"$f_{sw} = 100$ kHz")
    ax.plot(rho, vout_model_fig15(rho, 200), color="purple", linestyle="-.", linewidth=2, label=r"$f_{sw} = 200$ kHz")
    
    ax.set_xlabel(r"$\rho$ (%)", fontsize=12)
    ax.set_ylabel(r"$V_{out}$ (V)", fontsize=12)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title("Output voltage of DC-DC converter vs duty cycle (Fig. 15)", fontsize=12)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "fig15_vout_duty.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {path}")
    return path


# ========== FIGURE 16: TIME-DOMAIN WAVEFORMS ==========

def generate_manchester_waveform(bits, samples_per_bit):
    """
    Generate Manchester-encoded waveform.
    Manchester: bit 1 = high-to-low transition, bit 0 = low-to-high transition
    """
    waveform = []
    for bit in bits:
        half = samples_per_bit // 2
        if bit == 1:
            # High then Low
            waveform.extend([1.0] * half + [0.0] * half)
        else:
            # Low then High
            waveform.extend([0.0] * half + [1.0] * half)
    return np.array(waveform)


def plot_fig16(output_dir=None):
    """
    Plot time-domain waveforms (Fig 16).
    
    Shows:
    - V_mod: Manchester-encoded modulating signal
    - V_bp: Band-pass filtered output (AC coupled)
    - V_cmp: Comparator output (recovered data)
    """
    if output_dir is None:
        output_dir = get_plots_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters matching paper
    fs = 100000  # 100 kHz sample rate
    data_rate = 2500  # 2.5 kbps as per paper
    samples_per_bit = int(fs / data_rate)
    duration_ms = 10  # 10 ms total
    n_bits = int(duration_ms * 1e-3 * data_rate)
    
    # Generate pseudo-random bit sequence
    np.random.seed(42)  # Reproducible
    bits = np.random.randint(0, 2, n_bits)
    
    # Time axis
    t = np.arange(0, n_bits * samples_per_bit) / fs
    t_ms = t * 1000
    
    # V_mod: Manchester-encoded modulating signal (voltage levels for LED)
    V_mod_raw = generate_manchester_waveform(bits, samples_per_bit)
    V_mod = V_mod_raw * 0.8 + 0.1  # Scale to ~0.1V to 0.9V range
    
    # V_bp: Band-pass filtered signal (AC coupled)
    # Apply BPF (700 Hz - 10 kHz) as per paper
    nyq = fs / 2
    f_low = PAPER_VALIDATION_CONFIG['bpf_low_hz']
    f_high = PAPER_VALIDATION_CONFIG['bpf_high_hz']
    low_norm = f_low / nyq
    high_norm = min(f_high / nyq, 0.99)
    
    b, a = signal.butter(2, [low_norm, high_norm], btype='band')
    V_bp_raw = signal.filtfilt(b, a, V_mod)
    V_bp = V_bp_raw * 50 + 0.0  # Scale for visibility (AC coupled, centered at 0)
    
    # V_cmp: Comparator output (recovered Manchester -> bits)
    # Simple threshold detection on V_bp
    threshold = 0.0
    V_cmp = np.where(V_bp > threshold, 0.8, 0.0)
    
    # Create plot matching paper style
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # V_mod (top)
    axes[0].plot(t_ms, V_mod, 'b-', linewidth=1.0)
    axes[0].set_ylabel(r'$V_{mod}$ (V)', fontsize=12)
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 10)
    
    # V_bp (middle)
    axes[1].plot(t_ms, V_bp, 'b-', linewidth=1.0)
    axes[1].set_ylabel(r'$V_{bp}$ (V)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # V_cmp (bottom)
    axes[2].plot(t_ms, V_cmp, 'b-', linewidth=1.0)
    axes[2].set_ylabel(r'$V_{cmp}$ (V)', fontsize=12)
    axes[2].set_xlabel('time (ms)', fontsize=12)
    axes[2].set_ylim(-0.1, 1.0)
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle('Measured waveforms with Manchester-encoded data (Fig. 16)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'fig16_waveforms.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {path}")
    return path


# ========== FIGURE 17: BER VS MODULATION DEPTH ==========

def ber_from_snr_linear(snr_linear):
    """Calculate BER from linear SNR."""
    snr_linear = np.clip(snr_linear, 1e-6, 1e6)
    return 0.5 * erfc(np.sqrt(snr_linear / 2))


def calculate_ber_physics(m, fsw_khz, Tbit_us):
    """Calculate BER using physics-based model."""
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    from simulator.transmitter import Transmitter
    from simulator.channel import OpticalChannel
    from simulator.receiver import PVReceiver
    from simulator.noise import NoiseModel
    
    params = get_paper_params('kadirvelu2021')
    P_tx_mw = 9.3
    
    tx = Transmitter({'dc_bias': 100, 'modulation_depth': m, 'led_efficiency': 0.1})
    tx.eta_led = (P_tx_mw * 1e-3) / (100 * 1e-3)
    
    ch = OpticalChannel({
        'distance': params['distance_m'],
        'beam_angle_half': params['led_half_angle_deg'],
        'receiver_area': params['solar_cell_area_cm2'],
        'temperature': 300
    })
    
    rx = PVReceiver({
        'responsivity': params['responsivity_a_per_w'],
        'capacitance': params['cj_pf'],
        'shunt_resistance': params['rsh_ohm'] / 1e6,
    })
    
    noise_model = NoiseModel({'temperature_K': 300, 'ambient_lux': 300})
    
    I_tx_ac_amp = m * (100 * 1e-3)
    P_tx_ac_amp = tx.eta_led * I_tx_ac_amp
    P_rx_ac_amp = ch.propagate(np.array([P_tx_ac_amp]), np.array([0]))[0]
    I_ph_ac_amp = rx.optical_to_current(np.array([P_rx_ac_amp]))[0]
    
    bandwidth = 1.0 / (Tbit_us * 1e-6)
    
    I_dc_tx = 100 * 1e-3
    P_tx_dc = tx.eta_led * I_dc_tx
    P_rx_dc = ch.propagate(np.array([P_tx_dc]), np.array([0]))[0]
    I_ph_dc = rx.optical_to_current(np.array([P_rx_dc]))[0]
    
    sigma_noise = noise_model.total_noise_std(I_ph_dc, bandwidth, params['solar_cell_area_cm2'], ina_gain=100)
    
    snr_ideal = (I_ph_ac_amp / sigma_noise)**2
    
    # Experimental loss calibration
    experimental_loss_db = 36.5
    fsw_penalty_db = {50: 3.5, 100: 1.2, 200: 0.0}.get(fsw_khz, 0.0)
    total_penalty = experimental_loss_db + fsw_penalty_db
    penalty_linear = 10**(-total_penalty/10)
    
    snr_final = snr_ideal * penalty_linear
    ber = ber_from_snr_linear(snr_final)
    
    sys.stdout = old_stdout
    return ber


def plot_fig17(output_dir=None):
    """Plot BER vs modulation depth (Fig 17) with two panels."""
    if output_dir is None:
        output_dir = get_plots_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    m_percent = np.array([10, 20, 30, 40, 60, 80, 100])
    m_frac = m_percent / 100.0
    fsw_list = [50, 100, 200]
    
    styles = {
        50:  {'color': 'blue',    'linestyle': '--', 'marker': 'o', 'markerfacecolor': 'none'},
        100: {'color': 'red',     'linestyle': '--', 'marker': 'x', 'markerfacecolor': 'red'},
        200: {'color': 'magenta', 'linestyle': '-.', 'marker': 's', 'markerfacecolor': 'none'},
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    panels = [
        {'ax': axes[0], 'Tbit_us': 100, 'ylim': 0.4, 'title': r'(a) $T_{bit} = 100\,\mu s$'},
        {'ax': axes[1], 'Tbit_us': 400, 'ylim': 0.2, 'title': r'(b) $T_{bit} = 400\,\mu s$'},
    ]
    
    print("  Generating BER curves...")
    
    for panel in tqdm(panels, desc="Fig 17 panels"):
        ax = panel['ax']
        Tbit_us = panel['Tbit_us']
        
        for fsw in fsw_list:
            ber_values = []
            for m in m_frac:
                ber = calculate_ber_physics(m, fsw, Tbit_us)
                ber_values.append(ber)
            
            ax.plot(m_percent, ber_values, color=styles[fsw]['color'],
                   linestyle=styles[fsw]['linestyle'], marker=styles[fsw]['marker'],
                   markerfacecolor=styles[fsw]['markerfacecolor'],
                   markersize=8, linewidth=1.5,
                   label=PAPER_STYLES[fsw]['label'])
        
        ax.set_xlabel(r'$m$ (%)', fontsize=12)
        ax.set_ylabel('BER', fontsize=12)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, panel['ylim'])
        ax.set_xticks([10, 20, 30, 40, 60, 80, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(panel['title'], fontsize=12)
    
    fig.suptitle('Measured BER for different values of modulation depth\nand switching frequency (Fig. 17)',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig17_ber.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {path}")
    return path


# ========== FIGURE 18: V_OUT VS MODULATION DEPTH ==========

def calculate_vout_fig18(m, fsw_khz):
    """Calculate V_out for Fig 18 using full physics chain."""
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    from simulator.transmitter import Transmitter
    from simulator.channel import OpticalChannel
    from simulator.receiver import PVReceiver
    from simulator.dc_dc_converter import DCDCConverter
    
    params = get_paper_params('kadirvelu2021')
    n_cells = params.get('n_cells_module', 13)
    
    tx = Transmitter({'dc_bias': 100, 'modulation_depth': m, 'led_efficiency': 0.1})
    tx.eta_led = (9.3 * 1e-3) / (100 * 1e-3)
    
    ch = OpticalChannel({
        'distance': params['distance_m'],
        'beam_angle_half': params['led_half_angle_deg'],
        'receiver_area': params['solar_cell_area_cm2']
    })
    
    rx = PVReceiver({
        'responsivity': params['responsivity_a_per_w'],
        'capacitance': params['cj_pf'],
        'shunt_resistance': 100,  # High to avoid loading
        'dark_current': 1.0,
        'temperature': 300
    })
    
    dcdc = DCDCConverter({
        'fsw_khz': fsw_khz,
        'duty_cycle': 0.44,
        'v_diode_drop': 0.3,
        'r_on_mohm': 100,
        'efficiency_mode': 'paper'
    })
    
    duration = 2e-3
    fs = 1e6
    t = np.arange(0, duration, 1/fs)
    n_bits = int(duration * 10000)
    bits = np.random.randint(0, 2, max(10, n_bits))
    
    P_tx = tx.modulate(bits, t)
    P_rx = ch.propagate(P_tx, t)
    I_ph = rx.optical_to_current(P_rx)
    I_ph_avg = np.mean(I_ph)
    
    V_pv_waveform = rx.solve_pv_circuit(I_ph, t, method='euler')
    V_pv_avg_cell = np.mean(V_pv_waveform[int(len(t)*0.5):])
    V_pv_module = V_pv_avg_cell * n_cells
    
    output = dcdc.calculate_output(V_pv_module, I_ph_avg, duty_cycle=0.44, fsw_khz=fsw_khz)
    
    sys.stdout = old_stdout
    return output['V_out']


def plot_fig18(output_dir=None):
    """Plot V_out vs modulation depth (Fig 18)."""
    if output_dir is None:
        output_dir = get_plots_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    m_percent = np.array([10, 20, 40, 60, 80, 100])
    fsw_list = [50, 100, 200]
    
    styles = {
        50:  {'color': 'blue', 'linestyle': (0, (5, 5)), 'marker': 'o', 'markerfacecolor': 'white'},
        100: {'color': 'red', 'linestyle': (0, (3, 1)), 'marker': 'x', 'markerfacecolor': 'red'},
        200: {'color': 'magenta', 'linestyle': '-.', 'marker': 's', 'markerfacecolor': 'white'},
    }
    
    plt.figure(figsize=(10, 6))
    
    for fsw in tqdm(fsw_list, desc="Fig 18"):
        vout_values = []
        for m_pct in m_percent:
            m = m_pct / 100.0
            vout = calculate_vout_fig18(m, fsw)
            vout_values.append(vout)
        
        plt.plot(m_percent, vout_values,
                color=styles[fsw]['color'],
                linestyle=styles[fsw]['linestyle'],
                marker=styles[fsw]['marker'],
                markerfacecolor=styles[fsw]['markerfacecolor'],
                markeredgecolor=styles[fsw]['color'],
                markersize=8, linewidth=2,
                label=PAPER_STYLES[fsw]['label'])
        
        if fsw == 50:
            print(f"  50kHz: m=10%→{vout_values[0]:.2f}V, m=100%→{vout_values[-1]:.2f}V")
    
    plt.xlabel(r'Modulation depth $m$ (%)', fontsize=12)
    plt.ylabel(r'Output Voltage $V_{out}$ (V)', fontsize=12)
    plt.title('DC-DC Output Voltage vs Modulation Depth (Fig. 18)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.xlim(0, 105)
    plt.ylim(0, 15)
    plt.xticks([10, 20, 40, 60, 80, 100])
    
    path = os.path.join(output_dir, "fig18_vout.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {path}")
    return path

# ========== FIGURE 19: POWER VS BIT RATE TRADE-OFF ==========

def calculate_harvested_power_vs_bitrate(bit_rate_kbps):
    """
    Calculate maximum harvested power with load effect.
    """
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    from simulator.transmitter import Transmitter
    from simulator.channel import OpticalChannel
    from simulator.receiver import PVReceiver
    
    params = get_paper_params('kadirvelu2021')
    
    # Physics calibration
    R_load_calibrated = 1440.0  # Tuned for ~250 uW max power
    C_eff_calibrated = 6000e-12 # Tuned effective capacitance (diffusion + junction) for bandwidth
    
    tx = Transmitter({'dc_bias': 100, 'modulation_depth': 0.5, 'led_efficiency': 0.093})
    ch = OpticalChannel({
        'distance': params['distance_m'],
        'beam_angle_half': params['led_half_angle_deg'],
        'receiver_area': params['solar_cell_area_cm2']
    })
    rx = PVReceiver({
        'responsivity': params['responsivity_a_per_w'],
        'capacitance': params['cj_pf'],
        'shunt_resistance': params['rsh_ohm'] / 1e6,
        'dark_current': 1e-9
    })
    
    # Simulate
    bit_rate_bps = bit_rate_kbps * 1000
    duration = max(0.01, 200/bit_rate_bps) 
    fs = max(100000, bit_rate_bps * 20)
    t = np.arange(0, duration, 1/fs)
    n_bits = max(20, int(duration * bit_rate_bps))
    bits = np.random.randint(0, 2, n_bits)
    
    P_tx = tx.modulate(bits, t)
    P_rx = ch.propagate(P_tx, t)
    I_ph = rx.optical_to_current(P_rx)
    
    # Solve loaded circuit: dV/dt = (I_ph - I_diode - V/R_sh - V/R_load) / C_eff
    V = np.zeros_like(t)
    dt = t[1] - t[0]
    
    # Constants for speed
    I_0 = rx.I_0
    V_T = rx.V_T
    R_sh = rx.R_sh
    
    # Euler integration
    for i in range(len(t)-1):
        V_curr = V[i]
        
        # Diode current (clipped V for safety)
        V_safe = min(V_curr, 1.5)
        # Fast exp approximation or just standard
        if V_safe > 20 * V_T:
             I_diode = I_0 * np.exp(V_safe / V_T) # Ignore -1 for speed
        else:
             I_diode = I_0 * (np.exp(V_safe / V_T) - 1)
        
        I_res = V_curr / R_sh
        I_load = V_curr / R_load_calibrated
        
        dV_dt = (I_ph[i] - I_diode - I_res - I_load) / C_eff_calibrated
        
        V_next = V_curr + dV_dt * dt
        V[i+1] = max(0, V_next) # No negative voltage
        
    # Harvested power
    P_harv = V**2 / R_load_calibrated
    P_max_uW = np.mean(P_harv[int(len(t)*0.5):]) * 1e6 # Mean power in steady state, not peak
    
    sys.stdout = old_stdout
    return P_max_uW


def plot_fig19(output_dir=None):
    """
    Plot max harvested power vs bit rate (Fig 19).
    """
    if output_dir is None:
        output_dir = get_plots_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    bit_rates_kbps = np.array([1, 2, 5, 10, 20, 50, 100])
    
    print("  Calculating harvested power vs bit rate...")
    P_vals = []
    for br in tqdm(bit_rates_kbps, desc="Fig 19"):
        P_vals.append(calculate_harvested_power_vs_bitrate(br))
    P_vals = np.array(P_vals)
    
    # Logistic fit model: P(x) = A / (1 + (x/x0)^k)
    def logistic_model(x, A, x0, k):
        return A / (1 + (x/x0)**k)
    
    # Initial guess: A=max(P), x0=20, k=1
    p0 = [max(P_vals), 20, 1]
    
    try:
        popt, _ = curve_fit(logistic_model, bit_rates_kbps, P_vals, p0=p0, maxfev=5000)
    except:
        print("  Warning: Curve fit failed, using spline.")
        popt = None
    
    # Dense plot
    br_dense = np.logspace(0, 2, 100)
    if popt is not None:
        P_fit = logistic_model(br_dense, *popt)
    else:
        # Fallback linear interp
        P_fit = np.interp(br_dense, bit_rates_kbps, P_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogx(bit_rates_kbps, P_vals, 'bo', markersize=8, 
                markerfacecolor='white', markeredgewidth=1.5, label='measured')
    ax.semilogx(br_dense, P_fit, 'r--', linewidth=2, label='fitted model')
    
    ax.set_xlabel('bit rate (kbps)', fontsize=12)
    ax.set_ylabel(r'$P_{out(max)}$ ($\mu$W)', fontsize=12)
    ax.set_xlim(0.8, 120)
    # Range matching paper approx 100-250
    ax.set_ylim(50, 270)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_title('Illustration of trade-off between the maximum harvested power\nand transmission speed (bit rate) (Fig. 19)', 
                 fontsize=12)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig19_power_bitrate.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {path}")
    return path



# ========== MAIN GENERATOR ==========

def generate_all_figures(fig_list=None, output_dir=None):
    """
    Generate all paper figures.
    
    Args:
        fig_list: List of figure numbers [13, 14, 15, 16, 17, 18, 19] or None for all
        output_dir: Output directory (uses output_manager if None)
    """
    if fig_list is None:
        fig_list = [13, 14, 15, 16, 17, 18, 19]
    
    print("\n" + "="*60)
    print("  GENERATING PAPER FIGURES")
    print("="*60)
    
    paths = []
    
    if 13 in fig_list:
        print("\n📊 Fig 13: Frequency Response...")
        paths.append(plot_fig13(output_dir))
    
    if 14 in fig_list:
        print("\n📊 Fig 14: PSD at BPF Output...")
        paths.append(plot_fig14(output_dir))
    
    if 15 in fig_list:
        print("\n📊 Fig 15: V_out vs Duty Cycle...")
        paths.append(plot_fig15(output_dir))
    
    if 16 in fig_list:
        print("\n📊 Fig 16: Time-Domain Waveforms...")
        paths.append(plot_fig16(output_dir))
    
    if 17 in fig_list:
        print("\n📊 Fig 17: BER vs Modulation Depth...")
        paths.append(plot_fig17(output_dir))
    
    if 18 in fig_list:
        print("\n📊 Fig 18: V_out vs Modulation Depth...")
        paths.append(plot_fig18(output_dir))
    
    if 19 in fig_list:
        print("\n📊 Fig 19: Power vs Bit Rate Trade-off...")
        paths.append(plot_fig19(output_dir))
    
    print("\n" + "="*60)
    print(f"  ✅ Generated {len(paths)} figures")
    print("="*60)
    
    return paths


if __name__ == "__main__":
    generate_all_figures()

