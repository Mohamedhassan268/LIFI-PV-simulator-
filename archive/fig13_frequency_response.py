# fix_fig13_frequency_response.py


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

from utils.constants import PAPER_VALIDATION_CONFIG


def compute_pv_cell_response(freqs, R_sh, C_j, R_load):
    """
    Compute PV cell small-signal frequency response.
    
    The PV cell acts as a current source with parallel RC:
    H_pv(s) = Z_load / (Z_load + Z_pv)
    
    For simplified model:
    H_pv(jω) = 1 / (1 + jω·R_sh·C_j)
    
    This is a low-pass filter with f_c = 1/(2π·R_sh·C_j)
    
    Args:
        freqs: Frequency array (Hz)
        R_sh: Shunt resistance (Ω) - from paper: 138.8 Ω
        C_j: Junction capacitance (F) - from paper: 798 nF
        R_load: Load resistance (Ω) - from paper: 1360 Ω
        
    Returns:
        H_pv: Complex transfer function
    """
    omega = 2 * np.pi * freqs
    
    # Time constant
    tau = R_sh * C_j
    f_c = 1 / (2 * np.pi * tau)
    print(f"  PV cell cutoff frequency: {f_c:.1f} Hz")
    
    # First-order low-pass response
    H_pv = 1 / (1 + 1j * omega * tau)
    
    return H_pv


def compute_bpf_response(freqs, f_low=700, f_high=10000, order=2, fs=500000):
    """
    Compute Butterworth bandpass filter frequency response.
    
    Args:
        freqs: Frequency array (Hz)
        f_low: Lower cutoff (Hz)
        f_high: Upper cutoff (Hz)
        order: Filter order
        fs: Sample rate for digital filter design
        
    Returns:
        H_bpf: Complex transfer function
    """
    nyq = fs / 2
    low_norm = f_low / nyq
    high_norm = f_high / nyq
    
    # Design the filter
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    
    # Compute frequency response
    w, H_bpf = signal.freqz(b, a, worN=freqs, fs=fs)
    
    return H_bpf


def compute_ina_response(freqs, gain=100, f_3db=1e6):
    omega = 2 * np.pi * freqs
    omega_3db = 2 * np.pi * f_3db
    
    H_ina = gain / (1 + 1j * omega / omega_3db)
    
    return H_ina


def compute_total_system_response():
    print("\n" + "="*60)
    print("COMPUTING SYSTEM FREQUENCY RESPONSE (Fig. 13)")
    print("="*60)
    
    # Frequency range (as in paper: 1 Hz to 1 MHz, log scale)
    freqs = np.logspace(0, 6, 1000)  # 1 Hz to 1 MHz
    
    # Paper parameters
    R_sh = PAPER_VALIDATION_CONFIG['rsh_ohm']  # 138.8 Ω
    C_j = PAPER_VALIDATION_CONFIG['cj_pf'] * 1e-12  # 798 pF to F
    R_load = PAPER_VALIDATION_CONFIG['rload_ohm']  # 1360 Ω
    
    print(f"\n  Parameters:")
    print(f"    R_sh = {R_sh} Ω")
    print(f"    C_j = {C_j*1e9} nF")
    print(f"    R_load = {R_load} Ω")
    
    # 1. PV cell response (low-pass)
    print("\n  Computing PV cell response...")
    H_pv = compute_pv_cell_response(freqs, R_sh, C_j, R_load)
    
    # 2. BPF response
    print("  Computing BPF response...")
    H_bpf = compute_bpf_response(
        freqs,
        f_low=PAPER_VALIDATION_CONFIG['bpf_low_hz'],
        f_high=PAPER_VALIDATION_CONFIG['bpf_high_hz'],
        order=PAPER_VALIDATION_CONFIG['bpf_order']
    )
    
    # 3. INA response (flat gain, high bandwidth)
    print("  Computing INA response...")
    H_ina = compute_ina_response(freqs, gain=PAPER_VALIDATION_CONFIG['ina_gain'])
    
    # Total response
    H_total = H_pv * H_bpf * H_ina
    
    # Normalize to 0 dB at peak
    magnitude = np.abs(H_total)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)
    magnitude_db = magnitude_db - np.max(magnitude_db)  # Normalize to 0 dB peak
    
    print(f"\n  Peak frequency: {freqs[np.argmax(magnitude)]:.0f} Hz")
    print(f"  Peak magnitude: {np.max(magnitude_db):.1f} dB (normalized)")
    
    return freqs, magnitude_db, H_pv, H_bpf, H_ina


def plot_fig13_corrected(output_dir='outputs/plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    freqs, magnitude_db, H_pv, H_bpf, H_ina = compute_total_system_response()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main system response
    ax.semilogx(freqs, magnitude_db, 'b-', linewidth=2.5, label='Modelled (Total)')
    
    # Individual components (optional, for debugging)
    mag_pv = 20 * np.log10(np.abs(H_pv) + 1e-12)
    mag_bpf = 20 * np.log10(np.abs(H_bpf) + 1e-12)
    
    ax.semilogx(freqs, mag_pv - np.max(mag_pv), 'g--', linewidth=1.5, 
                alpha=0.5, label='PV cell (LP)')
    ax.semilogx(freqs, mag_bpf - np.max(mag_bpf), 'r--', linewidth=1.5,
                alpha=0.5, label='BPF')
    
    # Axis settings (match paper)
    ax.set_xlabel('frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('magnitude (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Modelled and measured frequency response (Fig. 13)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim([1, 1e6])
    ax.set_ylim([-50, 10])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig13_frequency_response_corrected.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Saved: {filepath}")
    return filepath, freqs, magnitude_db


if __name__ == "__main__":
    plot_fig13_corrected()
