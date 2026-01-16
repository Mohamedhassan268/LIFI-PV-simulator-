# utils/plotting.py
"""
Visualization utilities for Li-Fi + PV simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_simulation_results(result, n_samples_to_plot=10000, output_dir='outputs/plots'):
    """
    Create 4-panel plot showing all signal stages.
    
    Args:
        result (dict): Simulation results from run_single_simulation()
        n_samples_to_plot (int): Number of samples to show (for clarity)
        output_dir (str): Directory to save plots
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    t = result['t']
    bits = result['bits']
    P_tx = result['P_tx']
    P_rx = result['P_rx']
    I_ph = result['I_ph']
    V_pv = result['V_pv']
    
    # Limit to first n_samples for clarity
    n = min(n_samples_to_plot, len(t))
    t_plot = t[:n] * 1e3  # Convert to milliseconds
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle('Li-Fi + PV Simulation Results', fontsize=16, fontweight='bold')
    
    # ========== PLOT 1: Transmitted Optical Power ==========
    axes[0].plot(t_plot, P_tx[:n] * 1e3, 'r-', linewidth=1.5, label='P_tx')
    axes[0].set_ylabel('TX Power (mW)', fontsize=11, fontweight='bold')
    axes[0].set_title('Layer 1: Transmitter (OOK Modulation)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Add annotation
    axes[0].text(0.02, 0.95, f'Bit rate: {result["config"]["simulation"]["data_rate_bps"]/1e6:.1f} Mbps', 
                 transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== PLOT 2: Received Optical Power ==========
    axes[1].plot(t_plot, P_rx[:n] * 1e6, 'b-', linewidth=1.5, label='P_rx')
    axes[1].set_ylabel('RX Power (μW)', fontsize=11, fontweight='bold')
    axes[1].set_title('Layer 2: Channel (Lambertian Path Loss)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Add annotation
    path_loss_db = 10 * np.log10(P_rx.mean() / P_tx.mean())
    axes[1].text(0.02, 0.95, f'Path loss: {path_loss_db:.1f} dB', 
                 transform=axes[1].transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # ========== PLOT 3: Photocurrent ==========
    axes[2].plot(t_plot, I_ph[:n] * 1e6, 'g-', linewidth=1.5, label='I_ph')
    axes[2].set_ylabel('Photocurrent (μA)', fontsize=11, fontweight='bold')
    axes[2].set_title('Layer 3a: Photodetection (P → I conversion)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    # Add annotation
    responsivity = result['config']['receiver']['responsivity']
    axes[2].text(0.02, 0.95, f'Responsivity: {responsivity:.2f} A/W', 
                 transform=axes[2].transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # ========== PLOT 4: PV Junction Voltage (ODE Solution) ==========
    axes[3].plot(t_plot, V_pv[:n] * 1e3, 'm-', linewidth=1.5, label='V_pv')
    axes[3].set_ylabel('Junction Voltage (mV)', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
    axes[3].set_title('Layer 3b: PV Cell Circuit (Nonlinear ODE Solution)', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')
    
    # Add annotation
    V_final = V_pv[-1] * 1e3
    axes[3].text(0.02, 0.95, f'Final voltage: {V_final:.2f} mV', 
                 transform=axes[3].transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    filename = os.path.join(output_dir, 'simulation_results.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {filename}")
    
    # Close to free memory
    plt.close()
    
    # ========== BONUS: Zoomed-in view of first few bits ==========
    plot_zoomed_bits(result, output_dir)
    
    return filename


def plot_zoomed_bits(result, output_dir, n_bits_to_show=10):
    """
    Create a zoomed-in plot showing individual bit transitions.
    
    Args:
        result (dict): Simulation results
        output_dir (str): Directory to save plots
        n_bits_to_show (int): Number of bits to display
    """
    
    # Extract data
    t = result['t']
    P_tx = result['P_tx']
    V_pv = result['V_pv']
    bits = result['bits']
    
    # Calculate samples per bit
    sps = len(t) // len(bits)
    n_samples = n_bits_to_show * sps
    
    t_zoom = t[:n_samples] * 1e6  # Convert to microseconds
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f'Zoomed View: First {n_bits_to_show} Bits', fontsize=14, fontweight='bold')
    
    # Plot 1: TX Power with bit overlay
    axes[0].plot(t_zoom, P_tx[:n_samples] * 1e3, 'r-', linewidth=2)
    axes[0].set_ylabel('TX Power (mW)', fontsize=11, fontweight='bold')
    axes[0].set_title('Transmitted OOK Signal', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Overlay bit values
    for i in range(n_bits_to_show):
        t_bit_center = (i + 0.5) * sps / (len(t) / t[-1]) * 1e6
        axes[0].text(t_bit_center, P_tx[i*sps] * 1e3 * 1.1, str(bits[i]), 
                     ha='center', va='bottom', fontsize=10, fontweight='bold',
                     color='darkred')
    
    # Plot 2: PV Voltage response
    axes[1].plot(t_zoom, V_pv[:n_samples] * 1e3, 'm-', linewidth=2)
    axes[1].set_ylabel('PV Voltage (mV)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Time (μs)', fontsize=11, fontweight='bold')
    axes[1].set_title('PV Cell Voltage Response', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save
    filename = os.path.join(output_dir, 'zoomed_bit_view.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {filename}")
    
    plt.close()
    
    return filename


# ========== PAPER VALIDATION PLOTS (Kadirvelu et al. IEEE 2021) ==========

def plot_frequency_response(freqs, magnitude_db, output_dir='outputs/plots', 
                            filename='fig13_frequency_response.png'):
    """
    Plot frequency response (Fig. 13 style).
    
    Magnitude (dB) vs frequency (Hz), log-log scale.
    
    Args:
        freqs (ndarray): Frequency array (Hz)
        magnitude_db (ndarray): Magnitude in dB
        output_dir (str): Output directory
        filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter valid frequencies (positive only)
    valid_mask = freqs > 0
    freqs_plot = freqs[valid_mask]
    mag_plot = magnitude_db[valid_mask]
    
    ax.semilogx(freqs_plot, mag_plot, 'b-', linewidth=2, label='Simulated')
    
    # Mark BPF cutoffs (700 Hz and 10 kHz)
    ax.axvline(x=700, color='r', linestyle='--', alpha=0.7, label='f_low = 700 Hz')
    ax.axvline(x=10000, color='r', linestyle='--', alpha=0.7, label='f_high = 10 kHz')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Magnitude (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Frequency Response of Data Link (Fig. 13)', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim([100, 100000])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {filepath}")
    return filepath


def plot_harvested_power(fsw_khz, power_uw, mod_depths=None, output_dir='outputs/plots',
                         filename='fig15_harvested_power.png', target_power_uw=223):
    """
    Plot harvested power vs switching frequency (Fig. 15 style).
    
    Args:
        fsw_khz (list/ndarray): Switching frequencies (kHz)
        power_uw (dict or ndarray): Power values (µW), keyed by mod_depth if dict
        mod_depths (list): Modulation depths (if power_uw is dict)
        output_dir (str): Output directory
        filename (str): Output filename
        target_power_uw (float): Paper target power for reference line
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['b', 'g', 'r', 'm', 'c']
    markers = ['o', 's', '^', 'D', 'v']
    
    if isinstance(power_uw, dict):
        for i, (mod, powers) in enumerate(power_uw.items()):
            ax.plot(fsw_khz, powers, f'{colors[i%len(colors)]}-{markers[i%len(markers)]}', 
                    linewidth=2, markersize=8, label=f'm = {mod}')
    else:
        ax.plot(fsw_khz, power_uw, 'b-o', linewidth=2, markersize=8, label='Simulated')
    
    # Target line from paper
    ax.axhline(y=target_power_uw, color='r', linestyle='--', linewidth=2, 
               label=f'Paper Target = {target_power_uw} µW')
    
    ax.set_xlabel('Switching Frequency (kHz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Harvested Power (µW)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Harvesting vs Switching Frequency (Fig. 15)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {filepath}")
    return filepath


def plot_ber_vs_modulation(mod_depths, ber_data, output_dir='outputs/plots',
                           filename='fig17_ber_vs_modulation.png', target_ber=1.008e-3):
    """
    Plot BER vs modulation depth for different data rates and fsw (Fig. 17 style).
    
    Args:
        mod_depths (list): Modulation depth values
        ber_data (dict): Nested dict {data_rate: {fsw: [ber values]}}
        output_dir (str): Output directory
        filename (str): Output filename
        target_ber (float): Paper target BER for reference line
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['b', 'g', 'r', 'm', 'c', 'orange']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    linestyles = ['-', '--', '-.', ':']
    
    data_rates = list(ber_data.keys())
    
    for ax_idx, data_rate in enumerate(data_rates[:2]):
        ax = axes[ax_idx]
        fsw_data = ber_data[data_rate]
        
        for i, (fsw, bers) in enumerate(fsw_data.items()):
            # Handle case where bers might have NaN or zeros
            bers_plot = np.array(bers)
            bers_plot = np.where(bers_plot <= 0, 1e-10, bers_plot)  # Replace 0 with small value
            
            ax.semilogy(mod_depths, bers_plot, 
                       f'{colors[i%len(colors)]}-{markers[i%len(markers)]}',
                       linewidth=2, markersize=8, label=f'fsw = {fsw} kHz')
        
        # Target BER line
        ax.axhline(y=target_ber, color='k', linestyle='--', linewidth=2,
                   label=f'Target = {target_ber:.2e}')
        
        ax.set_xlabel('Modulation Depth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Bit Error Rate (BER)', fontsize=12, fontweight='bold')
        
        # Handle both bps and kbps conventions
        if data_rate == 2500 or data_rate == 2.5:
            rate_str = '2.5 kbps (Tbit=400µs)'
        elif data_rate == 10000 or data_rate == 10:
            rate_str = '10 kbps (Tbit=100µs)'
        else:
            rate_str = f'{data_rate} bps'
            
        ax.set_title(f'BER vs Modulation Depth @ {rate_str}', fontsize=12, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim([1e-6, 1])
    
    fig.suptitle('BER Performance (Fig. 17)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {filepath}")
    return filepath


def plot_time_domain_waveforms(t, V_mod, V_bp, bits_rx, n_bits_show=10,
                               output_dir='outputs/plots', filename='fig16_time_domain.png'):
    """
    Plot time-domain waveforms (Fig. 16 style).
    
    Shows: Vmod (modulated input), Vbp (after BPF), Vcmp (comparator output).
    
    Args:
        t (ndarray): Time vector (s)
        V_mod (ndarray): Modulated input signal (V)
        V_bp (ndarray): Bandpass filter output (V)
        bits_rx (ndarray): Received bits (for Vcmp-like display)
        n_bits_show (int): Number of bits to display
        output_dir (str): Output directory
        filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate samples per bit
    n_bits = len(bits_rx)
    sps = len(t) // n_bits
    n_samples = min(n_bits_show * sps, len(t))
    
    # Create upsampled bits_rx for Vcmp
    bits_expanded = np.repeat(bits_rx, sps)[:len(t)]
    
    t_plot = t[:n_samples] * 1e3  # Convert to ms
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Vmod (modulated input)
    axes[0].plot(t_plot, V_mod[:n_samples] * 1e3, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Vmod (mV)', fontsize=11, fontweight='bold')
    axes[0].set_title('Modulated Input Signal (INA Output)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Vbp (after BPF)
    axes[1].plot(t_plot, V_bp[:n_samples] * 1e3, 'g-', linewidth=1.5)
    axes[1].set_ylabel('Vbp (mV)', fontsize=11, fontweight='bold')
    axes[1].set_title('Bandpass Filter Output', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Vcmp (comparator output / recovered bits)
    axes[2].plot(t_plot, bits_expanded[:n_samples], 'r-', linewidth=2)
    axes[2].set_ylabel('Vcmp', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
    axes[2].set_title('Comparator Output (Recovered Bits)', fontsize=12)
    axes[2].set_ylim([-0.2, 1.2])
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle('Time-Domain Waveforms (Fig. 16)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {filepath}")
    return filepath


def plot_psd(freqs_khz, psd_db, output_dir='outputs/plots',
             filename='fig14_psd.png', with_capacitors=True):
    """
    Plot Power Spectral Density at BPF output (Fig. 14 style).
    
    Args:
        freqs_khz (ndarray): Frequency array in kHz
        psd_db (ndarray): PSD in dB (V²rms/Hz)
        output_dir (str): Output directory
        filename (str): Output filename
        with_capacitors (bool): Label for capacitor configuration
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    label = 'with capacitors' if with_capacitors else 'without capacitors'
    ax.semilogy(freqs_khz, 10**(psd_db/10), 'b-', linewidth=2, label=label)
    
    ax.set_xlabel('Frequency (kHz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSD (V²rms/Hz)', fontsize=12, fontweight='bold')
    ax.set_title('Power Spectral Density at BPF Output (Fig. 14)', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim([0, 100])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {filepath}")
    return filepath


def plot_dcdc_vout_vs_duty(duty_cycles, vout_data, output_dir='outputs/plots',
                           filename='fig15_dcdc_vout_vs_duty.png'):
    """
    Plot DC-DC output voltage vs duty cycle (Fig. 15 style).
    
    Args:
        duty_cycles (ndarray): Duty cycle values (0-1 or 0-100%)
        vout_data (dict): {fsw: vout_array} for different switching frequencies
        output_dir (str): Output directory
        filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['b', 'r', 'm']
    linestyles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    
    # Convert duty cycle to percentage if needed
    duty_pct = duty_cycles * 100 if np.max(duty_cycles) <= 1 else duty_cycles
    
    for i, (fsw, vout) in enumerate(vout_data.items()):
        ax.plot(duty_pct, vout, 
                f'{colors[i%len(colors)]}{linestyles[i%len(linestyles)]}{markers[i%len(markers)]}',
                linewidth=2, markersize=6, label=f'fsw = {fsw} kHz')
    
    ax.set_xlabel('ρ (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vout (V)', fontsize=12, fontweight='bold')
    ax.set_title('DC-DC Output Voltage vs Duty Cycle (Fig. 15)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim([0, 50])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {filepath}")
    return filepath


def plot_dcdc_vout_vs_modulation(mod_depths, vout_data, output_dir='outputs/plots',
                                  filename='fig18_dcdc_vout_vs_mod.png'):
    """
    Plot DC-DC output voltage vs modulation depth (Fig. 18 style).
    
    Args:
        mod_depths (ndarray): Modulation depth values (0-1 or 0-100%)
        vout_data (dict): {fsw: vout_array} for different switching frequencies
        output_dir (str): Output directory
        filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['b', 'r', 'm']
    linestyles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    
    # Convert mod depth to percentage if needed
    mod_pct = mod_depths * 100 if np.max(mod_depths) <= 1 else mod_depths
    
    for i, (fsw, vout) in enumerate(vout_data.items()):
        ax.plot(mod_pct, vout,
                f'{colors[i%len(colors)]}{linestyles[i%len(linestyles)]}{markers[i%len(markers)]}',
                linewidth=2, markersize=6, label=f'fsw = {fsw} kHz')
    
    ax.set_xlabel('m (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vout (V)', fontsize=12, fontweight='bold')
    ax.set_title('DC-DC Output Voltage vs Modulation Depth (Fig. 18)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim([0, 100])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {filepath}")
    return filepath


def plot_power_vs_bitrate(bitrates_kbps, power_uw, output_dir='outputs/plots',
                          filename='fig19_power_vs_bitrate.png', 
                          measured_data=None, fitted_model=None):
    """
    Plot harvested power vs bit rate (Fig. 19 style).
    
    Args:
        bitrates_kbps (ndarray): Bit rates in kbps
        power_uw (ndarray): Harvested power in µW
        output_dir (str): Output directory
        filename (str): Output filename
        measured_data (tuple): (bitrates, power) for measured points (optional)
        fitted_model (tuple): (bitrates, power) for fitted line (optional)
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main simulated data
    ax.semilogx(bitrates_kbps, power_uw, 'b-', linewidth=2, label='Simulated')
    
    # Measured data points (if provided)
    if measured_data is not None:
        ax.semilogx(measured_data[0], measured_data[1], 'ro', 
                    markersize=8, label='Measured')
    
    # Fitted model (if provided)
    if fitted_model is not None:
        ax.semilogx(fitted_model[0], fitted_model[1], 'r--',
                    linewidth=2, label='Fitted model')
    
    ax.set_xlabel('Bit Rate (kbps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pout(max) (µW)', fontsize=12, fontweight='bold')
    ax.set_title('Harvested Power vs Bit Rate (Fig. 19)', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim([1, 100])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {filepath}")
    return filepath


if __name__ == "__main__":
    print("This module contains plotting functions.")
    print("Run main.py to generate plots.")
