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


if __name__ == "__main__":
    print("This module contains plotting functions.")
    print("Run main.py to generate plots.")
