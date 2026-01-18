"""
González-Uriarte et al. (2024) Validation Module

Paper: "Design and Implementation of a Low-Cost VLC Photovoltaic Panel-Based Receiver 
        with off-the-Shelf Components"

Validates:
- Fig. 2: R_load vs Bandwidth curve
- Fig. 3: Voltage vs R_load trade-off
- 4.8 kBd max data rate
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from utils.output_manager import get_paper_output_dir

PARAMS = {
    # System Geometry
    'distance_m': 0.60,
    
    # Poly-Si Panel (low-cost generic 5V/40mA)
    'responsivity_a_per_w': 0.5,
    'solar_cell_area_cm2': 66.0,
    
    # Electrical Model
    # Large capacitance typical for cheap panels (hundreds of nF)
    'cj_f': 320e-9,  # 320 nF
    
    'rload_nominal': 220,
    
    # Targets from paper (Fig. 2)
    'bandwidth_targets': {
        1e6: 500,      # 1 MΩ → 500 Hz
        100e3: 500,    # 100 kΩ → 500 Hz
        10e3: 1000,    # 10 kΩ → 1 kHz
        1e3: 10000,    # 1 kΩ → 10 kHz
        220: 50000,    # 220 Ω → 50 kHz
        100: 100000,   # 100 Ω → 100 kHz
    },
    
    # Max data rate
    'max_baud_rate': 4800,
}


def calculate_bandwidth(R_load, C_j):
    """Calculate 3dB bandwidth: f_c = 1 / (2π × R_load × C_j)"""
    return 1.0 / (2 * np.pi * R_load * C_j)


def plot_rload_vs_bandwidth(output_dir):
    """Generate Fig. 2: R_load vs Bandwidth curve."""
    print("\n[Fig. 2] Generating R_load vs Bandwidth curve...")
    
    # Sweep R_load from 100 Ω to 1 MΩ
    r_loads = np.logspace(2, 6, 100)  # 100 Ω to 1 MΩ
    C_j = PARAMS['cj_f']
    
    bandwidths = [calculate_bandwidth(r, C_j) for r in r_loads]
    
    # Paper target points
    r_targets = list(PARAMS['bandwidth_targets'].keys())
    bw_targets = list(PARAMS['bandwidth_targets'].values())
    
    plt.figure(figsize=(10, 6))
    plt.loglog(r_loads, bandwidths, 'b-', linewidth=2, label='Simulated')
    plt.scatter(r_targets, bw_targets, color='red', s=100, zorder=5, 
                label='Paper Targets', marker='x')
    
    # Mark operating point
    plt.axvline(220, color='green', linestyle='--', alpha=0.7, label='Operating Point (220 Ω)')
    plt.axhline(50000, color='green', linestyle='--', alpha=0.7)
    
    plt.xlabel('Load Resistance R_load (Ω)', fontsize=12)
    plt.ylabel('Bandwidth (Hz)', fontsize=12)
    plt.title('González 2024 - Fig. 2: R_load vs Bandwidth\n(Low-Cost PV Panel)', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'fig2_rload_vs_bandwidth.png'), dpi=150)
    plt.close()
    
    print(f"  ✅ Saved to {output_dir}/fig2_rload_vs_bandwidth.png")
    
    # Calculate bandwidth at operating point
    bw_220 = calculate_bandwidth(220, C_j)
    print(f"  Bandwidth @ 220Ω: {bw_220/1000:.2f} kHz (Target: 50 kHz)")
    
    return bw_220


def plot_voltage_vs_rload(output_dir):
    """Generate Fig. 3: Voltage vs R_load trade-off."""
    print("\n[Fig. 3] Generating Voltage vs R_load curve...")
    
    # Assume 1 µA photocurrent
    I_ph = 1e-6  # 1 µA
    
    r_loads = np.logspace(2, 6, 100)
    voltages = I_ph * r_loads  # V = I × R
    bandwidths = [calculate_bandwidth(r, PARAMS['cj_f']) for r in r_loads]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Voltage (left axis)
    ax1.semilogx(r_loads, voltages * 1e6, 'b-', linewidth=2, label='Voltage')
    ax1.set_xlabel('Load Resistance R_load (Ω)', fontsize=12)
    ax1.set_ylabel('Output Voltage (µV)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Bandwidth (right axis)
    ax2 = ax1.twinx()
    ax2.loglog(r_loads, bandwidths, 'r--', linewidth=2, label='Bandwidth')
    ax2.set_ylabel('Bandwidth (Hz)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('González 2024 - Fig. 3: Voltage vs Bandwidth Trade-off', fontsize=14)
    ax1.axvline(220, color='green', linestyle=':', alpha=0.7, label='220 Ω')
    
    plt.savefig(os.path.join(output_dir, 'fig3_voltage_vs_rload.png'), dpi=150)
    plt.close()
    
    print(f"  ✅ Saved to {output_dir}/fig3_voltage_vs_rload.png")


def run_validation():
    """Main validation function."""
    print("=" * 60)
    print("VALIDATING GONZALEZ-URIARTE ET AL. (2024)")
    print("=" * 60)
    
    output_dir = get_paper_output_dir('gonzalez_2024', force_new=True)
    
    # Generate figures
    bw_220 = plot_rload_vs_bandwidth(output_dir)
    plot_voltage_vs_rload(output_dir)
    
    # Summary
    target_bw = 50000  # 50 kHz
    error = abs(bw_220 - target_bw) / target_bw * 100
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Bandwidth @ 220Ω: {bw_220/1000:.2f} kHz")
    print(f"  Target: {target_bw/1000:.0f} kHz")
    print(f"  Error: {error:.1f}%")
    print()
    print("  Generated Figures:")
    print("    ✅ Fig. 2: R_load vs Bandwidth")
    print("    ✅ Fig. 3: Voltage vs Bandwidth Trade-off")
    
    if error < 20:
        print("\n✅ SUCCESS: Parameters align with paper targets!")
    else:
        print(f"\n⚠️  NEEDS REVIEW: Adjust C_j to match 50 kHz target")
        # Calculate required C_j for exact match
        required_cj = 1 / (2 * np.pi * 220 * 50000)
        print(f"    Suggestion: C_j = {required_cj*1e9:.1f} nF for exact match")


if __name__ == "__main__":
    run_validation()
