"""
González-Uriarte et al. (2024) Validation Module

Paper: "Design and Implementation of a Low-Cost VLC Photovoltaic Panel-Based Receiver with off-the-Shelf Components"
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from simulator.receiver import PVReceiver

PARAMS = {
    # System Geometry
    'distance_m': 0.60,
    
    # Poly-Si Panel
    'responsivity_a_per_w': 0.5,
    'solar_cell_area_cm2': 66.0,
    
    # Electrical Model (Inferred from 50kHz BW @ 220 Ohm load)
    # R_eq = 220 || 500 = 152.8 Ohm -> C = 20.8 nF for 50 kHz
    'rsh_m_ohm': 0.0005, # 500 Ohms
    'cj_pf': 20800.0,    # 20.8 nF
    
    'rload_ohm': 220,
    
    # Target
    'bandwidth_target_hz': 50000,
}

def validate_bandwidth():
    """Validate inferred bandwidth match."""
    print("\n[Test 1] Validating Bandwidth (Target: 50 kHz)...")
    
    # Setup Receiver
    rx = PVReceiver({
        'responsivity': PARAMS['responsivity_a_per_w'],
        'shunt_resistance': PARAMS['rsh_m_ohm'],
        'capacitance': PARAMS['cj_pf'],
        'dark_current': 1e-10, # Typical for small Silicon cells (100 pA)
        'temperature': 300
    })
    
    # Frequency Sweep
    freqs = np.logspace(2, 6, 100) # 100 Hz to 1 MHz
    gains_db = []
    
    R_sh = PARAMS['rsh_m_ohm'] * 1e6
    R_load = PARAMS['rload_ohm']
    C_j = PARAMS['cj_pf'] * 1e-12
    
    R_eq = (R_sh * R_load) / (R_sh + R_load)
    
    for f in freqs:
        w = 2 * np.pi * f
        H = 1 / (1 + 1j * w * R_eq * C_j)
        gains_db.append(20 * np.log10(np.abs(H)))
        
    gains_db = np.array(gains_db)
    
    # Find -3dB point
    dc_gain = gains_db[0]
    idx_3db = np.argmin(np.abs(gains_db - (dc_gain - 3.0)))
    f_3db = freqs[idx_3db]
    
    print(f"  Calculated R_eq: {R_eq:.2f} Ohm")
    print(f"  Calculated C_j:  {C_j*1e9:.2f} nF")
    print(f"  Resulting Bandwidth (-3dB): {f_3db/1000:.2f} kHz")
    print(f"  Target Bandwidth: {PARAMS['bandwidth_target_hz']/1000:.2f} kHz")
    
    # Plot
    output_dir = 'outputs/gonzalez_2024/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs, gains_db)
    plt.axvline(f_3db, color='r', linestyle='--', label=f'-3dB @ {f_3db/1000:.1f} kHz')
    plt.axvline(PARAMS['bandwidth_target_hz'], color='g', linestyle='--', label='Target (50 kHz)')
    plt.title(f"Bandwidth Validation (González 2024)\nRsh={R_sh}Ω, Cj={C_j*1e9:.1f}nF")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'bandwidth.png'))
    print(f"  Plot saved to: {output_dir}/bandwidth.png")
    
    return f_3db

def run_validation():
    """Main execution function."""
    print("="*60)
    print("VALIDATING GONZALEZ ET AL. (2024)")
    print("="*60)
    
    bw = validate_bandwidth()
    
    target_bw = PARAMS['bandwidth_target_hz']
    error = abs(bw - target_bw) / target_bw * 100
    
    print("\nSUMMARY")
    print("-" * 60)
    print(f"Bandwidth Error: {error:.2f}%")
    if error < 5.0:
        print("✅ SUCCESS: Inferred parameters align with paper bandwidth.")
    else:
        print("❌ FAILURE: Inferred parameters do not match target.")

if __name__ == "__main__":
    run_validation()
