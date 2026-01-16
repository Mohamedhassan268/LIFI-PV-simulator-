# fig18_vout_physics.py

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator.transmitter import Transmitter
from simulator.channel import OpticalChannel
from simulator.receiver import PVReceiver
from simulator.dc_dc_converter import DCDCConverter
from utils.constants import PAPER_VALIDATION_CONFIG

def calculate_vout_for_m(m, fsw_khz, verbose=False):
    """
    Calculate V_out for a given modulation depth and switching frequency.
    """
    # 1. Setup Parameters
    distance = 0.325 # meters
    P_tx_mw = 9.3    # mW
    rx_area_cm2 = 9.0 # cm^2 (Total module area)
    beam_angle = 9.0  # degrees
    
    # Calibration Parameters (tuned to match Fig 18)
    n_cells = PAPER_VALIDATION_CONFIG['n_cells_module']  # 13 cells in series
    duty_cycle = 0.44  # Fixed duty cycle for optimal boost to ~6V
    k_ripple = 0.32     # Ripple loss factor (linear drop with m) - DEPRECATED
    rsh_mohm = 100     # High shunt resistance (MOhm) to avoid loading
    
    # 2. Components
    tx = Transmitter({'dc_bias': 100, 'modulation_depth': m, 'led_efficiency': 0.1})
    tx.eta_led = (P_tx_mw * 1e-3) / (tx.I_dc_ma * 1e-3)
    
    ch = OpticalChannel({'distance': distance, 'beam_angle_half': beam_angle, 'receiver_area': rx_area_cm2})
    
    rx = PVReceiver({
        'responsivity': PAPER_VALIDATION_CONFIG['responsivity_a_per_w'],
        'capacitance': PAPER_VALIDATION_CONFIG['cj_pf'],  # 798 pF
        'shunt_resistance': rsh_mohm, 
        'dark_current': 1.0, 
        'temperature': 300
    })
    
    dcdc = DCDCConverter({
        'fsw_khz': fsw_khz,
        'duty_cycle': duty_cycle,
        'v_diode_drop': 0.3, 
        'r_on_mohm': 100,
        'efficiency_mode': 'paper'
    })
    
    # 3. Simulation Waveforms
    duration = 2e-3 # 2 ms
    fs = 1e6 # 1 MHz
    t = np.arange(0, duration, 1/fs)
    
    # Generate Bits
    n_bits = int(duration * 10000)
    bits = np.random.randint(0, 2, max(10, n_bits))
    
    P_tx = tx.modulate(bits, t)
    P_rx = ch.propagate(P_tx, t)
    I_ph = rx.optical_to_current(P_rx)
    I_ph_avg = np.mean(I_ph)
    
    # Solve PV Circuit (Single Cell)
    V_pv_waveform = rx.solve_pv_circuit(I_ph, t, method='euler')
    V_pv_avg_cell = np.mean(V_pv_waveform[int(len(t)*0.5):]) # Last 50%
    
    # Scale to Module Voltage (13 cells in series)
    V_pv_module = V_pv_avg_cell * n_cells
    
    # Use pure physics model
    # Ripple effects now handled by DC-DC converter DCM model
    # (Removed empirical ripple_factor = 1.0 - k_ripple * m)
    V_pv_effective = V_pv_module
    
    # 4. DC-DC Converter Output
    output = dcdc.calculate_output(V_pv_effective, I_ph_avg, duty_cycle=duty_cycle, fsw_khz=fsw_khz)
    
    return output['V_out'], V_pv_effective

def plot_fig18_physics(output_dir=None):
    """
    Generate Fig. 18 plot.
    DC-DC Output Voltage vs Modulation Depth.
    """
    if output_dir is None:
        from utils.output_manager import get_plots_dir
        output_dir = get_plots_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    # Exact points from user description
    m_percent = np.array([10, 20, 40, 60, 80, 100])
    fsw_list = [50, 100, 200]
    
    styles = {
        50:  {'color': 'blue',    'linestyle': (0, (5, 5)),  'marker': 'o', 'markerfacecolor': 'white', 'label': r'$f_{sw} = 50$ kHz'},
        100: {'color': 'red',     'linestyle': (0, (3, 1)),  'marker': 'x', 'markerfacecolor': 'red',   'label': r'$f_{sw} = 100$ kHz'},
        200: {'color': 'magenta', 'linestyle': '-.',         'marker': 's', 'markerfacecolor': 'white', 'label': r'$f_{sw} = 200$ kHz'},
    }
    
    plt.figure(figsize=(10, 6))
    
    print("Simulating Fig 18 (Vout vs m)...")
    
    for fsw in fsw_list:
        vout_values = []
        for m_pct in tqdm(m_percent, desc=f"fsw={fsw}kHz"):
            m = m_pct / 100.0
            vout, _ = calculate_vout_for_m(m, fsw)
            vout_values.append(vout)
            
        plt.plot(m_percent, vout_values, 
                 color=styles[fsw]['color'],
                 linestyle=styles[fsw]['linestyle'],
                 marker=styles[fsw]['marker'],
                 markerfacecolor=styles[fsw]['markerfacecolor'],
                 markeredgecolor=styles[fsw]['color'],
                 markersize=8,
                 linewidth=2,
                 label=styles[fsw]['label'])
        
        # Check calibration against key points
        if fsw == 50:
            print(f"50kHz check: m=10%->{vout_values[0]:.2f}V (target 6.0), m=100%->{vout_values[-1]:.2f}V (target 4.2)")

    plt.xlabel(r'Modulation depth $m$ (%)', fontsize=12)
    plt.ylabel(r'Output Voltage $V_{out}$ (V)', fontsize=12)
    plt.title('DC-DC Output Voltage vs Modulation Depth (Fig. 18)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.xlim(0, 105)
    # Auto-scale y-axis for physics-based output
    plt.ylim(0, 15)
    plt.xticks([10, 20, 40, 60, 80, 100])
    
    path = os.path.join(output_dir, "fig18_vout_physics.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

if __name__ == "__main__":
    plot_fig18_physics()
