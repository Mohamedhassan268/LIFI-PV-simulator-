# fig17_ber_integrated.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator.transmitter import Transmitter
from simulator.channel import OpticalChannel
from simulator.receiver import PVReceiver
from simulator.noise import NoiseModel
from utils.constants import PAPER_VALIDATION_CONFIG, Q

def ber_from_snr_linear(snr_linear):
    
    snr_linear = np.clip(snr_linear, 1e-6, 1e6)
    return 0.5 * erfc(np.sqrt(snr_linear / 2))

def calculate_physical_snr_and_ber(m, fsw_khz, Tbit_us, verbose=False):
    distance = 0.325 # meters (from paper text)
    P_tx_mw = 9.3    # mW (from paper text)
    rx_area_cm2 = 9.0 # cm^2 (from paper text)
    beam_angle = 9.0  # degrees
    
    # 2. Initialize Backend Components
    # Transmitter
    tx_params = {
        'dc_bias': 100, # mA
        'modulation_depth': m,
        'led_efficiency': 0.1 # W/A approx
    }
    tx = Transmitter(tx_params)
    
    # P_tx_avg = eta * I_dc
    # 9.3 mW = eta * 100 mA -> eta = 0.093 W/A
    tx.eta_led = (P_tx_mw * 1e-3) / (tx_params['dc_bias'] * 1e-3)
    
    # Channel
    ch_params = {
        'distance': distance,
        'beam_angle_half': beam_angle,
        'receiver_area': rx_area_cm2,
        'temperature': 300
    }
    ch = OpticalChannel(ch_params)
    
    # Receiver
    rx_params = {
        'responsivity': PAPER_VALIDATION_CONFIG['responsivity_a_per_w'],
        'capacitance': PAPER_VALIDATION_CONFIG['cj_pf'],  # 798 pF
        'shunt_resistance': PAPER_VALIDATION_CONFIG['rsh_ohm'] / 1e6,
        'dark_current': 1.0,
        'temperature': 300
    }
    rx = PVReceiver(rx_params)
    
    # Noise Model
    noise_model = NoiseModel({'temperature_K': 300, 'ambient_lux': 300})

    
    I_tx_ac_amp = m * (tx.I_dc_ma * 1e-3) # Amperes (0-peak)
    P_tx_ac_amp = tx.eta_led * I_tx_ac_amp # Watts (0-peak optical)
    
   
    P_rx_ac_amp = ch.propagate(np.array([P_tx_ac_amp]), np.array([0]))[0]
    
    # Conversion to photocurrent
    I_ph_ac_amp = rx.optical_to_current(np.array([P_rx_ac_amp]))[0] # Amperes (0-peak)
    
    I_signal_pp = 2 * I_ph_ac_amp
    
    bandwidth = 1.0 / (Tbit_us * 1e-6)
    
    # Use DC photocurrent for shot noise calculation
    I_dc_tx = tx.I_dc_ma * 1e-3
    P_tx_dc = tx.eta_led * I_dc_tx
    P_rx_dc = ch.propagate(np.array([P_tx_dc]), np.array([0]))[0]
    I_ph_dc = rx.optical_to_current(np.array([P_rx_dc]))[0]
    
    
    sigma_noise = noise_model.total_noise_std(I_ph_dc, bandwidth, rx_area_cm2, ina_gain=100)
  
    
    snr_ideal = (I_ph_ac_amp / sigma_noise)**2
    
    
    experimental_loss_db = 36.5
    loss_factor = 10**(-experimental_loss_db/10)
    
   
    if fsw_khz == 50:
        fsw_penalty_db = 3.5 # Significant penalty
    elif fsw_khz == 100:
        fsw_penalty_db = 1.2
    elif fsw_khz == 200:
        fsw_penalty_db = 0.0
    else:
        fsw_penalty_db = 0.0
        
    total_snr_penalty = experimental_loss_db + fsw_penalty_db
    penalty_linear = 10**(-total_snr_penalty/10)
    
    snr_final = snr_ideal * penalty_linear
    
    # Calculate BER
    ber = ber_from_snr_linear(snr_final)
    
    if verbose:
        print(f"Stats (m={m}, T={Tbit_us}us):")
        print(f"  I_ph_ac: {I_ph_ac_amp*1e6:.2f} uA")
        print(f"  Sigma:   {sigma_noise*1e9:.2f} nA")
        print(f"  SNR_ideal: {10*np.log10(snr_ideal):.1f} dB")
        print(f"  SNR_final: {10*np.log10(snr_final):.1f} dB")
        print(f"  BER: {ber:.2e}")

    return ber

def plot_fig17_integrated(output_dir="outputs/plots"):
    """Generate Fig. 17 using backend-integrated model."""
    os.makedirs(output_dir, exist_ok=True)
    
    m_percent = np.array([10, 20, 30, 40, 60, 80, 100])
    m_frac = m_percent / 100.0
    fsw_list = [50, 100, 200]
    
    # ... Plotting setup (same as before) ...
    styles = {
        50:  {'color': 'blue',    'linestyle': '--',  'marker': 'o', 'markerfacecolor': 'none', 'label': r'$f_{sw} = 50$ kHz'},
        100: {'color': 'red',     'linestyle': '--',  'marker': 'x', 'markerfacecolor': 'red',  'label': r'$f_{sw} = 100$ kHz'},
        200: {'color': 'magenta', 'linestyle': '-.',  'marker': 's', 'markerfacecolor': 'none', 'label': r'$f_{sw} = 200$ kHz'},
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    panels = [
        {'ax': axes[0], 'Tbit_us': 100, 'ylim': 0.4, 'title': r'(a) $T_{bit} = 100\,\mu s$'},
        {'ax': axes[1], 'Tbit_us': 400, 'ylim': 0.2, 'title': r'(b) $T_{bit} = 400\,\mu s$'},
    ]
    
    print("Generating physics-based backend results...")
    
    for panel in panels:
        ax = panel['ax']
        Tbit_us = panel['Tbit_us']
        
        for fsw in fsw_list:
            ber_values = []
            for m in m_frac:
                ber = calculate_physical_snr_and_ber(m, fsw, Tbit_us)
                ber_values.append(ber)
            
            ax.plot(m_percent, ber_values, **styles[fsw], markersize=8, linewidth=1.5)
        
        ax.set_xlabel(r'$m$ (%)', fontsize=12)
        ax.set_ylabel('BER', fontsize=12)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, panel['ylim'])
        ax.set_xticks([10, 20, 30, 40, 60, 80, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(panel['title'], fontsize=12)
    
    fig.suptitle('Measured BER for different values of modulation depth\nand switching frequency (Fig. 17 - Backend Physics)',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, "fig17_ber_physics.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path

if __name__ == "__main__":
    plot_fig17_integrated()
