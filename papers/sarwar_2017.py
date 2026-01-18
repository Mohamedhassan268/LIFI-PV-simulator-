"""
Sarwar et al. (2017) Validation Module

Paper: "Visible Light Communication using Silicon Photodiode with RGB LED"
Key Target: 15 Mbps using OFDM with active DC-DC bias (Simulated here with generic params)

This originally existed as `validate_ofdm_sarwar.py`.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from simulator.transmitter import Transmitter
from simulator.demodulator import Demodulator
from simulator.receiver import PVReceiver
from simulator.channel import OpticalChannel
from utils.output_manager import get_paper_output_dir

# Parameters approximating Sarwar et al. 2017
# Note: Since they didn't provide physics specs, we use a "High-Bandwidth Silicon" model
PARAMS = {
    'qam_order': 16,
    'n_fft': 256,
    'cp_len': 32,
    
    # Target: 15 Mbps
    # Bandwidth usage: 15 Mbps / 4 bits/sym = 3.75 Msps
    # If using N=256, effective carriers N/2-1 = 127
    # 4 bits * 127 = 508 bits per OFDM sym
    # Symbol rate = Target / 508 = 29.5 k sym/sec
    # Sample rate = N_total * sym_rate = (256+32) * 29.5k = 8.5 MS/s
    'sample_rate': 10e6, # 10 MS/s
    'data_rate_target': 15e6,
    
    # Physics (Generic Silicon)
    'rsh_ohm': 2000,       # High quality silicon
    'cj_pf': 200,          # Low capacitance (small cell)
    'responsivity': 0.6,
    'distance': 2.0,       # 2 meters (Sarwar)
}

def run_validation():
    output_dir = get_paper_output_dir('sarwar_2017')
    # Folder already created by get_paper_output_dir
    
    print("="*60)
    print("VALIDATING SARWAR et al. (2017) - OFDM Benchmark")
    print("="*60)
    
    # 1. Setup Components
    tx = Transmitter({
        'dc_bias': 100, # mA
        'modulation_depth': 0.5,
        'sample_rate': PARAMS['sample_rate']
    })
    
    demod = Demodulator({'sample_rate': PARAMS['sample_rate']})
    
    # 2. Generate Data
    n_ofdm_symbols = 50
    # Bits calculation: (N/2 - 1) data carriers * 4 bits/symbol
    bits_per_symbol = (PARAMS['n_fft'] // 2 - 1) * int(np.log2(PARAMS['qam_order']))
    total_bits = n_ofdm_symbols * bits_per_symbol
    
    bits_tx = np.random.randint(0, 2, total_bits)
    
    print(f"Generating {total_bits} bits...")
    
    # 3. Transmit (OFDM)
    # Define time vector for the duration of the transmission
    # Total samples = Syms * (N+CP)
    n_samples = n_ofdm_symbols * (PARAMS['n_fft'] + PARAMS['cp_len'])
    t = np.arange(n_samples) / PARAMS['sample_rate']
    
    P_tx = tx.modulate_ofdm(
        bits_tx, t, 
        qam_order=PARAMS['qam_order'], 
        n_fft=PARAMS['n_fft'], 
        cp_len=PARAMS['cp_len']
    )
    
    print(f"Signal Generated. Length: {len(P_tx)}")
    
    # 4. Channel (Line of Sight Air)
    ch = OpticalChannel({'distance': PARAMS['distance'], 'receiver_area': 1.0}) # 1cm2 for standard PD
    P_rx = ch.propagate(P_tx, t)
    
    # 5. Receiver (Physics Model)
    # Using generic silicon parameters
    rx = PVReceiver({
        'responsivity': PARAMS['responsivity'],
        'shunt_resistance': PARAMS['rsh_ohm'] / 1e6, # Ohm -> MOhm
        'capacitance': PARAMS['cj_pf'],
    })
    
    # PV Output
    # Note: OFDM is high bandwidth. We expect attenuation here.
    I_ph = rx.optical_to_current(P_rx)
    V_pv = rx.solve_pv_circuit(I_ph, t, R_load=50, verbose=False) # 50 Ohm matching
    
    # 6. Demodulate
    bits_rx, symbols_rx = demod.demodulate_ofdm(
        V_pv, 
        n_fft=PARAMS['n_fft'], 
        cp_len=PARAMS['cp_len'], 
        qam_order=PARAMS['qam_order']
    )
    
    # 7. Metrics
    # BER
    min_len = min(len(bits_tx), len(bits_rx))
    errors = np.sum(bits_tx[:min_len] != bits_rx[:min_len])
    ber = errors / min_len
    
    # Throughput Calculation
    throughput_mbps = PARAMS['sample_rate'] * 4 * (127/288) / 1e6
    
    print("\nRESULTS")
    print("-" * 60)
    print(f"BER: {ber:.6f} ({errors} errors)")
    print(f"Throughput: {throughput_mbps:.2f} Mbps (Approx)")
    
    # Plot Constellation
    plt.figure(figsize=(6, 6))
    plt.scatter(symbols_rx.real, symbols_rx.imag, alpha=0.5, s=10)
    plt.title(f"Received Constellation (BER={ber:.4f})")
    plt.grid(True)
    plt.xlabel("I")
    plt.ylabel("Q")
    plot_path = os.path.join(output_dir, 'ofdm_constellation.png')
    plt.savefig(plot_path)
    print(f"Constellation saved to {plot_path}")
    
    if ber < 0.1:
        print("✅ SUCCESS: OFDM Pipeline functional.")
    else:
        print("❌ FAILURE: High BER (Check Signal Power / SNR).")
    
    return True


if __name__ == "__main__":
    run_validation()
