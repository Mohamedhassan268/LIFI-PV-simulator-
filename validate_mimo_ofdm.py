import numpy as np
import matplotlib.pyplot as plt
from simulator.transmitter import Transmitter
from simulator.channel import OpticalChannel
from simulator.receiver import MIMOPVReceiver
from simulator.demodulator import Demodulator

def run_mimo_benchmark():
    print("="*60)
    print("VALIDATING MIMO & ADAPTIVE OFDM (Library Refactor)")
    print("="*60)
    
    # 1. Setup MIMO Geometry (Room 5x5m)
    # 4 Transmitters on Ceiling (z=3m)
    tx_pos = [
        [1.5, 1.5, 3.0], 
        [3.5, 1.5, 3.0],
        [1.5, 3.5, 3.0],
        [3.5, 3.5, 3.0]
    ]
    
    # 4 Receivers on Desk (z=0.8m) directly under Tx
    rx_pos = [
        [1.5, 1.5, 0.8],
        [3.5, 1.5, 0.8],
        [1.5, 3.5, 0.8],
        [3.5, 3.5, 0.8]
    ]
    
    ch = OpticalChannel({'receiver_area': 1.0, 'beam_angle_half': 45})
    
    # 2. Compute Channel Matrix
    H = ch.compute_h_matrix(tx_pos, rx_pos)
    
    print("MIMO Channel Matrix H (4x4):")
    print(np.array2string(H, precision=2, suppress_small=True))
    
    # 3. Simulate MIMO Transmission
    # Generate 4 independent streams
    tx = Transmitter({'dc_bias': 100, 'modulation_depth': 0.5})
    t = np.linspace(0, 1e-4, 1000)
    
    # Stream 1: Simple Sine
    s1 = 100 + 50 * np.sin(2*np.pi*1e5*t)
    # Stream 2: Simple Cosine
    s2 = 100 + 50 * np.cos(2*np.pi*1e5*t)
    # Stream 3: Square
    s3 = 100 + 50 * np.sign(np.sin(2*np.pi*5e4*t))
    # Stream 4: DC
    s4 = np.ones_like(t) * 100
    
    P_tx_mimo = np.vstack([s1, s2, s3, s4]) * 1e-3 # Convert mA to Watts (approx)
    
    # Propagate (Optical)
    P_rx_optical = ch.propagate(P_tx_mimo, t, H_matrix=H)
    
    # PV Receiver (MIMO)
    rx_mimo = MIMOPVReceiver(num_receivers=4, params={'responsivity': 0.5})
    I_ph_mimo = rx_mimo.optical_to_current(P_rx_optical)
    
    # Solve Physics (Optional, skipping here for speed, just using I_ph as proxy for signal)
    # In full chain: V_mimo = rx_mimo.solve_pv_circuit(I_ph_mimo, t)
    
    print(f"\nPropagation Complete. Rx Shape: {I_ph_mimo.shape}")
    
    # 4. MIMO Demodulator (Zero-Forcing)
    demod = Demodulator({'sample_rate': 1e7})
    
    # Note: Optical Channel H is Power-to-Power.
    # Receiver: I_ph = R * P_rx
    # So effective H_eff = R * H
    # We want to estimate P_tx from I_ph
    # I_ph = R * H * P_tx
    # P_tx = inv(R*H) * I_ph
    
    R_resp = 0.5
    H_eff = R_resp * H
    
    P_est_mimo = demod.demodulate_mimo(I_ph_mimo, H_eff, method='zf')
    
    # Error check
    mse = np.mean((P_est_mimo - P_tx_mimo)**2)
    print(f"Zero-Forcing MSE: {mse:.2e} (Should be near zero)")
    
    if mse < 1e-10:
         print("✅ MIMO Spatial Demultiplexing Successful (Library Class)!")
    else:
         print("❌ MIMO Demux Failed")
         
    # 5. Adaptive Bit Loading Test
    print("\n[Adaptive Bit Loading Test]")
    n_carriers = 31 
    snr_profile = np.linspace(35, 0, n_carriers) # dB
    
    qam_orders = []
    for snr in snr_profile:
        if snr > 25: qam_orders.append(256)
        elif snr > 20: qam_orders.append(64)
        elif snr > 15: qam_orders.append(16)
        elif snr > 10: qam_orders.append(4)
        else: qam_orders.append(2) 
        
    print(f"Assigned QAM Orders: {qam_orders}")
    
    bits_needed = sum([int(np.log2(q)) for q in qam_orders])
    bits = np.random.randint(0, 2, bits_needed)
    
    try:
        tx.modulate_ofdm(bits, np.linspace(0, 1e-5, 100), qam_order=qam_orders, n_fft=64)
        print("✅ Adaptive `modulate_ofdm` (2-256 QAM) ran successfully.")
    except Exception as e:
        print(f"❌ Adaptive Modulation Failed: {e}")

if __name__ == "__main__":
    run_mimo_benchmark()

