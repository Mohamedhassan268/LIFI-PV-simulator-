# main.py - 7-LAYER ARCHITECTURE WITH TIA (FULLY INTEGRATED)
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import signal as sp_signal
from simulator.transmitter import Transmitter
from simulator.channel import OpticalChannel
from simulator.receiver import PVReceiver
from simulator.demodulator import Demodulator


def run_complete_simulation_with_tia(config=None, export_csv=True, verbose=True):
    
    
    # ========== DEFAULT CONFIGURATION ==========
    if config is None:
        config = {
            'simulation': {
                'data_rate_bps': 1e6,
                'samples_per_bit': 500,
                'n_bits': 1000
            },
            'transmitter': {
                'dc_bias': 200,
                'modulation_depth': 0.9,
                'led_efficiency': 0.08
            },
            'channel': {
                'distance': 1.0,
                'beam_angle_half': 30,
                'receiver_area': 1.0
            },
            'receiver': {
                'responsivity': 0.42,
                'capacitance': 100,
                'shunt_resistance': 1.0,
                'dark_current': 1.0,
                'temperature': 300
            },
            'tia': {
                'R_tia': 50e3,  # 50 kΩ
                'f_3db': 3e6    # 3 MHz
            }
        }
    
    if verbose:
        print("\n" + "="*80)
        print("🚀 Li-Fi + PV SIMULATOR - 7-LAYER WITH TIA DUAL-PATH")
        print("="*80)
    
    # Extract parameters
    f_b = config['simulation']['data_rate_bps']
    sps = config['simulation']['samples_per_bit']
    n_bits = config['simulation']['n_bits']
    fs = f_b * sps
    t = np.arange(int(n_bits * sps)) / fs
    
    if verbose:
        print(f"\n📋 System Configuration:")
        print(f"  Data rate: {f_b/1e6:.1f} Mbps")
        print(f"  Sample rate: {fs/1e6:.1f} MHz")
        print(f"  Samples/bit: {sps}")
        print(f"  Total bits: {n_bits}")
        print(f"  Simulation time: {t[-1]*1e3:.2f} ms")
    
    # ========== LAYER 1: TRANSMITTER ==========
    if verbose:
        print(f"\n{'='*80}")
        print("LAYER 1️⃣: TRANSMITTER (OOK Modulation)")
        print(f"{'='*80}")
    
    bits_tx = np.random.randint(0, 2, n_bits)
    tx = Transmitter(config['transmitter'])
    P_tx = tx.modulate(bits_tx, t)
    bits_expanded = np.repeat(bits_tx, sps)
    
    if verbose:
        print(f"\n  Generated {n_bits} bits: {np.sum(bits_tx==1)} ones, {np.sum(bits_tx==0)} zeros")
        print(f"  Optical Power:")
        print(f"    Bit=1: {np.mean(P_tx[bits_expanded==1])*1e3:.3f} mW")
        print(f"    Bit=0: {np.mean(P_tx[bits_expanded==0])*1e3:.3f} mW")
        print(f"    Swing: {(P_tx.max() - P_tx.min())*1e3:.3f} mW")
    
    # ========== LAYER 2: CHANNEL ==========
    if verbose:
        print(f"\n{'='*80}")
        print("LAYER 2️⃣: CHANNEL (Lambertian Propagation + Noise)")
        print(f"{'='*80}")
    
    ch = OpticalChannel(config['channel'])
    P_rx_clean = ch.propagate(P_tx, t)
    
    # Convert to photocurrent and add noise
    rx_temp = PVReceiver(config['receiver'])
    I_ph_clean = rx_temp.optical_to_current(P_rx_clean)
    B = fs / 2
    noise = ch.add_noise(P_rx_clean, I_ph_clean, B, verbose=False)
    I_ph_noisy = I_ph_clean + noise
    
    if verbose:
        path_loss_db = 10 * np.log10(P_rx_clean.mean() / P_tx.mean())
        snr_optical = 10 * np.log10(np.var(I_ph_clean) / np.var(noise)) if np.var(noise) > 0 else np.inf
        print(f"\n  Path Loss: {path_loss_db:.1f} dB")
        print(f"  Received Power (clean):")
        print(f"    Bit=1: {np.mean(P_rx_clean[bits_expanded==1])*1e6:.3f} μW")
        print(f"    Bit=0: {np.mean(P_rx_clean[bits_expanded==0])*1e6:.3f} μW")
        print(f"  Noise: {np.std(noise)*1e9:.3f} nA (std dev)")
        print(f"  Optical SNR: {snr_optical:.1f} dB")
    
    # ========== LAYER 3: RECEIVER (DUAL-PATH) ==========
    if verbose:
        print(f"\n{'='*80}")
        print("LAYER 3️⃣: RECEIVER (PV Cell + TIA Dual-Path)")
        print(f"{'='*80}")
    
    rx = PVReceiver(config['receiver'])
    
    # Energy path: PV ODE solver
    V_pv = rx.solve_pv_circuit(I_ph_noisy, t, verbose=False)
    
    # Communication path: TIA amplifier
    R_tia = config['tia']['R_tia']
    f_3db_tia = config['tia']['f_3db']
    V_tia = rx.apply_tia(I_ph_noisy, t, R_tia=R_tia, f_3db=f_3db_tia)
    
    if verbose:
        print(f"\n  Photocurrent (with noise):")
        print(f"    Range: {I_ph_noisy.min()*1e6:.3f} to {I_ph_noisy.max()*1e6:.3f} μA")
        print(f"    Mean: {I_ph_noisy.mean()*1e6:.3f} μA")
        
        print(f"\n  Energy Path (PV ODE for harvesting):")
        print(f"    Bit=1: {np.mean(V_pv[bits_expanded==1])*1e3:.2f} mV")
        print(f"    Bit=0: {np.mean(V_pv[bits_expanded==0])*1e3:.2f} mV")
        print(f"    Swing: {(V_pv.max() - V_pv.min())*1e3:.2f} mV")
        
        print(f"\n  Communication Path (TIA for data):")
        print(f"    R_TIA: {R_tia/1e3:.0f} kΩ")
        print(f"    f_3dB: {f_3db_tia/1e6:.1f} MHz")
        print(f"    Bit=1: {np.mean(V_tia[bits_expanded==1])*1e3:.2f} mV")
        print(f"    Bit=0: {np.mean(V_tia[bits_expanded==0])*1e3:.2f} mV")
        print(f"    Swing: {(V_tia.max() - V_tia.min())*1e3:.2f} mV")
    
    # ========== LAYER 4: POST-PROCESSING (DUAL-PATH) ==========
    if verbose:
        print(f"\n{'='*80}")
        print("LAYER 4️⃣: POST-PROCESSING (Data Path via TIA + Energy Path via PV)")
        print(f"{'='*80}")
    
    # Create demodulator
    demod = Demodulator({
        'sample_rate': fs,
        'hpf_cutoff': 50e3,  # 50 kHz (optimized for TIA)
        'lpf_cutoff': 2e6,   # 2 MHz (optimized for TIA)
        'filter_order': 2
    })
    
    # Data path: HPF + LPF on TIA output
    V_data_hpf = demod.apply_hpf(V_tia)
    V_data_lpf = demod.apply_lpf(V_data_hpf)
    
    # Energy path: DC extraction via very low-pass filter
    lpf_dc_cutoff = 1000  # Hz
    lpf_dc_norm = lpf_dc_cutoff / (fs / 2)
    b_dc, a_dc = sp_signal.butter(2, lpf_dc_norm, btype='low')
    V_energy_dc = sp_signal.filtfilt(b_dc, a_dc, V_pv)
    
    if verbose:
        print(f"\n  DATA PATH (via TIA):")
        print(f"    HPF cutoff: 50 kHz")
        print(f"    LPF cutoff: 2 MHz")
        print(f"    After HPF: {V_data_hpf.min()*1e3:.2f} to {V_data_hpf.max()*1e3:.2f} mV")
        print(f"    After LPF: {V_data_lpf.min()*1e3:.2f} to {V_data_lpf.max()*1e3:.2f} mV")
        print(f"    Bit=1: {np.mean(V_data_lpf[bits_expanded==1])*1e3:.2f} mV")
        print(f"    Bit=0: {np.mean(V_data_lpf[bits_expanded==0])*1e3:.2f} mV")
        
        print(f"\n  ENERGY PATH (via PV ODE):")
        print(f"    DC filter cutoff: {lpf_dc_cutoff} Hz")
        print(f"    Mean DC: {np.mean(V_energy_dc)*1e3:.2f} mV")
    
    # ========== LAYER 5: BIT RECOVERY ==========
    if verbose:
        print(f"\n{'='*80}")
        print("LAYER 5️⃣: BIT RECOVERY (Sampling + Threshold Decision)")
        print(f"{'='*80}")
    
    sample_indices = np.arange(n_bits) * sps + sps // 2
    samples = V_data_lpf[sample_indices]
    
    threshold = (np.mean(samples[bits_tx==0]) + np.mean(samples[bits_tx==1])) / 2
    bits_rx = (samples > threshold).astype(int)
    
    if verbose:
        print(f"\n  Sampling:")
        print(f"    Bit=1 samples: mean={samples[bits_tx==1].mean()*1e3:.2f} mV, std={samples[bits_tx==1].std()*1e3:.2f} mV")
        print(f"    Bit=0 samples: mean={samples[bits_tx==0].mean()*1e3:.2f} mV, std={samples[bits_tx==0].std()*1e3:.2f} mV")
        print(f"    Separation: {(samples[bits_tx==1].mean() - samples[bits_tx==0].mean())*1e3:.2f} mV")
        print(f"\n  Decision:")
        print(f"    Threshold: {threshold*1e3:.2f} mV")
        
        # Show first 10 decisions
        print(f"\n  First 10 bit decisions:")
        print(f"  {'Bit#':<5} {'TX':<4} {'Sample(mV)':<12} {'RX':<4} {'Match':<6}")
        print(f"  {'-'*45}")
        for i in range(min(10, n_bits)):
            match = '✓' if bits_tx[i] == bits_rx[i] else '✗'
            print(f"  {i:<5} {bits_tx[i]:<4} {samples[i]*1e3:>10.2f}  {bits_rx[i]:<4} {match:<6}")
    
    # ========== LAYER 6: METRICS CALCULATION ==========
    if verbose:
        print(f"\n{'='*80}")
        print("LAYER 6️⃣: METRICS (BER + SNR + Harvested Energy)")
        print(f"{'='*80}")
    
    # BER
    errors = np.sum(bits_tx != bits_rx)
    ber = errors / n_bits
    
    # SNR (electrical, after TIA + filters)
    P_signal = np.var(V_data_lpf)
    P_noise_est = np.var(V_tia - V_energy_dc)
    snr_db = 10 * np.log10(P_signal / P_noise_est) if P_noise_est > 0 else np.inf
    
    # Q-factor
    mu_1 = np.mean(samples[bits_tx == 1])
    mu_0 = np.mean(samples[bits_tx == 0])
    sigma_1 = np.std(samples[bits_tx == 1])
    sigma_0 = np.std(samples[bits_tx == 0])
    q_factor = abs(mu_1 - mu_0) / (sigma_1 + sigma_0 + 1e-12)
    
    # Eye opening
    eye_open = abs(mu_1 - mu_0) / (abs(mu_1) + abs(mu_0) + 1e-12) * 100
    
    # Harvested energy
    P_harv_inst = V_energy_dc * I_ph_noisy
    P_harv_avg = np.mean(P_harv_inst)
    E_harv_total = np.trapezoid(P_harv_inst, t)
    
    if verbose:
        print(f"\n  COMMUNICATION METRICS:")
        print(f"    BER: {ber:.6f} ({ber*100:.4f}%)")
        print(f"    Bit errors: {errors} / {n_bits}")
        print(f"    SNR: {snr_db:.2f} dB")
        print(f"    Q-factor: {q_factor:.2f}")
        print(f"    Eye opening: {eye_open:.1f}%")
        
        print(f"\n  ENERGY HARVESTING METRICS:")
        print(f"    Avg power: {P_harv_avg*1e6:.3f} μW")
        print(f"    Total energy: {E_harv_total*1e12:.2f} pJ")
        print(f"    Duration: {t[-1]*1e3:.2f} ms")
        
        # Performance assessment
        if ber < 0.001:
            print(f"\n  ✅ EXCELLENT! BER < 0.1%")
        elif ber < 0.01:
            print(f"\n  ✅ GOOD! BER < 1%")
        elif ber < 0.1:
            print(f"\n  ⚠️  ACCEPTABLE: BER < 10%")
        else:
            print(f"\n  ❌ POOR: BER > 10%")
    
    # ========== LAYER 7: OUTPUT & STORAGE ==========
    if verbose:
        print(f"\n{'='*80}")
        print("LAYER 7️⃣: OUTPUT (Results Packaging + CSV Export)")
        print(f"{'='*80}")
    
    results = {
        # Configuration
        'distance_m': config['channel']['distance'],
        'data_rate_kbps': f_b / 1e3,
        'dc_bias_ma': config['transmitter']['dc_bias'],
        'modulation_depth': config['transmitter']['modulation_depth'],
        'capacitance_pf': config['receiver']['capacitance'],
        'r_tia_kohm': R_tia / 1e3,
        'f_3db_tia_mhz': f_3db_tia / 1e6,
        
        # Communication metrics
        'ber': ber,
        'errors': errors,
        'snr_db': snr_db,
        'q_factor': q_factor,
        'eye_opening_percent': eye_open,
        
        # Energy metrics
        'harvested_power_avg_uw': P_harv_avg * 1e6,
        'harvested_energy_pj': E_harv_total * 1e12,
        
        # Signal levels
        'received_power_avg_uw': np.mean(P_rx_clean) * 1e6,
        'photocurrent_avg_ua': np.mean(I_ph_clean) * 1e6,
        'v_pv_rms_mv': np.std(V_pv) * 1e3,
        'v_tia_rms_mv': np.std(V_tia) * 1e3,
        'v_data_rms_mv': np.std(V_data_lpf) * 1e3,
        
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'n_bits': n_bits,
        'samples_per_bit': sps,
    }
    
    # Export to CSV
    if export_csv:
        output_dir = Path('outputs/csv')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame([results])
        csv_file = output_dir / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        
        if verbose:
            print(f"\n  ✓ Results exported to CSV:")
            print(f"    File: {csv_file}")
            print(f"    Columns: {len(results)}")
    
    if verbose:
        print(f"\n{'='*80}")
        print("✅ SIMULATION COMPLETE")
        print(f"{'='*80}\n")
    
    return {
        'results': results,
        'waveforms': {
            't': t,
            'bits_tx': bits_tx,
            'P_tx': P_tx,
            'P_rx': P_rx_clean,
            'I_ph': I_ph_noisy,
            'V_pv': V_pv,
            'V_tia': V_tia,
            'V_data_lpf': V_data_lpf,
            'V_energy_dc': V_energy_dc,
            'samples': samples,
            'bits_rx': bits_rx,
        },
        'config': config
    }


if __name__ == "__main__":
    print("\n🔬 Running 7-layer simulation with TIA dual-path...")
    output = run_complete_simulation_with_tia(config=None, export_csv=True, verbose=True)
