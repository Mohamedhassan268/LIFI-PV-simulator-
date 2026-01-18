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
from utils.output_manager import get_general_output_dir


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
                'responsivity': 0.457,     # Paper: GaAs 0.457 A/W (was 0.42)
                'capacitance': 798,         # Paper: 798 pF (was 100)
                'shunt_resistance': 0.1388, # Paper: 138.8 Ω = 0.1388 MΩ (was 1.0)
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
    
    # Create summary results dictionary
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
    
    # Export to CSV and save plots
    if export_csv:
        # Create output folders
        output_dir = Path(get_general_output_dir())  # outputs/general/Jan16_11pm_test1/
        csv_dir = Path('outputs/csv')  # CSVs stay in central location
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create DataFrame with ALL samples + ALL metadata
        waveform_df = pd.DataFrame({
            # Time axis
            'time_s': t,
            'time_ms': t * 1e3,
            
            # Waveform data
            'bit_tx': bits_expanded,
            'P_tx_mW': P_tx * 1e3,
            'P_rx_mW': P_rx_clean * 1e3,
            'I_ph_uA': I_ph_noisy * 1e6,
            'V_pv_mV': V_pv * 1e3,
            'V_tia_mV': V_tia * 1e3,
            'V_data_hpf_mV': V_data_hpf * 1e3,
            'V_data_lpf_mV': V_data_lpf * 1e3,
            'V_energy_dc_mV': V_energy_dc * 1e3,
            'P_harv_uW': P_harv_inst * 1e6,
        })
        
        # Add all configuration parameters as constant columns
        waveform_df['distance_m'] = config['channel']['distance']
        waveform_df['data_rate_kbps'] = f_b / 1e3
        waveform_df['dc_bias_ma'] = config['transmitter']['dc_bias']
        waveform_df['modulation_depth'] = config['transmitter']['modulation_depth']
        waveform_df['capacitance_pf'] = config['receiver']['capacitance']
        waveform_df['r_tia_kohm'] = R_tia / 1e3
        waveform_df['f_3db_tia_mhz'] = f_3db_tia / 1e6
        
        # Add all performance metrics as constant columns
        waveform_df['ber'] = ber
        waveform_df['errors'] = errors
        waveform_df['snr_db'] = snr_db
        waveform_df['q_factor'] = q_factor
        waveform_df['eye_opening_percent'] = eye_open
        waveform_df['harvested_power_avg_uw'] = P_harv_avg * 1e6
        waveform_df['harvested_energy_pj'] = E_harv_total * 1e12
        waveform_df['received_power_avg_uw'] = np.mean(P_rx_clean) * 1e6
        waveform_df['photocurrent_avg_ua'] = np.mean(I_ph_clean) * 1e6
        waveform_df['v_pv_rms_mv'] = np.std(V_pv) * 1e3
        waveform_df['v_tia_rms_mv'] = np.std(V_tia) * 1e3
        waveform_df['v_data_rms_mv'] = np.std(V_data_lpf) * 1e3
        
        # Add simulation metadata
        waveform_df['timestamp'] = datetime.now().isoformat()
        waveform_df['n_bits'] = n_bits
        waveform_df['samples_per_bit'] = sps
        
        # Save waveforms to central CSV folder
        waveform_file = csv_dir / f"waveforms_{timestamp}.csv"
        waveform_df.to_csv(waveform_file, index=False)
        
        # Save summary metrics to test folder
        summary_df = pd.DataFrame([results])
        summary_file = output_dir / f"summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        if verbose:
            print(f"\n  ✓ Results exported:")
            print(f"    Test folder: {output_dir}")
            print(f"    Summary: {summary_file}")
            print(f"    Waveforms: {waveform_file}")
            print(f"      - Rows: {len(waveform_df):,} samples")
            print(f"      - File size: ~{waveform_file.stat().st_size / 1024:.1f} KB")
        
        # ========== GENERATE PLOTS ==========
        import matplotlib.pyplot as plt
        
        # Plot 1: Time-Domain Waveforms (first 50 bits)
        n_bits_show = min(50, n_bits)
        n_samples_show = n_bits_show * sps
        t_show = t[:n_samples_show] * 1e6  # Convert to microseconds
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        axes[0].plot(t_show, P_tx[:n_samples_show] * 1e3, 'b-', linewidth=0.5)
        axes[0].set_ylabel('P_tx (mW)')
        axes[0].set_title(f'Time-Domain Waveforms (First {n_bits_show} bits)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t_show, P_rx_clean[:n_samples_show] * 1e6, 'g-', linewidth=0.5)
        axes[1].set_ylabel('P_rx (μW)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(t_show, V_tia[:n_samples_show] * 1e3, 'r-', linewidth=0.5)
        axes[2].set_ylabel('V_tia (mV)')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(t_show, V_data_lpf[:n_samples_show] * 1e3, 'm-', linewidth=0.5)
        axes[3].set_ylabel('V_data (mV)')
        axes[3].set_xlabel('Time (μs)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'time_domain_waveforms.png', dpi=150)
        plt.close()
        
        # Plot 2: Eye Diagram
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(min(200, n_bits - 1)):
            start_idx = i * sps
            end_idx = start_idx + 2 * sps
            if end_idx <= len(V_data_lpf):
                segment = V_data_lpf[start_idx:end_idx] * 1e3
                t_eye = np.linspace(0, 2, len(segment))
                ax.plot(t_eye, segment, 'b-', alpha=0.1, linewidth=0.5)
        
        ax.set_xlabel('Time (bit periods)')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_title(f'Eye Diagram (Q={q_factor:.2f}, BER={ber:.2e})')
        ax.grid(True, alpha=0.3)
        ax.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='Sample point')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'eye_diagram.png', dpi=150)
        plt.close()
        
        # Plot 3: Signal Constellation / Sampled Values
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(samples[bits_tx == 0] * 1e3, bins=30, alpha=0.7, label='Bit 0', color='blue')
        ax.hist(samples[bits_tx == 1] * 1e3, bins=30, alpha=0.7, label='Bit 1', color='red')
        ax.axvline(threshold * 1e3, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold*1e3:.2f} mV)')
        ax.set_xlabel('Sample Amplitude (mV)')
        ax.set_ylabel('Count')
        ax.set_title(f'Sample Distribution (Separation={abs(mu_1-mu_0)*1e3:.2f} mV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_distribution.png', dpi=150)
        plt.close()
        
        # Plot 4: Energy Harvesting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t * 1e3, P_harv_inst * 1e6, 'g-', linewidth=0.5, alpha=0.5, label='Instantaneous')
        ax.axhline(P_harv_avg * 1e6, color='r', linestyle='--', linewidth=2, label=f'Average ({P_harv_avg*1e6:.3f} μW)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Harvested Power (μW)')
        ax.set_title('Energy Harvesting Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'energy_harvesting.png', dpi=150)
        plt.close()
        
        if verbose:
            print(f"\n  ✓ Plots saved:")
            print(f"    - time_domain_waveforms.png")
            print(f"    - eye_diagram.png")
            print(f"    - sample_distribution.png")
            print(f"    - energy_harvesting.png")
    
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

# ========== PAPER VALIDATION MODE (Kadirvelu et al. IEEE TGCN 2021) ==========

def run_paper_validation_mode(verbose=True):
    """
    Run complete paper validation including parameter sweeps and all plots.
    
    Reproduces key results from:
    "Kadirvelu et al., A Circuit for Simultaneous Reception of Data and Power
     Using a Solar Cell, IEEE TGCN, 2021."
    
    Generates:
    - Fig 13: Frequency response
    - Fig 15: Harvested power vs fsw
    - Fig 16: Time-domain waveforms
    - Fig 17: BER vs modulation depth
    - CSV export of all sweep results
    - Validation report comparing to paper targets
    """
    from utils.constants import (
        PAPER_VALIDATION_CONFIG, BER_TEST_CONFIG, PAPER_TARGETS
    )
    from utils.plotting import (
        plot_frequency_response, plot_harvested_power,
        plot_ber_vs_modulation, plot_time_domain_waveforms
    )
    from scipy import fft
    
    print("\n" + "="*80)
    print("🔬 PAPER VALIDATION MODE - Kadirvelu et al. IEEE TGCN 2021")
    print("="*80)
    print("\nTarget conditions:")
    print(f"  Distance: {PAPER_VALIDATION_CONFIG['distance_m']} m")
    print(f"  Radiated power: {PAPER_VALIDATION_CONFIG['radiated_power_mw']} mW")
    print(f"  Solar cell area: {PAPER_VALIDATION_CONFIG['solar_cell_area_cm2']} cm²")
    print(f"  Responsivity: {PAPER_VALIDATION_CONFIG['responsivity_a_per_w']} A/W")
    print(f"  Receiver chain: Rsense({PAPER_VALIDATION_CONFIG['rsense_ohm']}Ω) → "
          f"INA({PAPER_VALIDATION_CONFIG['ina_gain']}×) → "
          f"BPF({PAPER_VALIDATION_CONFIG['bpf_low_hz']}-{PAPER_VALIDATION_CONFIG['bpf_high_hz']} Hz)")
    
    # Result storage
    all_results = []
    ber_sweep_data = {}  # {data_rate: {fsw: [ber values for each mod_depth]}}
    power_sweep_data = {}  # {mod_depth: [power values for each fsw]}
    
    # ========== PARAMETER SWEEPS ==========
    print("\n" + "-"*80)
    print("Running parameter sweeps...")
    print("-"*80)
    
    data_rates = BER_TEST_CONFIG['data_rates_bps']
    mod_depths = BER_TEST_CONFIG['mod_depths']
    fsw_list = BER_TEST_CONFIG['fsw_khz']
    
    # Use fewer bits for initial sweeps (speed), then full 1M for final validation
    n_bits_sweep = 10000  # 10k for sweeps (faster)
    n_bits_final = 100000  # 100k for final validation (balance speed/accuracy)
    
    # Initialize nested dicts
    for rate in data_rates:
        ber_sweep_data[rate] = {}
        for fsw in fsw_list:
            ber_sweep_data[rate][fsw] = []
    
    for mod in mod_depths:
        power_sweep_data[mod] = []
    
    total_runs = len(data_rates) * len(mod_depths) * len(fsw_list)
    run_count = 0
    
    for data_rate in data_rates:
        for mod_depth in mod_depths:
            for fsw in fsw_list:
                run_count += 1
                
                # Calculate samples per bit for this configuration
                # BPF needs sufficient samples: fs > 2 * f_high (10 kHz)
                # Use at least 50 samples per bit for low data rates
                min_fs = 50000  # 50 kHz minimum sample rate for BPF
                tbit = 1.0 / data_rate
                sps = max(50, int(min_fs * tbit))
                fs = data_rate * sps
                
                # Ensure fs is high enough for BPF
                if fs < 50000:
                    sps = int(50000 / data_rate) + 1
                    fs = data_rate * sps
                
                if verbose and run_count % 5 == 1:
                    print(f"\n  [{run_count}/{total_runs}] Rate={data_rate/1000:.1f}kbps, "
                          f"m={mod_depth}, fsw={fsw}kHz, fs={fs/1000:.0f}kHz")
                
                result = run_single_paper_validation(
                    data_rate_bps=data_rate,
                    mod_depth=mod_depth,
                    fsw_khz=fsw,
                    n_bits=n_bits_sweep,
                    sps=sps,
                    verbose=False
                )
                
                # Store results
                all_results.append(result)
                ber_sweep_data[data_rate][fsw].append(result['ber'])
                
                # Store power only once per fsw/mod combination
                if data_rate == data_rates[0]:
                    if len(power_sweep_data[mod_depth]) < len(fsw_list):
                        power_sweep_data[mod_depth].append(result['harvested_power_uw'])
    
    # ========== FREQUENCY RESPONSE ANALYSIS (Fig 13) ==========
    print("\n" + "-"*80)
    print("Generating frequency response (Fig. 13)...")
    print("-"*80)
    
    # Run a longer simulation for frequency analysis
    freq_result = run_single_paper_validation(
        data_rate_bps=2500,
        mod_depth=0.5,
        fsw_khz=100,
        n_bits=1000,
        sps=200,
        verbose=False,
        return_waveforms=True
    )
    
    # Compute frequency response using FFT
    t = freq_result['t']
    V_ina = freq_result['V_ina']
    V_bp = freq_result['V_bp']
    
    N = len(t)
    dt = t[1] - t[0]
    freqs = fft.fftfreq(N, dt)
    
    # FFT of input and output
    X = fft.fft(V_ina)
    Y = fft.fft(V_bp)
    
    # Transfer function H = Y/X (avoid divide by zero)
    H = np.divide(Y, X, out=np.zeros_like(Y), where=np.abs(X) > 1e-12)
    magnitude_db = 20 * np.log10(np.abs(H) + 1e-12)
    
    # Plot only positive frequencies
    pos_mask = freqs > 0
    plot_frequency_response(freqs[pos_mask], magnitude_db[pos_mask])
    
    # ========== HARVESTED POWER PLOT (Fig 15) ==========
    print("\n" + "-"*80)
    print("Generating harvested power plot (Fig. 15)...")
    print("-"*80)
    
    plot_harvested_power(
        fsw_khz=fsw_list,
        power_uw=power_sweep_data,
        target_power_uw=PAPER_TARGETS['harvested_power_uw']
    )
    
    # ========== TIME-DOMAIN WAVEFORMS (Fig 16) ==========
    print("\n" + "-"*80)
    print("Generating time-domain waveforms (Fig. 16)...")
    print("-"*80)
    
    plot_time_domain_waveforms(
        t=freq_result['t'],
        V_mod=freq_result['V_ina'],
        V_bp=freq_result['V_bp'],
        bits_rx=freq_result['bits_rx'],
        n_bits_show=20
    )
    
    # ========== BER VS MODULATION PLOT (Fig 17) ==========
    print("\n" + "-"*80)
    print("Generating BER plot (Fig. 17)...")
    print("-"*80)
    
    plot_ber_vs_modulation(
        mod_depths=mod_depths,
        ber_data=ber_sweep_data,
        target_ber=PAPER_TARGETS['ber_target']
    )
    
    # ========== CSV EXPORT ==========
    print("\n" + "-"*80)
    print("Exporting results to CSV...")
    print("-"*80)
    
    output_dir = Path('outputs/csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame from all results
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = output_dir / f"paper_validation_sweep_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"  ✓ Sweep results: {csv_file}")
    
    # ========== VALIDATION REPORT ==========
    print("\n" + "="*80)
    print("📊 VALIDATION REPORT")
    print("="*80)
    
    # Find closest result to paper target conditions
    target_rate = PAPER_TARGETS['ber_data_rate_bps']
    target_mod = PAPER_TARGETS['ber_mod_depth']
    
    closest_result = None
    for r in all_results:
        if r['data_rate_bps'] == target_rate and r['mod_depth'] == target_mod:
            closest_result = r
            break
    
    if closest_result is None:
        closest_result = all_results[0]
    
    # Summary table
    print("\n  Target Comparison:")
    print("  " + "-"*60)
    print(f"  {'Metric':<30} {'Simulated':<15} {'Paper Target':<15}")
    print("  " + "-"*60)
    
    # Harvested power
    avg_power = np.mean([r['harvested_power_uw'] for r in all_results 
                         if r['mod_depth'] == 0.5])
    target_power = PAPER_TARGETS['harvested_power_uw']
    power_match = "✓" if 0.3 < avg_power/target_power < 3.0 else "✗"
    print(f"  {'Harvested Power (µW)':<30} {avg_power:<15.2f} {target_power:<15} {power_match}")
    
    # BER at target conditions
    sim_ber = closest_result['ber']
    target_ber = PAPER_TARGETS['ber_target']
    ber_match = "✓" if sim_ber < 0.1 else "✗"  # Order of magnitude check
    print(f"  {'BER @ 2.5kbps, 50% mod':<30} {sim_ber:<15.2e} {target_ber:<15.2e} {ber_match}")
    
    print("  " + "-"*60)
    
    # Notes
    print("\n  Notes:")
    print("  • Simulator uses OOK encoding (paper uses Manchester)")
    print("  • Harvested power uses simplified approximation (no full DC-DC converter)")
    print("  • Absolute BER may differ due to encoding and noise model differences")
    print("  • Trends and relative behavior should match paper figures")
    
    # Save validation report
    report = {
        'timestamp': timestamp,
        'simulated_power_uw': avg_power,
        'target_power_uw': target_power,
        'simulated_ber': sim_ber,
        'target_ber': target_ber,
        'distance_m': PAPER_VALIDATION_CONFIG['distance_m'],
        'radiated_power_mw': PAPER_VALIDATION_CONFIG['radiated_power_mw'],
        'data_rate_bps': target_rate,
        'mod_depth': target_mod,
    }
    
    report_df = pd.DataFrame([report])
    report_file = output_dir / f"paper_validation_report_{timestamp}.csv"
    report_df.to_csv(report_file, index=False)
    print(f"\n  ✓ Validation report: {report_file}")
    
    print("\n" + "="*80)
    print("✅ PAPER VALIDATION COMPLETE")
    print("="*80)
    print(f"\n  Generated plots in: outputs/plots/")
    print(f"  Generated CSVs in: outputs/csv/")
    
    return {
        'all_results': all_results,
        'ber_sweep': ber_sweep_data,
        'power_sweep': power_sweep_data,
        'report': report
    }


def run_single_paper_validation(data_rate_bps=2500, mod_depth=0.5, fsw_khz=100,
                                 n_bits=10000, sps=100, verbose=False,
                                 return_waveforms=False):
    """
    Run a single simulation with paper validation parameters.
    
    Args:
        data_rate_bps: Data rate in bps (default 2500 = 2.5 kbps)
        mod_depth: Modulation depth (0-1)
        fsw_khz: Switching frequency in kHz (for reference, not actively used in OOK)
        n_bits: Number of bits to simulate
        sps: Samples per bit
        verbose: Print debug info
        return_waveforms: If True, include full waveforms in output
        
    Returns:
        dict: Simulation results
    """
    from utils.constants import PAPER_VALIDATION_CONFIG
    
    # Setup parameters from paper config
    config = {
        'simulation': {
            'data_rate_bps': data_rate_bps,
            'samples_per_bit': sps,
            'n_bits': n_bits
        },
        'transmitter': {
            'dc_bias': 100,  # mA - adjusted to get ~9.3 mW with efficiency
            'modulation_depth': mod_depth,
            'led_efficiency': 0.093  # W/A - tuned to get 9.3 mW at 100mA
        },
        'channel': {
            'distance': PAPER_VALIDATION_CONFIG['distance_m'],
            'beam_angle_half': PAPER_VALIDATION_CONFIG['led_half_angle_deg'],
            'receiver_area': PAPER_VALIDATION_CONFIG['solar_cell_area_cm2']
        },
        'receiver': {
            'responsivity': PAPER_VALIDATION_CONFIG['responsivity_a_per_w'],
            'capacitance': PAPER_VALIDATION_CONFIG['cj_pf'],  # 798 pF (direct)
            'shunt_resistance': PAPER_VALIDATION_CONFIG['rsh_ohm'] / 1e6,  # Ohm to MOhm (138.8 Ω = 0.0001388 MOhm)
            'dark_current': 1.0,
            'temperature': 300
        }
    }
    
    # Derived parameters
    f_b = config['simulation']['data_rate_bps']
    fs = f_b * sps
    t = np.arange(int(n_bits * sps)) / fs
    
    # ========== TRANSMITTER (MANCHESTER ENCODING) ==========
    bits_tx = np.random.randint(0, 2, n_bits)
    tx = Transmitter(config['transmitter'])
    # Use Manchester encoding as per paper
    P_tx = tx.modulate(bits_tx, t, encoding='manchester')
    bits_expanded = np.repeat(bits_tx, sps)
    
    # ========== CHANNEL ==========
    ch = OpticalChannel(config['channel'])
    P_rx = ch.propagate(P_tx, t)
    
    # ========== RECEIVER (PAPER MODE) ==========
    rx = PVReceiver(config['receiver'])
    I_ph = rx.optical_to_current(P_rx)
    
    # Add noise
    B = fs / 2
    noise = ch.add_noise(P_rx, I_ph, B, verbose=False)
    I_ph_noisy = I_ph + noise
    
    # Energy path: PV ODE
    V_pv = rx.solve_pv_circuit(I_ph_noisy, t, verbose=False)
    
    # Data path: Paper receiver chain (Rsense → INA → BPF)
    chain_result = rx.paper_receiver_chain(
        I_ph_noisy, t,
        R_sense=PAPER_VALIDATION_CONFIG['rsense_ohm'],
        ina_gain=PAPER_VALIDATION_CONFIG['ina_gain'],
        f_low=PAPER_VALIDATION_CONFIG['bpf_low_hz'],
        f_high=PAPER_VALIDATION_CONFIG['bpf_high_hz'],
        bpf_order=PAPER_VALIDATION_CONFIG['bpf_order'],
        verbose=verbose
    )
    
    V_bp = chain_result['V_bp']
    V_ina = chain_result['V_ina']
    
    # ========== BIT RECOVERY (MANCHESTER DECODING) ==========
    # Manchester differential decoding: look at slope at bit center
    # Negative slope (falling edge) = '1', Positive slope (rising edge) = '0'
    bits_rx = np.zeros(n_bits, dtype=int)
    
    for i in range(n_bits):
        # Bit center
        bit_center = i * sps + sps // 2
        
        # Look at slope around center
        window = max(1, sps // 8)
        idx_before = max(0, bit_center - window)
        idx_after = min(len(V_bp) - 1, bit_center + window)
        
        slope = V_bp[idx_after] - V_bp[idx_before]
        
        # Negative slope (falling edge) = '1'
        # Positive slope (rising edge) = '0'
        if slope < 0:
            bits_rx[i] = 1
        else:
            bits_rx[i] = 0
    
    # ========== METRICS ==========
    errors = np.sum(bits_tx != bits_rx)
    ber = errors / n_bits
    
    # Harvested power estimation
    # P_harv ≈ V_pv^2 / R_load (simplified)
    R_load = PAPER_VALIDATION_CONFIG['rload_ohm']
    P_harv_inst = V_pv**2 / R_load
    P_harv_avg = np.mean(P_harv_inst) * 1e6  # µW
    
    result = {
        'data_rate_bps': data_rate_bps,
        'mod_depth': mod_depth,
        'fsw_khz': fsw_khz,
        'n_bits': n_bits,
        'ber': ber,
        'errors': errors,
        'harvested_power_uw': P_harv_avg,
        'received_power_uw': np.mean(P_rx) * 1e6,
        'photocurrent_ua': np.mean(I_ph) * 1e6,
    }
    
    if return_waveforms:
        result['t'] = t
        result['bits_tx'] = bits_tx
        result['bits_rx'] = bits_rx
        result['V_ina'] = V_ina
        result['V_bp'] = V_bp
        result['V_pv'] = V_pv
        result['I_ph'] = I_ph_noisy
    
    return result


if __name__ == "__main__":
    import argparse
    
    # ========== PV CELL PRESETS ==========
    CELL_PRESETS = {
        'gaas': {
            'responsivity': 0.457,
            'capacitance': 798,
            'shunt_resistance': 0.1388,
            'description': 'High efficiency GaAs (Kadirvelu baseline)'
        },
        'poly-si': {
            'responsivity': 0.5,
            'capacitance': 500,
            'shunt_resistance': 0.01,
            'description': 'Low-cost Poly-Si for IoT'
        },
        'c-si': {
            'responsivity': 0.42,
            'capacitance': 300,
            'shunt_resistance': 0.1,
            'description': 'Standard crystalline silicon'
        },
        'high-speed': {
            'responsivity': 0.6,
            'capacitance': 100,
            'shunt_resistance': 1.0,
            'description': 'Low capacitance for OFDM (Sarwar 15 Mbps)'
        },
        'custom': {
            'description': 'User-defined parameters'
        }
    }
    
    # ========== VALIDATION FUNCTION ==========
    def validate_and_clamp(value, min_val, max_val, name):
        """Validate and clamp parameter to safe thresholds."""
        if value < min_val or value > max_val:
            print(f"⚠️  WARNING: {name}={value} out of range [{min_val}, {max_val}]")
            clamped = max(min_val, min(max_val, value))
            print(f"   Clamping to {clamped}")
            return clamped
        return value
    
    # ========== ARGUMENT PARSER ==========
    parser = argparse.ArgumentParser(
        description='Li-Fi + PV Simulator - Enhanced CLI with Layer-by-Layer Configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default simulation
  python main.py
  
  # Custom layer configuration
  python main.py --tx-mod-depth 0.7 --ch-distance 2.0 --rx-cell poly-si
  
  # Energy-first mode (no TIA)
  python main.py --no-tia --sim-rate 10000
  
  # Paper validation modes
  python main.py --validate
  python main.py --figures all
  python main.py --sensitivity
        """
    )
    
    # ========== LAYER 1: TRANSMITTER ==========
    tx_group = parser.add_argument_group('Layer 1: Transmitter')
    tx_group.add_argument('--tx-dc-bias', type=float, default=200,
                          help='LED DC bias current (mA) [10-500]')
    tx_group.add_argument('--tx-mod-depth', type=float, default=0.9,
                          help='Modulation depth [0.1-1.0]')
    tx_group.add_argument('--tx-efficiency', type=float, default=0.08,
                          help='LED wall-plug efficiency (W/A) [0.01-0.5]')
    
    # ========== LAYER 2: CHANNEL ==========
    ch_group = parser.add_argument_group('Layer 2: Channel')
    ch_group.add_argument('--ch-distance', type=float, default=1.0,
                          help='TX-RX distance (m) [0.1-10]')
    ch_group.add_argument('--ch-beam-angle', type=float, default=30,
                          help='LED half-power beam angle (deg) [5-60]')
    ch_group.add_argument('--ch-rx-area', type=float, default=1.0,
                          help='Receiver active area (cm²) [0.1-100]')
    
    # ========== LAYER 3: RECEIVER ==========
    rx_group = parser.add_argument_group('Layer 3: Receiver (PV Cell)')
    rx_group.add_argument('--rx-cell', type=str, default='gaas',
                          choices=['gaas', 'poly-si', 'c-si', 'high-speed', 'custom'],
                          help='PV cell preset or "custom" for manual parameters')
    rx_group.add_argument('--rx-responsivity', type=float, default=None,
                          help='Responsivity (A/W) [0.1-1.0] (overrides preset)')
    rx_group.add_argument('--rx-capacitance', type=float, default=None,
                          help='Junction capacitance (pF) [10-5000] (overrides preset)')
    rx_group.add_argument('--rx-shunt-r', type=float, default=None,
                          help='Shunt resistance (MΩ) [0.01-10] (overrides preset)')
    
    # ========== LAYER 3B: TIA ==========
    tia_group = parser.add_argument_group('Layer 3b: TIA (Transimpedance Amplifier)')
    tia_group.add_argument('--no-tia', action='store_true',
                           help='Disable TIA, use direct PV output (energy-first mode)')
    tia_group.add_argument('--tia-r', type=float, default=50e3,
                           help='Transimpedance (Ω) [1e3-1e6]')
    tia_group.add_argument('--tia-bw', type=float, default=3e6,
                           help='3dB bandwidth (Hz) [100e3-50e6]')
    
    # ========== SIMULATION CONTROL ==========
    sim_group = parser.add_argument_group('Simulation Control')
    sim_group.add_argument('--sim-rate', type=float, default=1e6,
                           help='Data rate (bps) [100-50e6]')
    sim_group.add_argument('--sim-bits', type=int, default=1000,
                           help='Number of bits to simulate [100-1e6]')
    sim_group.add_argument('--sim-sps', type=int, default=500,
                           help='Samples per bit [10-2000]')
    
    # ========== OUTPUT CONTROLS ==========
    out_group = parser.add_argument_group('Output Controls')
    out_group.add_argument('--out-dir', type=str, default='outputs/',
                           help='Output directory')
    out_group.add_argument('--out-no-csv', action='store_true',
                           help='Disable CSV export')
    out_group.add_argument('--out-no-plots', action='store_true',
                           help='Disable plot generation')
    out_group.add_argument('--out-downsample', type=int, default=1,
                           help='Save every Nth sample to CSV')
    
    # ========== LEGACY MODE FLAGS ==========
    mode_group = parser.add_argument_group('Paper Validation Modes (Legacy)')
    mode_group.add_argument('--figures', '-f', type=str, default=None,
                            help='Generate figures: "all" or comma-separated "13,15,18"')
    mode_group.add_argument('--validate', '-v', action='store_true',
                            help='Run paper validation mode')
    mode_group.add_argument('--sensitivity', '-s', action='store_true',
                            help='Run sensitivity analysis')
    mode_group.add_argument('--all', '-a', action='store_true',
                            help='Run everything (figures + validation + sensitivity)')
    mode_group.add_argument('--quiet', '-q', action='store_true',
                            help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # ========== CHECK FOR LEGACY MODES ==========
    if args.all:
        args.figures = 'all'
        args.validate = True
        args.sensitivity = True
    
    ran_something = False
    
    # Generate figures
    if args.figures:
        from utils.figures import generate_all_figures
        
        if args.figures.lower() == 'all':
            fig_list = [13, 14, 15, 16, 17, 18, 19]
        else:
            fig_list = [int(f.strip()) for f in args.figures.split(',')]
        
        generate_all_figures(fig_list)
        ran_something = True
    
    # Run sensitivity analysis
    if args.sensitivity:
        from sensitivity_analysis import run_full_sensitivity_analysis
        run_full_sensitivity_analysis()
        ran_something = True
    
    # Run validation
    if args.validate:
        run_paper_validation_mode(verbose=not args.quiet)
        ran_something = True
    
    # ========== RUN CUSTOM SIMULATION ==========
    if not ran_something:
        # Apply cell preset
        if args.rx_cell != 'custom':
            preset = CELL_PRESETS[args.rx_cell]
            print(f"\n📱 Using PV cell preset: {args.rx_cell.upper()}")
            print(f"   {preset['description']}")
            
            if args.rx_responsivity is None:
                args.rx_responsivity = preset['responsivity']
            if args.rx_capacitance is None:
                args.rx_capacitance = preset['capacitance']
            if args.rx_shunt_r is None:
                args.rx_shunt_r = preset['shunt_resistance']
        else:
            print(f"\n📱 Using CUSTOM PV cell parameters")
            # Default to GaAs if not specified
            if args.rx_responsivity is None:
                args.rx_responsivity = 0.457
            if args.rx_capacitance is None:
                args.rx_capacitance = 798
            if args.rx_shunt_r is None:
                args.rx_shunt_r = 0.1388
        
        # Validate and clamp parameters
        args.tx_dc_bias = validate_and_clamp(args.tx_dc_bias, 10, 500, 'TX DC Bias')
        args.tx_mod_depth = validate_and_clamp(args.tx_mod_depth, 0.1, 1.0, 'TX Modulation Depth')
        args.tx_efficiency = validate_and_clamp(args.tx_efficiency, 0.01, 0.5, 'TX Efficiency')
        
        args.ch_distance = validate_and_clamp(args.ch_distance, 0.1, 10.0, 'Channel Distance')
        args.ch_beam_angle = validate_and_clamp(args.ch_beam_angle, 5, 60, 'Channel Beam Angle')
        args.ch_rx_area = validate_and_clamp(args.ch_rx_area, 0.1, 100, 'Channel RX Area')
        
        args.rx_responsivity = validate_and_clamp(args.rx_responsivity, 0.1, 1.0, 'RX Responsivity')
        args.rx_capacitance = validate_and_clamp(args.rx_capacitance, 10, 5000, 'RX Capacitance')
        args.rx_shunt_r = validate_and_clamp(args.rx_shunt_r, 0.01, 10, 'RX Shunt R')
        
        args.tia_r = validate_and_clamp(args.tia_r, 1e3, 1e6, 'TIA Transimpedance')
        args.tia_bw = validate_and_clamp(args.tia_bw, 100e3, 50e6, 'TIA Bandwidth')
        
        args.sim_rate = validate_and_clamp(args.sim_rate, 100, 50e6, 'Simulation Rate')
        args.sim_bits = int(validate_and_clamp(args.sim_bits, 100, 1e6, 'Simulation Bits'))
        args.sim_sps = int(validate_and_clamp(args.sim_sps, 10, 2000, 'Samples Per Bit'))
        
        # Build config dictionary
        config = {
            'simulation': {
                'data_rate_bps': args.sim_rate,
                'samples_per_bit': args.sim_sps,
                'n_bits': args.sim_bits
            },
            'transmitter': {
                'dc_bias': args.tx_dc_bias,
                'modulation_depth': args.tx_mod_depth,
                'led_efficiency': args.tx_efficiency
            },
            'channel': {
                'distance': args.ch_distance,
                'beam_angle_half': args.ch_beam_angle,
                'receiver_area': args.ch_rx_area
            },
            'receiver': {
                'responsivity': args.rx_responsivity,
                'capacitance': args.rx_capacitance,
                'shunt_resistance': args.rx_shunt_r,
                'dark_current': 1.0,
                'temperature': 300
            },
            'tia': {
                'R_tia': args.tia_r,
                'f_3db': args.tia_bw
            }
        }
        
        # Display configuration summary
        print(f"\n🔬 Li-Fi + PV Simulator - Custom Configuration")
        print(f"{'='*80}")
        
        # Run simulation
        enable_csv = not args.out_no_csv
        
        if args.no_tia:
            print("⚠️  TIA DISABLED - Using direct PV output (energy-first mode)")
            print("   Communication bandwidth limited by PV cell RC time constant")
        
        run_complete_simulation_with_tia(
            config=config,
            export_csv=enable_csv,
            verbose=not args.quiet
        )
