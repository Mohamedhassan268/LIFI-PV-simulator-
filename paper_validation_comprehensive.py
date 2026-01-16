# paper_validation_comprehensive.py
"""
Comprehensive Paper Validation Module.

Generates all 7 figures from Kadirvelu et al. IEEE TGCN 2021:
- Fig. 13: Frequency response
- Fig. 14: PSD at BPF output
- Fig. 15: DC-DC Vout vs duty cycle
- Fig. 16: Time-domain waveforms
- Fig. 17: BER vs modulation depth
- Fig. 18: DC-DC Vout vs modulation depth
- Fig. 19: Harvested power vs bit rate

Also generates:
- CSV 1: Signal waveform data
- CSV 2: ML dataset (20k samples)
- CSV 3: ML dataset (50k samples)

Uses pure physical modeling - no artificial calibration.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from scipy import signal

# Simulator modules
from simulator.transmitter import Transmitter
from simulator.channel import OpticalChannel
from simulator.receiver import PVReceiver
from simulator.demodulator import Demodulator
from simulator.noise import NoiseModel
from simulator.dc_dc_converter import DCDCConverter

# Configuration
from utils.constants import (
    PAPER_VALIDATION_CONFIG, BER_TEST_CONFIG, PAPER_TARGETS,
    DCDC_EFFICIENCY_CONFIG, NOISE_MODEL_CONFIG, ML_DATASET_CONFIG
)

# Plotting
from utils.plotting import (
    plot_frequency_response, plot_harvested_power, plot_ber_vs_modulation,
    plot_time_domain_waveforms, plot_psd, plot_dcdc_vout_vs_duty,
    plot_dcdc_vout_vs_modulation, plot_power_vs_bitrate
)


def run_single_simulation(distance_m, P_tx_mw, mod_depth, data_rate_kbps,
                          fsw_khz, duty_cycle, beam_angle_deg, rx_area_cm2,
                          ambient_lux, temperature_K, n_bits=10000, sps=20,
                          verbose=False, return_waveforms=False):
    """
    Run a single simulation with specified parameters.
    
    Pure physical modeling - all outputs calculated from first principles.
    
    Args:
        distance_m: TX-RX distance in meters
        P_tx_mw: Transmit optical power in mW
        mod_depth: Modulation depth (0-1)
        data_rate_kbps: Data rate in kbps
        fsw_khz: DC-DC switching frequency in kHz
        duty_cycle: DC-DC duty cycle (0-1)
        beam_angle_deg: LED half-angle in degrees
        rx_area_cm2: Receiver area in cm²
        ambient_lux: Background light level in lux
        temperature_K: Temperature in Kelvin
        n_bits: Number of bits to simulate
        sps: Samples per bit
        verbose: Print debug info
        return_waveforms: Include full waveform data
        
    Returns:
        dict: Simulation results including BER, SNR, power metrics
    """
    # Data rate and timing
    data_rate_bps = data_rate_kbps * 1000
    f_b = data_rate_bps
    fs = f_b * sps
    
    # Ensure sample rate high enough for BPF
    min_fs = 50000  # 50 kHz minimum
    if fs < min_fs:
        sps = int(np.ceil(min_fs / f_b))
        fs = f_b * sps
    
    t = np.arange(int(n_bits * sps)) / fs
    
    # Setup configuration
    config = {
        'transmitter': {
            'dc_bias': 100,  # mA
            'modulation_depth': mod_depth,
            'led_efficiency': P_tx_mw / 100.0 / 1000.0 * 1000  # Adjusted for target power
        },
        'channel': {
            'distance': distance_m,
            'beam_angle_half': beam_angle_deg,
            'receiver_area': rx_area_cm2
        },
        'receiver': {
            'responsivity': PAPER_VALIDATION_CONFIG['responsivity_a_per_w'],
            'capacitance': PAPER_VALIDATION_CONFIG['cj_pf'],  # pF (direct, no conversion)
            'shunt_resistance': PAPER_VALIDATION_CONFIG['rsh_ohm'] / 1e6,  # Ω to MΩ
            'dark_current': 1.0,
            'temperature': temperature_K
        }
    }
    
    # Initialize noise model
    noise_model = NoiseModel({
        'temperature_K': temperature_K,
        'ambient_lux': ambient_lux,
    })
    
    # Initialize DC-DC converter with paper efficiency
    dcdc = DCDCConverter({
        'fsw_khz': fsw_khz,
        'duty_cycle': duty_cycle,
        'efficiency_mode': 'paper',
    })
    
    # ========== TRANSMITTER (MANCHESTER) ==========
    bits_tx = np.random.randint(0, 2, n_bits)
    tx = Transmitter(config['transmitter'])
    P_tx = tx.modulate(bits_tx, t, encoding='manchester')
    
    # ========== CHANNEL ==========
    ch = OpticalChannel(config['channel'])
    P_rx = ch.propagate(P_tx, t)
    
    # ========== RECEIVER ==========
    rx = PVReceiver(config['receiver'])
    I_ph = rx.optical_to_current(P_rx)
    
    # Add physical noise
    bandwidth = fs / 2
    noise = noise_model.generate_noise(len(I_ph), I_ph, bandwidth, rx_area_cm2)
    I_ph_noisy = I_ph + noise
    
    # Energy path: PV ODE
    V_pv = rx.solve_pv_circuit(I_ph_noisy, t, verbose=False)
    
    # Data path: Paper receiver chain (Physically Rigorous)
    # Step 1: Current Sense
    V_sense = rx.apply_current_sense(I_ph_noisy, PAPER_VALIDATION_CONFIG['rsense_ohm'])
    
    # Step 2: INA with GBW Limitation (Physics Model)
    # Using 'ina_gain' (100) and hardcoded 700kHz GBW if not in config, but config is preferred
    V_ina = rx.apply_ina_gain_physics(
        V_sense, t, 
        gain_dc=PAPER_VALIDATION_CONFIG.get('ina_gain_dc', 100),
        gbw_hz=PAPER_VALIDATION_CONFIG.get('ina_gbw_hz', 700000)
    )
    
    # Step 3: BPF with Component Logic (Physics Model)
    # Using the calculated capacitor values that solve the spec contradiction
    # R = 10k for both stages
    V_bp = rx.apply_bpf_physics(
        V_ina, t,
        Rhp=PAPER_VALIDATION_CONFIG.get('bpf_rhp_ohm', 10000),
        Chp=PAPER_VALIDATION_CONFIG.get('bpf_chp_value_f', 22.7e-9), # 700 Hz
        Rlp=PAPER_VALIDATION_CONFIG.get('bpf_rhp_ohm', 10000),       # Same R
        Clp=PAPER_VALIDATION_CONFIG.get('bpf_clp_value_f', 1.6e-9)   # 10 kHz
    )
    
    # ========== BIT RECOVERY (MANCHESTER DECODING) ==========
    bits_rx = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        bit_center = i * sps + sps // 2
        window = max(1, sps // 8)
        idx_before = max(0, bit_center - window)
        idx_after = min(len(V_bp) - 1, bit_center + window)
        slope = V_bp[idx_after] - V_bp[idx_before]
        if slope < 0:
            bits_rx[i] = 1
        else:
            bits_rx[i] = 0
    
    # ========== METRICS ==========
    errors = np.sum(bits_tx != bits_rx)
    ber = errors / n_bits
    
    # SNR calculation
    I_signal_pp = np.max(I_ph) - np.min(I_ph)
    snr_db = noise_model.calculate_snr(I_signal_pp, I_ph, bandwidth, rx_area_cm2)
    
    # Received power
    P_rx_avg = np.mean(P_rx) * 1e6  # µW
    I_ph_avg = np.mean(I_ph) * 1e6  # µA
    
    # DC-DC output
    V_pv_avg = np.mean(V_pv)
    I_pv_avg = np.mean(I_ph) * 0.9  # Approximate PV current
    dcdc_result = dcdc.calculate_output(V_pv_avg, I_pv_avg, duty_cycle, fsw_khz)
    
    P_harvest_uw = dcdc_result['P_out'] * 1e6  # µW
    V_dcdc = dcdc_result['V_out']
    efficiency = dcdc_result['efficiency']
    
    # Eye opening (simplified)
    sample_indices = np.arange(n_bits) * sps + sps // 2
    samples = V_bp[sample_indices]
    V_high = np.mean(samples[bits_tx == 1]) if np.sum(bits_tx == 1) > 0 else 0
    V_low = np.mean(samples[bits_tx == 0]) if np.sum(bits_tx == 0) > 0 else 0
    eye_opening = (V_high - V_low) / (np.max(V_bp) - np.min(V_bp) + 1e-12) * 100
    
    # Q-factor
    std_high = np.std(samples[bits_tx == 1]) if np.sum(bits_tx == 1) > 1 else 1e-12
    std_low = np.std(samples[bits_tx == 0]) if np.sum(bits_tx == 0) > 1 else 1e-12
    q_factor = (V_high - V_low) / (std_high + std_low + 1e-12)
    
    # Path loss
    path_loss_db = -10 * np.log10(P_rx_avg / (P_tx_mw * 1000) + 1e-12)
    
    result = {
        # Inputs (for ML)
        'distance_m': distance_m,
        'P_tx_mW': P_tx_mw,
        'mod_depth': mod_depth,
        'data_rate_kbps': data_rate_kbps,
        'fsw_khz': fsw_khz,
        'duty_cycle': duty_cycle,
        'beam_angle_deg': beam_angle_deg,
        'rx_area_cm2': rx_area_cm2,
        'ambient_lux': ambient_lux,
        'temperature_K': temperature_K,
        
        # Outputs
        'ber': ber,
        'snr_db': snr_db,
        'P_rx_uW': P_rx_avg,
        'P_harvest_uW': P_harvest_uw,
        'V_dcdc_V': V_dcdc,
        'I_ph_uA': I_ph_avg,
        'eye_opening_percent': eye_opening,
        'q_factor': q_factor,
        'efficiency_percent': efficiency * 100,
        
        # Derived
        'path_loss_dB': path_loss_db,
        'errors': errors,
        'n_bits': n_bits,
    }
    
    if return_waveforms:
        result['t'] = t
        result['P_tx'] = P_tx
        result['P_rx'] = P_rx
        result['I_ph'] = I_ph
        result['noise'] = noise
        result['V_pv'] = V_pv
        result['V_sense'] = V_sense
        result['V_ina'] = V_ina
        result['V_bp'] = V_bp
        result['bits_tx'] = bits_tx
        result['bits_rx'] = bits_rx
    
    if verbose:
        print(f"  BER: {ber:.4f} ({errors}/{n_bits})")
        print(f"  SNR: {snr_db:.1f} dB")
        print(f"  P_harvest: {P_harvest_uw:.2f} µW")
    
    return result


def generate_ml_dataset(n_samples, output_csv, verbose=True):
    """
    Generate ML-ready dataset with varied parameters.
    
    Args:
        n_samples: Number of samples to generate
        output_csv: Output CSV path
        verbose: Print progress
        
    Returns:
        DataFrame: Generated dataset
    """
    print(f"\n{'='*60}")
    print(f"GENERATING ML DATASET ({n_samples} samples)")
    print(f"{'='*60}")
    
    config = ML_DATASET_CONFIG
    results = []
    
    for i in range(n_samples):
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{n_samples} ({100*(i+1)/n_samples:.1f}%)")
        
        # Random parameters within ranges
        distance_m = np.random.uniform(*config['distance_m_range'])
        P_tx_mw = np.random.uniform(*config['P_tx_mw_range'])
        mod_depth = np.random.uniform(*config['mod_depth_range'])
        data_rate_kbps = np.random.uniform(*config['data_rate_kbps_range'])
        fsw_khz = np.random.uniform(*config['fsw_khz_range'])
        duty_cycle = np.random.uniform(*config['duty_cycle_range'])
        beam_angle_deg = np.random.uniform(*config['beam_angle_deg_range'])
        rx_area_cm2 = np.random.uniform(*config['rx_area_cm2_range'])
        ambient_lux = np.random.uniform(*config['ambient_lux_range'])
        temperature_K = np.random.uniform(*config['temperature_K_range'])
        
        # Run simulation (fewer bits for speed)
        result = run_single_simulation(
            distance_m=distance_m,
            P_tx_mw=P_tx_mw,
            mod_depth=mod_depth,
            data_rate_kbps=data_rate_kbps,
            fsw_khz=fsw_khz,
            duty_cycle=duty_cycle,
            beam_angle_deg=beam_angle_deg,
            rx_area_cm2=rx_area_cm2,
            ambient_lux=ambient_lux,
            temperature_K=temperature_K,
            n_bits=1000,  # Fewer bits for ML dataset generation
            verbose=False
        )
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Saved: {output_csv}")
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {len(df.columns)}")
    
    return df


def generate_waveform_csv(output_csv, verbose=True):
    """
    Generate signal waveform CSV (full time-domain data).
    
    Args:
        output_csv: Output CSV path
        verbose: Print progress
        
    Returns:
        DataFrame: Waveform data
    """
    print(f"\n{'='*60}")
    print("GENERATING WAVEFORM CSV")
    print(f"{'='*60}")
    
    # Run simulation with paper parameters and return waveforms
    result = run_single_simulation(
        distance_m=PAPER_VALIDATION_CONFIG['distance_m'],
        P_tx_mw=PAPER_VALIDATION_CONFIG['radiated_power_mw'],
        mod_depth=0.5,
        data_rate_kbps=10,
        fsw_khz=100,
        duty_cycle=0.5,
        beam_angle_deg=PAPER_VALIDATION_CONFIG['led_half_angle_deg'],
        rx_area_cm2=PAPER_VALIDATION_CONFIG['solar_cell_area_cm2'],
        ambient_lux=300,
        temperature_K=300,
        n_bits=100,  # 100 bits
        sps=100,  # High resolution
        return_waveforms=True,
        verbose=True
    )
    
    # Create waveform DataFrame
    df = pd.DataFrame({
        'time_s': result['t'],
        'time_ms': result['t'] * 1000,
        'P_tx_mW': result['P_tx'] * 1000,
        'P_rx_uW': result['P_rx'] * 1e6,
        'I_ph_uA': result['I_ph'] * 1e6,
        'noise_uA': result['noise'] * 1e6,
        'V_pv_mV': result['V_pv'] * 1000,
        'V_sense_uV': result['V_sense'] * 1e6,
        'V_ina_mV': result['V_ina'] * 1000,
        'V_bp_mV': result['V_bp'] * 1000,
    })
    
    # Add bit labels (repeated for each sample)
    sps = len(result['t']) // len(result['bits_tx'])
    df['bit_tx'] = np.repeat(result['bits_tx'], sps)[:len(df)]
    df['bit_rx'] = np.repeat(result['bits_rx'], sps)[:len(df)]
    
    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Saved: {output_csv}")
    print(f"    Rows: {len(df)}")
    print(f"    Duration: {result['t'][-1]*1000:.2f} ms")
    
    return df


def run_all_figures_validation(n_bits=10000, output_dir='outputs'):
    """
    Run comprehensive validation generating all 7 paper figures.
    
    Args:
        n_bits: Bits per simulation point
        output_dir: Output directory
        
    Returns:
        dict: All validation results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = os.path.join(output_dir, 'plots')
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("    COMPREHENSIVE PAPER VALIDATION")
    print("    Kadirvelu et al., IEEE TGCN 2021")
    print("    Pure Physical Modeling - No Artificial Calibration")
    print("="*70)
    
    results = {}
    
    # ========== FIG 13: FREQUENCY RESPONSE ==========
    print("\n[1/7] Generating Fig. 13: Frequency Response...")
    result_freq = run_single_simulation(
        distance_m=0.325, P_tx_mw=9.3, mod_depth=0.5, data_rate_kbps=10,
        fsw_khz=100, duty_cycle=0.5, beam_angle_deg=9, rx_area_cm2=9,
        ambient_lux=300, temperature_K=300, n_bits=500, sps=100,
        return_waveforms=True
    )
    
    # FFT for frequency response
    V_bp = result_freq['V_bp']
    V_ina = result_freq['V_ina']
    fs = len(result_freq['t']) / result_freq['t'][-1]
    
    freq_fft = np.fft.rfftfreq(len(V_bp), 1/fs)
    H_bp = np.abs(np.fft.rfft(V_bp))
    H_ina = np.abs(np.fft.rfft(V_ina)) + 1e-12
    H_ratio = H_bp / H_ina
    mag_db = 20 * np.log10(H_ratio + 1e-12)
    
    plot_frequency_response(freq_fft, mag_db, output_dir=plots_dir)
    results['fig13'] = {'freqs': freq_fft, 'magnitude_db': mag_db}
    
    # ========== FIG 14: PSD ==========
    print("[2/7] Generating Fig. 14: PSD...")
    f_psd, psd = signal.welch(V_bp, fs=fs, nperseg=min(4096, len(V_bp)))
    psd_db = 10 * np.log10(psd + 1e-20)
    plot_psd(f_psd/1000, psd_db, output_dir=plots_dir)
    results['fig14'] = {'freqs_khz': f_psd/1000, 'psd_db': psd_db}
    
    # ========== FIG 15: DC-DC VOUT vs DUTY CYCLE ==========
    print("[3/7] Generating Fig. 15: DC-DC Vout vs Duty Cycle...")
    duty_cycles = np.linspace(0.05, 0.5, 15)
    vout_data = {}
    
    for fsw in [50, 100, 200]:
        vout_list = []
        for D in duty_cycles:
            result = run_single_simulation(
                distance_m=0.325, P_tx_mw=9.3, mod_depth=0.5, data_rate_kbps=10,
                fsw_khz=fsw, duty_cycle=D, beam_angle_deg=9, rx_area_cm2=9,
                ambient_lux=300, temperature_K=300, n_bits=100, verbose=False
            )
            vout_list.append(result['V_dcdc_V'])
        vout_data[fsw] = np.array(vout_list)
    
    plot_dcdc_vout_vs_duty(duty_cycles, vout_data, output_dir=plots_dir)
    results['fig15'] = {'duty_cycles': duty_cycles, 'vout_data': vout_data}
    
    # ========== FIG 16: TIME-DOMAIN WAVEFORMS ==========
    print("[4/7] Generating Fig. 16: Time-Domain Waveforms...")
    result_td = run_single_simulation(
        distance_m=0.325, P_tx_mw=9.3, mod_depth=0.5, data_rate_kbps=2.5,
        fsw_khz=100, duty_cycle=0.5, beam_angle_deg=9, rx_area_cm2=9,
        ambient_lux=300, temperature_K=300, n_bits=20, sps=200,
        return_waveforms=True
    )
    
    plot_time_domain_waveforms(
        result_td['t'], result_td['V_ina'], result_td['V_bp'],
        result_td['bits_rx'], n_bits_show=20, output_dir=plots_dir
    )
    results['fig16'] = result_td
    
    # ========== FIG 17: BER vs MODULATION DEPTH ==========
    print("[5/7] Generating Fig. 17: BER vs Modulation Depth...")
    mod_depths = np.array([0.10, 0.20, 0.33, 0.50, 0.65, 0.80, 1.0])
    ber_data = {}
    
    for data_rate in [2.5, 10]:
        ber_data[data_rate] = {}
        for fsw in [50, 100, 200]:
            ber_list = []
            for m in mod_depths:
                result = run_single_simulation(
                    distance_m=0.325, P_tx_mw=9.3, mod_depth=m, 
                    data_rate_kbps=data_rate, fsw_khz=fsw, duty_cycle=0.5,
                    beam_angle_deg=9, rx_area_cm2=9, ambient_lux=300, 
                    temperature_K=300, n_bits=n_bits, verbose=False
                )
                ber_list.append(result['ber'])
            ber_data[data_rate][fsw] = np.array(ber_list)
    
    plot_ber_vs_modulation(mod_depths, ber_data, output_dir=plots_dir)
    results['fig17'] = {'mod_depths': mod_depths, 'ber_data': ber_data}
    
    # ========== FIG 18: DC-DC VOUT vs MODULATION DEPTH ==========
    print("[6/7] Generating Fig. 18: DC-DC Vout vs Modulation Depth...")
    vout_mod_data = {}
    
    for fsw in [50, 100, 200]:
        vout_list = []
        for m in mod_depths:
            result = run_single_simulation(
                distance_m=0.325, P_tx_mw=9.3, mod_depth=m, data_rate_kbps=10,
                fsw_khz=fsw, duty_cycle=0.5, beam_angle_deg=9, rx_area_cm2=9,
                ambient_lux=300, temperature_K=300, n_bits=100, verbose=False
            )
            vout_list.append(result['V_dcdc_V'])
        vout_mod_data[fsw] = np.array(vout_list)
    
    plot_dcdc_vout_vs_modulation(mod_depths, vout_mod_data, output_dir=plots_dir)
    results['fig18'] = {'mod_depths': mod_depths, 'vout_data': vout_mod_data}
    
    # ========== FIG 19: HARVESTED POWER vs BIT RATE ==========
    print("[7/7] Generating Fig. 19: Harvested Power vs Bit Rate...")
    bitrates = np.array([1, 2, 5, 10, 20, 50, 100])
    power_list = []
    
    for br in bitrates:
        result = run_single_simulation(
            distance_m=0.325, P_tx_mw=9.3, mod_depth=0.5, data_rate_kbps=br,
            fsw_khz=100, duty_cycle=0.5, beam_angle_deg=9, rx_area_cm2=9,
            ambient_lux=300, temperature_K=300, n_bits=max(100, int(10000/br)),
            verbose=False
        )
        power_list.append(result['P_harvest_uW'])
    
    power_uw = np.array(power_list)
    plot_power_vs_bitrate(bitrates, power_uw, output_dir=plots_dir)
    results['fig19'] = {'bitrates_kbps': bitrates, 'power_uw': power_uw}
    
    # ========== GENERATE CSVs ==========
    print("\n" + "-"*60)
    print("GENERATING CSV FILES")
    print("-"*60)
    
    # CSV 1: Waveform data
    waveform_csv = os.path.join(csv_dir, f'waveform_data_{timestamp}.csv')
    generate_waveform_csv(waveform_csv)
    
    # CSV 2: ML dataset 20k
    ml_20k_csv = os.path.join(csv_dir, f'ml_dataset_20k_{timestamp}.csv')
    generate_ml_dataset(20000, ml_20k_csv)
    
    # CSV 3: ML dataset 50k
    ml_50k_csv = os.path.join(csv_dir, f'ml_dataset_50k_{timestamp}.csv')
    generate_ml_dataset(50000, ml_50k_csv)
    
    # ========== VALIDATION REPORT ==========
    print("\n" + "="*70)
    print("    VALIDATION REPORT")
    print("="*70)
    
    # Target comparison
    target_ber = PAPER_TARGETS['ber_target']
    actual_ber = ber_data[2.5][100][-3]  # 50% mod depth at 2.5 kbps
    
    target_power = PAPER_TARGETS['harvested_power_uw']
    actual_power = power_uw[3]  # At 10 kbps
    
    print(f"\n  BER @ 2.5 kbps, 50% mod:")
    print(f"    Simulated: {actual_ber:.4f}")
    print(f"    Target:    {target_ber:.4f}")
    
    print(f"\n  Harvested Power @ 10 kbps:")
    print(f"    Simulated: {actual_power:.2f} µW")
    print(f"    Target:    {target_power:.2f} µW")
    
    print("\n" + "="*70)
    print("    VALIDATION COMPLETE")
    print("="*70)
    print(f"\n  Plots saved to: {plots_dir}")
    print(f"  CSVs saved to:  {csv_dir}")
    
    return results


if __name__ == "__main__":
    run_all_figures_validation(n_bits=10000)
