# quick_test_figures.py
"""Quick test of 7 figures generation (no ML datasets)."""

from paper_validation_comprehensive import run_single_simulation, run_all_figures_validation
from utils.constants import PAPER_VALIDATION_CONFIG, BER_TEST_CONFIG, PAPER_TARGETS
from utils.plotting import (
    plot_frequency_response, plot_psd, plot_dcdc_vout_vs_duty,
    plot_time_domain_waveforms, plot_ber_vs_modulation,
    plot_dcdc_vout_vs_modulation, plot_power_vs_bitrate
)
import numpy as np
from scipy import signal
import os
from datetime import datetime

def quick_test_all_figures():
    """Test all 7 figures with minimal simulation time."""
    print("\n" + "="*70)
    print("QUICK TEST: ALL 7 FIGURES")
    print("="*70)
    
    plots_dir = 'outputs/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # ========== FIG 13: FREQUENCY RESPONSE ==========
    print("\n[1/7] Fig 13: Frequency Response...")
    result = run_single_simulation(
        0.325, 9.3, 0.5, 10, 100, 0.5, 9, 9, 300, 300,
        n_bits=200, sps=100, return_waveforms=True
    )
    V_bp = result['V_bp']
    V_ina = result['V_ina']
    fs = len(result['t']) / result['t'][-1]
    freq_fft = np.fft.rfftfreq(len(V_bp), 1/fs)
    H_bp = np.abs(np.fft.rfft(V_bp))
    H_ina = np.abs(np.fft.rfft(V_ina)) + 1e-12
    H_ratio = H_bp / H_ina
    mag_db = 20 * np.log10(H_ratio + 1e-12)
    plot_frequency_response(freq_fft, mag_db, output_dir=plots_dir)
    print("  [OK]")
    
    # ========== FIG 14: PSD ==========
    print("[2/7] Fig 14: PSD...")
    f_psd, psd = signal.welch(V_bp, fs=fs, nperseg=min(2048, len(V_bp)))
    psd_db = 10 * np.log10(psd + 1e-20)
    plot_psd(f_psd/1000, psd_db, output_dir=plots_dir)
    print("  [OK]")
    
    # ========== FIG 15: DC-DC VOUT vs DUTY CYCLE ==========
    print("[3/7] Fig 15: DC-DC Vout vs Duty Cycle...")
    duty_cycles = np.linspace(0.05, 0.5, 10)
    vout_data = {}
    for fsw in [50, 100, 200]:
        vout_list = []
        for D in duty_cycles:
            r = run_single_simulation(0.325, 9.3, 0.5, 10, fsw, D, 9, 9, 300, 300, n_bits=50)
            vout_list.append(r['V_dcdc_V'])
        vout_data[fsw] = np.array(vout_list)
    plot_dcdc_vout_vs_duty(duty_cycles, vout_data, output_dir=plots_dir)
    print("  [OK]")
    
    # ========== FIG 16: TIME-DOMAIN WAVEFORMS ==========
    print("[4/7] Fig 16: Time-Domain Waveforms...")
    result_td = run_single_simulation(
        0.325, 9.3, 0.5, 2.5, 100, 0.5, 9, 9, 300, 300,
        n_bits=20, sps=200, return_waveforms=True
    )
    plot_time_domain_waveforms(
        result_td['t'], result_td['V_ina'], result_td['V_bp'],
        result_td['bits_rx'], n_bits_show=20, output_dir=plots_dir
    )
    print("  [OK]")
    
    # ========== FIG 17: BER vs MODULATION DEPTH ==========
    print("[5/7] Fig 17: BER vs Modulation Depth...")
    mod_depths = np.array([0.10, 0.33, 0.50, 0.80, 1.0])
    ber_data = {}
    for data_rate in [2.5, 10]:
        ber_data[data_rate] = {}
        for fsw in [50, 100, 200]:
            ber_list = []
            for m in mod_depths:
                r = run_single_simulation(
                    0.325, 9.3, m, data_rate, fsw, 0.5, 9, 9, 300, 300, n_bits=1000
                )
                ber_list.append(r['ber'])
            ber_data[data_rate][fsw] = np.array(ber_list)
    plot_ber_vs_modulation(mod_depths, ber_data, output_dir=plots_dir)
    print("  [OK]")
    
    # ========== FIG 18: DC-DC VOUT vs MODULATION DEPTH ==========
    print("[6/7] Fig 18: DC-DC Vout vs Modulation Depth...")
    vout_mod_data = {}
    for fsw in [50, 100, 200]:
        vout_list = []
        for m in mod_depths:
            r = run_single_simulation(0.325, 9.3, m, 10, fsw, 0.5, 9, 9, 300, 300, n_bits=50)
            vout_list.append(r['V_dcdc_V'])
        vout_mod_data[fsw] = np.array(vout_list)
    plot_dcdc_vout_vs_modulation(mod_depths, vout_mod_data, output_dir=plots_dir)
    print("  [OK]")
    
    # ========== FIG 19: HARVESTED POWER vs BIT RATE ==========
    print("[7/7] Fig 19: Harvested Power vs Bit Rate...")
    bitrates = np.array([1, 5, 10, 50, 100])
    power_list = []
    for br in bitrates:
        r = run_single_simulation(0.325, 9.3, 0.5, br, 100, 0.5, 9, 9, 300, 300, n_bits=100)
        power_list.append(r['P_harvest_uW'])
    power_uw = np.array(power_list)
    plot_power_vs_bitrate(bitrates, power_uw, output_dir=plots_dir)
    print("  [OK]")
    
    print("\n" + "="*70)
    print("ALL 7 FIGURES GENERATED SUCCESSFULLY!")
    print(f"Output: {plots_dir}")
    print("="*70)


if __name__ == "__main__":
    quick_test_all_figures()
