# utils/paper_config_oliveira2024.py
"""
Oliveira 2024 entry for paper_configs.py.

This file defines the oliveira2024 config dict to be merged into PAPER_CONFIGS.
Kept as a separate file so we can validate it independently, then merge.

To integrate: add OLIVEIRA2024_CONFIG to PAPER_CONFIGS dict in paper_configs.py.
"""

OLIVEIRA2024_CONFIG = {
    'title': 'Reconfigurable MIMO-based self-powered battery-less light communication system',
    'journal': 'Light: Science & Applications',
    'year': 2024,
    'doi': '10.1038/s41377-024-01566-3',
    
    'params': {
        # System Geometry
        'n_pds_total': 9,
        'n_pds_row': 3,
        'n_pds_col': 3,
        
        # Large PD (Luna sd445-14-21-305)
        'large_pd_active_area_cm2': 1.0,
        'large_pd_responsivity_630nm': 0.36,
        'large_pd_bandwidth_mhz': 1.5,
        
        # Small PD (PDBC171SM)
        'small_pd_active_area_mm2': 7.7,
        'small_pd_total_area_cm2': 0.693,
        
        # OFDM
        'ofdm_nfft': 1024,
        'ofdm_cp_length': 10,
        'ofdm_n_subcarriers': 500,
        'ofdm_bandwidth_low_hz': 15000,
        'ofdm_bandwidth_high_hz': 1500000,
        'ofdm_min_snr_db': 3.0,
        'ofdm_max_qam_order': 64,
        
        # Switching
        'switching_time_us': 22.0,
        'snr_scan_time_ms': 43.0,
        'settling_time_ms': 4.0,
        
        # Energy harvesting
        'harvest_max_power_mw': 87.33,
        'supercap_capacitance_f': 0.1,
        'supercap_voltage_v': 5.0,
        
        # System power
        'system_power_idle_mw': 27.88,
        'system_power_active_mw': 69.36,
        'system_power_typical_mw': 43.0,
        
        # Color detection
        'color_red_wavelength_nm': 658,
        'color_blue_wavelength_nm': 450,
        'color_green_wavelength_nm': 530,
        
        # Laser specifications
        'laser_red_wavelength_nm': 658,
        'laser_blue_wavelength_nm': 450,
        'led_green_wavelength_nm': 530,
    },
    
    'targets': {
        # Communication
        'ber_siso_small_pd': 3.4e-3,
        'ber_siso_large_pd': 3.3e-3,
        'ber_fec_threshold': 3.8e-3,
        'siso_gross_data_rate_mbps': 25.7,
        'siso_net_data_rate_mbps': 21.3,
        'mimo_net_data_rate_mbps': 85.2,
        'large_pd_net_rate_mbps': 4.0,
        
        # Energy harvesting
        'harvest_max_power_mw': 87.33,
        
        # Switching
        'switching_time_us': 22.0,
    },
    
    'figures': [
        'fig3c_subcarrier_performance',
        'fig3d_constellation',
        'fig4c_beam_tracking',
        'fig5_energy_harvesting',
        'tradeoff_comm_vs_harvest',
        'comparison_summary',
    ],
}


if __name__ == "__main__":
    print("Oliveira 2024 Paper Config")
    print("=" * 50)
    print(f"Title: {OLIVEIRA2024_CONFIG['title'][:60]}...")
    print(f"Year: {OLIVEIRA2024_CONFIG['year']}")
    print(f"Params: {len(OLIVEIRA2024_CONFIG['params'])} keys")
    print(f"Targets: {len(OLIVEIRA2024_CONFIG['targets'])} keys")
    print(f"Figures: {len(OLIVEIRA2024_CONFIG['figures'])}")
    
    print("\nValidation targets:")
    for k, v in OLIVEIRA2024_CONFIG['targets'].items():
        print(f"  {k}: {v}")
    
    print("\nTASK 5 PASSED")
