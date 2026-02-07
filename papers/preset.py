"""
Oliveira 2024 Paper Preset Configuration
=========================================

Paper: "Reconfigurable MIMO-based self-powered battery-less light communication system"
Journal: Light: Science & Applications (2024) 13:218
DOI: https://doi.org/10.1038/s41377-024-01566-3

This preset contains all locked parameters from the paper for validation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

# =============================================================================
# PAPER PRESET
# =============================================================================

OLIVEIRA_2024_PARAMS = {
    # ===== SYSTEM GEOMETRY =====
    'n_pds_total': 9,                    # 3x3 PD matrix
    'n_pds_row': 3,
    'n_pds_col': 3,
    
    # ===== LARGE PD SPECIFICATIONS (Luna sd445-14-21-305) =====
    'large_pd_active_area_cm2': 1.0,     # 1 cm² each
    'large_pd_package_area_cm2': 2.24,   # 2.24 cm² package
    'large_pd_responsivity_630nm': 0.36, # A/W at 630nm
    'large_pd_bandwidth_mhz': 1.5,       # Limited by PSoC ADC
    
    # ===== SMALL PD SPECIFICATIONS (PDBC171SM) =====
    'small_pd_active_area_mm2': 7.7,     # 7.7 mm² each
    'small_pd_total_area_cm2': 0.693,    # 69.3 mm² total (9 PDs)
    
    # ===== OFDM PARAMETERS =====
    'ofdm_bandwidth_low_hz': 15000,      # 0.015 MHz
    'ofdm_bandwidth_high_hz': 1500000,   # 1.5 MHz
    'ofdm_nfft': 1024,                   # FFT size
    'ofdm_cp_length': 10,                # Cyclic prefix
    'ofdm_n_subcarriers': 500,           # NSC in paper
    'ofdm_min_snr_db': 3.0,              # Minimum SNR per subcarrier
    'ofdm_max_qam_order': 64,            # Maximum QAM order
    
    # ===== DATA RATES (from paper) =====
    'siso_gross_data_rate_mbps': 25.7,   # Single PD
    'siso_net_data_rate_mbps': 21.3,     # After FEC overhead
    'mimo_gross_data_rate_mbps': 102.8,  # 4 channels x 25.7
    'mimo_net_data_rate_mbps': 85.2,     # After FEC overhead
    'large_pd_gross_rate_mbps': 5.3,     # Large PD SISO
    'large_pd_net_rate_mbps': 4.0,       # Large PD net
    
    # ===== BER PERFORMANCE =====
    'ber_siso_large_pd': 3.3e-3,         # Large PD average BER
    'ber_siso_small_pd': 3.4e-3,         # Small PD average BER
    'ber_fec_threshold': 3.8e-3,         # FEC limit
    
    # ===== SWITCHING PARAMETERS =====
    'switching_time_us': 22.0,           # Max switching time
    'snr_scan_time_ms': 43.0,            # Full SNR scan duration
    'settling_time_ms': 4.0,             # Per-PD settling time
    
    # ===== ENERGY HARVESTING (Table 1) =====
    'harvest_unilluminated_lux': 14,
    'harvest_unilluminated_voc': 2.57,
    'harvest_unilluminated_isc_ma': 0.0078,
    
    'harvest_incandescent_60w_lux': 1275,
    'harvest_incandescent_60w_voc': 3.82,
    'harvest_incandescent_60w_isc_ma': 2.75,
    
    'harvest_focused_75w_lux': 17780,
    'harvest_focused_75w_voc': 5.06,
    'harvest_focused_75w_isc_ma': 21.36,
    
    'harvest_shaded_sunlight_lux': 1852,
    'harvest_shaded_sunlight_voc': 3.84,
    'harvest_shaded_sunlight_isc_ma': 14.10,
    
    'harvest_direct_sunlight_lux': 55900,
    'harvest_direct_sunlight_voc': 4.10,
    'harvest_direct_sunlight_isc_ma': 21.30,
    
    'harvest_max_power_mw': 87.33,       # Direct sunlight
    
    # ===== SYSTEM POWER CONSUMPTION =====
    'system_power_idle_mw': 27.88,       # 3 MHz clock
    'system_power_active_mw': 69.36,     # 48 MHz clock
    'system_power_typical_mw': 43.0,     # Balanced operation
    
    # ===== SUPERCAPACITOR =====
    'supercap_capacitance_f': 0.1,       # 0.1 F
    'supercap_voltage_v': 5.0,           # 5 V
    'supercap_charge_time_single_pd_min': 32.2,
    'supercap_charge_time_no_pd_min': 31.0,
    
    # ===== COLOR DETECTION =====
    'color_blue_wavelength_nm': 450,
    'color_green_wavelength_nm': 530,
    'color_red_wavelength_nm': 658,
    'color_sensor_blue_peak_nm': 470,
    'color_sensor_green_peak_nm': 550,
    'color_sensor_red_peak_nm': 620,
    
    # ===== CUBE SYSTEM SPECS =====
    'cube_pd_area_mm2': 7.7,
    'cube_plane_area_mm2': 69.3,
    'cube_efficiency_mw_per_mm2': 0.149,
    'large_efficiency_mw_per_mm2': 0.12,
    
    # ===== LASER SPECIFICATIONS =====
    'laser_blue_wavelength_nm': 450,     # Osram PL-TB450B
    'laser_red_wavelength_nm': 658,      # Mitsubishi ML101J23
    'led_green_wavelength_nm': 530,      # High-power green LED
}

# Energy harvesting data as structured array for plotting
ENERGY_HARVESTING_TABLE = [
    # (lux, Voc, Isc_mA, source_name)
    (14, 2.57, 0.0078, 'Unilluminated'),
    (1275, 3.82, 2.75, 'Incandescent 60W'),
    (17780, 5.06, 21.36, 'Focused 75W'),
    (1852, 3.84, 14.10, 'Shaded Sunlight'),
    (55900, 4.10, 21.30, 'Direct Sunlight'),
]

# Quadrant configurations (10 total, from Fig. 4d)
QUADRANT_CONFIGS = {
    'I':    [0, 1, 3, 4],      # Top-left 2x2
    'II':   [1, 2, 4, 5],      # Top-right 2x2
    'III':  [3, 4, 6, 7],      # Bottom-left 2x2
    'IV':   [4, 5, 7, 8],      # Bottom-right 2x2
    'V':    [0, 1, 2],         # Top row
    'VI':   [3, 4, 5],         # Middle row
    'VII':  [6, 7, 8],         # Bottom row
    'VIII': [0, 3, 6],         # Left column
    'IX':   [1, 4, 7],         # Middle column
    'X':    [2, 5, 8],         # Right column
}

# Mode configurations
OPERATING_MODES = {
    'siso_max_harvest': {
        'comm_pds': 1,
        'harvest_pds': 8,
        'data_rate_mbps': 25.7,
        'power_fraction': 8/9,
    },
    'mimo_quadrant': {
        'comm_pds': 4,
        'harvest_pds': 5,
        'data_rate_mbps': 85.2,
        'power_fraction': 5/9,
    },
    'all_comm': {
        'comm_pds': 9,
        'harvest_pds': 0,
        'data_rate_mbps': 130.0,  # Estimated
        'power_fraction': 0,
    },
}


def get_oliveira_params() -> Dict[str, Any]:
    """Return a copy of the Oliveira 2024 parameters."""
    return OLIVEIRA_2024_PARAMS.copy()


def get_harvest_table():
    """Return energy harvesting data table."""
    return ENERGY_HARVESTING_TABLE.copy()


def get_quadrant_config(quadrant_name: str):
    """Return PD indices for a specific quadrant."""
    return QUADRANT_CONFIGS.get(quadrant_name, [])


def calculate_harvested_power(lux: float, n_pds: int = 9) -> float:
    """
    Estimate harvested power from illuminance.
    
    Uses linear interpolation of Table 1 data.
    
    Args:
        lux: Illuminance in lux
        n_pds: Number of PDs used for harvesting
        
    Returns:
        Estimated power in mW
    """
    import numpy as np
    
    # Extract data points
    lux_values = np.array([14, 1275, 1852, 17780, 55900])
    # Power = Voc x Isc (approximate MPP)
    power_values = np.array([
        2.57 * 0.0078,      # 0.02 mW
        3.82 * 2.75,        # 10.5 mW
        3.84 * 14.10,       # 54.1 mW
        5.06 * 21.36,       # 108 mW (but capped at 87.33)
        4.10 * 21.30,       # 87.3 mW
    ])
    
    # Cap at measured maximum
    power_values = np.minimum(power_values, 87.33)
    
    # Interpolate (log-linear for lux)
    log_lux = np.log10(lux_values)
    log_lux_query = np.log10(max(lux, 1))
    
    power = np.interp(log_lux_query, log_lux, power_values)
    
    # Scale by number of PDs
    return power * (n_pds / 9)


def calculate_data_rate(n_comm_pds: int, pd_type: str = 'small') -> float:
    """
    Calculate achievable data rate based on number of communication PDs.
    
    Args:
        n_comm_pds: Number of PDs used for communication
        pd_type: 'small' or 'large'
        
    Returns:
        Net data rate in Mbps
    """
    if pd_type == 'large':
        base_rate = 4.0  # Mbps per PD
    else:
        base_rate = 21.3  # Mbps per PD
    
    # Efficiency factor for MIMO (accounts for interference)
    if n_comm_pds == 1:
        efficiency = 1.0
    elif n_comm_pds <= 4:
        efficiency = 0.85
    else:
        efficiency = 0.75
    
    return base_rate * n_comm_pds * efficiency


if __name__ == "__main__":
    print("Oliveira 2024 Paper Parameters")
    print("=" * 50)
    
    params = get_oliveira_params()
    
    print("\nKey Parameters:")
    print(f"  PD Matrix: {params['n_pds_row']}x{params['n_pds_col']}")
    print(f"  SISO Data Rate: {params['siso_gross_data_rate_mbps']} Mbps (gross)")
    print(f"  MIMO Data Rate: {params['mimo_net_data_rate_mbps']} Mbps (net)")
    print(f"  Max Harvest Power: {params['harvest_max_power_mw']} mW")
    print(f"  System Power: {params['system_power_typical_mw']} mW")
    
    print("\nEnergy Harvesting Table:")
    for lux, voc, isc, name in ENERGY_HARVESTING_TABLE:
        power = voc * isc
        print(f"  {name:20s}: {lux:6d} lux -> {power:6.2f} mW")
