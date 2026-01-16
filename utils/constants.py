# utils/constants.py

"""
Physical constants and default parameters for Li-Fi + PV system.

All constants follow SI units unless otherwise specified.
"""

import numpy as np

# ========== PHYSICAL CONSTANTS ==========
Q = 1.6e-19  # Electron charge (Coulombs)
K_B = 1.38e-23  # Boltzmann constant (J/K)
H = 6.626e-34  # Planck constant (J·s)
C_LIGHT = 3.0e8  # Speed of light (m/s)

# ========== LED/TRANSMITTER PARAMETERS ==========
DC_BIAS_MA = 12  # DC bias current (mA)
MODULATION_DEPTH = 0.5  # Modulation depth (0.1-0.9)
LED_EFFICIENCY_W_PER_A = 0.08  # LED optical efficiency (W/A)
LED_BEAM_HALF_ANGLE_DEG = 9  # LED half-power angle (degrees)
LED_WAVELENGTH_NM = 650  # Red LED wavelength (nm)

# ========== PHOTODIODE/RECEIVER PARAMETERS ==========
PHOTODIODE_RESPONSIVITY_A_PER_W = 0.457  # A/W (Si @ 650nm)
PHOTODIODE_DARK_CURRENT_NA = 1.0  # Dark current (nA)
PHOTODIODE_JUNCTION_CAP_PF = 798  # Junction capacitance (pF) - CRITICAL!
PHOTODIODE_SHUNT_RESISTANCE_MOHM = 0.1388  # Shunt resistance (MΩ)

# ========== CHANNEL PARAMETERS ==========
RX_AREA_CM2 = 9.0  # Receiver photodiode area (cm²)
RX_AREA_M2 = RX_AREA_CM2 * 1e-4  # Receiver area in m²
TRANSIMPEDANCE_LOAD_OHMS = 50  # TIA load resistance (Ω)
ROOM_TEMPERATURE_K = 300  # Room temperature (K)

# ========== PAPER VALIDATION CONFIG (Kadirvelu et al. IEEE TGCN 2021) ==========
# "A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell"

PAPER_VALIDATION_CONFIG = {
    # Link geometry
    'distance_m': 0.325,              # 32.5 cm
    'radiated_power_mw': 9.3,         # 9.3 mW LED power (red LED + lens)
    'led_half_angle_deg': 9,          # Fraen 9° lens
    
    # GaAs Solar Cell Module (13 cells in series)
    'solar_cell_area_cm2': 9.0,       # 5cm × 1.8cm = 9 cm²
    'responsivity_a_per_w': 0.457,    # GaAs responsivity (A/W)
    'n_cells_module': 13,             # Number of cells in series
    
    # Current-sense receiver chain
    'rsense_ohm': 1.0,                # Current-sense resistor (Ω)
    'ina_gain': 100,                  # INA gain (40 dB)
    'bpf_low_hz': 700,                # BPF lower cutoff (Hz)
    'bpf_high_hz': 10000,             # BPF upper cutoff (Hz)
    'bpf_order': 2,                   # 2nd order Butterworth
    
    # Small-signal circuit parameters (from paper analysis)
    'rsh_ohm': 138800.0,              # Shunt resistance (Ω) -> 138.8 kΩ
    'cj_pf': 798000.0,                # Junction capacitance (pF) -> 798 nF
    'cload_uf': 10,                   # Load capacitance (µF)
    'rload_ohm': 1360,                # Load resistance ~1.36 kΩ
}

# DC-DC Converter efficiency from paper measurements
DCDC_EFFICIENCY_CONFIG = {
    # Measured efficiency vs switching frequency
    50: 0.67,     # 67% at fsw = 50 kHz
    100: 0.564,   # 56.4% at fsw = 100 kHz
    200: 0.42,    # 42% at fsw = 200 kHz
    
    # Default parameters
    'default_duty_cycle': 0.5,
    'v_diode_drop': 0.3,      # Volts
    'r_on_mohm': 100,         # MOSFET on-resistance in mΩ
}

# Noise model parameters
NOISE_MODEL_CONFIG = {
    'temperature_K': 300,             # Room temperature
    'load_resistance_ohm': 50,        # TIA/load resistance
    'ina_noise_nv_sqrt_hz': 10,       # INA input noise density (typical)
    'ambient_lux_indoor': 300,        # Typical indoor lighting
    'ambient_lux_range': (100, 500),  # Indoor range
    'ambient_responsivity': 0.4,      # A/W for ambient light
}

# BER test conditions from paper
BER_TEST_CONFIG = {
    'n_bits': 1_000_000,              # 1M bits for BER measurement
    'n_bits_fast': 10_000,            # 10k for development
    'n_bits_balanced': 10_000,        # 10k for balanced speed
    'n_bits_accurate': 100_000,       # 100k for accurate results
    'tbit_us': [400, 100],            # Tbit = 400µs (2.5 kbps), 100µs (10 kbps)
    'data_rates_bps': [2500, 10000],  # Corresponding data rates
    'mod_depths': [0.10, 0.20, 0.33, 0.50, 0.65, 0.80, 1.0],  # Extended range
    'fsw_khz': [50, 100, 200],        # Switching frequencies (kHz)
}

# Target values for validation report
PAPER_TARGETS = {
    'harvested_power_uw': 223,        # µW at 0.325m, 9.3mW radiated
    'ber_target': 1.008e-3,           # At 2.5 kbps, 50% mod depth
    'ber_data_rate_bps': 2500,        # Data rate for BER target
    'ber_mod_depth': 0.50,            # Mod depth for BER target
}

# ========== ML DATASET CONFIGURATION ==========

ML_DATASET_CONFIG = {
    # Sample sizes
    'n_samples_20k': 20_000,
    'n_samples_50k': 50_000,
    
    # Primary variations (high impact)
    'distance_m_range': (0.1, 1.0),
    'P_tx_mw_range': (1.0, 20.0),
    'mod_depth_range': (0.1, 0.9),
    'data_rate_kbps_range': (1, 100),
    
    # Secondary variations (medium impact)
    'fsw_khz_range': (20, 500),
    'duty_cycle_range': (0.1, 0.5),
    'beam_angle_deg_range': (5, 60),
    'rx_area_cm2_range': (1, 20),
    
    # Environmental variations (indoor)
    'ambient_lux_range': (100, 500),
    'temperature_K_range': (280, 330),
}

# Feature names for ML datasets
ML_FEATURE_NAMES = {
    'inputs': [
        'distance_m', 'P_tx_mW', 'mod_depth', 'data_rate_kbps',
        'fsw_khz', 'duty_cycle', 'beam_angle_deg', 'rx_area_cm2',
        'ambient_lux', 'temperature_K'
    ],
    'outputs': [
        'ber', 'snr_db', 'P_rx_uW', 'P_harvest_uW', 'V_dcdc_V',
        'I_ph_uA', 'eye_opening_percent', 'q_factor', 'efficiency_percent'
    ],
    'derived': [
        'path_loss_dB', 'noise_floor_dBm', 'link_margin_dB'
    ]
}

# ========== HELPER FUNCTIONS ==========

def thermal_voltage(T):
    """
    Calculate thermal voltage V_T = k_B·T / q
    
    Args:
        T (float): Temperature in Kelvin
    
    Returns:
        float: Thermal voltage in Volts (~26 mV at 300K)
    """
    return K_B * T / Q


def lambertian_order(beam_half_angle_deg):
    """
    Calculate Lambertian order from LED half-power angle.
    
    Formula: m_L = -ln(2) / ln(cos(Φ_1/2))
    
    Args:
        beam_half_angle_deg (float): LED half-power angle in degrees
    
    Returns:
        float: Lambertian order m_L (typically 1-10)
    """
    phi_rad = np.deg2rad(beam_half_angle_deg)
    cos_phi = np.cos(phi_rad)
    
    if cos_phi <= 0 or cos_phi >= 1:
        raise ValueError(f"Invalid beam angle: {beam_half_angle_deg}°")
    
    m_L = -np.log(2.0) / np.log(cos_phi)
    return m_L


def photodiode_responsivity(wavelength_nm, quantum_efficiency=0.8):
    """
    Calculate photodiode responsivity from wavelength.
    
    Formula: R = η · (q·λ) / (h·c)
    
    Args:
        wavelength_nm (float): Wavelength in nanometers
        quantum_efficiency (float): Quantum efficiency (0-1)
    
    Returns:
        float: Responsivity in A/W
    """
    wavelength_m = wavelength_nm * 1e-9
    R = quantum_efficiency * (Q * wavelength_m) / (H * C_LIGHT)
    return R


# ========== VALIDATION ==========

def validate_constants():
    """Run validation checks on constants."""
    print("\n" + "="*60)
    print("VALIDATING PHYSICAL CONSTANTS")
    print("="*60)
    
    # Check thermal voltage
    V_T = thermal_voltage(300)
    assert 0.025 < V_T < 0.027, f"V_T = {V_T:.4f} V (expected ~0.026 V)"
    print(f"✓ Thermal voltage: {V_T*1e3:.2f} mV at 300K")
    
    # Check Lambertian order
    m_L = lambertian_order(30)
    assert 1 < m_L < 3, f"m_L = {m_L:.2f} (expected 1-3 for 30°)"
    print(f"✓ Lambertian order: {m_L:.2f} for 30° beam")
    
    # Check responsivity
    R = photodiode_responsivity(650, 0.8)
    assert 0.35 < R < 0.45, f"R = {R:.3f} A/W (expected ~0.42 for Si@650nm)"
    print(f"✓ Responsivity: {R:.3f} A/W for 650nm")
    
    # Check time constant stability
    tau_rc = PHOTODIODE_JUNCTION_CAP_PF * 1e-12 * PHOTODIODE_SHUNT_RESISTANCE_MOHM * 1e6
    print(f"✓ RC time constant: {tau_rc*1e6:.1f} μs")
    
    if tau_rc < 1e-6:
        print(f"  ⚠️  WARNING: τ = {tau_rc*1e6:.1f} μs is very small!")
        print(f"  For dt = 2 μs (1 MHz @ 500 sps), may be unstable")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    validate_constants()
