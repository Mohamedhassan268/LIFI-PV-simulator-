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
DC_BIAS_MA = 50  # DC bias current (mA)
MODULATION_DEPTH = 0.5  # Modulation depth (0.1-0.9)
LED_EFFICIENCY_W_PER_A = 0.08  # LED optical efficiency (W/A)
LED_BEAM_HALF_ANGLE_DEG = 30  # LED half-power angle (degrees)
LED_WAVELENGTH_NM = 650  # Red LED wavelength (nm)

# ========== PHOTODIODE/RECEIVER PARAMETERS ==========
PHOTODIODE_RESPONSIVITY_A_PER_W = 0.42  # A/W (Si @ 650nm)
PHOTODIODE_DARK_CURRENT_NA = 1.0  # Dark current (nA)
PHOTODIODE_JUNCTION_CAP_PF = 100  # Junction capacitance (pF) - CRITICAL!
PHOTODIODE_SHUNT_RESISTANCE_MOHM = 1.0  # Shunt resistance (MΩ)

# ========== CHANNEL PARAMETERS ==========
RX_AREA_CM2 = 1.0  # Receiver photodiode area (cm²)
RX_AREA_M2 = RX_AREA_CM2 * 1e-4  # Receiver area in m²
TRANSIMPEDANCE_LOAD_OHMS = 50  # TIA load resistance (Ω)
ROOM_TEMPERATURE_K = 300  # Room temperature (K)

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
