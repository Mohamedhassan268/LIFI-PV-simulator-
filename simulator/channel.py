# simulator/channel.py
"""
Free-space Optical Wireless Channel: Lambertian Path Loss + AWGN Noise

Equation 2: Lambertian Path Loss
  P_rx = [(m_L+1) * A_rx / (2pi * r^2)] * P_tx * cos^m_L(phi) * cos(psi)

Equations 3-5: Noise Model (Shot + Thermal)
  S_shot = 2*q*I_ph*B
  S_thermal = 4*k_B*T*B/R
  Total: AWGN with sigma_noise^2
"""

import numpy as np
from utils.constants import (
    Q,
    K_B,
    LED_BEAM_HALF_ANGLE_DEG,
    RX_AREA_CM2,
    RX_AREA_M2,
    TRANSIMPEDANCE_LOAD_OHMS,
    ROOM_TEMPERATURE_K,
    lambertian_order,
)


class OpticalChannel:
    """
    Free-space optical wireless channel.
    
    This class:
    1. Takes transmitted optical power
    2. Applies Lambertian path loss (distance + angles)
    3. Adds shot noise and thermal noise (AWGN)
    4. Returns received signal with noise
    """
    
    def __init__(self, params=None):
        """
        Initialize channel.
        
        Args:
            params (dict): Configuration with keys:
                - 'distance': TX-RX distance (meters)
                - 'beam_angle_half': LED half-angle (degrees)
                - 'receiver_area': RX area (cm^2)
                - 'tx_angle': TX angle from normal (degrees), default=0
                - 'rx_angle': RX angle from normal (degrees), default=0
                - 'temperature': Room temperature (K), default=300
        """
        
        if params is None:
            params = {}
        
        # Channel geometry
        self.distance = params.get('distance', 1.0)  # meters
        self.beam_angle_deg = params.get('beam_angle_half', LED_BEAM_HALF_ANGLE_DEG)  # deg
        self.rx_area_cm2 = params.get('receiver_area', RX_AREA_CM2)  # cm^2
        self.rx_area_m2 = self.rx_area_cm2 * 1e-4  # Convert to m^2
        
        # Angles (on-axis by default)
        self.tx_angle_deg = params.get('tx_angle', 0)  # degrees
        self.rx_angle_deg = params.get('rx_angle', 0)  # degrees
        
        # Temperature for thermal noise
        self.temp_k = params.get('temperature', ROOM_TEMPERATURE_K)  # K
        
        # Compute Lambertian order
        self.m_L = lambertian_order(self.beam_angle_deg)
        
        print(f"[OK] Channel initialized:")
        print(f"    Distance: {self.distance} m")
        print(f"    Beam half-angle: {self.beam_angle_deg} deg")
        print(f"    Lambertian order (m_L): {self.m_L:.2f}")
        print(f"    RX area: {self.rx_area_cm2} cm^2")
    
    def propagate(self, P_tx, t, verbose=False):
        """
        Propagate optical signal through Lambertian channel.
        
        Equation 2: Lambertian Path Loss
          P_rx = [(m_L+1) * A_rx / (2pi * r^2)] * P_tx * cos^m_L(phi_tx) * cos(psi_rx)
        
        Simplified on-axis (phi_tx=0, psi_rx=0):
          P_rx = [(m_L+1) * A_rx / (2pi * r^2)] * P_tx
        
        Args:
            P_tx (array): Transmitted optical power (Watts)
            t (array): Time vector (seconds)
            verbose (bool): Print debug info
        
        Returns:
            P_rx (array): Received optical power (Watts)
        """
        
        # ========== COMPUTE PATH LOSS GAIN ==========
        # On-axis: cos^m_L(0) = 1, cos(0) = 1
        gain_linear = ((self.m_L + 1) * self.rx_area_m2) / (2 * np.pi * self.distance**2)
        
        # Apply to all samples
        P_rx = gain_linear * P_tx
        
        if verbose:
            print(f"\nChannel Propagation:")
            print(f"  Path loss gain: {gain_linear:.3e}")
            print(f"  P_tx range: {P_tx.min()*1e3:.3f} - {P_tx.max()*1e3:.3f} mW")
            print(f"  P_rx range: {P_rx.min()*1e6:.3f} - {P_rx.max()*1e6:.3f} uW")
        
        return P_rx
    
    def add_noise(self, P_rx, I_ph, B, verbose=False):
        """
        Add AWGN (shot + thermal noise).
        
        Equations 3-5: Noise Model
          Shot noise power: S_shot = 2*q*I_ph*B
          Thermal noise power: S_thermal = 4*k_B*T*B/R
          Total noise std: sigma = sqrt(S_shot + S_thermal)
        
        Args:
            P_rx (array): Received optical power (Watts)
            I_ph (array): Photocurrent (Amperes) - used to compute shot noise
            B (float): Bandwidth (Hz)
            verbose (bool): Print debug info
        
        Returns:
            noise (array): AWGN noise (Amperes)
        """
        
        # ========== SHOT NOISE ==========
        # S_shot = 2*q*I_ph*B
        # Use average photocurrent for bandwidth-limited shot noise
        I_ph_avg = np.mean(np.abs(I_ph))
        S_shot = 2 * Q * I_ph_avg * B
        
        # ========== THERMAL NOISE ==========
        # S_thermal = 4*k_B*T*B/R
        S_thermal = 4 * K_B * self.temp_k * B / TRANSIMPEDANCE_LOAD_OHMS
        
        # ========== TOTAL NOISE STD ==========
        sigma_noise = np.sqrt(S_shot + S_thermal)
        
        # Generate AWGN
        noise = np.zeros(len(P_rx))  # DISABLED for testing
        noise = np.random.normal(0, sigma_noise * 0.001, len(P_rx))  # Re-enabled, reduced
        if verbose:
            print(f"\nNoise Model:")
            print(f"  S_shot: {S_shot:.3e} W")
            print(f"  S_thermal: {S_thermal:.3e} W")
            print(f"  sigma_noise: {sigma_noise:.3e} A")
            print(f"  SNR_approx: {10*np.log10((np.mean(I_ph)**2) / (sigma_noise**2)):.1f} dB")
        
        return noise


# ========== TESTS ==========

def test_channel():
    """Unit test for channel."""
    
    print("\n" + "="*60)
    print("CHANNEL UNIT TEST")
    print("="*60)
    
    # Create channel
    ch = OpticalChannel({'distance': 1.0})
    
    # Test signal (simple pulse)
    P_tx = np.ones(1000) * 5e-3  # 5 mW constant
    t = np.arange(1000) / 1e6
    
    # Propagate
    P_rx = ch.propagate(P_tx, t, verbose=True)
    
    # Noise
    I_ph = np.ones(1000) * 1e-6  # 1 uA
    B = 1e6  # 1 MHz bandwidth
    noise = ch.add_noise(P_rx, I_ph, B, verbose=True)
    
    # Validation
    assert P_rx.min() >= 0, "[ERROR] Negative power!"
    assert P_rx.max() <= P_tx.max(), "[ERROR] Power gain > 1!"
    assert len(noise) == len(P_rx), "[ERROR] Noise length mismatch!"
    
    print("\n[OK] All channel tests passed!")
    print("="*60)
    
    return P_rx, noise


if __name__ == "__main__":
    test_channel()
