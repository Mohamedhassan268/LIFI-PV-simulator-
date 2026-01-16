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
    
    def compute_h_matrix(self, tx_pos_list, rx_pos_list, tx_normal_list=None, rx_normal_list=[[0,0,1]]):
        """
        Compute the H (LxM) Channel Gain Matrix for MIMO.
        
        Args:
            tx_pos_list: List of [x,y,z] for each Transmitter (M)
            rx_pos_list: List of [x,y,z] for each Receiver (L)
            tx_normal_list: List of normals [nx,ny,nz], default all pointing down [0,0,-1]
            rx_normal_list: List of normals [nx,ny,nz], default all pointing up [0,0,1]
            
        Returns:
            H (ndarray): LxM matrix where H[i,j] is gain from Tx[j] to Rx[i]
        """
        M = len(tx_pos_list) # Num Transmitters
        L = len(rx_pos_list) # Num Receivers
        
        if tx_normal_list is None:
            tx_normal_list = [[0, 0, -1]] * M # Down
            
        if rx_normal_list is None or len(rx_normal_list) != L:
            rx_normal_list = [[0, 0, 1]] * L # Up
            
        H = np.zeros((L, M))
        
        for i in range(L): # For each Receiver
            rx_pos = np.array(rx_pos_list[i])
            rx_norm = np.array(rx_normal_list[i])
            
            for j in range(M): # For each Transmitter
                tx_pos = np.array(tx_pos_list[j])
                tx_norm = np.array(tx_normal_list[j])
                
                # Vector from Tx to Rx
                vec_d = rx_pos - tx_pos
                dist = np.linalg.norm(vec_d)
                
                if dist == 0:
                    H[i,j] = 0
                    continue
                    
                # Cosine of angle at Tx (Irradiance angle)
                # phi = angle between normal and vector to Rx
                cos_phi = np.dot(tx_norm, vec_d / dist)
                
                # Cosine of angle at Rx (Incidence angle)
                # psi = angle between normal and vector from Tx (which is -vec_d)
                cos_psi = np.dot(rx_norm, -vec_d / dist)
                
                # Check Field of View
                # Assuming typical 90 deg FOV for simplicity: cos_psi > 0
                if cos_phi > 0 and cos_psi > 0:
                    # Lambertian calc
                    # Gain = [(m+1)A / 2pi d^2] * cos^m(phi) * cos(psi)
                    gain = ((self.m_L + 1) * self.rx_area_m2) / (2 * np.pi * dist**2)
                    gain *= (cos_phi ** self.m_L)
                    gain *= cos_psi
                    H[i,j] = gain
                else:
                    H[i,j] = 0.0
                    
        return H

    def propagate(self, P_tx, t, H_matrix=None, verbose=False):
        """
        Propagate optical signal through Lambertian channel.
        Supports SISO (scalar) and MIMO (matrix) modes.
        
        Args:
            P_tx (array): 
                - If SISO: 1D array of power (Watts)
                - If MIMO: array shape (M_tx, N_samples)
            t (array): Time vector (seconds)
            H_matrix (ndarray, optional): Precomputed LxM gain matrix.
                If None, uses simple SISO distance model.
            verbose (bool): Print debug info
        
        Returns:
            P_rx (array):
                - If SISO: 1D array (Watts)
                - If MIMO: array shape (L_rx, N_samples)
        """
        
        # MIMO MODE
        if H_matrix is not None:
            # P_tx shape: (M, N) or (N,)
            # If 1D, assume 1 Tx but broadcast via matrix? No, must match dimension.
            
            # Ensure P_tx is (M, N)
            if P_tx.ndim == 1:
                P_tx = P_tx.reshape(1, -1)
                
            M_tx = H_matrix.shape[1]
            if P_tx.shape[0] != M_tx:
                raise ValueError(f"MIMO Mismatch: H expects {M_tx} Tx, got {P_tx.shape[0]}")
                
            # Matrix Multiply: P_rx = H * P_tx
            # (L, M) * (M, N) -> (L, N)
            P_rx = np.dot(H_matrix, P_tx)
            
            if verbose:
                print(f"\nMIMO Propagation:")
                print(f"  H shape: {H_matrix.shape}")
                print(f"  Input power shape: {P_tx.shape}")
                print(f"  Output power shape: {P_rx.shape}")
                
            return P_rx

        # SISO MODE (Default)
        # On-axis: cos^m_L(0) = 1, cos(0) = 1
        gain_linear = ((self.m_L + 1) * self.rx_area_m2) / (2 * np.pi * self.distance**2)
        
        # Apply to all samples
        P_rx = gain_linear * P_tx
        
        if verbose:
            print(f"\nChannel Propagation (SISO):")
            print(f"  Path loss gain: {gain_linear:.3e}")
            if hasattr(P_tx, 'min'):
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
        
        # Generate AWGN with physical noise level
        # Use full physical noise (removed 0.001 artificial reduction)
        noise = np.random.normal(0, sigma_noise, len(P_rx))
        
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
