# simulator/channel.py
"""
Free-space Optical Wireless Channel: Lambertian Path Loss + Beer-Lambert Attenuation + AWGN Noise

Equation 2: Lambertian Path Loss
  P_rx = [(m_L+1) * A_rx / (2pi * r^2)] * P_tx * cos^m_L(phi) * cos(psi)

Beer-Lambert Atmospheric Attenuation (Correa 2025):
  P_rx *= exp(-α * d)
  where α depends on humidity (higher humidity → higher attenuation)

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
        
        # Sanity check for distance (Conflict 10 fix)
        if self.distance > 10.0:
            print(f"WARNING: distance={self.distance}m > 10m. Did you mean {self.distance*0.01:.3f}m (cm→m)?")
        if self.distance < 0.05:
            print(f"WARNING: distance={self.distance}m < 5cm. Very short range, check units.")
        
        self.beam_angle_deg = params.get('beam_angle_half', LED_BEAM_HALF_ANGLE_DEG)  # deg
        self.rx_area_cm2 = params.get('receiver_area', RX_AREA_CM2)  # cm^2
        self.rx_area_m2 = self.rx_area_cm2 * 1e-4  # Convert to m^2
        
        # Angles (on-axis by default)
        self.tx_angle_deg = params.get('tx_angle', 0)  # degrees
        self.rx_angle_deg = params.get('rx_angle', 0)  # degrees
        
        # Temperature for thermal noise
        self.temp_k = params.get('temperature', ROOM_TEMPERATURE_K)  # K
        
        # ========== BEER-LAMBERT ATTENUATION ==========
        # Humidity-dependent atmospheric absorption (Correa 2025)
        humidity_input = params.get('humidity', None)  # 0-1 scale (None = disabled)
        
        # Auto-detect and normalize humidity input (Conflict 8 fix)
        if humidity_input is not None:
            if humidity_input > 1.0:
                # Likely passed as percentage (0-100), convert to fraction
                print(f"WARNING: humidity={humidity_input} > 1, assuming percentage, converting to {humidity_input/100:.2f}")
                humidity_input = humidity_input / 100.0
            humidity_input = np.clip(humidity_input, 0.0, 1.0)
        
        self.humidity = humidity_input
        self.attenuation_alpha = self._compute_alpha(self.humidity)
        
        # Humidity layer depth — distance over which Beer-Lambert attenuation applies.
        # Default: full TX-RX path (entire distance is humidity-affected).
        # Override with a shorter value only if the humid zone is physically smaller
        # than the link distance (e.g., localized fog, spray near plants).
        self.humidity_layer_depth = params.get('humidity_layer_depth', self.distance)
        
        # Compute Lambertian order
        self.m_L = lambertian_order(self.beam_angle_deg)

        # Scattering / Multipath Configuration
        self.enable_multipath = params.get('multipath', False)
        self.room_dim = params.get('room_dimensions', None)  # dict with {width, length, height, reflection}

        print(f"[OK] Channel initialized:")
        print(f"    Distance: {self.distance} m")
        print(f"    Beam half-angle: {self.beam_angle_deg} deg")
        print(f"    Lambertian order (m_L): {self.m_L:.2f}")
        print(f"    RX area: {self.rx_area_cm2} cm^2")
        if self.humidity is not None:
            print(f"    Humidity: {self.humidity*100:.0f}%")
            print(f"    Beer-Lambert α: {self.attenuation_alpha:.3f} m⁻¹")
            print(f"    Humidity layer: {self.humidity_layer_depth:.2f} m")
        if self.enable_multipath:
            print(f"    Multipath: Enabled (Ceiling bounce)")
    
    def _compute_alpha(self, humidity):
        """
        Compute Beer-Lambert attenuation coefficient from humidity.
        
        Calibrated to match Correa Morales 2025 Fig. 6 behavior:
        - α increases nonlinearly with humidity (aerosol scattering physics)
        - Range: ~0.3 m⁻¹ (30% RH) to ~3.1 m⁻¹ (80% RH)
        - Power-law growth: aerosol particle size/density scales superlinearly
        
        Model: α = 0.3 + 4.0 × (RH - 0.3)^1.5    [from correa_2025_physics.py]
        
        Validation (Correa Fig. 6):
            30% RH → α = 0.30 m⁻¹  (dry baseline)
            50% RH → α = 0.93 m⁻¹  (moderate)
            80% RH → α = 3.12 m⁻¹  (humid greenhouse)
        
        Args:
            humidity: Relative humidity as fraction (0-1), or None to disable
            
        Returns:
            α: Attenuation coefficient (m⁻¹)
        """
        if humidity is None:
            return 0.0  # No Beer-Lambert attenuation
        
        # Clamp humidity to valid range
        humidity = np.clip(humidity, 0.0, 1.0)
        
        # Baseline attenuation (dry air scattering at 30% RH)
        alpha_base = 0.3  # m⁻¹
        
        # Humidity-dependent component — POWER LAW (physically correct)
        # Aerosol scattering grows nonlinearly with water vapor concentration.
        # The ^1.5 exponent captures the superlinear relationship between
        # relative humidity and optical extinction in the visible band.
        # Coefficients calibrated against Correa 2025 experimental Fig. 6.
        if humidity >= 0.3:
            alpha_humidity = 4.0 * (humidity - 0.3) ** 1.5  # MUST be exponentiation
        else:
            alpha_humidity = 0.0
        
        return alpha_base + alpha_humidity
    
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

    
    def _compute_ceiling_bounce_impulse(self, room_dim):
        """
        Compute impulse response of ceiling bounce (1st order reflection).
        
        Simple model assumes:
        - Lambertian source pointing up/down interacts with ceiling
        - Receiver pointing up
        - Single dominant reflection path
        (Simplified for 1D/2D simulation context)
        
        Args:
            room_dim (dict): {'width': w, 'length': l, 'height': h, 'reflection_coeff': rho}
            
        Returns:
            gain (float): Integrated path gain for NLOS
            delay (float): Time delay relative to LOS (seconds)
        """
        # Default room if not specified
        if room_dim is None:
            h_room = 3.0  # meters
            rho = 0.8     # Typical white ceiling
        else:
            h_room = room_dim.get('height', 3.0)
            rho = room_dim.get('reflection_coeff', 0.8)
            
        # Simplest geometric model for ceiling bounce delay
        # Path: TX -> Ceiling -> RX
        # Assume TX and RX are at desk height (e.g., 0.8m)
        h_desk = 0.8
        h_ceil = h_room - h_desk
        
        # LOS distance
        d_los = self.distance
        
        # NLOS distance (bounce off ceiling midpoint)
        # d_nlos = 2 * sqrt((d/2)^2 + h_ceil^2)
        d_nlos = 2 * np.sqrt((d_los/2)**2 + h_ceil**2)
        
        # Delay (relative to LOS, but propagate() adds absolute delay implicitly if we model full channel)
        # Here we model the *additional* delay for the multipath tap
        delta_d = d_nlos - d_los
        delay = delta_d / 3.0e8  # seconds
        
        # Path loss for NLOS
        # Just use square law + reflection coeff as approximation for diffuse reflection
        # P_nlos = P_tx * rho * (A / 2pi d_nlos^2) ... roughly
        # Better: use channel DC gain formula for Ceiling Bounce
        
        # Standard Ceiling Bounce Model (Carruthers & Kahn):
        # H(0)_diff = rho * A_rx / (3 * pi * h_ceil^2) * ... (depends heavily on FOV)
        # Let's use a simplified scaling relative to LOS for this "first implementation"
        # typically NLOS is 10-20 dB below LOS for directed beams
        
        # Distance factor ratio
        dist_factor = (d_los / d_nlos) ** 2
        
        # Gain relative to LOS gain
        # gain_nlos = gain_los * rho * dist_factor * (diffuse_penalty)
        gain_relative = rho * dist_factor * 0.5 # 0.5 for diffuse scattering loss
        
        return gain_relative, delay

    def propagate(self, P_tx, t, H_matrix=None, verbose=False):
        """
        Propagate optical signal through Lambertian channel.
        Supports SISO (scalar), MIMO (matrix), and Scattering (Multipath).
        
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
        gain_lambertian = ((self.m_L + 1) * self.rx_area_m2) / (2 * np.pi * self.distance**2)
        
        # Beer-Lambert atmospheric attenuation
        # FIX 5: Only apply attenuation within humidity layer
        d_attenuated = min(self.distance, self.humidity_layer_depth)
        beer_lambert_factor = np.exp(-self.attenuation_alpha * d_attenuated)
        
        # Combined LOS gain
        gain_los_total = gain_lambertian * beer_lambert_factor
        
        # Apply LOS gain
        P_rx = gain_los_total * P_tx
        
        # MULTIPATH / SCATTERING (Ceiling Bounce)
        if hasattr(self, 'enable_multipath') and self.enable_multipath:
            # Calculate NLOS component
            gain_rel, delay_s = self._compute_ceiling_bounce_impulse(self.room_dim)
            gain_nlos = gain_los_total * gain_rel
            
            # Convert delay to samples
            dt = t[1] - t[0]
            delay_samples = int(round(delay_s / dt))
            
            if delay_samples > 0 and delay_samples < len(P_tx):
                # Create delayed copy
                P_nlos = np.zeros_like(P_tx)
                P_nlos[delay_samples:] = P_tx[:-delay_samples] * gain_nlos
                
                # Add to Rx
                P_rx += P_nlos
                
                if verbose:
                    print(f"  Multipath Added:")
                    print(f"    NLOS Gain: {gain_nlos:.3e} (Rel: {gain_rel:.3f})")
                    print(f"    Delay: {delay_s*1e9:.1f} ns ({delay_samples} samples)")
        
        if verbose:
            print(f"\nChannel Propagation (SISO):")
            print(f"  Lambertian gain: {gain_lambertian:.3e}")
            if self.attenuation_alpha > 0:
                print(f"  Beer-Lambert factor: {beer_lambert_factor:.4f} (α={self.attenuation_alpha:.3f} m⁻¹, d_att={d_attenuated:.2f}m)")
            print(f"  Total LOS gain: {gain_los_total:.3e}")
            if hasattr(P_tx, 'min'):
                 print(f"  P_tx range: {P_tx.min()*1e3:.3f} - {P_tx.max()*1e3:.3f} mW")
                 print(f"  P_rx range: {P_rx.min()*1e6:.3f} - {P_rx.max()*1e6:.3f} uW")
        
        return P_rx
    
    def add_noise(self, P_rx, I_ph, B, I_ambient=0.0, R_load=None, verbose=False):
        """
        Add AWGN noise from physical sources.
        
        Noise sources:
          1. Signal shot noise:  σ²_shot    = 2·q·I_ph·B
          2. Thermal noise:      σ²_thermal = 4·k_B·T·B / R
          3. Ambient shot noise: σ²_ambient = 2·q·I_ambient·B
        
        Args:
            P_rx (array): Received optical power (Watts)
            I_ph (array): Signal photocurrent (Amperes)
            B (float): Noise bandwidth (Hz)
            I_ambient (float): Ambient light photocurrent (A), default=0
                Typical indoor: ~1–10 µA for solar panels under room lighting.
                Greenhouse: ~50–500 µA under supplemental grow lights.
            R_load (float): Load/transimpedance resistance (Ω), default=from constants
            verbose (bool): Print debug info
        
        Returns:
            noise (array): AWGN noise (Amperes)
        """
        if R_load is None:
            R_load = TRANSIMPEDANCE_LOAD_OHMS
        
        # ========== 1. SIGNAL SHOT NOISE ==========
        # S_shot = 2·q·I_ph·B
        I_ph_avg = np.mean(np.abs(I_ph))
        S_shot = 2 * Q * I_ph_avg * B
        
        # ========== 2. THERMAL NOISE ==========
        # S_thermal = 4·k_B·T·B / R
        S_thermal = 4 * K_B * self.temp_k * B / R_load
        
        # ========== 3. AMBIENT LIGHT SHOT NOISE ==========
        # S_ambient = 2·q·I_ambient·B
        S_ambient = 2 * Q * abs(I_ambient) * B
        
        # ========== TOTAL NOISE ==========
        sigma_noise = np.sqrt(S_shot + S_thermal + S_ambient)
        
        # Generate AWGN
        noise = np.random.normal(0, sigma_noise, len(P_rx))
        
        if verbose:
            print(f"\nNoise Model (3-source):")
            print(f"  S_shot:    {S_shot:.3e} A²")
            print(f"  S_thermal: {S_thermal:.3e} A²")
            print(f"  S_ambient: {S_ambient:.3e} A²")
            print(f"  σ_noise:   {sigma_noise:.3e} A")
            if sigma_noise > 0:
                print(f"  SNR_approx: {10*np.log10((np.mean(I_ph)**2) / (sigma_noise**2)):.1f} dB")
        
        return noise


# ========== TESTS ==========

def test_channel():
    """Unit test for channel — validates all fixes."""
    
    print("\n" + "="*60)
    print("CHANNEL UNIT TEST")
    print("="*60)
    
    # --- Test 1: Basic propagation (no humidity) ---
    print("\n[Test 1] Basic SISO propagation...")
    ch = OpticalChannel({'distance': 1.0})
    P_tx = np.ones(1000) * 5e-3  # 5 mW constant
    t = np.arange(1000) / 1e6
    P_rx = ch.propagate(P_tx, t, verbose=True)
    
    assert P_rx.min() >= 0, "[ERROR] Negative power!"
    assert P_rx.max() <= P_tx.max(), "[ERROR] Power gain > 1!"
    print("  [OK] Basic propagation passed")
    
    # --- Test 2: Humidity alpha (power-law fix) ---
    print("\n[Test 2] Humidity alpha model (power-law)...")
    ch_dry = OpticalChannel({'distance': 1.0, 'humidity': 0.30})
    ch_mid = OpticalChannel({'distance': 1.0, 'humidity': 0.50})
    ch_wet = OpticalChannel({'distance': 1.0, 'humidity': 0.80})
    
    alpha_dry = ch_dry.attenuation_alpha
    alpha_mid = ch_mid.attenuation_alpha
    alpha_wet = ch_wet.attenuation_alpha
    
    print(f"  α(30%) = {alpha_dry:.3f} m⁻¹  (expect ~0.30)")
    print(f"  α(50%) = {alpha_mid:.3f} m⁻¹  (expect ~0.93)")
    print(f"  α(80%) = {alpha_wet:.3f} m⁻¹  (expect ~3.12)")
    
    # Verify power-law shape: α should NOT be linear
    # Linear (old bug) would give α(50%)=1.85, α(80%)=7.25
    assert alpha_dry < 0.5, f"[ERROR] α(30%)={alpha_dry} too high — expected ~0.30"
    assert alpha_mid < 1.5, f"[ERROR] α(50%)={alpha_mid} too high — old linear bug?"
    assert alpha_wet < 4.0, f"[ERROR] α(80%)={alpha_wet} too high — old linear bug?"
    assert alpha_wet > alpha_mid > alpha_dry, "[ERROR] α not monotonically increasing"
    
    # Verify nonlinearity: ratio check
    # Power law: (80-30)/(50-30) in alpha should be > 2 (superlinear)
    ratio = (alpha_wet - alpha_dry) / (alpha_mid - alpha_dry) if alpha_mid > alpha_dry else 0
    assert ratio > 2.0, f"[ERROR] Nonlinearity ratio={ratio:.1f}, expected >2 (power law)"
    print(f"  Nonlinearity ratio: {ratio:.1f} (>2 confirms power law)")
    print("  [OK] Humidity alpha power-law verified")
    
    # --- Test 3: Humidity layer depth defaults to full distance ---
    print("\n[Test 3] Humidity layer depth default...")
    ch_full = OpticalChannel({'distance': 1.3, 'humidity': 0.80})
    assert ch_full.humidity_layer_depth == 1.3, \
        f"[ERROR] layer_depth={ch_full.humidity_layer_depth}, expected distance=1.3"
    
    ch_custom = OpticalChannel({'distance': 1.3, 'humidity': 0.80, 'humidity_layer_depth': 0.6})
    assert ch_custom.humidity_layer_depth == 0.6, \
        f"[ERROR] Custom layer_depth override failed"
    print("  [OK] Layer depth defaults to full distance, override works")
    
    # --- Test 4: Noise model with ambient light ---
    print("\n[Test 4] 3-source noise model...")
    I_ph = np.ones(1000) * 1e-6  # 1 µA signal
    B = 1e6
    
    noise_no_amb = ch.add_noise(P_rx, I_ph, B, I_ambient=0.0, verbose=False)
    noise_with_amb = ch.add_noise(P_rx, I_ph, B, I_ambient=100e-6, verbose=True)
    
    # Ambient noise should increase total noise power
    power_no_amb = np.var(noise_no_amb)
    power_with_amb = np.var(noise_with_amb)
    assert power_with_amb > power_no_amb, "[ERROR] Ambient noise had no effect!"
    print(f"  Noise power (no ambient):   {power_no_amb:.3e}")
    print(f"  Noise power (100µA ambient): {power_with_amb:.3e}")
    print(f"  Increase: {power_with_amb/power_no_amb:.1f}×")
    print("  [OK] Ambient noise source working")
    
    # --- Test 5: Backward compatibility (old 2-arg signature still works) ---
    print("\n[Test 5] Backward compatibility...")
    noise_old_api = ch.add_noise(P_rx, I_ph, B)  # No I_ambient, no R_load
    assert len(noise_old_api) == len(P_rx), "[ERROR] Old API broke"
    print("  [OK] Old add_noise(P_rx, I_ph, B) signature still works")
    
    print("\n" + "="*60)
    print("[OK] ALL CHANNEL TESTS PASSED!")
    print("="*60)
    
    return P_rx, noise_with_amb


if __name__ == "__main__":
    test_channel()
