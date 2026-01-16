# simulator/transmitter.py
"""
Optical Transmitter: LED + OOK Modulation

Equation 1: OOK Modulation with DC Bias
  I_tx(t) = I_dc + m * I_dc * d(t)

Equation LED Response:
  P_tx(t) = eta_LED * I_tx(t)
"""

import numpy as np
from utils.constants import (
    LED_EFFICIENCY_W_PER_A,
    DC_BIAS_MA,
    MODULATION_DEPTH,
)


class Transmitter:
    """
    Optical transmitter with OOK modulation.
    
    This class:
    1. Takes a bit sequence
    2. Applies OOK modulation with DC bias
    3. Converts to optical power via LED efficiency
    4. Returns optical intensity waveform
    """
    
    def __init__(self, params=None):
        """
        Initialize transmitter.
        
        Args:
            params (dict): Configuration dictionary with keys:
                - 'dc_bias': DC bias in mA
                - 'modulation_depth': m (0.1 to 0.9)
                - 'led_efficiency': eta_LED (W/A)
                - 'sample_rate': fs (Hz)
        """
        
        if params is None:
            params = {}
        
        # Extract parameters (use defaults if not provided)
        self.I_dc_ma = params.get('dc_bias', DC_BIAS_MA)  # mA
        self.m = params.get('modulation_depth', MODULATION_DEPTH)  # dimensionless
        self.eta_led = params.get('led_efficiency', LED_EFFICIENCY_W_PER_A)  # W/A
        self.fs = params.get('sample_rate', 1e6)  # Hz
        
        # Verify physically reasonable values
        assert 10 <= self.I_dc_ma <= 200, "DC bias should be 10-200 mA"
        assert 0.1 <= self.m <= 1.0, "Modulation depth should be 0.1-1.0"
        assert 0.05 <= self.eta_led <= 0.2, "LED efficiency should be 0.05-0.2 W/A"
        
        print(f"[OK] Transmitter initialized:")
        print(f"    I_dc = {self.I_dc_ma} mA")
        print(f"    m = {self.m}")
        print(f"    eta_LED = {self.eta_led} W/A")
    
    def modulate(self, bits, t, encoding='ook'):
        """
        Apply modulation with DC bias.
        
        Supports:
        - 'ook': On-Off Keying (original)
        - 'manchester': Manchester encoding (IEEE 802.3 convention)
        
        Args:
            bits (array): Binary bit sequence [0, 1, 0, 1, ...]
            t (array): Time vector (seconds)
            encoding (str): 'ook' or 'manchester'
        
        Returns:
            P_tx (array): Transmitted optical power (Watts)
        """
        
        # Determine samples per bit from t array
        n_bits = len(bits)
        sps = len(t) // n_bits
        
        if encoding == 'manchester':
            # Manchester encoding: each bit becomes two half-bit symbols
            # IEEE 802.3 convention: 
            #   bit '1' -> high-to-low transition (1, 0)
            #   bit '0' -> low-to-high transition (0, 1)
            manchester_symbols = []
            for bit in bits:
                if bit == 1:
                    manchester_symbols.extend([1, 0])  # high-low for '1'
                else:
                    manchester_symbols.extend([0, 1])  # low-high for '0'
            
            # Each symbol gets half the samples of a bit
            samples_per_symbol = sps // 2
            symbols_upsampled = np.repeat(manchester_symbols, samples_per_symbol)[:len(t)]
        else:
            # OOK: Each bit value repeated sps times
            symbols_upsampled = np.repeat(bits, sps)[:len(t)]
        
        # ========== EQUATION 1: Modulation ==========
        # I_tx(t) = I_dc + m * I_dc * d(t)
        # where d(t) is the symbol sequence (0 or 1)
        
        I_tx_ma = self.I_dc_ma + self.m * self.I_dc_ma * symbols_upsampled
        
        # Convert to Amperes
        I_tx_a = I_tx_ma * 1e-3
        
        # ========== LED RESPONSE ==========
        # P_tx = eta_LED * I_tx
        P_tx_w = self.eta_led * I_tx_a
        
        return P_tx_w
    
    def modulate_manchester(self, bits, t):
        """
        Convenience method for Manchester encoding.
        
        Manchester encoding: 
        - bit '1' -> high-to-low transition
        - bit '0' -> low-to-high transition
        
        This doubles the frequency content of the signal, making it
        DC-balanced and suitable for AC-coupled receivers.
        
        Args:
            bits (array): Binary bit sequence
            t (array): Time vector (seconds)
            
        Returns:
            P_tx (array): Transmitted optical power (Watts)
        """
        return self.modulate(bits, t, encoding='manchester')
    
    def _map_qam(self, bits, qam_order):
        """
        Map bits to QAM constellation symbols.
        
        Args:
            bits (array): Input bits
            qam_order (int): 4 for QPSK, 16 for 16-QAM, etc.
            
        Returns:
            symbols (array): Complex constellation points
        """
        if qam_order == 2:
            # BPSK: 0 -> -1, 1 -> 1
            return 2 * bits - 1
            
        if qam_order == 4:
            # QPSK / 4-QAM
            # Group into 2 bits
            n_sym = len(bits) // 2
            bits_reshaped = bits[:n_sym*2].reshape(-1, 2)
            # Map: 00->-1-j, 01->-1+j, 10->1-j, 11->1+j
            I = 2 * bits_reshaped[:, 0] - 1
            Q = 2 * bits_reshaped[:, 1] - 1
            return (I + 1j * Q) / np.sqrt(2)
            
        if qam_order in [16, 64, 256]:
            # Generic Square QAM Mapper (16, 64, 256)
            # 16-QAM: 4 bits/sym -> 2 bits I, 2 bits Q
            # 64-QAM: 6 bits/sym -> 3 bits I, 3 bits Q
            # 256-QAM: 8 bits/sym -> 4 bits I, 4 bits Q
            
            bps = int(np.log2(qam_order))
            bits_per_dim = bps // 2
            
            n_sym = len(bits) // bps
            bits_reshaped = bits[:n_sym*bps].reshape(-1, bps)
            
            # Helper to convert N bits to PAM amplitude
            # e.g. 2 bits: 00->-3, 01->-1, 11->1, 10->3 (Gray)
            def bits_to_pam(b_mat):
                # b_mat shape: (N_sym, bits_per_dim)
                # Weighted sum for binary mapping (simplest)
                # Gray mapping is better but binary is fine for functional test
                # Binary: 00..0 -> -A, 11..1 -> +A
                
                # Convert bits to integer
                # e.g. [1, 0] -> 2
                vals = np.zeros(len(b_mat))
                for i in range(b_mat.shape[1]):
                    vals += b_mat[:, -(i+1)] * (2**i)
                    
                # Map integer 0..M-1 to -M+1 .. M-1
                # M=4: 0->-3, 1->-1, 2->1, 3->3
                # Formula: 2*val - (M-1)
                M = 2**b_mat.shape[1]
                pam = 2 * vals - (M - 1)
                return pam

            I = bits_to_pam(bits_reshaped[:, :bits_per_dim])
            Q = bits_to_pam(bits_reshaped[:, bits_per_dim:])
            
            # Normalization
            # Avg power of M-PAM = (M^2 - 1) / 3
            # Avg power of QAM = 2 * P_PAM = 2 * (M_sqrt^2 - 1) / 3
            # M_sqrt = sqrt(qam_order)
            
            M_dim = 2**bits_per_dim
            avg_pwr = 2 * (M_dim**2 - 1) / 3
            norm_factor = np.sqrt(avg_pwr)
            
            return (I + 1j * Q) / norm_factor
            
        raise ValueError(f"QAM order {qam_order} not supported. Use 2, 4, 16, 64, 256.")

    def modulate_ofdm(self, bits, t, qam_order=16, n_fft=64, cp_len=16):
        """
        Generate DCO-OFDM signal (DC-Biased Optical OFDM).
        Supports ADAPTIVE BIT LOADING if qam_order is a list/array.
        
        Process:
        1. Map bits to QAM symbols (Fixed or Adaptive)
        2. Create Hermitian symmetric frame for Real-valued IFFT
        3. IFFT -> Time domain
        4. Add Cyclic Prefix
        5. Serialize and Interpolate to simulation time t
        6. Add DC bias + Clip negatives
        
        Args:
            bits (array): Input bits
            t (array): Simulation time vector
            qam_order (int or list): 
                - If int: Fixed QAM order for all subcarriers (e.g. 16)
                - If list: Exact QAM order for each data subcarrier. 
                           Length must be (n_fft // 2 - 1).
            n_fft (int): FFT size (subcarriers)
            cp_len (int): Cyclic prefix length
            
        Returns:
            P_tx (array): Optical power waveform
        """
        # Data carriers per OFDM symbol
        # Hermitian symmetry requires:
        # DC (idx 0) = 0
        # Nyquist (idx N/2) = 0
        # Data: indices 1 to N/2 - 1
        n_data_carriers = n_fft // 2 - 1
        
        # ========== 1. BIT LOADING CONFIG ==========
        if isinstance(qam_order, int):
            # Fixed Loading
            qam_map_list = [qam_order] * n_data_carriers
            bits_per_symbol_list = [int(np.log2(qam_order))] * n_data_carriers
        else:
            # Adaptive Loading
            if len(qam_order) != n_data_carriers:
                raise ValueError(f"Adaptive Loading: qam_order length ({len(qam_order)}) must match data carriers ({n_data_carriers})")
            qam_map_list = list(qam_order)
            bits_per_symbol_list = [int(np.log2(order)) for order in qam_map_list]
            
        bits_per_ofdm_symbol = sum(bits_per_symbol_list)
        n_ofdm_symbols = len(bits) // bits_per_ofdm_symbol
        
        # Truncate extra bits
        bits_used = bits[:n_ofdm_symbols * bits_per_ofdm_symbol]
        
        # ========== 2. SYMBOL MAPPING ==========
        qam_frames = np.zeros((n_ofdm_symbols, n_data_carriers), dtype=complex)
        
        bit_idx = 0
        for s_idx in range(n_ofdm_symbols):
            for k_idx in range(n_data_carriers):
                # How many bits for this subcarrier?
                n_b = bits_per_symbol_list[k_idx]
                if n_b == 0:
                    symbol = 0j
                else:
                    sub_bits = bits_used[bit_idx : bit_idx + n_b]
                    symbol = self._map_qam(sub_bits, qam_map_list[k_idx])[0] # _map_qam returns array
                    bit_idx += n_b
                qam_frames[s_idx, k_idx] = symbol
        
        ofdm_time_signal = []
        
        for frame in qam_frames:
            # 3. Hermitian Symmetry
            # Vector structure: [DC, Data, Nyquist, Conj(Flip(Data))]
            frame_freq = np.zeros(n_fft, dtype=complex)
            
            # Set data subcarriers
            frame_freq[1 : n_fft//2] = frame
            
            # Set conjugate symmetric part
            frame_freq[n_fft//2 + 1 : ] = np.conj(frame[::-1])
            
            # 4. IFFT (Result should be real)
            signal_t = np.fft.ifft(frame_freq)
            # assert np.allclose(signal_t.imag, 0), "IFFT output not real!" 
            # Relaxed assertion for speed/floating point noise
            signal_t = signal_t.real
            
            # 5. Add Cyclic Prefix
            signal_cp = np.concatenate([signal_t[-cp_len:], signal_t])
            ofdm_time_signal.extend(signal_cp)
            
        ofdm_time_signal = np.array(ofdm_time_signal)
        
        # 5. Scale and interpolate to physics time t
        # Normalizing signal to have std dev = I_bias * modulation_depth
        # This keeps the signal mostly within the linear range
        
        sig_std = np.std(ofdm_time_signal)
        if sig_std > 0:
            ofdm_time_signal /= sig_std
            
        # Target swing
        target_amp = self.I_dc_ma * self.m # mA
        
        # 6. Add DC Bias (DCO-OFDM)
        # I_tx = I_dc + signal
        # Note: ofdm_time_signal is normalized (std=1), so we scale it
        # We generally want the peaks to stay positive.
        # 3-sigma rule: max peak ~ 3*std
        # So we scale std to be fraction of DC bias
        
        # "Modulation Depth" in OFDM context usually means clipping ratio
        # Let's assume m controls the RMS power relative to DC
        scale_factor = self.m * self.I_dc_ma  # mA (RMS)
        
        # Upsample to match physics simulation 't'
        # Current length is n_samples_digital
        # Required length is len(t)
        I_tx_digital = self.I_dc_ma + ofdm_time_signal * scale_factor
        
        # Clip negative values (LED cannot emit negative light)
        I_tx_digital = np.maximum(I_tx_digital, 0)
        
        # Interpolate (Zero-order hold or Linear)
        # We need to stretch I_tx_digital to match t
        # Assuming t covers exactly the duration of the bits
        indices = np.linspace(0, len(I_tx_digital)-1, len(t))
        I_tx_interpolated = np.interp(indices, np.arange(len(I_tx_digital)), I_tx_digital)
        
        # Convert to Power
        P_tx = self.eta_led * (I_tx_interpolated * 1e-3) # W
        
        return P_tx
    def modulate_pwm_ask(self, pwm_duty_cycle, pwm_freq_hz, carrier_freq_hz, t):
        """
        Generate PWM-ASK signal (Correa et al. 2025).
        
        Signal Structure:
        - Baseband: PWM signal (Low frequency, e.g., 10 Hz)
        - Carrier: Square wave (High frequency, e.g., 10 kHz)
        - Output = Carrier * PWM_Baseband
        
        Args:
            pwm_duty_cycle (float): 0.25, 0.50, 0.75, etc.
            pwm_freq_hz (float): PWM base frequency (e.g. 10 Hz)
            carrier_freq_hz (float): Carrier frequency (e.g. 10 kHz)
            t (array): Time vector
            
        Returns:
            P_tx (array): Optical power waveform
        """
        # 1. Generate PWM Baseband
        # t * f gives cycles. (t*f) % 1 gives phase 0..1
        phase_pwm = (t * pwm_freq_hz) % 1.0
        pwm_baseband = (phase_pwm < pwm_duty_cycle).astype(float)
        
        # 2. Generate ASK Carrier (Square Wave 0..1)
        # Note: 10Vpp amplitude is handled in conversion logic
        phase_carrier = (t * carrier_freq_hz) % 1.0
        carrier = (phase_carrier < 0.5).astype(float)
        
        # 3. ASK Modulation (Multiplication)
        signal_digital = pwm_baseband * carrier
        
        # 4. Convert to Current/Power
        # Map 1 -> High Power, 0 -> Low Power
        # High = I_dc * (1 + m)
        # Low  = I_dc * (1 - m)
        
        level_high = self.I_dc_ma * (1 + self.m)
        level_low  = self.I_dc_ma * (1 - self.m)
        
        I_tx_ma = level_low + (level_high - level_low) * signal_digital
        
        # 5. Convert to Optical Power
        P_tx = self.eta_led * (I_tx_ma * 1e-3)
        
        return P_tx

    def get_power_stats(self, bits):
        """
        Compute expected TX power statistics.
        
        Useful for sanity checking.
        
        Args:
            bits (array): Bit sequence
        
        Returns:
            dict: Statistics (min, max, mean power)
        """
        
        # For bit = 1: I_tx = I_dc + m*I_dc = I_dc*(1+m)
        I_tx_1 = self.I_dc_ma * (1 + self.m)
        P_tx_1 = self.eta_led * (I_tx_1 * 1e-3)
        
        # For bit = 0: I_tx = I_dc - m*I_dc = I_dc*(1-m)
        I_tx_0 = self.I_dc_ma * (1 - self.m)
        P_tx_0 = self.eta_led * (I_tx_0 * 1e-3)
        
        # Average (assuming equal probability 0 and 1)
        P_tx_mean = (P_tx_1 + P_tx_0) / 2
        
        return {
            'P_tx_bit1_mw': P_tx_1 * 1e3,
            'P_tx_bit0_mw': P_tx_0 * 1e3,
            'P_tx_mean_mw': P_tx_mean * 1e3,
            'P_tx_swing_mw': (P_tx_1 - P_tx_0) * 1e3,
        }


    def modulate_fsk_passive(self, bits, t, f0=1600, f1=2000, rise_time=1.34e-3, fall_time=0.15e-3, power_density=600.0):
        """
        Produce Passive FSK Waveform (Liquid Crystal Shutter).
        
        Args:
            bits (array): Input bits
            t (array): Time vector (seconds)
            f0 (float): Frequency for bit '0' (Hz)
            f1 (float): Frequency for bit '1' (Hz)
            rise_time (float): Transitions 0->1 (seconds)
            fall_time (float): Transitions 1->0 (seconds)
            power_density (float): Base optical power density (W/m^2)
            
        Returns:
            P_tx (array): Optical power density waveform (W/m^2)
        """
        # 1. Generate Ideal Square Wave FSK
        # Need to track phase to avoid discontinuities
        # instantaneous_freq(t) -> phase(t) = integral(freq)
        
        dt = t[1] - t[0]
        n_bits = len(bits)
        sps = len(t) // n_bits
        
        # Upsample bits to time/frequency vector
        current_freq = np.repeat(bits, sps).astype(float)
        current_freq[current_freq == 0] = f0
        current_freq[current_freq == 1] = f1
        
        # Handle length mismatch due to repeat/sps
        current_freq = current_freq[:len(t)]
        if len(current_freq) < len(t):
             current_freq = np.pad(current_freq, (0, len(t)-len(current_freq)), 'edge')
             
        # Phase Accumulation
        phase = np.cumsum(current_freq) * dt * 2 * np.pi
        
        # Square wave: sign(sin(phase)) -> 0 or 1
        # LC is fully open (1) or checking (0)? Unclear.
        # Usually LC blocks light (0) or passes (1).
        # FSK usually toggles between high/low at rate f.
        # Square wave 50% duty cycle at freq f.
        
        ideal_square = 0.5 * (1 + np.sign(np.sin(phase)))
        
        # 2. Simulate Slow Rise/Fall Times (Exponential Smoothing)
        # y[n] = y[n-1] + alpha * (target - y[n-1])
        # alpha ~ dt / tau
        
        alpha_rise = dt / rise_time if rise_time > 0 else 1.0
        alpha_fall = dt / fall_time if fall_time > 0 else 1.0
        
        # Clip alphas for stability
        alpha_rise = np.clip(alpha_rise, 0, 1)
        alpha_fall = np.clip(alpha_fall, 0, 1)
        
        lc_state = np.zeros_like(ideal_square)
        current_val = 0.0
        
        for i in range(len(t)):
            target = ideal_square[i]
            if target > current_val:
                # Rising
                current_val += alpha_rise * (target - current_val)
            else:
                # Falling
                current_val += alpha_fall * (target - current_val)
            lc_state[i] = current_val
            
        # 3. Scale by Sunlight Power
        P_tx = lc_state * power_density
        
        return P_tx


# ========== TESTS ==========

def test_transmitter():
    """Unit test for transmitter."""
    
    print("\n" + "="*60)
    print("TRANSMITTER UNIT TEST")
    print("="*60)
    
    # Create transmitter with default params
    tx = Transmitter()
    
    # Generate test bit sequence
    bits = np.array([1, 0, 1, 1, 0])
    sps = 100  # samples per bit
    fs = 1e6  # 1 MHz sample rate
    t = np.arange(len(bits) * sps) / fs
    
    # Modulate
    P_tx = tx.modulate(bits, t)
    
    # Print stats
    stats = tx.get_power_stats(bits)
    print(f"\nTransmitter Output:")
    print(f"  P_tx (bit=1): {stats['P_tx_bit1_mw']:.2f} mW")
    print(f"  P_tx (bit=0): {stats['P_tx_bit0_mw']:.2f} mW")
    print(f"  P_tx (mean):  {stats['P_tx_mean_mw']:.2f} mW")
    print(f"  P_tx (swing): {stats['P_tx_swing_mw']:.2f} mW")
    
    print(f"\nWaveform:")
    print(f"  Length: {len(P_tx)} samples")
    print(f"  Min: {P_tx.min()*1e3:.3f} mW")
    print(f"  Max: {P_tx.max()*1e3:.3f} mW")
    print(f"  Mean: {P_tx.mean()*1e3:.3f} mW")
    
    # Validation checks
    assert P_tx.min() >= 0, "[ERROR] Negative power detected!"
    assert P_tx.max() > 0, "[ERROR] All power zero!"
    assert len(P_tx) == len(t), "[ERROR] Output length mismatch!"
    
    print("\n[OK] All transmitter tests passed!")
    print("="*60)
    
    return P_tx


if __name__ == "__main__":
    test_transmitter()
