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
        assert 0.1 <= self.m <= 0.9, "Modulation depth should be 0.1-0.9"
        assert 0.05 <= self.eta_led <= 0.2, "LED efficiency should be 0.05-0.2 W/A"
        
        print(f"[OK] Transmitter initialized:")
        print(f"    I_dc = {self.I_dc_ma} mA")
        print(f"    m = {self.m}")
        print(f"    eta_LED = {self.eta_led} W/A")
    
    def modulate(self, bits, t):
        """
        Apply OOK modulation with DC bias.
        
        Equation 1: I_tx(t) = I_dc + m * I_dc * d(t)
        LED response: P_tx(t) = eta_LED * I_tx(t)
        
        Args:
            bits (array): Binary bit sequence [0, 1, 0, 1, ...]
            t (array): Time vector (seconds)
        
        Returns:
            P_tx (array): Transmitted optical power (Watts)
        """
        
        # Determine samples per bit from t array
        n_bits = len(bits)
        sps = len(t) // n_bits
        
        # Upsample bits to sample rate
        # Each bit value repeated sps times
        bits_upsampled = np.repeat(bits, sps)[:len(t)]
        
        # ========== EQUATION 1: OOK Modulation ==========
        # I_tx(t) = I_dc + m * I_dc * d(t)
        # where d(t) is the bit sequence (0 or 1)
        
        I_tx_ma = self.I_dc_ma + self.m * self.I_dc_ma * bits_upsampled
        
        # Convert to Amperes
        I_tx_a = I_tx_ma * 1e-3
        
        # ========== LED RESPONSE ==========
        # P_tx = eta_LED * I_tx
        P_tx_w = self.eta_led * I_tx_a
        
        return P_tx_w
    
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
