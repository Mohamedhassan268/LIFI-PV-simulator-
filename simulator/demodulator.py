# simulator/demodulator.py - CORRECTED VERSION

"""
Signal Processing & Demodulation for PV Receiver.

Components:
1. High-pass filter (remove DC bias)
2. Low-pass filter (reduce noise)
3. Matched filter / sampling
4. Decision circuit
5. BER calculation

CHANGES FROM ORIGINAL:
- Fixed inverted decision logic (< changed to >)
- Updated default filter cutoffs
- Reduced filter order from 4 to 2
"""

import numpy as np
from scipy import signal


class Demodulator:
    """
    Signal processing and bit recovery.
    
    This class:
    1. Filters the PV voltage signal
    2. Samples at bit centers
    3. Makes bit decisions
    4. Compares with transmitted bits
    5. Calculates BER
    """
    
    def __init__(self, params=None):
        """
        Initialize demodulator.
        
        Args:
            params (dict): Configuration with keys:
                - 'hpf_cutoff': High-pass cutoff (Hz)
                - 'lpf_cutoff': Low-pass cutoff (Hz)
                - 'filter_order': Filter order (default=2, CHANGED from 4)
                - 'sample_rate': fs (Hz)
        """
        if params is None:
            params = {}
        
        self.fs = params.get('sample_rate', 1e6)  # Hz
        
        # FIXED: More appropriate default cutoffs
        # HPF should be ~0.1x data rate (to remove DC)
        # LPF should be ~0.75x data rate (to filter noise but pass signal)
        self.hpf_cutoff = params.get('hpf_cutoff', 1e5)  # 100 kHz (was 1 kHz!)
        self.lpf_cutoff = params.get('lpf_cutoff', 7.5e5)  # 750 kHz (was 2 MHz!)
        self.filter_order = params.get('filter_order', 2)  # FIXED: Order-2 (was 4)
        
        # Design filters
        self._design_filters()
        
        print(f"[OK] Demodulator initialized (CORRECTED):")
        print(f"  HPF cutoff: {self.hpf_cutoff/1e3:.0f} kHz")
        print(f"  LPF cutoff: {self.lpf_cutoff/1e6:.2f} MHz")
        print(f"  Filter order: {self.filter_order}")
    
    def _design_filters(self):
        """Design Butterworth filters."""
        # Normalize cutoff frequencies
        nyquist = self.fs / 2
        
        # High-pass filter (remove DC)
        hpf_norm = self.hpf_cutoff / nyquist
        if hpf_norm >= 1.0:
            raise ValueError(f"HPF cutoff {self.hpf_cutoff} Hz >= Nyquist {nyquist} Hz")
        
        self.hpf_b, self.hpf_a = signal.butter(self.filter_order, hpf_norm,
                                               btype='high', analog=False)
        
        # Low-pass filter (remove high-frequency noise)
        lpf_norm = self.lpf_cutoff / nyquist
        if lpf_norm >= 1.0:
            raise ValueError(f"LPF cutoff {self.lpf_cutoff} Hz >= Nyquist {nyquist} Hz")
        
        self.lpf_b, self.lpf_a = signal.butter(self.filter_order, lpf_norm,
                                               btype='low', analog=False)
    
    def apply_hpf(self, signal_in):
        """
        Apply high-pass filter to remove DC bias.
        
        Args:
            signal_in (array): Input signal (V)
        
        Returns:
            signal_out (array): Filtered signal (V)
        """
        # Check for NaN before filtering
        if np.any(np.isnan(signal_in)):
            print("WARNING: NaN detected in input to HPF!")
            signal_in = np.nan_to_num(signal_in, nan=0.0)
        
        signal_out = signal.filtfilt(self.hpf_b, self.hpf_a, signal_in)
        return signal_out
    
    def apply_lpf(self, signal_in):
        """
        Apply low-pass filter to reduce noise.
        
        Args:
            signal_in (array): Input signal (V)
        
        Returns:
            signal_out (array): Filtered signal (V)
        """
        # Check for NaN before filtering
        if np.any(np.isnan(signal_in)):
            print("WARNING: NaN detected in input to LPF!")
            signal_in = np.nan_to_num(signal_in, nan=0.0)
        
        signal_out = signal.filtfilt(self.lpf_b, self.lpf_a, signal_in)
        return signal_out
    
    def sample_bits(self, signal_in, n_bits, sps):
        """
        Sample signal at bit centers.
        
        Args:
            signal_in (array): Filtered signal
            n_bits (int): Number of bits
            sps (int): Samples per symbol
        
        Returns:
            samples (array): Sampled values (one per bit)
        """
        # Sample at the center of each bit period
        sample_indices = np.arange(n_bits) * sps + sps // 2
        sample_indices = sample_indices.astype(int)
        
        # Clip to valid range
        sample_indices = np.clip(sample_indices, 0, len(signal_in) - 1)
        
        samples = signal_in[sample_indices]
        return samples
    
    def make_decisions(self, samples, threshold='auto'):
        """
        Convert samples to bit decisions.
        
        Args:
            samples (array): Sampled signal values
            threshold (float or 'auto'): Decision threshold
        
        Returns:
            bits_rx (array): Recovered bits (0 or 1)
            threshold_used (float): Actual threshold used
        """
        # Auto threshold: use midpoint between mean 0 and mean 1
        if threshold == 'auto':
            threshold_used = (np.max(samples) + np.min(samples)) / 2
        else:
            threshold_used = threshold
        
        # CRITICAL FIX: Changed from < to >
        # For OOK with positive modulation, bit=1 has HIGHER voltage
        bits_rx = (samples > threshold_used).astype(int)  # FIXED!
        
        return bits_rx, threshold_used
    
    def calculate_ber(self, bits_tx, bits_rx):
        """
        Calculate Bit Error Rate.
        
        Args:
            bits_tx (array): Transmitted bits
            bits_rx (array): Received bits
        
        Returns:
            dict: BER statistics
        """
        # Ensure same length
        n = min(len(bits_tx), len(bits_rx))
        bits_tx = bits_tx[:n]
        bits_rx = bits_rx[:n]
        
        # Count errors
        errors = np.sum(bits_tx != bits_rx)
        
        # Calculate BER
        ber = errors / n if n > 0 else 1.0
        
        return {
            'ber': ber,
            'errors': errors,
            'total_bits': n,
            'accuracy': 1 - ber,
        }
    
    def demodulate(self, V_pv, bits_tx, n_bits, sps, verbose=False):
        """
        Complete demodulation pipeline.
        
        Args:
            V_pv (array): PV voltage signal
            bits_tx (array): Transmitted bits (for comparison)
            n_bits (int): Number of bits
            sps (int): Samples per symbol
            verbose (bool): Print debug info
        
        Returns:
            dict: Demodulation results
        """
        # Step 1: High-pass filter
        V_hpf = self.apply_hpf(V_pv)
        
        # Step 2: Low-pass filter
        V_lpf = self.apply_lpf(V_hpf)
        
        # Step 3: Sample at bit centers
        samples = self.sample_bits(V_lpf, n_bits, sps)
        
        # Step 4: Make decisions (FIXED logic)
        bits_rx, threshold = self.make_decisions(samples, threshold='auto')
        
        # Step 5: Calculate BER
        ber_stats = self.calculate_ber(bits_tx, bits_rx)
        
        if verbose:
            print(f"\nDemodulation Results:")
            print(f"  HPF output range: {V_hpf.min()*1e3:.2f} to {V_hpf.max()*1e3:.2f} mV")
            print(f"  LPF output range: {V_lpf.min()*1e3:.2f} to {V_lpf.max()*1e3:.2f} mV")
            print(f"  Decision threshold: {threshold*1e3:.2f} mV")
            print(f"  BER: {ber_stats['ber']:.6f} ({ber_stats['errors']} errors / {ber_stats['total_bits']} bits)")
            print(f"  Accuracy: {ber_stats['accuracy']*100:.4f}%")
        
        return {
            'V_hpf': V_hpf,
            'V_lpf': V_lpf,
            'samples': samples,
            'bits_rx': bits_rx,
            'threshold': threshold,
            'ber_stats': ber_stats,
        }


# ========== TESTS ==========

def test_demodulator():
    """Unit test for demodulator."""
    print("\n" + "="*60)
    print("DEMODULATOR UNIT TEST (CORRECTED)")
    print("="*60)
    
    # Create test signal
    fs = 1e6
    sps = 100
    n_bits = 100
    
    # Generate test bits
    bits_tx = np.random.randint(0, 2, n_bits)
    
    # Create test signal (simple OOK)
    t = np.arange(n_bits * sps) / fs
    signal_test = np.repeat(bits_tx, sps) * 0.1 + 0.05  # 50-150 mV
    
    # Add noise
    signal_test += np.random.normal(0, 0.005, len(signal_test))  # 5 mV noise
    
    # Create demodulator with corrected defaults
    demod = Demodulator({'sample_rate': fs})
    
    # Demodulate
    result = demod.demodulate(signal_test, bits_tx, n_bits, sps, verbose=True)
    
    # Validation
    assert result['ber_stats']['ber'] <= 0.05, f"[ERROR] BER too high: {result['ber_stats']['ber']:.2%}"
    assert len(result['bits_rx']) == n_bits, "[ERROR] Wrong number of bits!"
    
    print("\n[OK] All demodulator tests passed!")
    print("="*60)
    
    return result


if __name__ == "__main__":
    test_demodulator()
