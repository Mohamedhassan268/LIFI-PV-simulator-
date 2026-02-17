# simulator/demodulator.py
"""
Signal Processing & Demodulation for PV Receiver.

Components:
1. High-pass filter (remove DC bias)
2. Low-pass filter (reduce noise)
3. Matched filter / sampling
4. Decision circuit (OOK + Manchester)
5. BER calculation
6. BER prediction functions (OOK, BPSK, M-QAM)
"""

import numpy as np
from scipy import signal
from scipy.special import erfc


# =============================================================================
# BER PREDICTION FUNCTIONS (module-level, no class needed)
# =============================================================================

def predict_ber_ook(snr_linear):
    """Predict BER for OOK: BER = 0.5 * erfc(sqrt(SNR/2))."""
    return 0.5 * erfc(np.sqrt(snr_linear / 2))

def predict_ber_ook_db(snr_db):
    """Predict BER for OOK given SNR in dB."""
    snr_linear = 10 ** (np.asarray(snr_db, dtype=float) / 10)
    return predict_ber_ook(snr_linear)

def predict_ber_bpsk(snr_linear):
    """Predict BER for BPSK: BER = 0.5 * erfc(sqrt(SNR))."""
    return 0.5 * erfc(np.sqrt(snr_linear))

def predict_ber_mqam(snr_linear, M):
    """Approximate BER for M-QAM."""
    if M <= 1:
        return 0.0
    k = np.log2(M)
    factor = (4 / k) * (1 - 1 / np.sqrt(M))
    arg = np.sqrt(3 * snr_linear / (M - 1))
    return factor * 0.5 * erfc(arg / np.sqrt(2))

def compute_eb_n0(snr_linear, bandwidth_hz, bit_rate_bps):
    """Compute Eb/N0 = SNR * (B / R)."""
    return snr_linear * (bandwidth_hz / bit_rate_bps)


# =============================================================================
# DEMODULATOR CLASS
# =============================================================================

class Demodulator:
    """
    Signal processing and bit recovery with filter caching.
    
    Filter coefficients are cached at the class level so that creating
    multiple Demodulator instances with the same parameters reuses
    the already-designed filters instead of calling signal.butter() again.
    """
    
    _filter_cache = {}
    
    def __init__(self, params=None):
        if params is None:
            params = {}
        
        self.fs = params.get('sample_rate', 1e6)
        self.hpf_cutoff = params.get('hpf_cutoff', 1e4)
        self.lpf_cutoff = params.get('lpf_cutoff', 4e5)
        self.filter_order = params.get('filter_order', 4)
        
        self._design_filters()
        
        print(f"[OK] Demodulator initialized:")
        print(f"    HPF cutoff: {self.hpf_cutoff/1e3:.0f} kHz")
        print(f"    LPF cutoff: {self.lpf_cutoff/1e6:.1f} MHz")
        print(f"    Filter order: {self.filter_order}")
    
    def _design_filters(self):
        """Design Butterworth filters with class-level caching."""
        nyquist = self.fs / 2
        
        hpf_key = (self.fs, self.hpf_cutoff, self.filter_order, 'high')
        if hpf_key in Demodulator._filter_cache:
            self.hpf_b, self.hpf_a = Demodulator._filter_cache[hpf_key]
        else:
            self.hpf_b, self.hpf_a = signal.butter(
                self.filter_order, self.hpf_cutoff / nyquist, btype='high')
            Demodulator._filter_cache[hpf_key] = (self.hpf_b, self.hpf_a)
        
        lpf_key = (self.fs, self.lpf_cutoff, self.filter_order, 'low')
        if lpf_key in Demodulator._filter_cache:
            self.lpf_b, self.lpf_a = Demodulator._filter_cache[lpf_key]
        else:
            self.lpf_b, self.lpf_a = signal.butter(
                self.filter_order, self.lpf_cutoff / nyquist, btype='low')
            Demodulator._filter_cache[lpf_key] = (self.lpf_b, self.lpf_a)
    
    def apply_hpf(self, signal_in):
        """Apply high-pass filter to remove DC bias."""
        return signal.filtfilt(self.hpf_b, self.hpf_a, signal_in)
    
    def apply_lpf(self, signal_in):
        """Apply low-pass filter to reduce noise."""
        return signal.filtfilt(self.lpf_b, self.lpf_a, signal_in)
    
    def sample_bits(self, signal_in, n_bits, sps):
        """Sample signal at bit centers."""
        indices = np.arange(n_bits) * sps + sps // 2
        indices = np.clip(indices.astype(int), 0, len(signal_in) - 1)
        return signal_in[indices]
    
    def make_decisions(self, samples, threshold='auto'):
        """Convert samples to bit decisions."""
        if threshold == 'auto':
            threshold_used = (np.max(samples) + np.min(samples)) / 2
        else:
            threshold_used = threshold
        bits_rx = (samples > threshold_used).astype(int)
        return bits_rx, threshold_used
    
    def decode_manchester(self, signal_in, n_bits, sps):
        """
        Decode Manchester-encoded signal by comparing half-bit energies.
        
        For each bit period, samples at 25% and 75% points:
            first_half > second_half -> bit 1
            first_half < second_half -> bit 0
        
        Args:
            signal_in (array): Analog signal
            n_bits (int): Number of data bits
            sps (int): Samples per original bit period
        Returns:
            bits_rx (ndarray): Decoded bits
        """
        bits_rx = np.zeros(n_bits, dtype=int)
        for i in range(n_bits):
            idx_first = min(i * sps + sps // 4, len(signal_in) - 1)
            idx_second = min(i * sps + 3 * sps // 4, len(signal_in) - 1)
            bits_rx[i] = 1 if signal_in[idx_first] > signal_in[idx_second] else 0
        return bits_rx
    
    def calculate_ber(self, bits_tx, bits_rx):
        """Calculate Bit Error Rate."""
        n = min(len(bits_tx), len(bits_rx))
        errors = int(np.sum(bits_tx[:n] != bits_rx[:n]))
        ber = errors / n
        return {'ber': ber, 'errors': errors, 'total_bits': n, 'accuracy': 1 - ber}
    
    def demodulate(self, V_pv, bits_tx, n_bits, sps, encoding='ook', verbose=False):
        """
        Complete demodulation pipeline.
        
        Args:
            V_pv (array): PV voltage signal
            bits_tx (array): Transmitted bits
            n_bits (int): Number of bits
            sps (int): Samples per symbol
            encoding (str): 'ook' or 'manchester'
            verbose (bool): Print debug info
        Returns:
            dict: Demodulation results
        """
        V_hpf = self.apply_hpf(V_pv)
        V_lpf = self.apply_lpf(V_hpf)
        
        if encoding == 'manchester':
            bits_rx = self.decode_manchester(V_lpf, n_bits, sps)
            threshold = None
        else:
            samples = self.sample_bits(V_lpf, n_bits, sps)
            bits_rx, threshold = self.make_decisions(samples)
        
        ber_stats = self.calculate_ber(bits_tx, bits_rx)
        
        if verbose:
            print(f"\nDemodulation Results ({encoding}):")
            print(f"  HPF range: {V_hpf.min()*1e3:.2f} to {V_hpf.max()*1e3:.2f} mV")
            print(f"  LPF range: {V_lpf.min()*1e3:.2f} to {V_lpf.max()*1e3:.2f} mV")
            if threshold is not None:
                print(f"  Threshold: {threshold*1e3:.2f} mV")
            print(f"  BER: {ber_stats['ber']:.6f} ({ber_stats['errors']}/{ber_stats['total_bits']})")
        
        result = {'V_hpf': V_hpf, 'V_lpf': V_lpf, 'bits_rx': bits_rx, 'ber_stats': ber_stats}
        if threshold is not None:
            result['threshold'] = threshold
        return result


# ========== TESTS ==========

def test_demodulator():
    """Unit test — OOK, Manchester, filter caching, BER functions."""
    
    print("\n" + "="*60)
    print("DEMODULATOR UNIT TEST")
    print("="*60)
    
    fs = 1e6; sps = 100; n_bits = 100
    bits_tx = np.random.randint(0, 2, n_bits)
    
    # Test 1: OOK
    print("\n[Test 1] OOK demodulation...")
    t = np.arange(n_bits * sps) / fs
    sig = np.repeat(bits_tx, sps).astype(float) * 0.1 + 0.05
    sig += np.random.normal(0, 0.005, len(sig))
    
    Demodulator._filter_cache.clear()
    # Use low HPF cutoff so OOK baseband survives filtering
    demod = Demodulator({'sample_rate': fs, 'hpf_cutoff': 100, 'lpf_cutoff': 4e5})
    result = demod.demodulate(sig, bits_tx, n_bits, sps, verbose=True)
    assert result['ber_stats']['ber'] <= 0.1
    print("  [OK]")
    
    # Test 2: Filter caching
    print("\n[Test 2] Filter caching...")
    sz1 = len(Demodulator._filter_cache)
    demod2 = Demodulator({'sample_rate': fs, 'hpf_cutoff': 100, 'lpf_cutoff': 4e5})  # Same as demod
    sz2 = len(Demodulator._filter_cache)
    assert sz2 == sz1, f"[ERROR] Cache grew: {sz1}->{sz2}"
    demod3 = Demodulator({'sample_rate': 2e6})  # Different params → new cache entries
    sz3 = len(Demodulator._filter_cache)
    assert sz3 > sz2
    print(f"  Cache: {sz1} (init) -> {sz2} (reuse) -> {sz3} (new params)  [OK]")
    
    # Test 3: Manchester
    print("\n[Test 3] Manchester demodulation...")
    manch = np.zeros(n_bits * sps)
    for i, bit in enumerate(bits_tx):
        s = i * sps; m = s + sps // 2
        manch[s:m] = 0.15 if bit == 1 else 0.05
        manch[m:s+sps] = 0.05 if bit == 1 else 0.15
    manch += np.random.normal(0, 0.01, len(manch))
    bits_m = demod.decode_manchester(manch, n_bits, sps)
    ber_m = np.sum(bits_m != bits_tx) / n_bits
    print(f"  Manchester BER: {ber_m:.4f}")
    assert ber_m < 0.05
    print("  [OK]")
    
    # Test 4: BER functions
    print("\n[Test 4] BER prediction functions...")
    ber10 = predict_ber_ook_db(10)
    assert 1e-4 < ber10 < 1e-2
    print(f"  OOK@10dB: {ber10:.2e}")
    assert predict_ber_bpsk(10) < predict_ber_ook(10)
    print(f"  BPSK < OOK at same SNR: confirmed")
    ber_16 = predict_ber_mqam(100, 16)
    print(f"  16-QAM@20dB: {ber_16:.2e}")
    print("  [OK]")
    
    print("\n" + "="*60)
    print("[OK] ALL DEMODULATOR TESTS PASSED!")
    print("="*60)
    return result

if __name__ == "__main__":
    test_demodulator()
