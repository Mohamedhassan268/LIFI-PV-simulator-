# simulator/demodulator.py - CORRECTED VERSION

"""
Signal Processing & Demodulation for PV Receiver.

Components:
1. High-pass filter (remove DC bias)
2. Low-pass filter (reduce noise)
3. Matched filter / sampling
4. Decision circuit
5. BER calculation
6. Physics-based BER prediction (Eb/N0 → BER)

CHANGES FROM ORIGINAL:
- Fixed inverted decision logic (< changed to >)
- Updated default filter cutoffs
- Reduced filter order from 4 to 2
- Added predict_ber_ook for analytical BER computation
"""

import numpy as np
from scipy import signal
from scipy.special import erfc


# ========== PHYSICS-BASED BER PREDICTION ==========

def predict_ber_ook(eb_n0_linear):
    """
    Predict BER from Eb/N0 for binary OOK/PWM-ASK over AWGN.
    
    Standard formula for coherent OOK with optimal threshold:
        BER = 0.5 × erfc(√(Eb/N0 / 2))
    
    This is equivalent to BER = Q(√(Eb/N0)) where Q(x) = 0.5×erfc(x/√2).
    
    Args:
        eb_n0_linear: Eb/N0 in linear scale (not dB)
        
    Returns:
        BER: Bit Error Rate (0-1)
    """
    # Ensure non-negative
    eb_n0_linear = np.maximum(eb_n0_linear, 1e-10)
    return 0.5 * erfc(np.sqrt(eb_n0_linear / 2))


def predict_ber_ook_db(eb_n0_db):
    """
    Predict BER from Eb/N0 in dB for binary OOK.
    
    Args:
        eb_n0_db: Eb/N0 in dB
        
    Returns:
        BER: Bit Error Rate (0-1)
    """
    eb_n0_linear = 10 ** (eb_n0_db / 10)
    return predict_ber_ook(eb_n0_linear)


def compute_eb_n0(P_rx, noise_psd, bit_rate, responsivity=0.5):
    """
    Compute Eb/N0 from system parameters.
    
    For optical OOK:
        Eb = (R × P_rx)² / bit_rate   (Energy per bit from photocurrent)
        N0 = noise_psd                 (Noise power spectral density in A²/Hz)
    
    Args:
        P_rx: Received optical power (W)
        noise_psd: Noise power spectral density (A²/Hz)
        bit_rate: Data rate (bps)
        responsivity: PV responsivity (A/W), default 0.5 for poly-Si
        
    Returns:
        Eb/N0 in linear scale
    """
    # Signal current
    I_signal = responsivity * P_rx
    
    # Energy per bit (signal power / bit rate)
    # For OOK, average power is I²/2 (half on, half off)
    signal_power = I_signal ** 2
    Eb = signal_power / bit_rate
    
    # Noise spectral density
    N0 = noise_psd
    
    # Avoid division by zero
    if N0 == 0:
        return 1e10  # Very high SNR
    
    return Eb / N0


def snr_to_ber_lookup(snr_db_array):
    """
    Generate BER vs SNR lookup table for plotting.
    
    Args:
        snr_db_array: Array of SNR values in dB
        
    Returns:
        ber_array: Corresponding BER values
    """
    return predict_ber_ook_db(snr_db_array)


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
    
    def calculate_ber(self, bits_tx, bits_rx, strict=False, mismatch_tolerance=0.1):
        """
        Calculate Bit Error Rate with enhanced mismatch handling.
        
        Phase 7 Enhancement:
        - strict mode: Raises exception if array lengths differ significantly
        - mismatch_tolerance: Max allowed length difference as fraction (default 10%)
        - sync_loss_rate: Reports missing/extra bits as separate metric
        
        Args:
            bits_tx (array): Transmitted bits
            bits_rx (array): Received bits
            strict (bool): If True, raise exception on length mismatch > tolerance
            mismatch_tolerance (float): Max allowed length difference (0-1)
        
        Returns:
            dict: BER statistics including:
                - ber: Bit error rate (errors / compared_bits)
                - errors: Number of bit errors
                - total_bits: Number of bits compared
                - accuracy: 1 - ber
                - sync_loss_rate: Fraction of bits lost due to length mismatch
                - length_mismatch: True if arrays were different lengths
        """
        len_tx = len(bits_tx)
        len_rx = len(bits_rx)
        
        # Calculate mismatch
        length_diff = abs(len_tx - len_rx)
        max_len = max(len_tx, len_rx)
        mismatch_fraction = length_diff / max_len if max_len > 0 else 0
        
        # Strict mode: raise exception if mismatch exceeds tolerance
        if strict and mismatch_fraction > mismatch_tolerance:
            raise ValueError(
                f"BER length mismatch too large: TX={len_tx}, RX={len_rx} "
                f"({mismatch_fraction*100:.1f}% > {mismatch_tolerance*100:.0f}% tolerance). "
                f"Check synchronization or use strict=False."
            )
        
        # Warn on any mismatch
        if length_diff > 0:
            print(f"WARNING: BER array length mismatch: TX={len_tx}, RX={len_rx} "
                  f"(using min={min(len_tx, len_rx)} bits)")
        
        # Use minimum length for comparison
        n = min(len_tx, len_rx)
        
        if n == 0:
            return {
                'ber': 1.0,
                'errors': 0,
                'total_bits': 0,
                'accuracy': 0.0,
                'sync_loss_rate': 1.0,
                'length_mismatch': True,
                'bits_tx_len': len_tx,
                'bits_rx_len': len_rx,
            }
        
        bits_tx_trunc = bits_tx[:n]
        bits_rx_trunc = bits_rx[:n]
        
        # Count errors
        errors = np.sum(bits_tx_trunc != bits_rx_trunc)
        
        # Calculate BER
        ber = errors / n
        
        # Sync loss rate: fraction of bits that couldn't be compared
        sync_loss_rate = length_diff / max_len if max_len > 0 else 0
        
        return {
            'ber': ber,
            'errors': int(errors),
            'total_bits': n,
            'accuracy': 1 - ber,
            'sync_loss_rate': sync_loss_rate,
            'length_mismatch': length_diff > 0,
            'bits_tx_len': len_tx,
            'bits_rx_len': len_rx,
        }
    
    def decode_manchester(self, signal_in, n_bits, sps):
        """
        Decode Manchester-encoded signal.
        
        Manchester encoding:
        - bit '1' -> high-to-low transition (first half high, second half low)
        - bit '0' -> low-to-high transition (first half low, second half high)
        
        Decoding strategy:
        - Sample first half and second half of each bit period
        - If first > second: bit = 1
        - If first < second: bit = 0
        
        Args:
            signal_in (array): Filtered signal
            n_bits (int): Number of bits to decode
            sps (int): Samples per bit period
            
        Returns:
            bits_rx (array): Recovered bits
        """
        bits_rx = np.zeros(n_bits, dtype=int)
        
        # Sample points: quarter and three-quarter of each bit period
        samples_per_half = sps // 2
        
        for i in range(n_bits):
            # Start of bit period
            bit_start = i * sps
            
            # Sample first half (at 1/4 of bit period)
            idx_first = bit_start + samples_per_half // 2
            # Sample second half (at 3/4 of bit period)
            idx_second = bit_start + samples_per_half + samples_per_half // 2
            
            # Ensure indices are valid
            idx_first = min(idx_first, len(signal_in) - 1)
            idx_second = min(idx_second, len(signal_in) - 1)
            
            # Compare first and second half
            if signal_in[idx_first] > signal_in[idx_second]:
                bits_rx[i] = 1  # high-to-low = '1'
            else:
                bits_rx[i] = 0  # low-to-high = '0'
        
        return bits_rx
    
    def decode_manchester_differential(self, signal_in, n_bits, sps):
        """
        Decode Manchester using differential detection (more robust).
        
        Looks at the slope/transition at bit center.
        
        Args:
            signal_in (array): Filtered signal  
            n_bits (int): Number of bits to decode
            sps (int): Samples per bit period
            
        Returns:
            bits_rx (array): Recovered bits
        """
        bits_rx = np.zeros(n_bits, dtype=int)
        
        for i in range(n_bits):
            # Bit center
            bit_center = i * sps + sps // 2
            
            # Look at slope around center
            window = max(1, sps // 8)
            idx_before = max(0, bit_center - window)
            idx_after = min(len(signal_in) - 1, bit_center + window)
            
            slope = signal_in[idx_after] - signal_in[idx_before]
            
            # Negative slope (falling edge) = '1'
            # Positive slope (rising edge) = '0'
            if slope < 0:
                bits_rx[i] = 1
            else:
                bits_rx[i] = 0
        
        return bits_rx
    
    def _qam_demap(self, symbols, qam_order):
        """
        Convert complex QAM symbols to bits.
        
        Args:
            symbols (array): Complex symbols
            qam_order (int): 4, 16, etc.
            
        Returns:
            bits (array): Decoded bits
        """
        bits = []
        
        if qam_order == 16:
            # Simple demapping for the binary map used in Tx
            # 16-QAM Levels: -3, -1, 1, 3
            # Thresholds: -2, 0, 2
            
            # Normalize power back (x sqrt(10))
            norm_factor = np.sqrt(10)
            sym_scaled = symbols * norm_factor
            
            I = sym_scaled.real
            Q = sym_scaled.imag
            
            # Helper to decode 4 levels back to 2 bits
            # Map used in Tx:
            # 00 -> -3 ( < -2 )
            # 01 -> -1 ( -2 < x < 0 )
            # 11 ->  1 ( 0 < x < 2 )
            # 10 ->  3 ( > 2 )
            
            def decode_levels(vals):
                b = np.zeros((len(vals), 2), dtype=int)
                # Bit 0 (Sign bit)
                # >0 -> 1, <0 -> 0
                b[vals > 0, 0] = 1
                
                # Bit 1 (Magnitude bit)
                # Inner levels (-1, 1) -> 1, Outer levels (-3, 3) -> 0
                abs_vals = np.abs(vals)
                b[(abs_vals < 2), 1] = 1
                return b
            
            b_I = decode_levels(I)
            b_Q = decode_levels(Q)
            
            # Interleave I and Q bits: [bI0, bI1, bQ0, bQ1]
            for i in range(len(I)):
                bits.extend([b_I[i,0], b_I[i,1], b_Q[i,0], b_Q[i,1]])
                
        elif qam_order == 4:
            # QPSK
            # >0 -> 1, <0 -> 0
            # sqrt(2) normalization
            sym_scaled = symbols * np.sqrt(2)
            bits = list(((sym_scaled.real > 0).astype(int) * 2 + (sym_scaled.imag > 0).astype(int)))
            # This is packed 0-3 int, need to unpack to bits... this is lazy.
            # Proper bit expansion:
            for s in sym_scaled:
                bits.append(1 if s.real > 0 else 0)
                bits.append(1 if s.imag > 0 else 0)
        
        return np.array(bits)

    def demodulate_mimo(self, P_rx_mimo, H_matrix, method='zf'):
        """
        Demodulate MIMO signal (Spatial Demultiplexing).
        
        Args:
            P_rx_mimo (ndarray): Received signal matrix (L_rx, N_samples)
            H_matrix (ndarray): Channel matrix (L_rx, M_tx)
            method (str): Equalizer method ('zf' for Zero-Forcing)
            
        Returns:
            P_est (ndarray): Estimated transmitted signals (M_tx, N_samples)
        """
        if method == 'zf':
            # Zero-Forcing: X_est = inv(H) * Y
            # For non-square H, use Pseudo-inverse: pinv(H) * Y
            try:
                H_inv = np.linalg.pinv(H_matrix)
                P_est = np.dot(H_inv, P_rx_mimo)
                return P_est
            except np.linalg.LinAlgError:
                print("Error: H matrix singular or invalid.")
                return np.zeros((H_matrix.shape[1], P_rx_mimo.shape[1]))
        else:
            raise ValueError(f"Unknown MIMO method: {method}")

    def demodulate_ofdm(self, signal_in, n_fft=64, cp_len=16, qam_order=16, verbose=False):
        """
        Demodulate DCO-OFDM signal.
        
        Process:
        1. Resample signal to symbol rate (downsample)
        2. Remove Cyclic Prefix
        3. FFT -> Frequency Domain
        4. Extract data subcarriers
        5. Channel Equalization (Zero Forcing)
        6. QAM Demapping
        
        Args:
            signal_in (array): Received time-domain signal
            n_fft (int): FFT subcarriers
            cp_len (int): Cyclic prefix length
            
        Returns:
            bits_rx (array): Recovered bits
        """
        # 1. Resample / Sync
        # We assume perfect synchronization for now.
        # We need to extract exactly the OFDM frames.
        
        # Calculate number of samples per frame
        n_frame_samples = n_fft + cp_len
        
        # Determine number of full frames
        n_frames = len(signal_in) // n_frame_samples
        
        # Truncate to whole frames
        signal_trunc = signal_in[:n_frames * n_frame_samples]
        
        # Reshape
        frames_time = signal_trunc.reshape(n_frames, n_frame_samples)
        
        # 2. Remove CP
        # Keep only the last n_fft samples of each frame
        frames_payload = frames_time[:, cp_len:]
        
        all_symbols = []
        
        # 3. FFT
        for frame_t in frames_payload:
            frame_f = np.fft.fft(frame_t)
            
            # 4. Extract Data Subcarriers
            # Indices: 1 to N/2 - 1
            data_subcarriers = frame_f[1 : n_fft//2]
            
            # 5. Channel Equalization (Placeholder)
            # In a real system, we'd use pilots. 
            # Here, we assume "blind" equalization or simple normalization.
            # DCO-OFDM suffers from attenuation at high frequencies.
            # Simple approach: automatic gain control per subcarrier? 
            # No, that destroys QAM info.
            # For now: Pass raw symbols. The constellation will look "shrunk" and rotated.
            # Ideally we pass H_est here.
            
            all_symbols.extend(data_subcarriers)
            
        all_symbols = np.array(all_symbols)
        
        # Quick Blind Equalization (Scalar): Center the constellation
        # Normalize power to match ideal QAM power (1.0)
        p_avg = np.mean(np.abs(all_symbols)**2)
        if p_avg > 0:
            all_symbols /= np.sqrt(p_avg)
        
        # 6. Demap
        bits_rx = self._qam_demap(all_symbols, qam_order)
        
        return bits_rx, all_symbols

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
