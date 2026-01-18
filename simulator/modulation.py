"""
Modulation schemes for Li-Fi / PV Simulator.

Implements:
- PWMASKModulator: PWM-ASK hybrid modulation (Correa 2025)
- AdaptiveOFDMModulator: OFDM with adaptive bit loading (Xu 2024)

Author: Phase 3 Implementation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union


# =============================================================================
# PWM-ASK MODULATOR (CORREA 2025 - CONFLICT 2.1)
# =============================================================================

class PWMASKModulator:
    """
    PWM-ASK Hybrid Modulator for greenhouse VLC (Correa 2025).
    
    Data is encoded in the duty cycle of a low-frequency PWM signal,
    carried by a higher-frequency ASK signal.
    
    Parameters:
        pwm_freq_hz: Base PWM frequency (default: 10 Hz)
        carrier_freq_hz: ASK carrier frequency (default: 10 kHz)  
        duty_cycles: Dict mapping bits to duty cycles
        sample_rate_hz: Sample rate for signal generation
    """
    
    def __init__(
        self,
        pwm_freq_hz: float = 10.0,
        carrier_freq_hz: float = 10000.0,
        duty_cycles: Optional[Dict[int, float]] = None,
        sample_rate_hz: float = 100000.0
    ):
        self.pwm_freq_hz = float(pwm_freq_hz)
        self.carrier_freq_hz = float(carrier_freq_hz)
        self.sample_rate_hz = float(sample_rate_hz)
        
        # Default duty cycles: 0 → 25%, 1 → 75%
        if duty_cycles is None:
            self.duty_cycles = {0: 0.25, 1: 0.75}
        else:
            self.duty_cycles = duty_cycles
        
        # Derived parameters
        self.pwm_period_s = 1.0 / self.pwm_freq_hz
        self.carrier_period_s = 1.0 / self.carrier_freq_hz
        self.samples_per_pwm_period = int(self.sample_rate_hz * self.pwm_period_s)
        
        print(f"[PWM-ASK] Initialized:")
        print(f"    PWM freq: {self.pwm_freq_hz} Hz (period: {self.pwm_period_s*1000:.1f} ms)")
        print(f"    Carrier freq: {self.carrier_freq_hz} Hz")
        print(f"    Duty cycles: 0→{self.duty_cycles[0]*100:.0f}%, 1→{self.duty_cycles[1]*100:.0f}%")
    
    def modulate(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate PWM-ASK modulated signal.
        
        Args:
            bits: Binary data array (0s and 1s)
            
        Returns:
            t: Time array (seconds)
            signal: Modulated signal array
        """
        bits = np.asarray(bits).flatten()
        n_bits = len(bits)
        
        # Total duration
        total_duration_s = n_bits * self.pwm_period_s
        n_samples = int(self.sample_rate_hz * total_duration_s)
        
        # Time array
        t = np.linspace(0, total_duration_s, n_samples, endpoint=False)
        
        # Initialize output
        signal = np.zeros(n_samples)
        
        # Generate signal for each bit
        for i, bit in enumerate(bits):
            duty = self.duty_cycles.get(int(bit), 0.5)
            
            # Time indices for this bit's period
            start_idx = int(i * self.samples_per_pwm_period)
            end_idx = int((i + 1) * self.samples_per_pwm_period)
            
            # Ensure we don't exceed array bounds
            end_idx = min(end_idx, n_samples)
            
            # Generate PWM envelope (square wave with duty cycle)
            bit_t = t[start_idx:end_idx] - t[start_idx]
            pwm_envelope = np.where(bit_t < duty * self.pwm_period_s, 1.0, 0.0)
            
            # Generate ASK carrier (square wave at carrier frequency)
            carrier = np.where(
                np.sin(2 * np.pi * self.carrier_freq_hz * bit_t) > 0,
                1.0, 0.0
            )
            
            # Apply carrier to envelope (OOK-style chopping)
            signal[start_idx:end_idx] = pwm_envelope * carrier
        
        return t, signal
    
    def get_bit_duration_ms(self) -> float:
        """Return duration of one bit in milliseconds."""
        return self.pwm_period_s * 1000


class PWMASKDemodulator:
    """
    PWM-ASK Demodulator using envelope detection.
    
    Process:
    1. Rectify the signal (absolute value)
    2. Low-pass filter to extract envelope
    3. Measure high-duration per period
    4. Threshold at 50% to decide bit
    """
    
    def __init__(
        self,
        pwm_freq_hz: float = 10.0,
        sample_rate_hz: float = 100000.0,
        threshold: float = 0.5
    ):
        self.pwm_freq_hz = float(pwm_freq_hz)
        self.sample_rate_hz = float(sample_rate_hz)
        self.threshold = threshold
        
        self.pwm_period_s = 1.0 / self.pwm_freq_hz
        self.samples_per_period = int(self.sample_rate_hz * self.pwm_period_s)
    
    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """
        Demodulate PWM-ASK signal to recover bits.
        
        Args:
            signal: Received signal array
            
        Returns:
            bits: Recovered binary data
        """
        from scipy import signal as sig
        
        signal = np.asarray(signal).flatten()
        
        # 1. Rectify (already positive for OOK, but ensure)
        rectified = np.abs(signal)
        
        # 2. Low-pass filter to extract envelope
        # Cutoff below carrier, above PWM
        # E.g., 500 Hz for 10 Hz PWM and 10 kHz carrier
        cutoff_hz = 500.0
        nyq = self.sample_rate_hz / 2
        
        if cutoff_hz < nyq:
            b, a = sig.butter(2, cutoff_hz / nyq, btype='low')
            envelope = sig.filtfilt(b, a, rectified)
        else:
            envelope = rectified
        
        # 3. Segment into bit periods and measure duty cycle
        n_bits = len(signal) // self.samples_per_period
        bits = []
        
        for i in range(n_bits):
            start_idx = i * self.samples_per_period
            end_idx = (i + 1) * self.samples_per_period
            
            segment = envelope[start_idx:end_idx]
            
            # Estimate duty cycle from envelope
            # Threshold at mid-level
            segment_max = np.max(segment)
            segment_threshold = segment_max * 0.5
            
            high_samples = np.sum(segment > segment_threshold)
            duty_estimate = high_samples / len(segment)
            
            # Decide bit based on duty cycle threshold
            bit = 1 if duty_estimate > self.threshold else 0
            bits.append(bit)
        
        return np.array(bits)


# =============================================================================
# ADAPTIVE OFDM MODULATOR (XU 2024 - CONFLICT 3.2)
# =============================================================================

class AdaptiveOFDMModulator:
    """
    OFDM with Adaptive Bit Loading (Xu 2024).
    
    Maximizes throughput by adapting constellation size per subcarrier
    based on the estimated SNR.
    
    Parameters:
        nfft: FFT size (default: 64)
        cp_length: Cyclic prefix length (default: 16)
        num_data_carriers: Number of data subcarriers (default: 52)
    """
    
    # Bit allocation thresholds (SNR in dB)
    BIT_ALLOCATION_TABLE = [
        (16.0, 6),   # SNR >= 16 dB → 64-QAM (6 bits)
        (10.0, 4),   # SNR >= 10 dB → 16-QAM (4 bits)
        (6.0, 2),    # SNR >= 6 dB  → QPSK (2 bits)
        (3.0, 1),    # SNR >= 3 dB  → BPSK (1 bit)
        (-np.inf, 0) # SNR < 3 dB   → Inactive (0 bits)
    ]
    
    def __init__(
        self,
        nfft: int = 64,
        cp_length: int = 16,
        num_data_carriers: Optional[int] = None,
        clipping_ratio_db: Optional[float] = None
    ):
        self.nfft = int(nfft)
        self.cp_length = int(cp_length)
        self.clipping_ratio_db = clipping_ratio_db
        
        # Default: Use ~52 data carriers for 64-point FFT (like WiFi)
        if num_data_carriers is None:
            self.num_data_carriers = self.nfft - 12  # Exclude DC, pilots, guards
        else:
            self.num_data_carriers = int(num_data_carriers)
        
        # Data carrier indices (symmetric around DC)
        # Example for 64-pt: -26 to -1, +1 to +26 (skip DC at 0)
        self.data_carrier_indices = self._get_data_carrier_indices()
        
        # Current bit allocation (updated by allocate_bits)
        self.bit_allocation = np.zeros(self.num_data_carriers, dtype=int)
        self.total_bits_per_symbol = 0
        
        print(f"[Adaptive OFDM] Initialized:")
        print(f"    NFFT: {self.nfft}, CP: {self.cp_length}")
        print(f"    Data carriers: {self.num_data_carriers}")
    
    def _get_data_carrier_indices(self) -> np.ndarray:
        """Get indices of data subcarriers in FFT output."""
        # Simple allocation: skip DC (index 0) and some guards
        half = self.num_data_carriers // 2
        # Negative frequencies (upper half of FFT)
        neg_indices = np.arange(self.nfft - half, self.nfft)
        # Positive frequencies (lower half of FFT, skip DC)
        pos_indices = np.arange(1, half + 1)
        return np.concatenate([neg_indices, pos_indices])
    
    def allocate_bits(self, snr_db_per_subcarrier: np.ndarray) -> np.ndarray:
        """
        Allocate bits to each subcarrier based on SNR.
        
        Args:
            snr_db_per_subcarrier: SNR values in dB for each data subcarrier
            
        Returns:
            bit_allocation: Number of bits per subcarrier
        """
        snr = np.asarray(snr_db_per_subcarrier).flatten()
        
        if len(snr) != self.num_data_carriers:
            raise ValueError(
                f"SNR array length ({len(snr)}) must match "
                f"num_data_carriers ({self.num_data_carriers})"
            )
        
        # Allocate bits based on thresholds
        allocation = np.zeros(len(snr), dtype=int)
        
        for i, snr_val in enumerate(snr):
            for threshold, bits in self.BIT_ALLOCATION_TABLE:
                if snr_val >= threshold:
                    allocation[i] = bits
                    break
        
        self.bit_allocation = allocation
        self.total_bits_per_symbol = int(np.sum(allocation))
        
        return allocation
    
    def get_throughput_bits_per_symbol(self) -> int:
        """Return total bits that can be transmitted per OFDM symbol."""
        return self.total_bits_per_symbol
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Modulate bits using adaptive OFDM.
        
        Args:
            bits: Input bit stream
            
        Returns:
            signal: Time-domain OFDM signal with CP
        """
        bits = np.asarray(bits).flatten()
        
        if self.total_bits_per_symbol == 0:
            raise ValueError("No bits allocated. Call allocate_bits() first.")
        
        # Number of complete OFDM symbols
        n_symbols = len(bits) // self.total_bits_per_symbol
        
        if n_symbols == 0:
            raise ValueError(
                f"Need at least {self.total_bits_per_symbol} bits, got {len(bits)}"
            )
        
        output_samples = []
        bit_idx = 0
        
        for sym_idx in range(n_symbols):
            # Create frequency-domain symbol
            freq_symbol = np.zeros(self.nfft, dtype=complex)
            
            for sc_idx, (carrier_idx, n_bits) in enumerate(
                zip(self.data_carrier_indices, self.bit_allocation)
            ):
                if n_bits == 0:
                    continue
                
                # Extract bits for this subcarrier
                sc_bits = bits[bit_idx:bit_idx + n_bits]
                bit_idx += n_bits
                
                # Map to constellation
                symbol = self._bits_to_symbol(sc_bits, n_bits)
                freq_symbol[carrier_idx] = symbol
            
            # IFFT to time domain
            time_symbol = np.fft.ifft(freq_symbol)
            
            # Add cyclic prefix
            cp = time_symbol[-self.cp_length:]
            ofdm_symbol = np.concatenate([cp, time_symbol])
            
            output_samples.extend(ofdm_symbol)
        
            output_samples.extend(ofdm_symbol)
        
        # Convert to array
        signal = np.array(output_samples)
        
        # Apply Clipping if configured
        if self.clipping_ratio_db is not None:
            # P_avg = mean(abs(x)^2)
            # PAPR reduction via clipping
            
            p_avg = np.mean(np.abs(signal)**2)
            # CR = Peak / Average (PAPR target) or Peak / RMS?
            # Usually Clipping Ratio (CR) is defined as A_max / sigma
            # where sigma = sqrt(P_avg)
            
            sigma = np.sqrt(p_avg)
            cr_linear = 10 ** (self.clipping_ratio_db / 20.0)
            threshold = sigma * cr_linear
            
            # Clip complex signal magnitude while preserving phase
            # x_clipped = x if |x| < A else A * x/|x|
            
            # Vectorized clipping
            # Avoid division by zero
            magnitudes = np.abs(signal)
            scale_factors = np.where(magnitudes > threshold, threshold / magnitudes, 1.0)
            signal = signal * scale_factors
            
        return signal
    
    def _bits_to_symbol(self, bits: np.ndarray, n_bits: int) -> complex:
        """Map bits to QAM symbol."""
        if n_bits == 0:
            return 0j
        elif n_bits == 1:
            # BPSK: 0 → -1, 1 → +1
            return 1.0 if bits[0] else -1.0
        elif n_bits == 2:
            # QPSK
            val = bits[0] * 2 + bits[1]
            qpsk_map = {0: -1-1j, 1: -1+1j, 2: 1-1j, 3: 1+1j}
            return qpsk_map[val] / np.sqrt(2)
        elif n_bits == 4:
            # 16-QAM
            val = sum(b << (3-i) for i, b in enumerate(bits[:4]))
            # Gray-coded 16-QAM
            re = (val >> 2) * 2 - 3  # -3, -1, +1, +3
            im = (val & 3) * 2 - 3
            return (re + 1j * im) / np.sqrt(10)
        elif n_bits == 6:
            # 64-QAM
            val = sum(b << (5-i) for i, b in enumerate(bits[:6]))
            re = (val >> 3) * 2 - 7  # -7, -5, ..., +5, +7
            im = (val & 7) * 2 - 7
            return (re + 1j * im) / np.sqrt(42)
        else:
            raise ValueError(f"Unsupported bit count: {n_bits}")
    
    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """
        Demodulate OFDM signal back to bits.
        
        Args:
            signal: Time-domain OFDM signal
            
        Returns:
            bits: Recovered bit stream
        """
        signal = np.asarray(signal).flatten()
        
        # Symbol length including CP
        symbol_length = self.nfft + self.cp_length
        n_symbols = len(signal) // symbol_length
        
        recovered_bits = []
        
        for sym_idx in range(n_symbols):
            start = sym_idx * symbol_length
            
            # Remove CP and take FFT
            time_symbol = signal[start + self.cp_length : start + symbol_length]
            freq_symbol = np.fft.fft(time_symbol)
            
            # Extract bits from each subcarrier
            for sc_idx, (carrier_idx, n_bits) in enumerate(
                zip(self.data_carrier_indices, self.bit_allocation)
            ):
                if n_bits == 0:
                    continue
                
                symbol = freq_symbol[carrier_idx]
                bits = self._symbol_to_bits(symbol, n_bits)
                recovered_bits.extend(bits)
        
        return np.array(recovered_bits)
    
    def _symbol_to_bits(self, symbol: complex, n_bits: int) -> List[int]:
        """Demodulate QAM symbol to bits (hard decision)."""
        if n_bits == 0:
            return []
        elif n_bits == 1:
            # BPSK
            return [1 if np.real(symbol) > 0 else 0]
        elif n_bits == 2:
            # QPSK
            b0 = 1 if np.real(symbol) > 0 else 0
            b1 = 1 if np.imag(symbol) > 0 else 0
            return [b0, b1]
        elif n_bits == 4:
            # 16-QAM
            re_scaled = np.real(symbol) * np.sqrt(10)
            im_scaled = np.imag(symbol) * np.sqrt(10)
            re_idx = int(np.clip(np.round((re_scaled + 3) / 2), 0, 3))
            im_idx = int(np.clip(np.round((im_scaled + 3) / 2), 0, 3))
            val = (re_idx << 2) | im_idx
            return [(val >> (3-i)) & 1 for i in range(4)]
        elif n_bits == 6:
            # 64-QAM
            re_scaled = np.real(symbol) * np.sqrt(42)
            im_scaled = np.imag(symbol) * np.sqrt(42)
            re_idx = int(np.clip(np.round((re_scaled + 7) / 2), 0, 7))
            im_idx = int(np.clip(np.round((im_scaled + 7) / 2), 0, 7))
            val = (re_idx << 3) | im_idx
            return [(val >> (5-i)) & 1 for i in range(6)]
        else:
            return [0] * n_bits


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_bit_loading(snr_db: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """
    Calculate bit loading based on SNR using standard thresholds.
    
    Args:
        snr_db: SNR value(s) in dB
        
    Returns:
        Number of bits per symbol
    """
    snr = np.asarray(snr_db)
    bits = np.zeros_like(snr, dtype=int)
    
    bits = np.where(snr >= 16.0, 6, bits)  # 64-QAM
    bits = np.where((snr >= 10.0) & (snr < 16.0), 4, bits)  # 16-QAM
    bits = np.where((snr >= 6.0) & (snr < 10.0), 2, bits)   # QPSK
    bits = np.where((snr >= 3.0) & (snr < 6.0), 1, bits)    # BPSK
    # snr < 3.0 → 0 bits (inactive)
    
    return int(bits) if np.isscalar(snr_db) else bits


# =============================================================================
# BFSK MODULATOR (XU 2024 - SUNLIGHT-DUO)
# =============================================================================

class BFSKModulator:
    """
    Binary Frequency Shift Keying (BFSK) Modulator for Sunlight-Duo (Xu 2024).
    
    Uses two frequencies to encode binary data:
    - Logic '0': 1600 Hz
    - Logic '1': 2000 Hz
    
    Parameters:
        freq_0: Frequency for bit '0' (default: 1600 Hz)
        freq_1: Frequency for bit '1' (default: 2000 Hz)
        sample_rate: Sample rate in Hz (default: 16000 Hz)
        bit_duration: Duration of each bit in seconds (default: 1/400 for 400 bps)
    """
    
    def __init__(
        self,
        freq_0: float = 1600.0,
        freq_1: float = 2000.0,
        sample_rate: float = 16000.0,
        bit_rate: float = 400.0
    ):
        self.freq_0 = freq_0
        self.freq_1 = freq_1
        self.sample_rate = sample_rate
        self.bit_rate = bit_rate
        self.bit_duration = 1.0 / bit_rate
        self.samples_per_bit = int(sample_rate / bit_rate)
        
        print(f"[BFSK] Initialized:")
        print(f"    Freq '0': {self.freq_0} Hz")
        print(f"    Freq '1': {self.freq_1} Hz")
        print(f"    Bit rate: {self.bit_rate} bps")
        print(f"    Samples/bit: {self.samples_per_bit}")
    
    def modulate(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modulate binary data using BFSK.
        
        Args:
            bits: Array of binary values (0 or 1)
            
        Returns:
            t: Time vector
            signal: BFSK modulated signal
        """
        bits = np.asarray(bits)
        n_bits = len(bits)
        n_samples = n_bits * self.samples_per_bit
        
        t = np.arange(n_samples) / self.sample_rate
        signal = np.zeros(n_samples)
        
        for i, bit in enumerate(bits):
            start_idx = i * self.samples_per_bit
            end_idx = start_idx + self.samples_per_bit
            bit_t = t[start_idx:end_idx] - t[start_idx]
            
            freq = self.freq_1 if bit == 1 else self.freq_0
            signal[start_idx:end_idx] = np.sin(2 * np.pi * freq * bit_t)
        
        return t, signal
    
    def demodulate(self, signal: np.ndarray, n_bits: int) -> np.ndarray:
        """
        Demodulate BFSK signal using correlation.
        
        Args:
            signal: BFSK modulated signal
            n_bits: Expected number of bits
            
        Returns:
            bits: Recovered binary values
        """
        bits = np.zeros(n_bits, dtype=int)
        
        for i in range(n_bits):
            start_idx = i * self.samples_per_bit
            end_idx = min(start_idx + self.samples_per_bit, len(signal))
            
            if end_idx <= start_idx:
                break
            
            segment = signal[start_idx:end_idx]
            t_local = np.arange(len(segment)) / self.sample_rate
            
            # Correlate with both frequencies
            ref_0 = np.sin(2 * np.pi * self.freq_0 * t_local)
            ref_1 = np.sin(2 * np.pi * self.freq_1 * t_local)
            
            corr_0 = np.abs(np.sum(segment * ref_0))
            corr_1 = np.abs(np.sum(segment * ref_1))
            
            bits[i] = 1 if corr_1 > corr_0 else 0
        
        return bits


class BFSKDemodulator:
    """
    BFSK Demodulator using envelope detection.
    """
    
    def __init__(self, freq_0: float = 1600.0, freq_1: float = 2000.0,
                 sample_rate: float = 16000.0, bit_rate: float = 400.0):
        self.freq_0 = freq_0
        self.freq_1 = freq_1
        self.sample_rate = sample_rate
        self.bit_rate = bit_rate
        self.samples_per_bit = int(sample_rate / bit_rate)
    
    def demodulate(self, signal: np.ndarray, n_bits: int) -> np.ndarray:
        """Demodulate BFSK signal."""
        return BFSKModulator(
            self.freq_0, self.freq_1, self.sample_rate, self.bit_rate
        ).demodulate(signal, n_bits)


if __name__ == "__main__":
    # Quick test
    print("\n=== PWM-ASK Test ===")
    pwm_mod = PWMASKModulator()
    bits = np.array([0, 1, 0, 1, 1, 0])
    t, signal = pwm_mod.modulate(bits)
    print(f"Generated {len(signal)} samples for {len(bits)} bits")
    print(f"Signal duration: {t[-1]*1000:.1f} ms")
    
    print("\n=== Adaptive OFDM Test ===")
    ofdm = AdaptiveOFDMModulator(nfft=64, cp_length=16)
    snr_profile = np.array([-5, 4, 8, 12, 20] * 10 + [10] * 2)  # 52 subcarriers
    allocation = ofdm.allocate_bits(snr_profile)
    print(f"SNR profile (first 5): {snr_profile[:5]}")
    print(f"Bit allocation (first 5): {allocation[:5]}")
    print(f"Total bits per symbol: {ofdm.get_throughput_bits_per_symbol()}")
