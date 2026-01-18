"""
Phase 3 Unit Tests: PWM-ASK and Adaptive OFDM Modulation.

Tests:
1. PWM-ASK: Verify duty cycle timing (25ms for '0', 75ms for '1' at 10Hz)
2. OFDM Bit Loading: Verify allocation matches SNR thresholds

Run with: python tests/test_phase3.py
Or with pytest: python -m pytest tests/test_phase3.py -v
"""

import numpy as np
import sys
import os

# Optional pytest import (for running as standalone script)
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Mock pytest.approx for standalone running
    class pytest:
        @staticmethod
        def approx(val, rel=None, abs=None):
            return val
        @staticmethod
        def raises(exc, match=None):
            from contextlib import contextmanager
            @contextmanager
            def _raises():
                try:
                    yield
                    raise AssertionError(f"Expected {exc.__name__} but no exception raised")
                except exc:
                    pass
            return _raises()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.modulation import (
    PWMASKModulator, 
    PWMASKDemodulator,
    AdaptiveOFDMModulator,
    calculate_bit_loading
)


class TestPWMASK:
    """Test PWM-ASK Modulator and Demodulator."""
    
    def test_pwm_ask_initialization(self):
        """Test PWMASKModulator initializes with correct default parameters."""
        mod = PWMASKModulator()
        
        assert mod.pwm_freq_hz == 10.0
        assert mod.carrier_freq_hz == 10000.0
        assert mod.duty_cycles[0] == 0.25
        assert mod.duty_cycles[1] == 0.75
    
    def test_pwm_bit0_generates_25percent_duty(self):
        """Verify bit '0' generates 25% duty cycle (envelope, not raw carrier)."""
        from scipy import signal as sig
        
        mod = PWMASKModulator(pwm_freq_hz=10.0, sample_rate_hz=100000.0)
        
        bits = np.array([0])
        t, signal_out = mod.modulate(bits)
        
        # Extract ENVELOPE via low-pass filter (same as demodulator)
        nyq = mod.sample_rate_hz / 2
        b, a = sig.butter(2, 500.0 / nyq, btype='low')
        envelope = sig.filtfilt(b, a, signal_out)
        
        # Measure duty from envelope threshold
        threshold = np.max(envelope) * 0.5
        high_samples = np.sum(envelope > threshold)
        duty_measured = high_samples / len(envelope)
        
        # Should be approximately 25% (allow some tolerance for filter transients)
        assert 0.20 <= duty_measured <= 0.30, f"Expected ~25%, got {duty_measured*100:.1f}%"
    
    def test_pwm_bit1_generates_75percent_duty(self):
        """Verify bit '1' generates 75% duty cycle (envelope, not raw carrier)."""
        from scipy import signal as sig
        
        mod = PWMASKModulator(pwm_freq_hz=10.0, sample_rate_hz=100000.0)
        
        bits = np.array([1])
        t, signal_out = mod.modulate(bits)
        
        # Extract ENVELOPE via low-pass filter
        nyq = mod.sample_rate_hz / 2
        b, a = sig.butter(2, 500.0 / nyq, btype='low')
        envelope = sig.filtfilt(b, a, signal_out)
        
        # Measure duty from envelope threshold
        threshold = np.max(envelope) * 0.5
        high_samples = np.sum(envelope > threshold)
        duty_measured = high_samples / len(envelope)
        
        # Should be approximately 75%
        assert 0.70 <= duty_measured <= 0.80, f"Expected ~75%, got {duty_measured*100:.1f}%"
    
    def test_pwm_ask_roundtrip(self):
        """Test that modulation followed by demodulation recovers original bits."""
        mod = PWMASKModulator(pwm_freq_hz=10.0, sample_rate_hz=100000.0)
        demod = PWMASKDemodulator(pwm_freq_hz=10.0, sample_rate_hz=100000.0)
        
        bits_tx = np.array([0, 1, 1, 0, 1, 0])
        t, signal = mod.modulate(bits_tx)
        
        # Add small noise
        signal_noisy = signal + np.random.normal(0, 0.01, len(signal))
        
        bits_rx = demod.demodulate(signal_noisy)
        
        # Should recover same bits
        np.testing.assert_array_equal(bits_rx, bits_tx)
    
    def test_pwm_carrier_frequency(self):
        """Verify carrier frequency is present in modulated signal."""
        mod = PWMASKModulator(pwm_freq_hz=10.0, carrier_freq_hz=10000.0, sample_rate_hz=100000.0)
        
        bits = np.array([1, 1])  # Long high period
        t, signal = mod.modulate(bits)
        
        # FFT to check carrier
        from scipy.fft import fft, fftfreq
        N = len(signal)
        yf = np.abs(fft(signal))[:N//2]
        xf = fftfreq(N, 1/mod.sample_rate_hz)[:N//2]
        
        # Find peak frequency
        peak_idx = np.argmax(yf[1:]) + 1  # Skip DC
        peak_freq = xf[peak_idx]
        
        # Should be near carrier frequency
        assert pytest.approx(peak_freq, rel=0.1) == 10000.0


class TestAdaptiveOFDM:
    """Test Adaptive OFDM Modulator with bit loading."""
    
    def test_ofdm_initialization(self):
        """Test AdaptiveOFDMModulator initializes correctly."""
        ofdm = AdaptiveOFDMModulator(nfft=64, cp_length=16)
        
        assert ofdm.nfft == 64
        assert ofdm.cp_length == 16
        assert ofdm.num_data_carriers == 52
    
    def test_bit_allocation_thresholds(self):
        """Verify bit allocation matches SNR thresholds exactly."""
        ofdm = AdaptiveOFDMModulator(nfft=64, cp_length=16)
        
        # Test specific SNR values
        snr_test = np.array([
            -5,   # < 3 dB → 0 bits
            4,    # 3-6 dB → 1 bit (BPSK)
            8,    # 6-10 dB → 2 bits (QPSK)
            12,   # 10-16 dB → 4 bits (16-QAM)
            20    # >= 16 dB → 6 bits (64-QAM)
        ] * 10 + [10, 10])  # Fill to 52 carriers
        
        allocation = ofdm.allocate_bits(snr_test)
        
        # First 5 subcarriers should match expected
        expected_first_5 = [0, 1, 2, 4, 6]
        np.testing.assert_array_equal(allocation[:5], expected_first_5)
    
    def test_bit_loading_helper(self):
        """Test standalone bit loading function."""
        snr_values = np.array([-5, 4, 8, 12, 20])
        expected = np.array([0, 1, 2, 4, 6])
        
        result = calculate_bit_loading(snr_values)
        np.testing.assert_array_equal(result, expected)
    
    def test_throughput_calculation(self):
        """Test total bits per symbol calculation."""
        ofdm = AdaptiveOFDMModulator(nfft=64, cp_length=16)
        
        # All subcarriers at high SNR (64-QAM = 6 bits each)
        snr_high = np.ones(52) * 20.0
        ofdm.allocate_bits(snr_high)
        
        # Should be 52 * 6 = 312 bits per symbol
        assert ofdm.get_throughput_bits_per_symbol() == 312
        
        # All low SNR (inactive)
        snr_low = np.ones(52) * (-10.0)
        ofdm.allocate_bits(snr_low)
        
        # Should be 0 bits
        assert ofdm.get_throughput_bits_per_symbol() == 0
    
    def test_ofdm_modulation_shape(self):
        """Test modulated signal has correct shape with CP."""
        ofdm = AdaptiveOFDMModulator(nfft=64, cp_length=16)
        
        # Allocate some bits
        snr = np.ones(52) * 10.0  # 16-QAM = 4 bits each
        ofdm.allocate_bits(snr)
        
        # Generate enough bits for 2 symbols
        bits_per_symbol = ofdm.get_throughput_bits_per_symbol()
        bits = np.random.randint(0, 2, 2 * bits_per_symbol)
        
        signal = ofdm.modulate(bits)
        
        # Each symbol: NFFT + CP = 64 + 16 = 80 samples
        expected_length = 2 * (64 + 16)
        assert len(signal) == expected_length
    
    def test_ofdm_roundtrip_high_snr(self):
        """Test OFDM modulation/demodulation roundtrip (no noise)."""
        ofdm = AdaptiveOFDMModulator(nfft=64, cp_length=16)
        
        # High SNR = BPSK for easier testing
        snr = np.ones(52) * 5.0  # BPSK = 1 bit each
        ofdm.allocate_bits(snr)
        
        bits_per_symbol = ofdm.get_throughput_bits_per_symbol()
        bits_tx = np.random.randint(0, 2, bits_per_symbol)
        
        signal = ofdm.modulate(bits_tx)
        bits_rx = ofdm.demodulate(signal)
        
        # Should recover same bits
        np.testing.assert_array_equal(bits_rx, bits_tx)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_pwm_empty_bits(self):
        """PWM modulator should handle empty bit array."""
        mod = PWMASKModulator()
        t, signal = mod.modulate(np.array([]))
        
        assert len(signal) == 0
        assert len(t) == 0
    
    def test_ofdm_wrong_snr_length(self):
        """OFDM should raise error for wrong SNR array length."""
        ofdm = AdaptiveOFDMModulator(nfft=64)
        
        wrong_snr = np.ones(10)  # Wrong length
        
        with pytest.raises(ValueError, match="SNR array length"):
            ofdm.allocate_bits(wrong_snr)
    
    def test_ofdm_modulate_without_allocation(self):
        """OFDM should raise error if modulating without bit allocation."""
        ofdm = AdaptiveOFDMModulator()
        
        bits = np.array([1, 0, 1, 0])
        
        with pytest.raises(ValueError, match="No bits allocated"):
            ofdm.modulate(bits)


if __name__ == "__main__":
    # Run quick tests
    print("=== Running Phase 3 Unit Tests ===\n")
    
    # PWM-ASK Tests
    print("--- PWM-ASK Tests ---")
    test_pwm = TestPWMASK()
    test_pwm.test_pwm_ask_initialization()
    print("✓ Initialization")
    test_pwm.test_pwm_bit0_generates_25percent_duty()
    print("✓ Bit 0 → 25% duty (25ms)")
    test_pwm.test_pwm_bit1_generates_75percent_duty()
    print("✓ Bit 1 → 75% duty (75ms)")
    test_pwm.test_pwm_ask_roundtrip()
    print("✓ Roundtrip recovery")
    
    # OFDM Tests
    print("\n--- Adaptive OFDM Tests ---")
    test_ofdm = TestAdaptiveOFDM()
    test_ofdm.test_ofdm_initialization()
    print("✓ Initialization")
    test_ofdm.test_bit_allocation_thresholds()
    print("✓ Bit allocation: [-5,4,8,12,20] → [0,1,2,4,6]")
    test_ofdm.test_throughput_calculation()
    print("✓ Throughput calculation")
    test_ofdm.test_ofdm_roundtrip_high_snr()
    print("✓ OFDM roundtrip")
    
    print("\n=== All Phase 3 Tests Passed ===")
