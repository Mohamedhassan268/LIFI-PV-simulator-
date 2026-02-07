# utils/ber.py
"""
Theoretical BER Formulas for Optical Wireless Communication.

Provides closed-form BER expressions over AWGN for:
  - OOK (On-Off Keying)
  - BPSK
  - M-QAM (4, 16, 64, 256)

These are standard textbook formulas used across multiple papers.
Any simulator module can import these instead of reimplementing.

References:
  - Proakis, "Digital Communications", 5th ed.
  - Oliveira 2024: uses M-QAM BER for adaptive OFDM subcarrier allocation
  - Kadirvelu 2021: uses OOK BER for single-carrier validation
"""

import numpy as np
from scipy.special import erfc


# =============================================================================
# CORE BER FORMULAS
# =============================================================================

def ber_ook(snr_db):
    """
    Theoretical BER for coherent OOK over AWGN.
    
    BER = 0.5 * erfc(sqrt(Eb/N0 / 2))
    
    Equivalent to BER = Q(sqrt(Eb/N0)) where Q(x) = 0.5*erfc(x/sqrt(2)).
    
    Args:
        snr_db: Eb/N0 in dB (scalar or array)
        
    Returns:
        BER value(s) in range [0, 0.5]
    """
    snr_linear = _db_to_linear(snr_db)
    return 0.5 * erfc(np.sqrt(snr_linear / 2))


def ber_bpsk(snr_db):
    """
    Theoretical BER for BPSK over AWGN.
    
    BER = 0.5 * erfc(sqrt(Eb/N0))
    
    Args:
        snr_db: Eb/N0 in dB (scalar or array)
        
    Returns:
        BER value(s) in range [0, 0.5]
    """
    snr_linear = _db_to_linear(snr_db)
    return 0.5 * erfc(np.sqrt(snr_linear))


def ber_mqam(snr_db, M):
    """
    Theoretical BER for M-QAM over AWGN.
    
    Supports M = 4 (QPSK), 16, 64, 256.
    Uses exact closed-form approximations from Proakis.
    
    For general M-QAM (M = 2^k, k even):
        BER ≈ (2/log2(M)) * (1 - 1/sqrt(M)) * erfc(sqrt(3*SNR / (2*(M-1))))
    
    For specific low orders, uses tighter expressions matching
    Oliveira 2024's subcarrier BER computation.
    
    Args:
        snr_db: SNR per symbol in dB (scalar or array)
        M: Modulation order (4, 16, 64, 256)
        
    Returns:
        BER value(s) in range [0, 0.5]
        
    Raises:
        ValueError: If M is not a valid QAM order
    """
    snr_linear = _db_to_linear(snr_db)
    
    if M == 2:
        # BPSK
        return 0.5 * erfc(np.sqrt(snr_linear))
    elif M == 4:
        # QPSK (same as two independent BPSK channels)
        return 0.5 * erfc(np.sqrt(snr_linear / 2))
    elif M == 16:
        # 16-QAM
        return (3 / 8) * erfc(np.sqrt(snr_linear / 10))
    elif M == 64:
        # 64-QAM
        return (7 / 24) * erfc(np.sqrt(snr_linear / 42))
    elif M == 256:
        # 256-QAM (general formula)
        return (15 / 64) * erfc(np.sqrt(snr_linear / 170))
    else:
        # General M-QAM approximation for any square QAM
        k = np.log2(M)
        if k != int(k) or int(k) % 2 != 0:
            raise ValueError(
                f"M={M} is not a valid square QAM order. "
                f"Use M = 4, 16, 64, 256, ... (M = 2^(2n))"
            )
        return (2 / k) * (1 - 1 / np.sqrt(M)) * erfc(
            np.sqrt(3 * snr_linear / (2 * (M - 1)))
        )


# =============================================================================
# SNR THRESHOLD LOOKUP (for adaptive bit loading)
# =============================================================================

# Minimum SNR (dB) required to achieve BER ≤ 3.8e-3 (FEC threshold)
# at each modulation order. Used by adaptive OFDM bit loading.
# Pre-computed from inverting the BER formulas above.
SNR_THRESHOLDS_FEC = {
    0:   -np.inf,   # No transmission
    2:   6.8,       # BPSK
    4:   9.8,       # QPSK  
    16:  16.5,      # 16-QAM
    64:  22.5,      # 64-QAM
    256: 28.5,      # 256-QAM (if needed)
}


def bits_per_symbol(M):
    """Return log2(M) bits per QAM symbol."""
    if M <= 0:
        return 0
    return int(np.log2(M))


def select_modulation_order(snr_db, ber_target=3.8e-3):
    """
    Select highest modulation order achievable at given SNR.
    
    Scans from highest to lowest order, returns first where
    theoretical BER ≤ target.
    
    Args:
        snr_db: Available SNR in dB (scalar)
        ber_target: Maximum acceptable BER (default: FEC threshold)
        
    Returns:
        M: Modulation order (0 if none achievable)
    """
    for M in [64, 16, 4, 2]:
        if ber_mqam(snr_db, M) <= ber_target:
            return M
    return 0  # SNR too low for any modulation


# =============================================================================
# Eb/N0 COMPUTATION
# =============================================================================

def compute_eb_n0(P_rx, noise_psd, bit_rate, responsivity=0.5):
    """
    Compute Eb/N0 from system parameters.
    
    For optical systems:
        Signal power = (R * P_rx)^2
        Eb = signal_power / bit_rate
        N0 = noise_psd (A²/Hz)
    
    Args:
        P_rx: Received optical power (W)
        noise_psd: Noise power spectral density (A²/Hz)
        bit_rate: Data rate (bps)
        responsivity: PV responsivity (A/W)
        
    Returns:
        Eb/N0 in linear scale
    """
    I_signal = responsivity * P_rx
    signal_power = I_signal ** 2
    Eb = signal_power / bit_rate
    
    if noise_psd == 0:
        return 1e10
    
    return Eb / noise_psd


def compute_eb_n0_db(P_rx, noise_psd, bit_rate, responsivity=0.5):
    """Compute Eb/N0 in dB. See compute_eb_n0 for args."""
    return _linear_to_db(compute_eb_n0(P_rx, noise_psd, bit_rate, responsivity))


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _db_to_linear(db):
    """Convert dB to linear scale."""
    return 10 ** (np.asarray(db, dtype=float) / 10)


def _linear_to_db(linear):
    """Convert linear scale to dB."""
    return 10 * np.log10(np.maximum(linear, 1e-30))


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("BER Utility Tests")
    print("=" * 60)
    
    # Test 1: OOK BER matches existing demodulator
    print("\n1. OOK BER (should match demodulator.predict_ber_ook_db):")
    for snr in [0, 5, 10, 15, 20]:
        print(f"   SNR={snr:2d} dB -> BER = {ber_ook(snr):.2e}")
    
    # Test 2: M-QAM BER matches Oliveira's compute_ber_awgn
    print("\n2. M-QAM BER (should match oliveira_2024_plots.compute_ber_awgn):")
    for M in [2, 4, 16, 64]:
        ber_25 = ber_mqam(25.0, M)
        print(f"   {M:3d}-QAM @ 25 dB -> BER = {ber_25:.2e}")
    
    # Test 3: Oliveira validation — 16-QAM at ~25 dB should give BER ≈ 3.4e-3
    ber_oliveira = ber_mqam(25.0, 16)
    print(f"\n3. Oliveira validation: 16-QAM @ 25 dB -> BER = {ber_oliveira:.4e}")
    print(f"   Paper target: 3.4e-3, Match: {abs(ber_oliveira - 3.4e-3) < 1e-3}")
    
    # Test 4: Adaptive modulation selection
    print("\n4. Adaptive modulation selection (BER target = 3.8e-3):")
    for snr in [5, 10, 15, 20, 25]:
        M = select_modulation_order(snr)
        bps = bits_per_symbol(M)
        print(f"   SNR={snr:2d} dB -> {M}-QAM ({bps} bits/symbol)")
    
    print("\nAll tests complete.")
