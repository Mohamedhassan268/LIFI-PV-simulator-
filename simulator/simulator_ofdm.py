"""
simulator/ofdm.py — Unified OFDM Engine for VLC/SLIPT
======================================================

Consolidates OFDM functionality from:
  - validate_sarwar_2017.py: Per-subcarrier BER/EVM, Gray-coded 16-QAM, ZF equalization
  - oliveira_2024_simulator.py: Adaptive bit/power loading, Hermitian symmetry, water-filling
  - simulator/modulation.py: AdaptiveOFDMModulator (bit allocation tables)
  - simulator/transmitter.py: modulate_ofdm() (DCO-OFDM)

Provides:
  - OFDMModem: Full TX/RX chain with adaptive or fixed modulation
  - Per-subcarrier analysis (BER, EVM, SNR)
  - Gray-coded M-QAM (BPSK, QPSK, 16-QAM, 64-QAM)
  - ZF equalization
  - Hermitian symmetry for real-valued optical output
  - Data rate calculation
  - Water-filling power allocation

Papers Served:
  - Sarwar 2017: Fixed 16-QAM, 80 subcarriers, 256-FFT, per-SC BER/EVM
  - Oliveira 2024: Adaptive M-QAM (up to 64-QAM), 500 subcarriers, 1024-FFT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.special import erfc


# =============================================================================
# GRAY-CODED M-QAM CONSTELLATION
# =============================================================================

# Gray code mapping: (b0, b1) -> PAM level for I and Q independently
# 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
_GRAY_4PAM = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
_GRAY_4PAM_INV = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}

# 8-PAM Gray: 000->-7, 001->-5, 011->-3, 010->-1, 110->+1, 111->+3, 101->+5, 100->+7
_GRAY_8PAM = {
    (0,0,0): -7, (0,0,1): -5, (0,1,1): -3, (0,1,0): -1,
    (1,1,0): 1,  (1,1,1): 3,  (1,0,1): 5,  (1,0,0): 7,
}
_GRAY_8PAM_INV = {v: k for k, v in _GRAY_8PAM.items()}


def _bits_to_qam(bits: np.ndarray, n_bits: int) -> complex:
    """
    Map bits to Gray-coded QAM symbol (normalized to unit average power).
    
    Supports: BPSK (1 bit), QPSK (2), 8-QAM (3), 16-QAM (4), 32-QAM (5), 64-QAM (6).
    """
    if n_bits == 0 or len(bits) < n_bits:
        return 0j
    
    if n_bits == 1:
        # BPSK: 0 -> -1, 1 -> +1
        return (2.0 * bits[0] - 1.0) + 0j
    
    elif n_bits == 2:
        # QPSK (Gray): 00->(-1,-1), 01->(-1,+1), 11->(+1,+1), 10->(+1,-1)
        I = 2.0 * bits[0] - 1.0
        Q = 2.0 * bits[1] - 1.0
        return (I + 1j * Q) / np.sqrt(2)
    
    elif n_bits == 3:
        # 8-QAM: Use rectangular 8-QAM (2×4 grid)
        # Map 3 bits: b0 selects I ∈ {-1, +1}, b1b2 selects Q ∈ {-3,-1,+1,+3}
        I = 2.0 * bits[0] - 1.0
        Q = _GRAY_4PAM.get((int(bits[1]), int(bits[2])), 0)
        # Avg power = (1 + 10/4)/2 * 2 = normalize
        return (I + 1j * Q) / np.sqrt(1 + 5)  # E[|x|²] = (1+5)/1 = 6 → /sqrt(6)
    
    elif n_bits == 4:
        # 16-QAM (Gray-coded)
        I = _GRAY_4PAM.get((int(bits[0]), int(bits[1])), 0)
        Q = _GRAY_4PAM.get((int(bits[2]), int(bits[3])), 0)
        return (I + 1j * Q) / np.sqrt(10)
    
    elif n_bits == 5:
        # 32-QAM: Cross constellation (standard)
        # Use 32-QAM as 4×8 rectangular (I: 4-PAM, Q: 8-PAM) 
        # with bits 0-1 for I, bits 2-4 for Q
        I = _GRAY_4PAM.get((int(bits[0]), int(bits[1])), 0)
        Q = _GRAY_8PAM.get((int(bits[2]), int(bits[3]), int(bits[4])), 0)
        return (I + 1j * Q) / np.sqrt(5 + 21)  # E = (5+21) = 26
    
    elif n_bits == 6:
        # 64-QAM (Gray-coded)
        I = _GRAY_8PAM.get((int(bits[0]), int(bits[1]), int(bits[2])), 0)
        Q = _GRAY_8PAM.get((int(bits[3]), int(bits[4]), int(bits[5])), 0)
        return (I + 1j * Q) / np.sqrt(42)
    
    return 0j


def _qam_to_bits(symbol: complex, n_bits: int) -> List[int]:
    """
    Hard-decision demodulate QAM symbol to bits (Gray-coded).
    """
    if n_bits == 0:
        return []
    
    if n_bits == 1:
        return [1 if symbol.real > 0 else 0]
    
    elif n_bits == 2:
        return [1 if symbol.real > 0 else 0,
                1 if symbol.imag > 0 else 0]
    
    elif n_bits == 3:
        # 8-QAM: I is BPSK, Q is 4-PAM
        s_I = symbol.real * np.sqrt(6)
        s_Q = symbol.imag * np.sqrt(6)
        b0 = 1 if s_I > 0 else 0
        # 4-PAM decision for Q
        def _dec4(v):
            if v < -2: return (0, 0)
            elif v < 0: return (0, 1)
            elif v < 2: return (1, 1)
            else:       return (1, 0)
        b12 = _dec4(s_Q)
        return [b0] + list(b12)
    
    elif n_bits == 4:
        s = symbol * np.sqrt(10)
        def _decide_4pam(v):
            if v < -2: return (0, 0)
            elif v < 0: return (0, 1)
            elif v < 2: return (1, 1)
            else:       return (1, 0)
        bi = _decide_4pam(s.real)
        bq = _decide_4pam(s.imag)
        return list(bi) + list(bq)
    
    elif n_bits == 5:
        # 32-QAM: I is 4-PAM, Q is 8-PAM
        s = symbol * np.sqrt(26)
        def _dec4(v):
            if v < -2: return (0, 0)
            elif v < 0: return (0, 1)
            elif v < 2: return (1, 1)
            else:       return (1, 0)
        def _dec8(v):
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            idx = np.argmin(np.abs(levels - v))
            mapping = [(0,0,0),(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,1,1),(1,0,1),(1,0,0)]
            return mapping[idx]
        bi = _dec4(s.real)
        bq = _dec8(s.imag)
        return list(bi) + list(bq)
    
    elif n_bits == 6:
        s = symbol * np.sqrt(42)
        def _decide_8pam(v):
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            idx = np.argmin(np.abs(levels - v))
            mapping = [(0,0,0),(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,1,1),(1,0,1),(1,0,0)]
            return mapping[idx]
        bi = _decide_8pam(s.real)
        bq = _decide_8pam(s.imag)
        return list(bi) + list(bq)
    
    return [0] * n_bits


# =============================================================================
# THEORETICAL BER
# =============================================================================

def ber_mqam(snr_db: float, M: int) -> float:
    """
    Theoretical BER for M-QAM over AWGN (Proakis).
    
    Args:
        snr_db: SNR per symbol in dB
        M: Modulation order (2, 4, 16, 64)
    Returns:
        BER
    """
    snr = 10 ** (snr_db / 10)
    if M == 2:
        return 0.5 * erfc(np.sqrt(snr))
    elif M == 4:
        return 0.5 * erfc(np.sqrt(snr / 2))
    elif M == 16:
        return (3/8) * erfc(np.sqrt(snr / 10))
    elif M == 64:
        return (7/24) * erfc(np.sqrt(snr / 42))
    else:
        # General approximation
        k = np.log2(M)
        return (2/k) * (1 - 1/np.sqrt(M)) * erfc(np.sqrt(3*snr / (2*(M-1))))


# =============================================================================
# BIT ALLOCATION TABLE
# =============================================================================

# (min_SNR_dB, bits_per_symbol, QAM_order)
BIT_ALLOCATION_TABLE = [
    (22.0, 6, 64),    # 64-QAM
    (16.0, 5, 32),    # 32-QAM  (Oliveira uses this threshold)
    (10.0, 4, 16),    # 16-QAM
    (7.0,  3, 8),     # 8-QAM
    (4.0,  2, 4),     # QPSK
    (1.0,  1, 2),     # BPSK
]


def allocate_bits(snr_per_sc: np.ndarray, 
                  table=None) -> np.ndarray:
    """
    Allocate bits per subcarrier based on SNR thresholds.
    
    Args:
        snr_per_sc: SNR in dB for each subcarrier
        table: Custom allocation table [(min_snr, bits, M), ...]
    Returns:
        bit_allocation: bits per subcarrier (array of ints)
    """
    if table is None:
        table = BIT_ALLOCATION_TABLE
    
    allocation = np.zeros(len(snr_per_sc), dtype=int)
    for i, snr in enumerate(snr_per_sc):
        for threshold, bits, _ in table:
            if snr >= threshold:
                allocation[i] = bits
                break
    return allocation


def water_filling_power(snr_per_sc: np.ndarray) -> np.ndarray:
    """
    Water-filling power allocation.
    
    Allocates more power to stronger subcarriers.
    Normalized to unit average power.
    """
    snr_lin = 10 ** (snr_per_sc / 10)
    mu = np.mean(1 / (snr_lin + 1e-10)) + 1
    power = np.maximum(0, mu - 1 / (snr_lin + 1e-10))
    if power.sum() > 0:
        power = power / power.sum() * len(snr_per_sc)
    return power


# =============================================================================
# CHANNEL RESPONSE MODEL (RC low-pass for solar panel)
# =============================================================================

def solar_panel_channel_response(n_subcarriers: int,
                                  signal_bandwidth_hz: float,
                                  junction_capacitance_pf: float,
                                  shunt_resistance_kohm: float,
                                  snr_base_db: float = 28.0,
                                  snr_floor_db: float = 17.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency-domain channel response for a solar panel receiver.
    
    Models the RC low-pass characteristic of the PV cell, and maps it to
    a per-subcarrier SNR profile.
    
    Used by: Sarwar 2017 (calibrated to match paper Fig 4 BER curve)
    
    Args:
        n_subcarriers: Number of data subcarriers
        signal_bandwidth_hz: Total signal bandwidth
        junction_capacitance_pf: C_j in pF
        shunt_resistance_kohm: R_sh in kΩ
        snr_base_db: SNR at DC (subcarrier 0)
        snr_floor_db: SNR floor at highest subcarrier
    
    Returns:
        H: Channel magnitude response (0-1) per subcarrier
        snr_db: SNR per subcarrier in dB
    """
    freqs = np.linspace(0, signal_bandwidth_hz, n_subcarriers)
    C = junction_capacitance_pf * 1e-12
    R = shunt_resistance_kohm * 1e3
    f_3db = 1 / (2 * np.pi * R * C)
    
    # First-order LP magnitude response
    H = 1 / np.sqrt(1 + (freqs / f_3db) ** 2)
    
    # SNR decreases with frequency following channel attenuation
    snr_db = snr_base_db - (snr_base_db - snr_floor_db) * (1 - H**2)
    
    return H, snr_db


# =============================================================================
# OFDM MODEM — UNIFIED ENGINE
# =============================================================================

class OFDMModem:
    """
    Unified OFDM modem with adaptive or fixed modulation.
    
    Supports both Sarwar 2017 (fixed 16-QAM) and Oliveira 2024 (adaptive M-QAM).
    
    Usage (Sarwar — fixed 16-QAM):
        modem = OFDMModem(nfft=256, cp_length=32, n_subcarriers=80)
        modem.set_fixed_modulation(qam_order=16)
        results = modem.simulate(n_symbols=1000, snr_per_sc=snr_db)
    
    Usage (Oliveira — adaptive):
        modem = OFDMModem(nfft=1024, cp_length=10, n_subcarriers=500)
        modem.set_adaptive_modulation(snr_per_sc=snr_db)
        results = modem.simulate(n_symbols=100, snr_per_sc=snr_db)
    """
    
    def __init__(self,
                 nfft: int = 256,
                 cp_length: int = 32,
                 n_subcarriers: int = 80,
                 bandwidth_hz: float = 5e6):
        self.nfft = nfft
        self.cp_length = cp_length
        self.n_subcarriers = n_subcarriers
        self.bandwidth_hz = bandwidth_hz
        self.subcarrier_spacing = bandwidth_hz / n_subcarriers
        
        # Bit and power allocation per subcarrier
        self.bit_allocation = np.zeros(n_subcarriers, dtype=int)
        self.power_allocation = np.ones(n_subcarriers)
        self._mode = None  # 'fixed' or 'adaptive'
    
    def set_fixed_modulation(self, qam_order: int = 16):
        """Set all subcarriers to the same QAM order."""
        bps = int(np.log2(qam_order))
        self.bit_allocation = np.full(self.n_subcarriers, bps, dtype=int)
        self.power_allocation = np.ones(self.n_subcarriers)
        self._mode = 'fixed'
    
    def set_adaptive_modulation(self, snr_per_sc: np.ndarray,
                                 use_water_filling: bool = False):
        """Allocate bits (and optionally power) based on per-SC SNR."""
        self.bit_allocation = allocate_bits(snr_per_sc[:self.n_subcarriers])
        if use_water_filling:
            self.power_allocation = water_filling_power(snr_per_sc[:self.n_subcarriers])
        else:
            self.power_allocation = np.ones(self.n_subcarriers)
        self._mode = 'adaptive'
    
    @property
    def total_bits_per_symbol(self) -> int:
        return int(np.sum(self.bit_allocation))
    
    def calculate_data_rate(self, sample_rate_hz: float) -> float:
        """
        Gross data rate in Mbps.
        
        GDR = (f_sample / (N_FFT + N_CP)) × Σ bits_per_sc
        """
        symbol_rate = sample_rate_hz / (self.nfft + self.cp_length)
        return symbol_rate * self.total_bits_per_symbol / 1e6
    
    # ----- TX -----
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        OFDM TX: bits → time-domain real signal (Hermitian symmetry).
        
        Returns:
            Real-valued time-domain signal with cyclic prefix.
        """
        n_symbols = len(bits) // max(1, self.total_bits_per_symbol)
        signal_out = []
        bit_idx = 0
        
        for _ in range(n_symbols):
            freq_data = np.zeros(self.nfft, dtype=complex)
            for sc in range(self.n_subcarriers):
                nb = self.bit_allocation[sc]
                if nb == 0:
                    continue
                sc_bits = bits[bit_idx:bit_idx + nb]
                bit_idx += nb
                sym = _bits_to_qam(sc_bits, nb)
                sym *= np.sqrt(self.power_allocation[sc])
                freq_data[sc + 1] = sym  # skip DC at index 0
            
            # Hermitian symmetry
            freq_data[self.nfft//2 + 1:] = np.conj(freq_data[1:self.nfft//2][::-1])
            
            # IFFT
            time_data = np.fft.ifft(freq_data).real
            
            # Cyclic prefix
            signal_out.append(np.concatenate([time_data[-self.cp_length:], time_data]))
        
        return np.concatenate(signal_out) if signal_out else np.array([])
    
    def demodulate(self, signal_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        OFDM RX: time-domain signal → bits + frequency-domain symbols.
        """
        sym_len = self.nfft + self.cp_length
        n_symbols = len(signal_in) // sym_len
        all_bits = []
        all_symbols = []
        
        for s in range(n_symbols):
            start = s * sym_len
            time_data = signal_in[start + self.cp_length: start + sym_len]
            freq_data = np.fft.fft(time_data)
            
            for sc in range(self.n_subcarriers):
                sym = freq_data[sc + 1]
                if self.power_allocation[sc] > 0:
                    sym /= np.sqrt(self.power_allocation[sc])
                all_symbols.append(sym)
                nb = self.bit_allocation[sc]
                all_bits.extend(_qam_to_bits(sym, nb))
        
        return np.array(all_bits, dtype=int), np.array(all_symbols)
    
    # ----- Per-Subcarrier Simulation (Sarwar/Oliveira style) -----
    
    def simulate(self, n_symbols: int,
                 snr_per_sc: np.ndarray,
                 channel_H: Optional[np.ndarray] = None,
                 equalization: str = 'ZF',
                 seed: Optional[int] = None) -> Dict:
        """
        Full per-subcarrier Monte Carlo simulation.
        
        For each OFDM symbol and each subcarrier:
          1. Generate random bits
          2. Map to QAM symbol (Gray-coded)
          3. Optionally apply channel H
          4. Add AWGN at the specified per-SC SNR
          5. Equalize (ZF)
          6. Hard-decision demodulate
          7. Count bit errors and compute EVM
        
        Args:
            n_symbols: Number of OFDM symbols to simulate
            snr_per_sc: SNR in dB per subcarrier (length = n_subcarriers)
            channel_H: Optional channel magnitude response (for ZF equalization)
            equalization: 'ZF' (default) or 'none'
            seed: Random seed for reproducibility
        
        Returns:
            Dict with keys:
              - ber_per_sc: BER per subcarrier
              - evm_per_sc: EVM (%) per subcarrier
              - overall_ber: Weighted overall BER
              - data_rate_mbps: (if sample rate provided in self)
              - tx_symbols: Sample of TX symbols
              - rx_symbols: Sample of RX symbols (equalized)
              - snr_per_sc: Input SNR profile
              - channel_H: Channel response used
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_sc = self.n_subcarriers
        snr = snr_per_sc[:n_sc]
        
        if channel_H is None:
            H = np.ones(n_sc)
        else:
            H = channel_H[:n_sc]
        
        err_per_sc = np.zeros(n_sc)
        tot_per_sc = np.zeros(n_sc)
        evm_per_sc = np.zeros(n_sc)
        sample_tx = []
        sample_rx = []
        
        for sym_idx in range(n_symbols):
            for sc in range(n_sc):
                nb = self.bit_allocation[sc]
                if nb == 0:
                    continue
                
                # 1. Random bits
                tx_bits = np.random.randint(0, 2, nb)
                
                # 2. QAM symbol
                tx_sym = _bits_to_qam(tx_bits, nb)
                
                # 3. Channel
                rx_sym = tx_sym * H[sc]
                
                # 4. AWGN
                snr_lin = 10 ** (snr[sc] / 10)
                sig_power = np.abs(rx_sym) ** 2
                noise_power = sig_power / snr_lin if snr_lin > 0 else 1e-10
                noise_std = np.sqrt(noise_power / 2)
                noise = noise_std * (np.random.randn() + 1j * np.random.randn())
                rx_noisy = rx_sym + noise
                
                # 5. Equalize
                if equalization == 'ZF' and abs(H[sc]) > 1e-10:
                    rx_eq = rx_noisy / H[sc]
                else:
                    rx_eq = rx_noisy
                
                # 6. Demodulate
                rx_bits = _qam_to_bits(rx_eq, nb)
                
                # 7. Count errors
                errors = sum(int(a) != int(b) for a, b in zip(tx_bits, rx_bits))
                err_per_sc[sc] += errors
                tot_per_sc[sc] += nb
                
                # EVM
                err_vec = rx_eq - tx_sym
                if abs(tx_sym) > 1e-10:
                    evm_per_sc[sc] += abs(err_vec) / abs(tx_sym) * 100
                
                # Sample symbols for constellation plot
                if sym_idx % max(1, n_symbols // 100) == 0:
                    sample_tx.append(tx_sym)
                    sample_rx.append(rx_eq)
        
        ber_per_sc = err_per_sc / np.maximum(tot_per_sc, 1)
        evm_per_sc /= n_symbols  # Average EVM per SC
        
        total_errors = np.sum(err_per_sc)
        total_bits = np.sum(tot_per_sc)
        overall_ber = total_errors / total_bits if total_bits > 0 else 0
        
        return {
            'ber_per_sc': ber_per_sc,
            'evm_per_sc': evm_per_sc,
            'overall_ber': overall_ber,
            'total_errors': int(total_errors),
            'total_bits': int(total_bits),
            'n_symbols': n_symbols,
            'n_subcarriers': n_sc,
            'bit_allocation': self.bit_allocation.copy(),
            'snr_per_sc': snr.copy(),
            'channel_H': H.copy(),
            'tx_symbols': np.array(sample_tx),
            'rx_symbols': np.array(sample_rx),
        }


# =============================================================================
# SPECTRAL EFFICIENCY & DATA RATE HELPERS
# =============================================================================

def spectral_efficiency(bit_allocation: np.ndarray, 
                         nfft: int, cp_length: int) -> float:
    """
    Spectral efficiency in bits/s/Hz.
    
    η = Σ bits_per_sc / (N_FFT + N_CP)
    """
    return np.sum(bit_allocation) / (nfft + cp_length)


def gross_data_rate(bit_allocation: np.ndarray,
                     nfft: int, cp_length: int,
                     sample_rate_hz: float) -> float:
    """
    Gross data rate in Mbps.
    
    GDR = f_sample / (N_FFT + N_CP) × Σ bits_per_sc
    """
    symbol_rate = sample_rate_hz / (nfft + cp_length)
    return symbol_rate * np.sum(bit_allocation) / 1e6


def net_data_rate(gross_rate_mbps: float, 
                   fec_overhead: float = 0.171) -> float:
    """
    Net data rate after FEC overhead.
    
    Default 17.1% overhead (RS code, used by Oliveira 2024).
    """
    return gross_rate_mbps * (1 - fec_overhead)


# =============================================================================
# TESTS
# =============================================================================

def test_ofdm_module():
    """Quick self-test."""
    print("=" * 60)
    print("OFDM MODULE SELF-TEST")
    print("=" * 60)
    
    # --- Test 1: Sarwar-style fixed 16-QAM ---
    print("\n[1] Sarwar-style: Fixed 16-QAM, 80 SCs, 256-FFT")
    modem = OFDMModem(nfft=256, cp_length=32, n_subcarriers=80, bandwidth_hz=5e6)
    modem.set_fixed_modulation(qam_order=16)
    
    H, snr = solar_panel_channel_response(
        80, 5e6, junction_capacitance_pf=500, shunt_resistance_kohm=10
    )
    results = modem.simulate(n_symbols=500, snr_per_sc=snr, channel_H=H, seed=42)
    
    rate = modem.calculate_data_rate(sample_rate_hz=15e6)
    print(f"  Data rate: {rate:.2f} Mbps (target: 15.03)")
    print(f"  BER: {results['overall_ber']:.4e} (target: 1.69e-3)")
    print(f"  SNR range: {snr.min():.1f} - {snr.max():.1f} dB")
    
    # --- Test 2: Oliveira-style adaptive ---
    print("\n[2] Oliveira-style: Adaptive, 500 SCs, 1024-FFT")
    modem2 = OFDMModem(nfft=1024, cp_length=10, n_subcarriers=500, bandwidth_hz=1.5e6)
    
    # Generate SNR profile (25 dB max, rolling off)
    sc_idx = np.arange(500)
    snr2 = 25 - 15 * (sc_idx / 500) ** 1.5 + np.random.normal(0, 1, 500)
    snr2 = np.clip(snr2, 0, 30)
    
    modem2.set_adaptive_modulation(snr2)
    results2 = modem2.simulate(n_symbols=100, snr_per_sc=snr2, seed=42)
    
    rate2 = modem2.calculate_data_rate(sample_rate_hz=3e6)
    active = np.sum(modem2.bit_allocation > 0)
    print(f"  Active SCs: {active}/500")
    print(f"  Bits/symbol: {modem2.total_bits_per_symbol}")
    print(f"  Data rate: {rate2:.2f} Mbps (target: ~25.7 gross)")
    print(f"  BER: {results2['overall_ber']:.4e}")
    
    # --- Test 3: QAM codec round-trip ---
    print("\n[3] Gray-coded QAM round-trip test")
    for nb, name in [(1, 'BPSK'), (2, 'QPSK'), (4, '16-QAM'), (6, '64-QAM')]:
        errors = 0
        trials = 1000
        for _ in range(trials):
            bits_in = np.random.randint(0, 2, nb)
            sym = _bits_to_qam(bits_in, nb)
            bits_out = _qam_to_bits(sym, nb)
            errors += sum(int(a) != int(b) for a, b in zip(bits_in, bits_out))
        print(f"  {name}: {errors} errors in {trials*nb} bits (should be 0)")
    
    # --- Test 4: Modulate/demodulate round-trip ---
    print("\n[4] Full OFDM TX/RX round-trip (no noise)")
    modem3 = OFDMModem(nfft=64, cp_length=8, n_subcarriers=28, bandwidth_hz=1e6)
    modem3.set_fixed_modulation(qam_order=4)
    n_bits = modem3.total_bits_per_symbol * 10
    tx_bits = np.random.randint(0, 2, n_bits)
    sig = modem3.modulate(tx_bits)
    rx_bits, _ = modem3.demodulate(sig)
    min_len = min(len(tx_bits), len(rx_bits))
    ber_rt = np.sum(tx_bits[:min_len] != rx_bits[:min_len]) / min_len
    print(f"  TX bits: {len(tx_bits)}, RX bits: {len(rx_bits)}")
    print(f"  Round-trip BER: {ber_rt:.4e} (should be ~0)")
    
    print("\n" + "=" * 60)
    print("SELF-TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_ofdm_module()
