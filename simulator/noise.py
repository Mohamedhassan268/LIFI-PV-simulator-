# simulator/noise.py
"""
Comprehensive Noise Model for Li-Fi + PV System.

Implements 6 physical noise sources:
1. Shot noise (quantum fluctuations in signal photocurrent)
2. Thermal noise (Johnson-Nyquist from load/TIA resistance)
3. Ambient light noise (shot noise from background DC current)
4. Amplifier noise (INA/TIA input voltage + current noise)
5. ADC quantization noise (digitization error)
6. Processing noise (threshold uncertainty, timing jitter)

Sources 1-4 were in the original model.
Sources 5-6 are NEW — required for accurate Correa 2025 greenhouse BER
where ADC and processing noise dominate at low SNR.

All calculations based on physical equations — no artificial calibration.
"""

import numpy as np
from utils.constants import Q, K_B


class NoiseModel:
    """
    Physical noise model for optical wireless communication.
    
    All noise sources are calculated from first principles using
    physical parameters. No empirical fudge factors.
    """
    
    def __init__(self, params=None):
        """
        Initialize noise model.
        
        Args:
            params (dict): Configuration with optional keys:
                - 'temperature_K': Operating temperature (default 300K)
                - 'load_resistance_ohm': TIA/load resistance (default 50Ω)
                - 'ina_noise_nv_sqrt_hz': INA input noise density (default 10 nV/√Hz)
                - 'ina_current_noise_pa_sqrt_hz': INA current noise (default 0.01 pA/√Hz)
                - 'ambient_lux': Background light level (default 0)
                - 'ambient_responsivity': Responsivity for ambient (default 0.4 A/W)
                - 'adc_bits': ADC resolution (default 12, set 0 to disable)
                - 'adc_vref': ADC reference voltage (default 3.3V)
                - 'amp_gain': Amplifier gain for ADC referral (default 11)
                - 'R_sense': Current-sense resistor for referral (default 1.0Ω)
                - 'processing_threshold_mv': Decision threshold uncertainty (default 10mV)
        """
        if params is None:
            params = {}
        
        self.T = params.get('temperature_K', 300)  # Kelvin
        self.R_load = params.get('load_resistance_ohm', 50)  # Ohms
        self.V_n_ina = params.get('ina_noise_nv_sqrt_hz', 10) * 1e-9  # V/√Hz
        self.I_n_ina = params.get('ina_current_noise_pa_sqrt_hz', 0.01) * 1e-12  # A/√Hz
        self.ambient_lux = params.get('ambient_lux', 0)  # lux
        self.ambient_resp = params.get('ambient_responsivity', 0.4)  # A/W
        
        # ADC parameters (NEW)
        self.adc_bits = params.get('adc_bits', 12)
        self.adc_vref = params.get('adc_vref', 3.3)  # V
        self.amp_gain = params.get('amp_gain', 11.0)  # from Correa circuit: 1 + R5/R6
        
        # Current referral
        self.R_sense = params.get('R_sense', 1.0)  # Ω
        
        # Processing noise (NEW)
        self.threshold_mv = params.get('processing_threshold_mv', 10.0)  # mV
        
        # Conversion: ~1 lux ≈ 1.46 µW/cm² for typical illumination
        self.lux_to_power_factor = 1.46e-6  # W/cm² per lux
        
    def shot_noise_variance(self, I_ph, bandwidth):
        """
        Source 1: Shot noise variance.
        
        Shot noise arises from quantum nature of light/current.
        σ²_shot = 2 · q · I_ph · B
        
        Args:
            I_ph (float or ndarray): Photocurrent in Amperes
            bandwidth (float): Noise bandwidth in Hz
            
        Returns:
            float: Shot noise variance in A²
        """
        I_avg = np.mean(np.abs(I_ph))
        return 2 * Q * I_avg * bandwidth
    
    def thermal_noise_variance(self, bandwidth):
        """
        Source 2: Thermal (Johnson-Nyquist) noise variance.
        
        Thermal noise from resistive elements.
        σ²_thermal = 4 · k_B · T · B / R
        
        Args:
            bandwidth (float): Noise bandwidth in Hz
            
        Returns:
            float: Thermal noise variance in A² (current-referred)
        """
        return 4 * K_B * self.T * bandwidth / self.R_load
    
    def ambient_noise_variance(self, bandwidth, rx_area_cm2=9.0):
        """
        Source 3: Ambient light shot noise variance.
        
        Background light creates DC photocurrent with associated shot noise.
        Dominates in greenhouse/outdoor environments.
        
        Args:
            bandwidth (float): Noise bandwidth in Hz
            rx_area_cm2 (float): Receiver area in cm²
            
        Returns:
            float: Ambient noise variance in A²
        """
        if self.ambient_lux <= 0:
            return 0.0
        
        # Convert lux to optical power on receiver
        P_ambient_w = self.ambient_lux * self.lux_to_power_factor * rx_area_cm2
        
        # Convert to photocurrent
        I_ambient = self.ambient_resp * P_ambient_w
        
        # Shot noise from ambient current
        return 2 * Q * I_ambient * bandwidth
    
    def amplifier_noise_variance(self, bandwidth, ina_gain=100):
        """
        Source 4: Amplifier (INA/TIA) input-referred noise variance.
        
        Includes both voltage noise and current noise contributions.
        σ²_amp = (V_n / R_sense)² · B + I_n² · B
        
        Args:
            bandwidth (float): Noise bandwidth in Hz
            ina_gain (float): INA voltage gain (unused in current-referred calc)
            
        Returns:
            float: Amplifier noise variance in A² (input-referred)
        """
        # Voltage noise referred to current through sense resistor
        I_n_from_V = self.V_n_ina / self.R_sense  # A/√Hz
        
        # Total amplifier current noise density
        I_n_total = np.sqrt(I_n_from_V**2 + self.I_n_ina**2)
        
        return I_n_total**2 * bandwidth
    
    def adc_noise_variance(self):
        """
        Source 5: ADC quantization noise variance (NEW).
        
        Quantization noise arises from finite ADC resolution.
        σ²_adc = (V_LSB)² / 12, referred to input current.
        
        For a 12-bit ADC with 3.3V reference:
            V_LSB = 3.3 / 4096 = 0.806 mV
            σ_V = 0.806 / √12 = 0.233 mV
            σ_I = σ_V / (R_load × gain) = 0.233mV / (50Ω × 11) ≈ 0.42 µA
        
        This is independent of bandwidth — it's a fixed noise floor.
        
        Returns:
            float: ADC noise variance in A² (input-referred)
        """
        if self.adc_bits <= 0:
            return 0.0
        
        # LSB voltage
        V_lsb = self.adc_vref / (2 ** self.adc_bits)
        
        # Quantization noise voltage RMS
        sigma_v_adc = V_lsb / np.sqrt(12)
        
        # Refer to input current: divide by transimpedance chain
        # V_adc = I_ph × R_sense × amp_gain → I_ph = V_adc / (R_sense × amp_gain)
        transimpedance = self.R_load * self.amp_gain
        if transimpedance <= 0:
            return 0.0
        
        sigma_i_adc = sigma_v_adc / transimpedance
        
        return sigma_i_adc ** 2
    
    def processing_noise_variance(self):
        """
        Source 6: Processing/threshold noise variance (NEW).
        
        Models decision threshold uncertainty from:
        - Comparator hysteresis
        - Timing jitter at sample points
        - SDM algorithm threshold sensitivity
        
        From Correa 2025: "error identified when detected pulse width
        deviated by more than 10 ms" — this timing uncertainty maps
        to an equivalent current noise floor.
        
        Returns:
            float: Processing noise variance in A² (input-referred)
        """
        if self.threshold_mv <= 0:
            return 0.0
        
        # Threshold voltage uncertainty
        sigma_v_proc = self.threshold_mv * 1e-3  # mV → V
        
        # Refer to input current
        transimpedance = self.R_load * self.amp_gain
        if transimpedance <= 0:
            return 0.0
        
        sigma_i_proc = sigma_v_proc / transimpedance
        
        return sigma_i_proc ** 2
    
    def total_noise_std(self, I_ph, bandwidth, rx_area_cm2=9.0, ina_gain=100, 
                        include_adc=True, include_processing=True, verbose=False):
        """
        Calculate total noise standard deviation from all sources.
        
        σ_total = √(σ²_shot + σ²_thermal + σ²_ambient + σ²_amp + σ²_adc + σ²_proc)
        
        Args:
            I_ph (float or ndarray): Signal photocurrent in Amperes
            bandwidth (float): Noise bandwidth in Hz
            rx_area_cm2 (float): Receiver area in cm²
            ina_gain (float): INA voltage gain
            include_adc (bool): Include ADC quantization noise (default True)
            include_processing (bool): Include processing noise (default True)
            verbose (bool): Print breakdown
            
        Returns:
            float: Total noise standard deviation in Amperes
        """
        sigma2_shot = self.shot_noise_variance(I_ph, bandwidth)
        sigma2_thermal = self.thermal_noise_variance(bandwidth)
        sigma2_ambient = self.ambient_noise_variance(bandwidth, rx_area_cm2)
        sigma2_amp = self.amplifier_noise_variance(bandwidth, ina_gain)
        sigma2_adc = self.adc_noise_variance() if include_adc else 0.0
        sigma2_proc = self.processing_noise_variance() if include_processing else 0.0
        
        sigma2_total = (sigma2_shot + sigma2_thermal + sigma2_ambient + 
                        sigma2_amp + sigma2_adc + sigma2_proc)
        sigma_total = np.sqrt(sigma2_total)
        
        if verbose:
            print(f"\n  Noise Breakdown (6-source model):")
            print(f"    1. Shot noise:    {np.sqrt(sigma2_shot)*1e9:.3f} nA (σ)")
            print(f"    2. Thermal noise: {np.sqrt(sigma2_thermal)*1e9:.3f} nA (σ)")
            print(f"    3. Ambient noise: {np.sqrt(sigma2_ambient)*1e9:.3f} nA (σ)")
            print(f"    4. Amplifier:     {np.sqrt(sigma2_amp)*1e9:.3f} nA (σ)")
            print(f"    5. ADC quant:     {np.sqrt(sigma2_adc)*1e9:.3f} nA (σ)")
            print(f"    6. Processing:    {np.sqrt(sigma2_proc)*1e9:.3f} nA (σ)")
            print(f"    ─────────────────────────────────")
            print(f"    TOTAL:            {sigma_total*1e9:.3f} nA (σ)")
            print(f"    Bandwidth:        {bandwidth/1e3:.1f} kHz")
            
            # Identify dominant source
            variances = {
                'Shot': sigma2_shot, 'Thermal': sigma2_thermal,
                'Ambient': sigma2_ambient, 'Amplifier': sigma2_amp,
                'ADC': sigma2_adc, 'Processing': sigma2_proc
            }
            dominant = max(variances, key=variances.get)
            pct = variances[dominant] / sigma2_total * 100 if sigma2_total > 0 else 0
            print(f"    Dominant: {dominant} ({pct:.0f}%)")
        
        return sigma_total
    
    def noise_breakdown(self, I_ph, bandwidth, rx_area_cm2=9.0, ina_gain=100):
        """
        Return full noise breakdown as a dictionary.
        
        Useful for plotting noise budgets or debugging.
        
        Args:
            I_ph, bandwidth, rx_area_cm2, ina_gain: Same as total_noise_std
            
        Returns:
            dict: Variance (A²) for each source + total sigma (A)
        """
        return {
            'shot_A2': self.shot_noise_variance(I_ph, bandwidth),
            'thermal_A2': self.thermal_noise_variance(bandwidth),
            'ambient_A2': self.ambient_noise_variance(bandwidth, rx_area_cm2),
            'amplifier_A2': self.amplifier_noise_variance(bandwidth, ina_gain),
            'adc_A2': self.adc_noise_variance(),
            'processing_A2': self.processing_noise_variance(),
            'sigma_total_A': self.total_noise_std(I_ph, bandwidth, rx_area_cm2, ina_gain),
        }
    
    def generate_noise(self, length, I_ph, bandwidth, rx_area_cm2=9.0, ina_gain=100,
                       include_adc=True, include_processing=True):
        """
        Generate AWGN noise samples.
        
        Args:
            length (int): Number of samples
            I_ph (float or ndarray): Signal photocurrent
            bandwidth (float): Noise bandwidth
            rx_area_cm2 (float): Receiver area
            ina_gain (float): INA gain
            include_adc (bool): Include ADC noise
            include_processing (bool): Include processing noise
            
        Returns:
            ndarray: Noise samples in Amperes
        """
        sigma = self.total_noise_std(I_ph, bandwidth, rx_area_cm2, ina_gain,
                                      include_adc, include_processing)
        return np.random.normal(0, sigma, length)
    
    def calculate_snr(self, I_signal_pp, I_ph, bandwidth, rx_area_cm2=9.0,
                      include_adc=True, include_processing=True):
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            I_signal_pp (float): Peak-to-peak signal current in Amperes
            I_ph (float or ndarray): Average photocurrent
            bandwidth (float): Noise bandwidth
            rx_area_cm2 (float): Receiver area
            include_adc (bool): Include ADC noise
            include_processing (bool): Include processing noise
            
        Returns:
            float: SNR in dB
        """
        sigma_noise = self.total_noise_std(I_ph, bandwidth, rx_area_cm2,
                                            include_adc=include_adc,
                                            include_processing=include_processing)
        
        # Signal power (RMS of AC component)
        I_signal_rms = I_signal_pp / (2 * np.sqrt(2))  # For square wave
        
        # SNR
        snr_linear = (I_signal_rms / sigma_noise) ** 2
        snr_db = 10 * np.log10(snr_linear + 1e-12)
        
        return snr_db


# ========== TESTS ==========

def test_noise_model():
    """Unit test for noise model — validates 6-source model."""
    
    print("\n" + "="*60)
    print("NOISE MODEL UNIT TEST (6-Source)")
    print("="*60)
    
    # --- Test 1: Basic 4-source model (backward compatible) ---
    print("\n[Test 1] Basic 4-source noise (backward compatible)...")
    noise = NoiseModel({
        'temperature_K': 300,
        'ambient_lux': 200,
        'adc_bits': 0,  # Disable ADC noise
        'processing_threshold_mv': 0,  # Disable processing noise
    })
    
    I_ph = 400e-6  # 400 µA
    bandwidth = 50e3  # 50 kHz
    sigma_4src = noise.total_noise_std(I_ph, bandwidth, rx_area_cm2=9.0, verbose=True)
    assert sigma_4src > 0, "[ERROR] Zero noise!"
    print(f"  σ_4src = {sigma_4src*1e9:.3f} nA  [OK]")
    
    # --- Test 2: Full 6-source model ---
    print("\n[Test 2] Full 6-source noise model...")
    noise6 = NoiseModel({
        'temperature_K': 300,
        'ambient_lux': 200,
        'load_resistance_ohm': 220,  # Correa: R_load = 220Ω
        'adc_bits': 12,
        'adc_vref': 3.3,
        'amp_gain': 11.0,  # Correa circuit: 1 + 10k/1k
        'processing_threshold_mv': 10.0,
    })
    
    sigma_6src = noise6.total_noise_std(I_ph, bandwidth, rx_area_cm2=9.0, verbose=True)
    assert sigma_6src > sigma_4src, \
        f"[ERROR] 6-source ({sigma_6src:.3e}) should be > 4-source ({sigma_4src:.3e})"
    print(f"  σ_6src = {sigma_6src*1e9:.3f} nA  (increase: {sigma_6src/sigma_4src:.2f}×)  [OK]")
    
    # --- Test 3: ADC noise scales with resolution ---
    print("\n[Test 3] ADC noise vs resolution...")
    for bits in [8, 10, 12, 14, 16]:
        nm = NoiseModel({'adc_bits': bits, 'load_resistance_ohm': 220, 'amp_gain': 11})
        adc_var = nm.adc_noise_variance()
        print(f"  {bits}-bit ADC: σ_adc = {np.sqrt(adc_var)*1e9:.3f} nA")
    
    # Higher resolution should give lower noise
    nm_8 = NoiseModel({'adc_bits': 8, 'load_resistance_ohm': 220, 'amp_gain': 11})
    nm_16 = NoiseModel({'adc_bits': 16, 'load_resistance_ohm': 220, 'amp_gain': 11})
    assert nm_8.adc_noise_variance() > nm_16.adc_noise_variance(), \
        "[ERROR] 8-bit should be noisier than 16-bit"
    print("  [OK] ADC noise decreases with resolution")
    
    # --- Test 4: Processing noise independence from bandwidth ---
    print("\n[Test 4] Processing noise is bandwidth-independent...")
    nm_proc = NoiseModel({'processing_threshold_mv': 10, 'load_resistance_ohm': 220, 'amp_gain': 11})
    proc_var = nm_proc.processing_noise_variance()
    print(f"  σ_proc = {np.sqrt(proc_var)*1e9:.3f} nA (fixed, not bandwidth-dependent)")
    assert proc_var > 0, "[ERROR] Processing noise is zero"
    print("  [OK]")
    
    # --- Test 5: SNR calculation ---
    print("\n[Test 5] SNR calculation...")
    I_pp = 100e-6  # 100 µA signal swing
    snr_4 = noise.calculate_snr(I_pp, I_ph, bandwidth, include_adc=False, include_processing=False)
    snr_6 = noise6.calculate_snr(I_pp, I_ph, bandwidth)
    print(f"  SNR (4-source): {snr_4:.1f} dB")
    print(f"  SNR (6-source): {snr_6:.1f} dB")
    assert snr_6 <= snr_4, "[ERROR] More noise should reduce SNR"
    print("  [OK] 6-source SNR ≤ 4-source SNR")
    
    # --- Test 6: Noise breakdown dict ---
    print("\n[Test 6] Noise breakdown API...")
    bd = noise6.noise_breakdown(I_ph, bandwidth)
    assert 'shot_A2' in bd and 'adc_A2' in bd and 'processing_A2' in bd
    assert bd['sigma_total_A'] > 0
    print(f"  Keys: {list(bd.keys())}")
    print("  [OK]")
    
    # --- Test 7: Generate noise samples ---
    print("\n[Test 7] Noise sample generation...")
    samples = noise6.generate_noise(10000, I_ph, bandwidth)
    measured_std = np.std(samples)
    expected_std = sigma_6src
    ratio = measured_std / expected_std
    print(f"  Expected σ: {expected_std*1e9:.3f} nA")
    print(f"  Measured σ: {measured_std*1e9:.3f} nA")
    print(f"  Ratio: {ratio:.3f} (expect ~1.0)")
    assert 0.9 < ratio < 1.1, f"[ERROR] Noise std ratio {ratio:.3f} too far from 1.0"
    print("  [OK]")
    
    print("\n" + "="*60)
    print("[OK] ALL NOISE MODEL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_noise_model()
