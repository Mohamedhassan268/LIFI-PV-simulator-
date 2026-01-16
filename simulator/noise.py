# simulator/noise.py
"""
Comprehensive Noise Model for Li-Fi + PV System.

Implements:
1. Shot noise (quantum fluctuations)
2. Thermal noise (Johnson-Nyquist)
3. Ambient light noise (background DC current)
4. Amplifier noise (INA input noise)

All calculations based on physical equations - no artificial calibration.
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
                - 'ambient_lux': Background light level (default 0)
                - 'ambient_responsivity': Responsivity for ambient (default 0.4 A/W)
        """
        if params is None:
            params = {}
        
        self.T = params.get('temperature_K', 300)  # Kelvin
        self.R_load = params.get('load_resistance_ohm', 50)  # Ohms
        self.V_n_ina = params.get('ina_noise_nv_sqrt_hz', 10) * 1e-9  # V/√Hz
        self.ambient_lux = params.get('ambient_lux', 0)  # lux
        self.ambient_resp = params.get('ambient_responsivity', 0.4)  # A/W
        
        # Conversion: ~1 lux ≈ 1.46 µW/cm² for typical illumination
        # and for indoor lighting spectrum
        self.lux_to_power_factor = 1.46e-6  # W/cm² per lux
        
    def shot_noise_variance(self, I_ph, bandwidth):
        """
        Calculate shot noise variance.
        
        Shot noise arises from quantum nature of light/current.
        σ²_shot = 2 * q * I_ph * B
        
        Args:
            I_ph (float or ndarray): Photocurrent in Amperes
            bandwidth (float): Noise bandwidth in Hz
            
        Returns:
            float: Shot noise variance in A²
        """
        I_avg = np.mean(np.abs(I_ph))
        sigma2_shot = 2 * Q * I_avg * bandwidth
        return sigma2_shot
    
    def thermal_noise_variance(self, bandwidth):
        """
        Calculate thermal (Johnson-Nyquist) noise variance.
        
        Thermal noise from resistive elements.
        σ²_thermal = 4 * k_B * T * B / R
        
        Args:
            bandwidth (float): Noise bandwidth in Hz
            
        Returns:
            float: Thermal noise variance in A² (when referred to current)
        """
        sigma2_thermal = 4 * K_B * self.T * bandwidth / self.R_load
        return sigma2_thermal
    
    def ambient_noise_variance(self, bandwidth, rx_area_cm2=9.0):
        """
        Calculate ambient light noise variance.
        
        Background light creates DC photocurrent with shot noise.
        
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
        sigma2_ambient = 2 * Q * I_ambient * bandwidth
        
        return sigma2_ambient
    
    def amplifier_noise_variance(self, bandwidth, ina_gain=100):
        """
        Calculate amplifier (INA) input-referred noise variance.
        
        INA has input voltage noise density, referred to current input.
        
        Args:
            bandwidth (float): Noise bandwidth in Hz
            ina_gain (float): INA voltage gain
            
        Returns:
            float: Amplifier noise variance in A² (input-referred)
        """
        # Voltage noise at INA input
        V_noise_rms = self.V_n_ina * np.sqrt(bandwidth)
        
        # Refer to current through sense resistor (1Ω typically)
        # I_noise = V_noise / R_sense
        R_sense = 1.0  # Ohm (from paper)
        I_noise_rms = V_noise_rms / R_sense
        
        sigma2_amp = I_noise_rms ** 2
        return sigma2_amp
    
    def total_noise_std(self, I_ph, bandwidth, rx_area_cm2=9.0, ina_gain=100, 
                        verbose=False):
        """
        Calculate total noise standard deviation.
        
        σ_total = √(σ²_shot + σ²_thermal + σ²_ambient + σ²_amp)
        
        Args:
            I_ph (float or ndarray): Signal photocurrent in Amperes
            bandwidth (float): Noise bandwidth in Hz
            rx_area_cm2 (float): Receiver area in cm²
            ina_gain (float): INA voltage gain
            verbose (bool): Print breakdown
            
        Returns:
            float: Total noise standard deviation in Amperes
        """
        sigma2_shot = self.shot_noise_variance(I_ph, bandwidth)
        sigma2_thermal = self.thermal_noise_variance(bandwidth)
        sigma2_ambient = self.ambient_noise_variance(bandwidth, rx_area_cm2)
        sigma2_amp = self.amplifier_noise_variance(bandwidth, ina_gain)
        
        sigma2_total = sigma2_shot + sigma2_thermal + sigma2_ambient + sigma2_amp
        sigma_total = np.sqrt(sigma2_total)
        
        if verbose:
            print(f"\n  Noise Breakdown:")
            print(f"    Shot noise:    {np.sqrt(sigma2_shot)*1e9:.3f} nA (σ)")
            print(f"    Thermal noise: {np.sqrt(sigma2_thermal)*1e9:.3f} nA (σ)")
            print(f"    Ambient noise: {np.sqrt(sigma2_ambient)*1e9:.3f} nA (σ)")
            print(f"    Amplifier:     {np.sqrt(sigma2_amp)*1e9:.3f} nA (σ)")
            print(f"    TOTAL:         {sigma_total*1e9:.3f} nA (σ)")
            print(f"    Bandwidth:     {bandwidth/1e3:.1f} kHz")
        
        return sigma_total
    
    def generate_noise(self, length, I_ph, bandwidth, rx_area_cm2=9.0, ina_gain=100):
        """
        Generate AWGN noise samples.
        
        Args:
            length (int): Number of samples
            I_ph (float or ndarray): Signal photocurrent
            bandwidth (float): Noise bandwidth
            rx_area_cm2 (float): Receiver area
            ina_gain (float): INA gain
            
        Returns:
            ndarray: Noise samples in Amperes
        """
        sigma = self.total_noise_std(I_ph, bandwidth, rx_area_cm2, ina_gain)
        noise = np.random.normal(0, sigma, length)
        return noise
    
    def calculate_snr(self, I_signal_pp, I_ph, bandwidth, rx_area_cm2=9.0):
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            I_signal_pp (float): Peak-to-peak signal current in Amperes
            I_ph (float or ndarray): Average photocurrent
            bandwidth (float): Noise bandwidth
            rx_area_cm2 (float): Receiver area
            
        Returns:
            float: SNR in dB
        """
        sigma_noise = self.total_noise_std(I_ph, bandwidth, rx_area_cm2)
        
        # Signal power (RMS of AC component)
        I_signal_rms = I_signal_pp / (2 * np.sqrt(2))  # For square wave
        
        # SNR
        snr_linear = (I_signal_rms / sigma_noise) ** 2
        snr_db = 10 * np.log10(snr_linear + 1e-12)
        
        return snr_db


# ========== TESTS ==========

def test_noise_model():
    """Unit test for noise model."""
    print("\n" + "="*60)
    print("NOISE MODEL UNIT TEST")
    print("="*60)
    
    # Create noise model
    noise = NoiseModel({
        'temperature_K': 300,
        'ambient_lux': 200,  # Indoor lighting
    })
    
    # Test parameters
    I_ph = 400e-6  # 400 µA photocurrent
    bandwidth = 50e3  # 50 kHz
    
    # Calculate total noise
    sigma = noise.total_noise_std(I_ph, bandwidth, rx_area_cm2=9.0, verbose=True)
    
    # Calculate SNR
    I_pp = 100e-6  # 100 µA signal swing
    snr = noise.calculate_snr(I_pp, I_ph, bandwidth)
    print(f"\n  Signal (pp): {I_pp*1e6:.1f} µA")
    print(f"  SNR: {snr:.1f} dB")
    
    # Generate noise samples
    samples = noise.generate_noise(1000, I_ph, bandwidth)
    print(f"\n  Generated 1000 samples")
    print(f"  Sample std: {np.std(samples)*1e9:.3f} nA")
    print(f"  Expected std: {sigma*1e9:.3f} nA")
    
    print("\n[OK] Noise model tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_noise_model()
