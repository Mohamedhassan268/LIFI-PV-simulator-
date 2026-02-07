# simulator/spectral.py
"""
Spectral Sensing and Wavelength Detection.

Provides models for optical wavelength identification using
multi-channel color sensors (e.g., RGB photodiodes).

Components:
  - SpectralResponseModel: Base class for spectral response curves
  - GaussianSpectralResponse: Gaussian-shaped response per channel
  - ColorSensor: N-channel sensor model (RGB, RGBW, etc.)
  - WavelengthClassifier: Dominance-based wavelength classification

Physics extracted from:
  - Oliveira 2024: Kingbright APS5130PD7C-P22 RGB sensor
    (3 channels, Gaussian response, binary LUT output)

Designed for generality — any number of channels, any response shape,
any classification rule. Oliveira's sensor is one configuration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


# =============================================================================
# SPECTRAL RESPONSE MODELS
# =============================================================================

class GaussianSpectralResponse:
    """
    Gaussian spectral response for a single sensor channel.
    
    R(λ) = peak_response * exp(-(λ - peak_nm)² / (2 * sigma_nm²))
    
    Args:
        peak_nm: Peak sensitivity wavelength (nm)
        sigma_nm: Gaussian width (nm), controls bandwidth
        peak_response: Maximum response at peak (dimensionless, default 1.0)
        label: Human-readable channel name (e.g., 'red', 'blue')
    """
    
    def __init__(self, peak_nm: float, sigma_nm: float,
                 peak_response: float = 1.0, label: str = ''):
        self.peak_nm = peak_nm
        self.sigma_nm = sigma_nm
        self.peak_response = peak_response
        self.label = label
    
    def __call__(self, wavelength_nm):
        """Evaluate response at given wavelength(s)."""
        wl = np.asarray(wavelength_nm, dtype=float)
        return self.peak_response * np.exp(
            -((wl - self.peak_nm) ** 2) / (2 * self.sigma_nm ** 2)
        )
    
    def bandwidth_fwhm(self):
        """Full width at half maximum (nm)."""
        return 2.355 * self.sigma_nm
    
    def __repr__(self):
        return (f"GaussianSpectralResponse("
                f"peak={self.peak_nm}nm, σ={self.sigma_nm}nm, "
                f"label='{self.label}')")


# =============================================================================
# COLOR SENSOR
# =============================================================================

class ColorSensor:
    """
    Multi-channel optical color sensor.
    
    Models a sensor with N spectral response channels. Each channel has
    its own response curve. Given an input wavelength and intensity,
    returns the response of each channel.
    
    Args:
        channels: List of (label, response_callable) pairs.
                  Each response_callable takes wavelength_nm → response.
    """
    
    def __init__(self, channels: List[Tuple[str, Callable]]):
        self.channels = channels
        self.n_channels = len(channels)
        self.labels = [ch[0] for ch in channels]
    
    def measure(self, wavelength_nm: float, intensity: float = 1.0) -> Dict[str, float]:
        """
        Measure sensor response to a monochromatic input.
        
        Args:
            wavelength_nm: Input wavelength (nm)
            intensity: Optical intensity (arbitrary units, default 1.0)
            
        Returns:
            Dict mapping channel label → response value
        """
        return {
            label: float(response_fn(wavelength_nm) * intensity)
            for label, response_fn in self.channels
        }
    
    def measure_spectrum(self, wavelengths_nm, spectrum_intensity) -> Dict[str, float]:
        """
        Measure sensor response to a broadband spectrum.
        
        Integrates each channel's response over the input spectrum.
        
        Args:
            wavelengths_nm: Array of wavelengths (nm)
            spectrum_intensity: Array of intensities at each wavelength
            
        Returns:
            Dict mapping channel label → integrated response
        """
        wl = np.asarray(wavelengths_nm, dtype=float)
        spec = np.asarray(spectrum_intensity, dtype=float)
        
        result = {}
        for label, response_fn in self.channels:
            channel_response = response_fn(wl)
            # Trapezoidal integration over spectrum
            result[label] = float(np.trapezoid(channel_response * spec, wl))
        
        return result
    
    def __repr__(self):
        return f"ColorSensor(channels={self.labels})"


# =============================================================================
# WAVELENGTH CLASSIFIER
# =============================================================================

class WavelengthClassifier:
    """
    Dominance-based wavelength classifier with configurable LUT output.
    
    Given a ColorSensor measurement, determines which channel dominates
    and maps to a binary or categorical output via a lookup table.
    
    Args:
        sensor: ColorSensor instance
        threshold_ratio: Minimum ratio of dominant channel to all others
                         for classification to succeed (default 1.5)
        lut: Dict mapping channel label → output value.
             Default output for ambiguous cases is configurable.
        default_output: Output when no channel dominates
    """
    
    def __init__(self, sensor: ColorSensor,
                 threshold_ratio: float = 1.5,
                 lut: Optional[Dict[str, any]] = None,
                 default_output=None):
        self.sensor = sensor
        self.threshold_ratio = threshold_ratio
        self.default_output = default_output
        
        # Default LUT: each channel maps to its own label
        if lut is None:
            self.lut = {label: label for label in sensor.labels}
        else:
            self.lut = lut
    
    def classify(self, wavelength_nm: float, intensity: float = 1.0) -> Dict:
        """
        Classify a monochromatic input by wavelength.
        
        Args:
            wavelength_nm: Input wavelength (nm)
            intensity: Signal intensity
            
        Returns:
            Dict with:
                'wavelength_nm': input wavelength
                'responses': per-channel response values
                'dominant_channel': label of dominant channel (or 'none')
                'is_dominant': bool — True if one channel clearly dominates
                'output': LUT output value for detected channel
        """
        responses = self.sensor.measure(wavelength_nm, intensity)
        
        # Find dominant channel
        max_label = max(responses, key=responses.get)
        max_value = responses[max_label]
        
        # Check dominance: max must exceed threshold_ratio * every other
        others = [v for k, v in responses.items() if k != max_label]
        is_dominant = all(
            max_value > self.threshold_ratio * v for v in others
        ) if others else True
        
        if is_dominant:
            dominant = max_label
            output = self.lut.get(max_label, self.default_output)
        else:
            dominant = 'none'
            output = self.default_output
        
        return {
            'wavelength_nm': wavelength_nm,
            'responses': responses,
            'dominant_channel': dominant,
            'is_dominant': is_dominant,
            'output': output,
        }
    
    def classify_batch(self, wavelengths_nm, intensity: float = 1.0) -> List[Dict]:
        """Classify multiple wavelengths. Returns list of classify() results."""
        return [self.classify(wl, intensity) for wl in wavelengths_nm]


# =============================================================================
# PRE-BUILT SENSOR CONFIGURATIONS
# =============================================================================

def kingbright_aps5130():
    """
    Create the Kingbright APS5130PD7C-P22 RGB sensor from Oliveira 2024.
    
    3-channel RGB sensor with Gaussian spectral responses.
    Peak wavelengths and widths from Oliveira's model.
    
    Returns:
        (sensor, classifier): Tuple of ColorSensor and WavelengthClassifier
                              configured with Oliveira's binary LUT.
    """
    # Channel responses (matching Oliveira 2024 parameters)
    blue = GaussianSpectralResponse(peak_nm=470, sigma_nm=30, label='blue')
    green = GaussianSpectralResponse(peak_nm=550, sigma_nm=40, label='green')
    red = GaussianSpectralResponse(peak_nm=620, sigma_nm=50, label='red')
    
    sensor = ColorSensor([
        ('blue', blue),
        ('green', green),
        ('red', red),
    ])
    
    # Oliveira's binary LUT (Table in paper)
    # red → (0, 1), blue → (1, 0), green → (1, 1), ambiguous → (0, 0)
    lut = {
        'red': (0, 1),
        'blue': (1, 0),
        'green': (1, 1),
    }
    
    classifier = WavelengthClassifier(
        sensor=sensor,
        threshold_ratio=1.5,
        lut=lut,
        default_output=(0, 0),
    )
    
    return sensor, classifier


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Spectral Sensing Module Tests")
    print("=" * 60)
    
    # Test 1: Kingbright sensor from Oliveira 2024
    sensor, classifier = kingbright_aps5130()
    print(f"\n1. Sensor: {sensor}")
    
    # Test 2: Known wavelengths from paper
    test_wavelengths = [
        (450, 'blue', (1, 0)),
        (530, 'green', (1, 1)),
        (658, 'red', (0, 1)),
    ]
    
    print("\n2. Wavelength classification (Oliveira validation):")
    all_pass = True
    for wl, expected_color, expected_output in test_wavelengths:
        result = classifier.classify(wl)
        match = result['output'] == expected_output
        all_pass = all_pass and match
        status = "PASS" if match else "FAIL"
        print(f"   {wl}nm -> dominant={result['dominant_channel']}, "
              f"output={result['output']} [{status}]")
    
    # Test 3: Ambiguous wavelength (between channels)
    result_ambiguous = classifier.classify(510)  # Between blue and green
    print(f"\n3. Ambiguous 510nm -> dominant={result_ambiguous['dominant_channel']}, "
          f"output={result_ambiguous['output']}")
    
    # Test 4: Channel responses across visible spectrum
    print("\n4. Response sweep (400-700nm):")
    wavelengths = [400, 450, 500, 550, 600, 650, 700]
    for wl in wavelengths:
        resp = sensor.measure(wl)
        bar = {k: f"{v:.3f}" for k, v in resp.items()}
        print(f"   {wl}nm: {bar}")
    
    # Test 5: Broadband spectrum measurement
    wl_range = np.linspace(400, 700, 100)
    flat_spectrum = np.ones_like(wl_range)
    broadband = sensor.measure_spectrum(wl_range, flat_spectrum)
    print(f"\n5. Flat broadband response: { {k: f'{v:.1f}' for k, v in broadband.items()} }")
    
    print(f"\n{'TASK 3 PASSED' if all_pass else 'TASK 3 FAILED'}")
