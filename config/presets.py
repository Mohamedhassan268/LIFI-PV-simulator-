"""
Paper-specific preset configurations for validation runs.

Each preset contains LOCKED parameters that exactly match the paper's experimental setup.
These should be used for paper validation to ensure reproducibility.

Usage:
    from config.presets import KADIRVELU_2021
    params = KADIRVELU_2021.get_params()
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass(frozen=True)  # Immutable to prevent accidental modification
class PaperPreset:
    """
    Base class for paper validation presets.
    
    frozen=True makes this immutable after creation.
    """
    name: str
    paper_ref: str
    description: str
    
    # Core parameters - stored as a tuple of (key, value) pairs for immutability
    _params: tuple = field(default_factory=tuple)
    
    def get_params(self) -> Dict[str, Any]:
        """Return a mutable copy of the preset parameters."""
        return dict(self._params)
    
    def get_receiver_params(self) -> Dict[str, Any]:
        """Return only receiver-related parameters."""
        p = self.get_params()
        return {
            'responsivity': p.get('responsivity_a_per_w', 0.457),
            'capacitance': p.get('junction_capacitance_pf', 798),
            'shunt_resistance': p.get('shunt_resistance_mohm', 138.8),
            'dark_current': p.get('dark_current_na', 1.0),
        }
    
    def get_channel_params(self) -> Dict[str, Any]:
        """Return only channel-related parameters."""
        p = self.get_params()
        return {
            'distance': p.get('distance_m'),
            'beam_angle_half': p.get('led_half_angle_deg', 9),
            'receiver_area': p.get('solar_cell_area_cm2', 9.0),
        }


# =============================================================================
# KADIRVELU ET AL. 2021 PRESET
# =============================================================================
# Paper: "A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell"
# IEEE Transactions on Green Communications and Networking, Vol. 5, No. 4, Dec 2021
# DOI: 10.1109/TGCN.2021.3087008

KADIRVELU_2021 = PaperPreset(
    name="Kadirvelu_2021",
    paper_ref="IEEE TGCN 2021, DOI: 10.1109/TGCN.2021.3087008",
    description="GaAs solar cell with DC-DC converter for simultaneous data and power",
    _params=tuple({
        # ===== CRITICAL: LOCKED GEOMETRY =====
        'distance_m': 0.325,              # 32.5 cm - DO NOT CHANGE
        
        # LED/Transmitter
        'radiated_power_mw': 9.3,
        'led_half_angle_deg': 9,          # Fraen 9° lens
        
        # GaAs Solar Cell
        'solar_cell_area_cm2': 9.0,       # 5cm × 1.8cm
        'responsivity_a_per_w': 0.457,
        'n_cells_module': 13,             # 13 cells in series
        
        # ===== CRITICAL: PAPER SPEC (Fig. 6, Section III-B) =====
        'shunt_resistance_mohm': 0.1388,      # 0.1388 MΩ = 138.8 kΩ
        'shunt_resistance_ohm': 138800.0,     # 138.8 kΩ in Ohms
        'junction_capacitance_pf': 798000.0,  # 798 nF = 798,000 pF
        'junction_capacitance_f': 798e-9,     # 798 nF in Farads
        
        # Receiver Chain
        'rsense_ohm': 0.02,               # 20 mΩ (Typical current shunt)
                                          # Gain = 0.02 * 100 = 2 V/A = 6 dB (Matches Fig 13 target)
        'ina_gain': 100,
        'bpf_low_hz': 75,                 # Per reference: 1/(2π×33k×64nF)
        'bpf_high_hz': 10000,
        'bpf_order': 2,
        
        # Load
        'rload_ohm': 1360,                # ~1.36 kΩ
        'cload_uf': 10,
        
        # DC-DC Converter Efficiencies (measured)
        'dcdc_efficiency_50khz': 0.67,
        'dcdc_efficiency_100khz': 0.564,
        'dcdc_efficiency_200khz': 0.42,
        
        # Target outputs for validation
        'target_harvested_power_uw': 223,
        'target_ber_at_2p5kbps_50pct': 1.008e-3,
    }.items())
)


# =============================================================================
# CORREA-MORALES ET AL. 2025 PRESET
# =============================================================================
# Paper: "Experimental design and performance evaluation of a solar panel-based 
#         visible light communication system for greenhouse applications"
# Scientific Reports, Vol. 15, Article 4518 (2025)
# DOI: 10.1038/s41598-025-29067-2

CORREA_2025_GREENHOUSE = PaperPreset(
    name="Correa_2025_Greenhouse",
    paper_ref="Scientific Reports 2025, DOI: 10.1038/s41598-025-29067-2",
    description="Solar panel VLC in greenhouse with PWM-ASK modulation",
    _params=tuple({
        # Transmitter
        'tx_power_electrical_w': 30.0,
        'led_efficiency': 0.10,
        
        # Modulation (PWM-ASK)
        'modulation': 'pwm_ask',
        'pwm_freq_hz': 10,
        'ask_carrier_hz': 10000,
        
        # Solar Panel Receiver
        'load_resistance_ohm': 220,
        'panel_capacitance_nf': 50.0,
        'responsivity_a_per_w': 0.5,
        'panel_area_cm2': 66.0,
        
        # Environment
        'humidity_model': 'beer_lambert',
        'distance_range_m': (0.3, 1.3),
        'humidity_range': (0.3, 0.8),  # 30% - 80% RH
    }.items())
)


# =============================================================================
# SARWAR ET AL. 2017 PRESET
# =============================================================================
# Paper: "Visible Light Communication Using a Solar-Panel Receiver"
# 16th International Conference on Optical Communications and Networks (ICOCN)
# DOI: 10.1109/ICOCN.2017.8121295

SARWAR_2017_OFDM = PaperPreset(
    name="Sarwar_2017_OFDM",
    paper_ref="ICOCN 2017, DOI: 10.1109/ICOCN.2017.8121295",
    description="High-speed 15 Mbps OFDM with silicon solar panel",
    _params=tuple({
        # Geometry
        'distance_m': 2.0,
        
        # Transmitter
        'led_power_w': 3.0,              # 3W Blue LED
        
        # HIGH-SPEED Solar Panel (reduced capacitance)
        'panel_area_cm2': 7.5,           # 25mm × 30mm
        'junction_capacitance_pf': 100,  # ~100 pF for high bandwidth
        'junction_capacitance_f': 100e-12,
        
        # OFDM Parameters
        'modulation': 'ofdm_16qam',
        'nfft': 64,
        'cp_length': 16,
        'target_data_rate_mbps': 15.03,
        'target_ber': 1.6883e-3,
    }.items())
)


# =============================================================================
# XU ET AL. 2024 (SUNLIGHT-DUO) PRESET
# =============================================================================
# Paper: "Sunlight-Duo: Exploiting Sunlight for Simultaneous Energy Harvesting & Communication"
# EWSN 2024, Pages 254-265

XU_2024_SUNLIGHT_DUO = PaperPreset(
    name="Xu_2024_Sunlight_Duo",
    paper_ref="EWSN 2024",
    description="Reconfigurable solar cell array with BFSK modulation",
    _params=tuple({
        # Array Configuration
        'num_cells': 16,
        'cell_area_cm2': 3.5,
        'configurations': ['2s-8p', '4s-4p', '8s-2p'],
        
        # Modulation (BFSK via LC shutter)
        'modulation': 'bfsk',
        'fsk_f0_hz': 1600,
        'fsk_f1_hz': 2000,
        'lc_rise_time_s': 1.34e-3,
        'lc_fall_time_s': 0.15e-3,
        
        # Data Rates
        'data_rate_standard_bps': 400,
        'data_rate_max_bps': 1200,
        
        # Energy Storage
        'supercap_f': 0.47,
    }.items())
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_preset(name: str) -> Optional[PaperPreset]:
    """
    Get a preset by name (case-insensitive).
    
    Args:
        name: Preset name like 'kadirvelu_2021' or 'KADIRVELU'
        
    Returns:
        PaperPreset or None if not found
    """
    presets = {
        'kadirvelu': KADIRVELU_2021,
        'kadirvelu_2021': KADIRVELU_2021,
        'correa': CORREA_2025_GREENHOUSE,
        'correa_2025': CORREA_2025_GREENHOUSE,
        'greenhouse': CORREA_2025_GREENHOUSE,
        'sarwar': SARWAR_2017_OFDM,
        'sarwar_2017': SARWAR_2017_OFDM,
        'xu': XU_2024_SUNLIGHT_DUO,
        'xu_2024': XU_2024_SUNLIGHT_DUO,
        'sunlight_duo': XU_2024_SUNLIGHT_DUO,
    }
    return presets.get(name.lower().replace('-', '_').replace(' ', '_'))


def list_presets() -> list:
    """Return list of available preset names."""
    return ['kadirvelu_2021', 'correa_2025', 'sarwar_2017', 'xu_2024']


if __name__ == "__main__":
    # Quick validation
    print("=== Paper Presets ===")
    for name in list_presets():
        preset = get_preset(name)
        print(f"\n{preset.name}:")
        print(f"  Ref: {preset.paper_ref}")
        params = preset.get_params()
        if 'distance_m' in params:
            print(f"  Distance: {params['distance_m']} m")
        if 'shunt_resistance_ohm' in params:
            print(f"  R_sh: {params['shunt_resistance_ohm']/1e6:.1f} MΩ")
