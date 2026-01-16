# utils/paper_configs.py
"""
Multi-Paper Validation Configuration.

Stores parameters and targets from published papers for validation.
Add new papers by extending PAPER_CONFIGS dictionary.
"""

import numpy as np

# ========== PAPER CONFIGURATIONS ==========

PAPER_CONFIGS = {
    # ========== Kadirvelu et al. IEEE TGCN 2021 ==========
    'kadirvelu2021': {
        'title': 'A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell',
        'journal': 'IEEE Transactions on Green Communications and Networking',
        'year': 2021,
        'doi': '10.1109/TGCN.2021.3062404',
        
        # Table I: System Parameters
        'params': {
            # Link Geometry
            'distance_m': 0.325,
            'radiated_power_mw': 9.3,
            'led_half_angle_deg': 9,
            
            # GaAs Solar Cell Module
            'solar_cell_area_cm2': 9.0,
            'responsivity_a_per_w': 0.457,
            'n_cells_module': 13,
            
            # Small-signal Circuit
            'rsh_ohm': 138800.0,
            'cj_pf': 798000.0,
            'cload_uf': 10,
            'rload_ohm': 1360,
            
            # Receiver Chain
            'rsense_ohm': 1.0,
            
            # INA322
            'ina_gain_dc': 1000,      # Open loop ~100dB, closed loop set by resistors effectively 60dB/40dB. 
                                      # Spec says effective 40dB (x100), but circuit has R1=33k, R2=33 -> 1000x w/ 700k GBW.
                                      # We use 1000 to model the physical circuit, allowing GBW to limit it naturaly.
            'ina_gbw_hz': 700000,     # 700 kHz
            
            # Bandpass Filter Components (2 stages)
            # High-pass: 700 Hz
            'bpf_rhp_ohm': 10000.0,
            'bpf_chp_pf': 482.0,      # 10k * 482p -> fc = 33 kHz? (Matches Spec Values)
                                      # Wait, spec says fL=700 Hz. 1/(2pi*10k*22n) = 700. 
                                      # 1/(2pi*10k*482p) = 33,000 Hz.
                                      # Spec contradiction again. 482pF is very small for 700Hz.
                                      # We will use VALUES derived from target frequencies to satisfy "Physics Logic"
            
            # Re-calculating components for target Frequencies:
            # fL = 700 Hz -> C = 1/(2*pi*10k*700) = 22.7 nF
            # fH = 10000 Hz -> C = 1/(2*pi*10k*10k) = 1.6 nF
            
            'bpf_chp_value_f': 22.7e-9,  # Override spec text with physics truth for 700Hz
            'bpf_clp_value_f': 1.6e-9,   # Override spec text with physics truth for 10kHz
            
            'bpf_stages': 2,
            
            # DC-DC Converter Efficiency
            'efficiency': {
                50: 0.67,
                100: 0.564,
                200: 0.42,
            },
        },
        
        # Validation Targets from Figures
        'targets': {
            # Fig 15: V_out vs Duty Cycle
            'fig15_peak_V_50khz': 3.5,
            'fig15_peak_duty_50khz': 0.08,
            'fig15_V_at_50pct_duty': 2.0,
            
            # Fig 18: V_out vs Modulation Depth  
            'fig18_V_at_m10_50khz': 6.0,
            'fig18_V_at_m100_50khz': 4.2,
            
            # Other targets
            'harvested_power_uw': 223,
            'ber_target': 1e-3,
            'ber_data_rate_bps': 2500,
            'ber_mod_depth': 0.5,
        },
        
        # Figures to reproduce
        'figures': ['fig13', 'fig14', 'fig15', 'fig17', 'fig18'],
    },

    # ========== González-Uriarte et al. IEEE LATINCOM 2024 ==========
    'gonzalez2024': {
        'title': 'Design and Implementation of a Low-Cost VLC Photovoltaic Panel-Based Receiver',
        'journal': '2024 IEEE LATINCOM',
        'year': 2024,
        'doi': '10.1109/LATINCOM62985.2024.10770662',
        
        # Inferred Parameters (See Inference Note)
        'params': {
            # System Geometry
            'distance_m': 0.60,
            
            # Poly-Si Panel (11x6 cm)
            'solar_cell_area_cm2': 66.0,
            'responsivity_a_per_w': 0.5,      # Typical for Poly-Si
            'n_cells_module': 1,              # Assumed single module output
            
            # Electrical Model (Inferred from 50kHz BW @ 220 Ohm load)
            # Req = Rload || Rsh = 220 || 500 = 152.8 Ohm
            # fc = 1/(2*pi*Req*Cj) -> Cj = 20.8 nF
            'rsh_ohm': 500.0,
            'cj_pf': 20800.0,                 # 20.8 nF
            'cload_uf': 0,                    # No external load C mentioned
            'rload_ohm': 220,                 # Explicitly stated
            
            # Receiver Chain (Simple resistor load + Amp)
            'rsense_ohm': 220.0,              # The load is the sense resistor
            'ina_gain_dc': 1,                 # Unity buffer or variable gain
            'ina_gbw_hz': 10000000,           # MCP6022 GBW = 10 MHz typically
            
            # Filters (Notch 100Hz, HP 159Hz)
            # We can model these if needed, or focus on raw bandwidth first
            'bpf_stages': 0,
        },
        
        'targets': {
            'bandwidth_hz': 50000,
            'ber_target': 0.0,
            'bit_rate_kbps': 4.8,
            'vout_pp_mv': 20.0, # at 10 kHz
        },
        
        'figures': ['fig_bandwidth', 'fig_ber'],
    },
    
    # ========== Template for Adding New Papers ==========
    # 'author_year': {
    #     'title': 'Paper Title',
    #     'journal': 'Journal Name',
    #     'year': 2024,
    #     'doi': 'xxx',
    #     'params': { ... },
    #     'targets': { ... },
    #     'figures': ['fig1', 'fig2'],
    # },
}


# ========== HELPER FUNCTIONS ==========

def get_paper_config(paper_id: str) -> dict:
    """
    Get configuration for a specific paper.
    
    Args:
        paper_id: Paper identifier (e.g., 'kadirvelu2021')
        
    Returns:
        dict: Paper configuration or None if not found
    """
    return PAPER_CONFIGS.get(paper_id)


def list_available_papers() -> list:
    """List all papers with validation support."""
    papers = []
    for paper_id, config in PAPER_CONFIGS.items():
        papers.append({
            'id': paper_id,
            'title': config['title'],
            'year': config['year'],
            'figures': config['figures'],
        })
    return papers


def get_paper_params(paper_id: str) -> dict:
    """Get simulation parameters for a paper."""
    config = get_paper_config(paper_id)
    return config['params'] if config else {}


def get_paper_targets(paper_id: str) -> dict:
    """Get validation targets for a paper."""
    config = get_paper_config(paper_id)
    return config['targets'] if config else {}


def compare_to_targets(paper_id: str, results: dict) -> dict:
    """
    Compare simulation results to paper targets.
    
    Args:
        paper_id: Paper identifier
        results: Dict of simulation results
        
    Returns:
        dict: Comparison with pass/fail for each target
    """
    targets = get_paper_targets(paper_id)
    comparison = {}
    
    for key, target in targets.items():
        if key in results:
            actual = results[key]
            # Allow 30% tolerance for physics differences
            tolerance = 0.3
            if isinstance(target, (int, float)) and target != 0:
                error = abs(actual - target) / abs(target)
                passed = error < tolerance
            else:
                passed = actual == target
                error = 0
            
            comparison[key] = {
                'target': target,
                'actual': actual,
                'error_pct': error * 100,
                'passed': passed,
            }
    
    return comparison


# ========== TEST ==========

if __name__ == "__main__":
    print("Available Papers for Validation:")
    print("-" * 60)
    for paper in list_available_papers():
        print(f"  {paper['id']}: {paper['title'][:50]}... ({paper['year']})")
        print(f"    Figures: {', '.join(paper['figures'])}")
    
    print("\nKadirvelu 2021 Parameters:")
    params = get_paper_params('kadirvelu2021')
    for key, val in params.items():
        if not isinstance(val, dict):
            print(f"  {key}: {val}")
