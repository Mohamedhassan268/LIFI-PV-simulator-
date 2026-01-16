# sensitivity_analysis.py
"""
Layer-by-Layer Sensitivity Analysis for Li-Fi + PV Simulator.

Analyzes parameter sensitivity across all 4 layers:
1. Transmitter (LED + OOK)
2. Channel (Lambertian + noise)
3. Receiver (PV cell + ODE)
4. DC-DC Converter (Boost)

Outputs:
- Sensitivity indices per parameter
- Tornado charts ranking parameter importance
- Layer-specific recommendations
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Simulator modules
from simulator.transmitter import Transmitter
from simulator.channel import OpticalChannel
from simulator.receiver import PVReceiver
from simulator.dc_dc_converter import DCDCConverter
from utils.constants import PAPER_VALIDATION_CONFIG


# ========== BASELINE CONFIGURATION ==========
BASELINE = {
    # Transmitter
    'dc_bias': 100,           # mA
    'modulation_depth': 0.5,  # 0-1
    'led_efficiency': 0.08,   # W/A
    
    # Channel
    'distance': 0.325,        # m
    'beam_angle_half': 9,     # deg
    'receiver_area': 9.0,     # cm²
    
    # Receiver
    'responsivity': 0.457,    # A/W
    'capacitance': 798,       # pF
    'shunt_resistance': 0.1388,  # MΩ
    'dark_current': 1.0,      # nA
    'temperature': 300,       # K
    
    # DC-DC
    'fsw_khz': 100,           # kHz
    'duty_cycle': 0.5,        # 0-1
    'v_diode_drop': 0.3,      # V
    'r_on_mohm': 100,         # mΩ
}

# ========== PARAMETER SWEEP RANGES ==========
PARAM_RANGES = {
    # Layer 1: Transmitter
    'dc_bias': np.linspace(50, 150, 11),
    'modulation_depth': np.linspace(0.1, 1.0, 10),
    'led_efficiency': np.linspace(0.05, 0.15, 11),
    
    # Layer 2: Channel
    'distance': np.linspace(0.1, 1.0, 10),
    'beam_angle_half': np.linspace(5, 60, 12),
    'receiver_area': np.linspace(1, 20, 10),
    
    # Layer 3: Receiver
    'responsivity': np.linspace(0.3, 0.6, 7),
    'capacitance': np.linspace(100, 2000, 10),
    'shunt_resistance': np.linspace(0.05, 0.5, 10),
    'dark_current': np.linspace(0.1, 10, 10),
    'temperature': np.linspace(280, 330, 6),
    
    # Layer 4: DC-DC
    'fsw_khz': np.array([50, 75, 100, 150, 200]),
    'duty_cycle': np.linspace(0.05, 0.6, 12),
    'v_diode_drop': np.linspace(0.1, 0.5, 5),
    'r_on_mohm': np.linspace(50, 200, 6),
}

# Map parameters to layers
LAYER_PARAMS = {
    'transmitter': ['dc_bias', 'modulation_depth', 'led_efficiency'],
    'channel': ['distance', 'beam_angle_half', 'receiver_area'],
    'receiver': ['responsivity', 'capacitance', 'shunt_resistance', 'dark_current', 'temperature'],
    'dc_dc': ['fsw_khz', 'duty_cycle', 'v_diode_drop', 'r_on_mohm'],
}


def run_simulation(config):
    """
    Run end-to-end simulation with given config.
    
    Returns dict with key output metrics.
    """
    # Simulation setup
    n_bits = 100
    sps = 50
    data_rate = 10000  # 10 kbps
    fs = data_rate * sps
    duration = n_bits / data_rate
    t = np.arange(0, duration, 1/fs)
    bits = np.random.randint(0, 2, n_bits)
    
    try:
        # Layer 1: Transmitter
        tx = Transmitter({
            'dc_bias': config['dc_bias'],
            'modulation_depth': config['modulation_depth'],
            'led_efficiency': config['led_efficiency'],
        })
        P_tx = tx.modulate(bits, t)
        
        # Layer 2: Channel
        ch = OpticalChannel({
            'distance': config['distance'],
            'beam_angle_half': config['beam_angle_half'],
            'receiver_area': config['receiver_area'],
        })
        P_rx = ch.propagate(P_tx, t)
        
        # Layer 3: Receiver
        rx = PVReceiver({
            'responsivity': config['responsivity'],
            'capacitance': config['capacitance'],
            'shunt_resistance': config['shunt_resistance'],
            'dark_current': config['dark_current'],
            'temperature': config['temperature'],
        })
        I_ph = rx.optical_to_current(P_rx)
        V_pv = rx.solve_pv_circuit(I_ph, t)
        
        # ========== MODULE SCALING ==========
        # Paper uses 13-cell GaAs module (cells in series)
        # V_module = V_cell × n_cells
        from utils.constants import PAPER_VALIDATION_CONFIG
        n_cells = PAPER_VALIDATION_CONFIG.get('n_cells_module', 13)
        V_pv_avg = np.mean(V_pv[len(V_pv)//2:])  # Steady-state single cell
        V_module = V_pv_avg * n_cells            # 13-cell module voltage
        
        # Layer 4: DC-DC
        dcdc = DCDCConverter({
            'fsw_khz': config['fsw_khz'],
            'duty_cycle': config['duty_cycle'],
            'v_diode_drop': config['v_diode_drop'],
            'r_on_mohm': config['r_on_mohm'],
        })
        I_pv_avg = np.mean(I_ph)
        dcdc_out = dcdc.calculate_output(V_module, I_pv_avg)
        
        return {
            'P_tx_mW': np.mean(P_tx) * 1e3,
            'P_rx_uW': np.mean(P_rx) * 1e6,
            'I_ph_uA': np.mean(I_ph) * 1e6,
            'V_pv_mV': V_pv_avg * 1e3,
            'V_module_mV': V_module * 1e3,
            'V_out': dcdc_out['V_out'],
            'P_out_uW': dcdc_out['P_out'] * 1e6,
            'efficiency': dcdc_out['efficiency'],
            'mode': dcdc_out.get('mode', 'N/A'),
            'path_loss_dB': -10 * np.log10(np.mean(P_rx) / np.mean(P_tx) + 1e-12),
            'success': True,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def calculate_sensitivity(param_name, sweep_values, baseline_config, output_metric='V_out'):
    """
    Calculate sensitivity index for one parameter.
    
    S = (ΔY/Y) / (ΔX/X) - normalized sensitivity
    """
    results = []
    
    for val in sweep_values:
        config = baseline_config.copy()
        config[param_name] = val
        output = run_simulation(config)
        if output['success']:
            results.append({
                'param_value': val,
                'output': output[output_metric]
            })
    
    if len(results) < 2:
        return {'sensitivity': 0, 'values': [], 'outputs': []}
    
    # Calculate normalized sensitivity at baseline
    baseline_idx = len(results) // 2
    Y_base = results[baseline_idx]['output']
    X_base = results[baseline_idx]['param_value']
    
    # Use finite difference
    if baseline_idx > 0:
        dY = results[baseline_idx + 1]['output'] - results[baseline_idx - 1]['output']
        dX = results[baseline_idx + 1]['param_value'] - results[baseline_idx - 1]['param_value']
    else:
        dY = results[1]['output'] - results[0]['output']
        dX = results[1]['param_value'] - results[0]['param_value']
    
    if X_base != 0 and Y_base != 0 and dX != 0:
        S = (dY / Y_base) / (dX / X_base)
    else:
        S = 0
    
    return {
        'sensitivity': S,
        'values': [r['param_value'] for r in results],
        'outputs': [r['output'] for r in results],
    }


def analyze_layer(layer_name, output_metric='V_out', verbose=True):
    """
    Run sensitivity analysis for all parameters in a layer.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  LAYER SENSITIVITY: {layer_name.upper()}")
        print(f"  Output Metric: {output_metric}")
        print(f"{'='*60}")
    
    params = LAYER_PARAMS[layer_name]
    sensitivities = {}
    
    for param in tqdm(params, desc=f"Analyzing {layer_name}"):
        sweep = PARAM_RANGES[param]
        result = calculate_sensitivity(param, sweep, BASELINE, output_metric)
        sensitivities[param] = result
        
        if verbose:
            print(f"  {param}: S = {result['sensitivity']:.3f}")
    
    return sensitivities


def generate_tornado_chart(sensitivities, layer_name, output_dir='outputs/sensitivity'):
    """
    Create tornado chart showing parameter importance.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by absolute sensitivity
    sorted_params = sorted(sensitivities.items(), 
                          key=lambda x: abs(x[1]['sensitivity']), 
                          reverse=True)
    
    params = [p[0] for p in sorted_params]
    values = [p[1]['sensitivity'] for p in sorted_params]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red' if v > 0 else 'blue' for v in values]
    y_pos = np.arange(len(params))
    
    ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.set_xlabel('Normalized Sensitivity Index', fontsize=12)
    ax.set_title(f'{layer_name.upper()} Layer - Parameter Sensitivity', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + 0.01 if v >= 0 else v - 0.15, i, f'{v:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f'{layer_name}_tornado.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {path}")
    return path


def generate_sweep_plots(sensitivities, layer_name, output_dir='outputs/sensitivity'):
    """
    Create parameter sweep plots for a layer.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_params = len(sensitivities)
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 4))
    
    if n_params == 1:
        axes = [axes]
    
    for ax, (param, data) in zip(axes, sensitivities.items()):
        if data['values']:
            ax.plot(data['values'], data['outputs'], 'b-o', linewidth=2, markersize=6)
            ax.set_xlabel(param, fontsize=11)
            ax.set_ylabel('V_out (V)', fontsize=11)
            ax.set_title(f'S = {data["sensitivity"]:.2f}', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{layer_name.upper()} Layer - Parameter Sweeps', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    path = os.path.join(output_dir, f'{layer_name}_sweeps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {path}")
    return path


def run_full_sensitivity_analysis(output_dir=None):
    """
    Main entry point - analyze all layers.
    
    Args:
        output_dir: Optional output directory. If None, uses output_manager.
    """
    # Use output manager for organized folders
    if output_dir is None:
        from utils.output_manager import get_sensitivity_dir
        output_dir = get_sensitivity_dir()
    
    print("\n" + "="*70)
    print("    FULL LAYER SENSITIVITY ANALYSIS")
    print("    Li-Fi + PV Simulator Optimization")
    print("="*70)
    
    # Suppress component init messages
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    # Test baseline simulation first
    sys.stdout = old_stdout
    print("\nTesting baseline configuration...")
    baseline_result = run_simulation(BASELINE)
    if not baseline_result['success']:
        print(f"ERROR: Baseline simulation failed: {baseline_result.get('error', 'Unknown')}")
        return None
    
    print(f"  Baseline V_out: {baseline_result['V_out']:.3f} V")
    print(f"  Baseline P_out: {baseline_result['P_out_uW']:.2f} µW")
    
    all_sensitivities = {}
    
    for layer in ['transmitter', 'channel', 'receiver', 'dc_dc']:
        sys.stdout = StringIO()  # Suppress layer output
        sensitivities = analyze_layer(layer, verbose=False)
        sys.stdout = old_stdout
        
        all_sensitivities[layer] = sensitivities
        
        # Generate visualizations
        generate_tornado_chart(sensitivities, layer, output_dir)
        generate_sweep_plots(sensitivities, layer, output_dir)
        
        # Print summary
        print(f"\n  {layer.upper()} Layer Results:")
        sorted_s = sorted(sensitivities.items(), 
                         key=lambda x: abs(x[1]['sensitivity']), 
                         reverse=True)
        for param, data in sorted_s:
            print(f"    {param}: S = {data['sensitivity']:.3f}")
    
    # Final report
    print("\n" + "="*70)
    print("    SENSITIVITY RANKING (All Layers)")
    print("="*70)
    
    all_params = []
    for layer, params in all_sensitivities.items():
        for param, data in params.items():
            all_params.append((f"{layer}.{param}", data['sensitivity']))
    
    all_params.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\n  Top 10 Most Sensitive Parameters:")
    for i, (param, s) in enumerate(all_params[:10], 1):
        print(f"    {i:2d}. {param}: S = {s:+.3f}")
    
    print("\n" + "="*70)
    print(f"    Analysis complete. Plots saved to: {output_dir}")
    print("="*70)
    
    return all_sensitivities


if __name__ == "__main__":
    results = run_full_sensitivity_analysis()
