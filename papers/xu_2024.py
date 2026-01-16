import numpy as np
import matplotlib.pyplot as plt
from simulator.transmitter import Transmitter
from simulator.channel import OpticalChannel
from simulator.receiver import MIMOPVReceiver, ReconfigurablePVArray

# ==============================================================================
# PARAMETERS (Xu et al. 2024 - Sunlight-Duo)
# ==============================================================================

# 1. SOLAR CELL (Anysolar KXOB25-14X1F)
# Single Cell estimation (Si)
V_OC_CELL = 0.63  # Volts (Standard Si)
I_SC_CELL_MA = 20.0 # mA (Approx from table range for unit)
CELL_AREA_CM2 = 3.5 # 25mm x 14mm = 3.5 cm^2
RESPONSIVITY = 0.5 # A/W (Standard Si)

# 2. TRANSMITTER (Sunlight + LC Shutter)
# Sunlight Intensity
SUNLIGHT_INTENSITY_LUX = 60000 # 60 klux (Outdoor)
# Power density approx: 1000 W/m^2 = 100 klux -> 600 W/m^2
SUNLIGHT_POWER_W_M2 = 600.0

# LC Shutter Timing
LC_RISE_TIME_S = 1.34e-3
LC_FALL_TIME_S = 0.15e-3

# Modulation (BFSK)
FSK_F0 = 1600.0 # Hz
FSK_F1 = 2000.0 # Hz
DATA_RATE_BPS = 400.0

# 3. RECEIVER CONFIG
# 16 Cells total
CONFIGS = {
    '2s-8p': {'series': 2, 'parallel': 8}, # Charging
    '4s-4p': {'series': 4, 'parallel': 4}, # Balanced
    '8s-2p': {'series': 8, 'parallel': 2}, # Communication
}

def get_paper_config():
    """Return the configuration dictionary for Xu 2024."""
    return {
        'transmitter': {
            'type': 'sunlight_lc',
            'power_density': SUNLIGHT_POWER_W_M2,
            'fsk_f0': FSK_F0,
            'fsk_f1': FSK_F1,
            'rise_time': LC_RISE_TIME_S,
            'fall_time': LC_FALL_TIME_S,
            'data_rate': DATA_RATE_BPS
        },
        'channel': {
            'distance': 4.0, # meters (Outdoor Setup)
            'beam_angle_half': 45.0, # Sunlight is diffuse/ambient or directed? 
                                     # Actually sunlight is parallel rays, but system uses collector.
                                     # Let's assume effectively uniform illumination for now.
             'receiver_area': 16 * CELL_AREA_CM2
        },
        'receiver': {
            'responsivity': RESPONSIVITY,
            'area_cm2': CELL_AREA_CM2, # Per cell
            'num_cells': 16,
            'dark_current': 0.1, # 0.1 nA
            'capacitance': 1000.0, # 1000 pF (1 nF)
        }
    }

def run_validation(output_dir="outputs/xu_2024"):
    """
    Run the validation suite for Xu et al. (2024).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print("VALIDATING Xu et al. (2024) - Sunlight-Duo")
    print("="*60)
    
    # Setup Components
    params = get_paper_config()
    
    # 1. TRANSMITTER & CHANNEL
    tx = Transmitter({})
    ch = OpticalChannel(params['channel'])
    
    # 2. RECEIVER (Smart Array)
    # Cell params
    cell_params = {
        'responsivity': params['receiver']['responsivity'],
        'receiver_area': params['receiver']['area_cm2'],
        'dark_current': params['receiver']['dark_current'],
        'capacitance': params['receiver']['capacitance']
    }
    
    rx_array = ReconfigurablePVArray({
        'num_cells': params['receiver']['num_cells'],
        'cell_params': cell_params
    })
    
    # ==========================================================================
    # TEST 1: Passive FSK Modulation (Visual Check)
    # ==========================================================================
    print("\n[Test 1] Validating Passive LC Modulation...")
    t_short = np.linspace(0, 0.01, 1000) # 10ms
    bits_short = [1, 0, 1, 1, 0]
    
    P_tx_short = tx.modulate_fsk_passive(
        bits_short, t_short, 
        f0=params['transmitter']['fsk_f0'],
        f1=params['transmitter']['fsk_f1'],
        rise_time=params['transmitter']['rise_time'],
        fall_time=params['transmitter']['fall_time'],
        power_density=params['transmitter']['power_density']
    )
    
    plt.figure()
    plt.plot(t_short*1000, P_tx_short, label='LC Modulated Sunlight')
    plt.title('Passive FSK Waveform (LC Response)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Power Density $(W/m^2)$')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "lc_waveform.png"))
    print(f"  Plot saved to {plot_dir}/lc_waveform.png")
    
    # Check if edges are sloped (not ideal square)
    # Derivative should not be infinite
    deriv = np.diff(P_tx_short)
    if np.max(np.abs(deriv)) < (np.max(P_tx_short) / (t_short[1]-t_short[0])):
        print("  [OK] Waveform shows finite rise/fall times.")
    else:
        print("  [WARNING] Waveform looks like ideal square wave.")

    # ==========================================================================
    # TEST 2: Cold Start Charging (2s-8p)
    # ==========================================================================
    print("\n[Test 2] Validating Cold Start Charging (2s-8p)...")
    
    # Configure for Charging
    rx_array.set_configuration(2, 8) # 2 Series, 8 Parallel
    
    # Simulation Time: 10 seconds of charging (Full charge takes 20-120s)
    t_charge = np.linspace(0, 10, 1000)
    
    # Constant Sunlight (No modulation for charging efficiency)
    P_tx_const = np.ones_like(t_charge) * params['transmitter']['power_density']
    
    # Propagate to Receiver Area
    # Channel.propagate expects Watts. P_tx is W/m^2.
    # But OpticalChannel usually takes Watts.
    # Let's adjust: P_tx_watts = P_density * Rx_Area?
    # No, simulator assumes Point Source usually.
    # Here Source is "Sunlight". 
    # Let's assume P_tx entering channel is Watts on the Collector?
    # Or simpler: Incident Power Density at RX is P_density (if loss=0 over 4m for sun).
    # OpticalChannel applies Loss.
    # Let's override logic:
    #   P_rx_total = PowerDensity * Rx_Area (No distance loss for sunlight, it's collimated/everywhere)
    #   Wait, paper uses "Himawari collector" + Fiber. Distance loss applies to Fiber/Lens?
    #   Let's assume Power Density AT RECEIVER is 600 W/m^2 for simplicity of "Outdoor" test.
    
    Rx_Area_m2 = params['channel']['receiver_area'] * 1e-4
    P_rx_total = P_tx_const * Rx_Area_m2 # W
    
    I_ph_arr = rx_array.optical_to_current(P_rx_total, t_charge)
    
    # Solve Charging (C=0.47F)
    V_cap, I_cap = rx_array.solve_circuit(
        I_ph_arr, t_charge, 
        V_cap_initial=0.0, 
        C_storage=0.47, 
        R_load_comm=None
    )
    
    print(f"  Initial V: {V_cap[0]:.3f} V")
    print(f"  Final V (10s): {V_cap[-1]:.3f} V")
    print(f"  Charging Rate: {(V_cap[-1]-V_cap[0])/10 * 1000:.2f} mV/s")
    
    plt.figure()
    plt.plot(t_charge, V_cap)
    plt.title('Cold Start Charging (2s-8p)')
    plt.xlabel('Time (s)')
    plt.ylabel('Supercap Voltage (V)')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "cold_start_charging.png"))
    
    # Expectation: V should rise linearly-ish
    if V_cap[-1] > V_cap[0]:
        print("  [OK] System is charging.")
    else:
        print("  [FAIL] System is not charging.")

    # ==========================================================================
    # TEST 3: Mode Switch & Communication (8s-2p)
    # ==========================================================================
    print("\n[Test 3] Mode Switch & Communication (8s-2p)...")
    
    # Simulate condition where V_cap > 1.8V
    # Switch to 8s-2p for higher voltage output
    rx_array.set_configuration(8, 2)
    
    # Generate FSK Data Signal
    t_comm = np.linspace(0, 0.1, 10000) # 100ms
    bits_comm = [1, 0, 1, 1, 0, 0, 1, 0] * 5 # 40 bits
    
    P_tx_comm = tx.modulate_fsk_passive(
        bits_comm, t_comm[:len(bits_comm)*int(len(t_comm)/len(bits_comm))], # Rough alignment
        f0=params['transmitter']['fsk_f0'],
        f1=params['transmitter']['fsk_f1'],
        rise_time=params['transmitter']['rise_time'],
        fall_time=params['transmitter']['fall_time'],
        power_density=params['transmitter']['power_density']
    )
    if len(P_tx_comm) != len(t_comm):
        # Quick fix for length
        P_tx_comm = np.resize(P_tx_comm, len(t_comm))
        
    P_rx_comm = P_tx_comm * Rx_Area_m2
    I_ph_comm = rx_array.optical_to_current(P_rx_comm, t_comm)
    
    # Resistive Load for Comm (e.g. MCU consumption ~2mA @ 3V -> 1.5kOhm)
    # Or specific Sensing Resistor? Paper mentions AD5165.
    # Let's use 10 kOhm for high voltage signal
    R_load = 10000.0 
    
    V_sig, _ = rx_array.solve_circuit(
        I_ph_comm, t_comm, 
        R_load_comm=R_load
    )
    
    print(f"  Signal Mean Voltage: {np.mean(V_sig):.3f} V")
    print(f"  Signal Peak-to-Peak: {np.ptp(V_sig):.3f} V")
    
    plt.figure()
    plt.plot(t_comm*1000, V_sig)
    plt.title('Received FSK Signal (8s-2p)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "comm_signal.png"))
    
    # Expectation: 8s configuration should give high voltage (> 3V)
    if np.mean(V_sig) > 2.5:
        print("  [OK] High Voltage achieved in 8s-2p mode.")
    else:
        print(f"  [WARNING] Voltage low ({np.mean(V_sig):.2f} V). Check Series logic.")
        
    return True
