"""
Correa Morales et al. (2025) Validation Module

Paper: "Experimental design and performance evaluation of a solar panel-based 
        visible light communication system for greenhouse applications"

Key Validation Targets:
1. Frequency Response: f_3dB approx 14 kHz (Given Ceq=50nF, R=220 Ohm)
2. Signal Chain: Rx -> AC Couple -> Amp -> ADC
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from simulator.receiver import PVReceiver
from simulator.transmitter import Transmitter
from simulator.channel import OpticalChannel

PARAMS = {
    # Transmitter (Correa uses 30W LED, 10 Vpp carrier)
    'tx_power_w': 30.0,
    'carrier_freq_hz': 10000, # 10 kHz
    'carrier_amp_vpp': 10.0, 
    
    # Receiver (Polycrystalline Silicon)
    'r_load_ohm': 220,
    
    # The paper states "Ceq = 50 nF" (Combined Junction + Input)
    # We will use this as the effective Cj for simulation
    'c_eq_nf': 50.0,
    
    # Since area/responsivity not given, we assume generic Poly-Si
    'responsivity': 0.5, # A/W
    'area_cm2': 66.0,    # 110x60 mm
    
    # Analog Front End
    'ac_coupling_r': 1000,    # 1 kOhm
    'ac_coupling_c': 100e-9,  # 100 nF
    'amp_gain': 11.0,         # 1 + 10k/1k
    'v_supply': 3.3,
    
    # Target
    'target_bw_hz': 14000,
}

def validate_frequency_response():
    """
    Validate the electrical bandwidth of the Receiver.
    Paper Eq 14: f_c = 1 / (2*pi * R_load * C_eq)
    """
    print("\n[Test 1] Validating Electrical Bandwidth...")
    
    # Setup Receiver
    rx = PVReceiver({
        'responsivity': PARAMS['responsivity'],
        'shunt_resistance': 0.001, # 1 kOhm (Assumed Rsh >> Rload)
        'capacitance': PARAMS['c_eq_nf'] * 1e3, # nF -> pF input
        'dark_current': 1e-9,
    })
    
    # Analytical Check
    R = PARAMS['r_load_ohm']
    C = PARAMS['c_eq_nf'] * 1e-9
    f_c_theory = 1 / (2 * np.pi * R * C)
    
    print(f"  R_load: {R} Ohm")
    print(f"  C_eq:   {C*1e9} nF")
    print(f"  Theoretical f_3dB: {f_c_theory/1000:.2f} kHz (Target: ~14 kHz)")
    
    # Simulation Check (Frequency Sweep)
    freqs = np.logspace(2, 5, 50) # 100 Hz to 100 kHz
    gains = []
    
    for f in freqs:
        # Simulate a sinusoidal current source
        t = np.linspace(0, 5/f, 1000)
        I_ph = 1e-3 * (1 + 0.5 * np.sin(2*np.pi*f*t)) # 1mA DC + Signal
        
        # Solve PV Circuit
        V_pv = rx.solve_pv_circuit(I_ph, t, R_load=PARAMS['r_load_ohm'], verbose=False)
        
        # Measure AC amplitude
        ac_amp = (np.max(V_pv) - np.min(V_pv)) / 2
        gains.append(ac_amp)
        
    gains = np.array(gains)
    gains_db = 20 * np.log10(gains / gains[0])
    
    # Find -3dB
    idx_3db = np.argmin(np.abs(gains_db + 3.0))
    f_simulated = freqs[idx_3db]
    
    print(f"  Simulated f_3dB:   {f_simulated/1000:.2f} kHz")
    
    # Plot
    output_dir = 'outputs/correa_2025/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure()
    plt.semilogx(freqs, gains_db, 'b-', label='Simulated')
    plt.axvline(f_c_theory, color='g', linestyle='--', label='Theory (Eq. 14)')
    plt.axvline(14000, color='r', linestyle=':', label='Paper Target (14 kHz)')
    plt.axhline(-3, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Correa 2025: Receiver Frequency Response')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'frequency_response.png'))
    print(f"  Plot saved to {output_dir}/frequency_response.png")
    
    return f_simulated

def validate_pwm_ask_chain():
    """
    Validate the Time-Domain waveform processing.
    Tx (PWM-ASK) -> Rx (PV) -> AC Couple -> Amp
    """
    print("\n[Test 2] Validating PWM-ASK Signal Chain...")
    
    # Config
    fs = 200000 # 200 kHz sample rate (>> 10 kHz carrier)
    duration = 0.2 # 200ms (2 cycles of 10Hz PWM)
    t = np.arange(int(fs * duration)) / fs
    
    # 1. Transmitter (PWM-ASK)
    tx = Transmitter({
        'dc_bias': 100, # mA
        'modulation_depth': 0.8,
        'led_efficiency': 0.2
    })
    
    # 10 Hz PWM, 50% Duty, 10 kHz Carrier
    P_tx = tx.modulate_pwm_ask(
        pwm_duty_cycle=0.5, 
        pwm_freq_hz=10, 
        carrier_freq_hz=10000, 
        t=t
    )
    
    # 2. Receiver (PV)
    rx = PVReceiver({
        'responsivity': PARAMS['responsivity'],
        'shunt_resistance': 0.001, # 1 kOhm
        'capacitance': PARAMS['c_eq_nf'] * 1e3,
    })
    
    # Optical Channel (Short range)
    ch = OpticalChannel({'distance': 0.4, 'receiver_area': PARAMS['area_cm2']})
    P_rx = ch.propagate(P_tx, t)
    
    I_ph = rx.optical_to_current(P_rx)
    V_pv = rx.solve_pv_circuit(I_ph, t, R_load=PARAMS['r_load_ohm'], verbose=False)
    
    # 3. Analog Front End
    # A. AC Coupling (High Pass)
    V_ac = rx.apply_rc_highpass(
        V_pv, t, 
        R_ohm=PARAMS['ac_coupling_r'], 
        C_f=PARAMS['ac_coupling_c']
    )
    
    # B. Amplifier
    V_amp = rx.apply_voltage_amplifier(
        V_ac, 
        gain_linear=PARAMS['amp_gain'],
        supply_voltage=PARAMS['v_supply']
    )
    
    # Plot
    output_dir = 'outputs/correa_2025/plots'
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3,1,1)
    plt.plot(t*1000, P_tx*1000, 'r')
    plt.title('Tx Optical Power (PWM-ASK)')
    plt.ylabel('mW')
    
    plt.subplot(3,1,2)
    plt.plot(t*1000, V_pv*1000, 'g')
    plt.title('PV Output Voltage (Raw)')
    plt.ylabel('mV')
    
    plt.subplot(3,1,3)
    plt.plot(t*1000, V_amp, 'b')
    plt.title('Amplified Output (After AC Coupling + Gain)')
    plt.ylabel('Volts')
    plt.xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_chain.png'))
    print(f"  Plot saved to {output_dir}/signal_chain.png")

def run_validation():
    print("="*60)
    print("VALIDATING CORREA Morales et al. (2025)")
    print("="*60)
    
    f_bw = validate_frequency_response()
    validate_pwm_ask_chain()
    
    target = PARAMS['target_bw_hz']
    error = abs(f_bw - target) / target * 100
    print("\nSUMMARY")
    print("-" * 60)
    print(f"Bandwidth: {f_bw/1000:.2f} kHz (Target: 14.0 kHz)")
    print(f"Error: {error:.2f}%")
    
    if error < 10:
        print("✅ SUCCESS: Simulation matches Correa 2025 bandwidth.")
    else:
        print("❌ FAILURE: Bandwidth mismatch.")

if __name__ == "__main__":
    run_validation()
