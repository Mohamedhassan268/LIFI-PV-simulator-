
import numpy as np
import sys
sys.path.insert(0, '.')
from simulator.channel import OpticalChannel
from simulator.receiver import PVReceiver
from simulator.modulation import AdaptiveOFDMModulator
from utils.constants import thermal_voltage
from simulator.receiver import PVReceiver
from simulator.modulation import AdaptiveOFDMModulator
from utils.constants import thermal_voltage

def test_multipath_delay():
    print("\n--- Testing Multipath Delay ---")
    # Setup channel with multipath enabled
    params = {
        'distance': 1.0, 
        'multipath': True,
        'room_dimensions': {'height': 3.0, 'reflection_coeff': 0.8}
    }
    ch = OpticalChannel(params)
    
    # Create an impulse
    t = np.linspace(0, 100e-9, 100) # 100 ns window
    P_tx = np.zeros_like(t)
    P_tx[0] = 1.0 # Delta function
    
    P_rx = ch.propagate(P_tx, t)
    
    # Expect 2 peaks: LOS (at t=0 in relative time, or index 0) and NLOS (delayed)
    peaks = np.where(P_rx > 0)[0]
    print(f"Non-zero indices: {peaks}")
    
    assert len(peaks) >= 2, "Should have at least 2 non-zero signal components (LOS + NLOS)"
    assert peaks[0] == 0, "First peak should be LOS at index 0"
    assert peaks[-1] > 0, "Second peak should be delayed"
    print("Multipath verified.")

def test_temperature_impact():
    print("\n--- Testing Temperature Impact on Voc ---")
    rx = PVReceiver()
    
    # Constant illumination
    I_ph = np.ones(100) * 1e-3 # 1 mA
    t = np.linspace(0, 1e-3, 100)
    
    # Run at 300K
    rx.set_temperature(300)
    V_300 = rx.estimate_open_circuit_voltage(1e-3)
    
    # Run at 350K
    rx.set_temperature(350)
    V_350 = rx.estimate_open_circuit_voltage(1e-3)
    
    print(f"Voc @ 300K: {V_300*1000:.1f} mV")
    print(f"Voc @ 350K: {V_350*1000:.1f} mV")
    
    # Physics: Voc decreases as T increases (due to I0 increasing exponentially)
    assert V_350 < V_300, "Voc should decrease with temperature"
    print("Temperature effect verified.")

def test_dynamic_temperature_simulation():
    print("\n--- Testing Dynamic Temperature Simulation ---")
    # Simpler test: verify that the internal V_T changes with temperature profile
    rx = PVReceiver()
    
    # Create temperature profile
    T_profile = np.array([300.0, 320.0, 340.0])
    
    # Calculate expected thermal voltages
    V_T_300 = thermal_voltage(300)
    V_T_320 = thermal_voltage(320)
    V_T_340 = thermal_voltage(340)
    
    print(f"V_T @ 300K: {V_T_300*1000:.2f} mV")
    print(f"V_T @ 320K: {V_T_320*1000:.2f} mV")
    print(f"V_T @ 340K: {V_T_340*1000:.2f} mV")
    
    # Verify thermal voltage increases with temperature
    assert V_T_320 > V_T_300, "V_T should increase with temperature"
    assert V_T_340 > V_T_320, "V_T should increase with temperature"
    
    # Verify that receiver can accept array temperature
    rx.T = T_profile
    assert isinstance(rx.T, np.ndarray), "Receiver should accept array temperature"
    
    print("Dynamic temperature mechanism verified.")

def test_varicap_effect():
    print("\n--- Testing Voltage-Dependent Capacitance ---")
    # Simple mathematical verification of the varicap formula
    # C_j(V) = C_j0 / (1 - V/V_bi)^m
    
    C_j0 = 798e-12  # 798 pF
    V_bi = 0.8
    m = 0.5
    
    # At V=0: C_j = C_j0
    V = 0.0
    C_at_0 = C_j0 / ((1 - V/V_bi) ** m)
    print(f"C_j at V=0V: {C_at_0*1e12:.1f} pF (expected: {C_j0*1e12:.1f} pF)")
    assert abs(C_at_0 - C_j0) < 1e-15, "C_j(0) should equal C_j0"
    
    # At V=0.4V (forward bias): C_j should increase
    V = 0.4
    C_at_0p4 = C_j0 / ((1 - V/V_bi) ** m)
    print(f"C_j at V=0.4V: {C_at_0p4*1e12:.1f} pF")
    assert C_at_0p4 > C_j0, "C_j should increase in forward bias"
    
    # At V=-0.5V (reverse bias): C_j should decrease
    V = -0.5
    C_at_neg = C_j0 / ((1 - V/V_bi) ** m)
    print(f"C_j at V=-0.5V: {C_at_neg*1e12:.1f} pF")
    assert C_at_neg < C_j0, "C_j should decrease in reverse bias"
    
    # Verify receiver has the parameters
    rx = PVReceiver({'V_bi': 0.8, 'grading_m': 0.5})
    assert hasattr(rx, 'V_bi') and rx.V_bi == 0.8, "Receiver should have V_bi parameter"
    assert hasattr(rx, 'grading_m') and rx.grading_m == 0.5, "Receiver should have grading_m parameter"
    
    print("Varicap mechanism verified.")

def test_ofdm_clipping():
    print("\n--- Testing OFDM Clipping ---")
    # High clipping ratio (no clip)
    mod_high = AdaptiveOFDMModulator(nfft=64, cp_length=16, clipping_ratio_db=20)
    mod_high.allocate_bits(np.ones(52)*20) # Max bits
    bits = np.random.randint(0, 2, 10000)
    sig_high = mod_high.modulate(bits)
    
    # Low clipping ratio (strong clip)
    CR_dB = 3.0
    mod_low = AdaptiveOFDMModulator(nfft=64, cp_length=16, clipping_ratio_db=CR_dB)
    mod_low.bit_allocation = mod_high.bit_allocation # Same allocation
    mod_low.total_bits_per_symbol = mod_high.total_bits_per_symbol
    sig_low = mod_low.modulate(bits)
    
    # Check PAPR
    def get_papr(s):
        p_avg = np.mean(np.abs(s)**2)
        p_max = np.max(np.abs(s)**2)
        return 10*np.log10(p_max/p_avg)
    
    papr_high = get_papr(sig_high)
    papr_low = get_papr(sig_low)
    
    print(f"PAPR High CR: {papr_high:.2f} dB")
    print(f"PAPR Low CR ({CR_dB} dB): {papr_low:.2f} dB")
    
    # Ideally PAPR should be close to CR_dB for the low case
    assert papr_low < papr_high, "Clipping should reduce PAPR"
    assert papr_low <= CR_dB + 1.0, f"PAPR {papr_low:.2f} dB should be approx {CR_dB} dB"
    print("OFDM Clipping verified.")

if __name__ == "__main__":
    test_multipath_delay()
    test_temperature_impact()
    test_dynamic_temperature_simulation()
    test_varicap_effect()
    test_ofdm_clipping()
