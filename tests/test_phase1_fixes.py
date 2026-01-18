# Quick test: verify constants and receiver can initialize
from utils.constants import PHOTODIODE_SHUNT_RESISTANCE_MOHM, PHOTODIODE_JUNCTION_CAP_PF, PAPER_VALIDATION_CONFIG
from simulator.receiver import PVReceiver
from simulator.channel import OpticalChannel

print('=== TESTING PHASE 1 FIXES ===')
print()

# 1. Constants test
print('1. CONSTANTS CHECK:')
print(f'   PHOTODIODE_SHUNT_RESISTANCE_MOHM = {PHOTODIODE_SHUNT_RESISTANCE_MOHM} MΩ')
print(f'   PHOTODIODE_JUNCTION_CAP_PF = {PHOTODIODE_JUNCTION_CAP_PF} pF')
print(f'   PAPER_CONFIG rsh_ohm = {PAPER_VALIDATION_CONFIG["rsh_ohm"]} Ω')
print(f'   PAPER_CONFIG cj_f = {PAPER_VALIDATION_CONFIG["cj_f"]} F')

# Verify values are correct
assert PHOTODIODE_SHUNT_RESISTANCE_MOHM == 138.8, f'FAIL: R_sh should be 138.8, got {PHOTODIODE_SHUNT_RESISTANCE_MOHM}'
assert PHOTODIODE_JUNCTION_CAP_PF == 798, f'FAIL: C_j should be 798, got {PHOTODIODE_JUNCTION_CAP_PF}'
assert abs(PAPER_VALIDATION_CONFIG['rsh_ohm'] - 138.8e6) < 1, f'FAIL: rsh_ohm should be 138.8e6'
assert abs(PAPER_VALIDATION_CONFIG['cj_f'] - 798e-12) < 1e-15, f'FAIL: cj_f should be 798e-12'
print('   [OK] All constant values correct!')
print()

# 2. Receiver test
print('2. RECEIVER INITIALIZATION TEST:')
rx = PVReceiver()  # Should use correct defaults
print(f'   [OK] PVReceiver initialized with R_sh = {rx.R_sh*1e-6:.1f} MΩ, C_j = {rx.C_j*1e12:.1f} pF')
print()

# 3. Channel with humidity auto-detection
print('3. CHANNEL HUMIDITY NORMALIZATION TEST:')
ch = OpticalChannel({'humidity': 80, 'distance': 0.5})  # Pass 80 should warn and convert to 0.8
print(f'   Normalized humidity = {ch.humidity}')
print(f'   Beer-Lambert alpha = {ch.attenuation_alpha:.3f} m^-1')
expected_alpha = 0.5 + 9.0 * (0.8 - 0.3) * 1.5  # = 0.5 + 6.75 = 7.25
print(f'   Expected alpha (at 80%) = {expected_alpha:.3f} m^-1')
assert abs(ch.attenuation_alpha - expected_alpha) < 0.01, f'FAIL: alpha mismatch'
print('   [OK] Humidity normalized and alpha correct!')
print()

# 4. Lambertian order check
print('4. LAMBERTIAN ORDER TEST:')
from utils.constants import lambertian_order
m_9 = lambertian_order(9)
m_30 = lambertian_order(30)
print(f'   m(9°) = {m_9:.2f} (expected ~57.4)')
print(f'   m(30°) = {m_30:.2f} (expected ~4.81)')
assert abs(m_30 - 4.81) < 0.1, f'FAIL: m(30) should be ~4.81, got {m_30}'
print('   [OK] Lambertian order calculations correct!')
print()

print('=== ALL PHASE 1 TESTS PASSED ===')
