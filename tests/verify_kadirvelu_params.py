"""
PV Pole Frequency Verification
Kadirvelu 2021 Paper Validation
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from papers.kadirvelu_2021 import PARAMS

print("=" * 60)
print("PV POLE FREQUENCY VERIFICATION")
print("=" * 60)

# Get parameters
rsh_ohm = PARAMS['rsh_ohm']
cj_pf = PARAMS['cj_pf']
cj_f = cj_pf * 1e-12  # Convert pF to F

# Calculate pole frequency
f_pole = 1 / (2 * np.pi * rsh_ohm * cj_f)

print(f"\nParameters from kadirvelu_2021.py:")
print(f"  R_sh = {rsh_ohm:,.0f} Ω = {rsh_ohm/1000:.1f} kΩ")
print(f"  C_j = {cj_pf:,.0f} pF = {cj_pf/1e6:.0f} nF")
print(f"  C_j = {cj_f:.2e} F")

print(f"\nPV Pole Frequency:")
print(f"  f_pole = 1 / (2π × R_sh × C_j)")
print(f"  f_pole = 1 / (2π × {rsh_ohm:,.0f} × {cj_f:.2e})")
print(f"  f_pole = {f_pole:.2f} Hz = {f_pole/1000:.3f} kHz")

print(f"\nPaper Target: ~1.44 kHz")

# Verify
if 1000 < f_pole < 2000:
    print(f"\n✓ PASS: PV pole is within expected range (1-2 kHz)")
else:
    print(f"\n✗ FAIL: PV pole is outside expected range")
    print(f"  Expected: 1000-2000 Hz, Got: {f_pole:.0f} Hz")

# Also verify BPF cutoffs
print(f"\n" + "=" * 60)
print("BPF CUTOFF VERIFICATION")
print("=" * 60)
print(f"\nBPF Parameters:")
print(f"  f_low = {PARAMS['bpf_low_hz']} Hz (target: ~75 Hz)")
print(f"  f_high = {PARAMS['bpf_high_hz']} Hz (target: ~10 kHz)")

# Check if pole is within BPF passband
if PARAMS['bpf_low_hz'] < f_pole < PARAMS['bpf_high_hz']:
    print(f"\n✓ PV pole ({f_pole:.0f} Hz) is within BPF passband")
else:
    print(f"\n⚠ PV pole is outside BPF passband - check configuration")

print("\n" + "=" * 60)
