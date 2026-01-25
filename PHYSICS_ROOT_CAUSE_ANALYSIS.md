# Kadirvelu 2021: Physics-First Root Cause Analysis
## "No Magic Numbers" - Complete Investigation Report

**Date**: 2026-01-25
**Principle**: Physics and Logic First - No Arbitrary Constants
**Status**: 🔴 CRITICAL - 7 Major Magic Numbers Identified

---

## Executive Summary

The validation script `papers/kadirvelu_2021.py` contains **seven instances of magic numbers** that hide the underlying physics. This report:
1. Identifies each magic number
2. Explains the ROOT CAUSE physics
3. Proposes physics-based replacements
4. Shows why curve-fitting fails to build understanding

---

## 🔴 Issue #1: Frequency Response (+10 dB Mystery Offset)

### Evidence
```python
# Line 98, 135 in kadirvelu_2021.py
ref_db = 20 * np.log10(np.abs(H_total) + 1e-12) + 10.0  # ← MAGIC +10 dB
gain_db = 20 * np.log10(gain_linear + 1e-12) + 10.0     # ← MAGIC +10 dB
```

### What Paper Shows
Fig. 13 shows frequency response with peak gain around +10 dB at mid-band frequencies.

### Root Cause Physics

The script calculates `H_total = H_pv × G_sense × G_ina × H_bpf`, which is the transfer function from **photocurrent I_ph** to **output voltage V_bp**.

But the paper measures from **LED driver input V_mod** to **output voltage V_bp**.

**The missing links:**

```
V_mod → I_led (LED driver: I = V/R_E)
I_led → P_e (LED radiant efficiency: Eq. 17, G_led = 0.88 W/A)
P_e → P_r (Optical path: Eq. 7, Lambertian + distance)
P_r → I_ph (Photodetection: Eq. 9, R_λ = 0.457 A/W)
I_ph → V_bp (Receiver chain: already modeled)
```

### Physics-Based Solution

```python
# From paper parameters
R_E = 1.0  # LED driver resistor (from Eq. 16)
G_led = 0.88  # W/A (from paper text after Eq. 17)
T_lens = 0.85  # Lens transmittance
m_lambert = -np.log(2) / np.log(np.cos(9 * np.pi/180))  # Mode number
distance_m = 0.325
theta = 0  # On-axis
beta = 0  # Normal incidence
A_cell = 9.0e-4  # 9 cm² = 9e-4 m²

# Optical path gain (Eq. 8)
G_op = ((m_lambert + 1) / (2 * np.pi * distance_m**2) *
        np.cos(theta)**m_lambert * np.cos(beta) * A_cell)

# LED driver gain
G_driver = G_led * T_lens / R_E  # W/V

# Complete system gain (V_mod to V_bp)
G_system = G_driver * G_op * R_lambda * R_sense * G_ina * H_bpf

# Convert to dB (NO MAGIC OFFSET)
ref_db = 20 * np.log10(np.abs(G_system))
```

### Why +10 dB "Worked"
It compensated for missing `G_driver × G_op ≈ 10^(10/20) ≈ 3.16`. But this masks the actual optical link budget.

### Impact
- Cannot predict performance at different distances
- Cannot optimize LED power vs data rate
- Hides whether system is noise-limited or signal-limited

---

## 🔴 Issue #2: DC-DC Vout vs Duty Cycle (Exponential Curve Fits)

### Evidence
```python
# Lines 264-268 in kadirvelu_2021.py
FITS = {
    50:  {"a": 2.24, "b": 0.19, "c": 0.67},  # ← MAGIC
    100: {"a": 1.92, "b": 0.18, "c": 0.77},
    200: {"a": 1.51, "b": 0.17, "c": 0.89},
}
vout_measured = p['a'] * rho * np.exp(-p['b'] * rho) + p['c']
```

### What Paper Shows
Fig. 15 shows V_out initially increases with duty cycle, then collapses as duty cycle approaches 50%.

### Root Cause Physics

This is **NOT** an exponential phenomenon. It's a **nonlinear circuit equilibrium problem**.

**Two competing equations:**

1. **PV Cell Supply** (Shockley diode equation):
```python
I_supply(V) = I_ph - I_0 * (exp(V / (n*V_T)) - 1) - V/R_sh
```

2. **DC-DC Converter Demand** (power balance):
```python
# Output power: P_out = V_out² / R_load
# Input power: P_in = V × I_in
# Efficiency: η = P_out / P_in
# Boost relation: V_out = V_in / (1 - D)

# Solve for I_in:
I_demand(V, D) = V / (η × (1-D)² × R_load)
```

**Operating Point**: Where these intersect.
```python
I_supply(V_op) = I_demand(V_op, D)
```

**As duty cycle increases:**
- Demand current ↑ (need more input power for higher V_out)
- PV voltage ↓ (moving left on I-V curve toward knee)
- Eventually: V collapses below diode turn-on → converter stops

### Physics-Based Solution

Already implemented in `dc_dc_converter.py:solve_operating_point()` but not used in Fig. 15!

```python
# CORRECT APPROACH
pv_params = {
    'I_ph': 0.00033,  # From measured I-V curve
    'I_0': 1e-9,
    'n': 1.5 * 13,  # 13 cells in series
    'T': 300,
    'R_sh': 13000
}

V_out_list = []
for D in duty_cycle_range:
    V_op, I_op = dcdc.solve_operating_point(
        pv_params, R_load=2000, duty_cycle=D, fsw_khz=fsw
    )
    result = dcdc.calculate_output(V_op, I_op, duty_cycle=D, fsw_khz=fsw)
    V_out_list.append(result['V_out'])
```

### Why Exponential Fit "Worked"
The PV diode equation naturally produces exponential-looking behavior. But the fitted curve ignores:
- Actual I_ph value
- Cell temperature
- Shunt resistance
- Switching losses

### Impact
- Cannot predict behavior with different PV cells
- Cannot optimize MPPT algorithm
- Misses physics of CCM/DCM transition

---

## 🔴 Issue #3: Transient Waveform Amplitude (60× Magic Gain)

### Evidence
```python
# Lines 329-330 in kadirvelu_2021.py
calibration_gain = 60.0  # ← MAGIC
Vbp = Vbp * PARAMS['ina_gain'] * calibration_gain * 0.4  # ← MORE MAGIC
```

### Root Cause Physics

Manchester encoding splits each bit into two pulses. The **frequency content** is not at f_bit, but at **f_bit/2** (fundamental) plus harmonics.

**The BPF response varies with frequency:**
```python
H_bpf(f) = |H_hp(f) × H_lp(f)|
```

Where:
- H_hp(f) = f / √(f² + f_low²)  ← High-pass
- H_lp(f) = f_high / √(f² + f_high²)  ← Low-pass

**For Manchester at bit rate R_b:**
- Fundamental: f₁ = R_b / 2
- 3rd harmonic: f₃ = 3R_b / 2
- 5th harmonic: f₅ = 5R_b / 2

**Amplitude after BPF:**
```python
V_bp_rms = sqrt(sum over harmonics of:
    (amplitude_n × |H_bpf(f_n)|)²
)
```

### Physics-Based Solution

```python
# Manchester spectrum (analytical)
def manchester_spectrum(bit_rate, modulation_depth, I_ph_dc):
    amplitudes = []
    frequencies = []

    for n in [1, 3, 5, 7, 9]:  # Odd harmonics
        f_n = n * bit_rate / 2
        # Manchester: sinc-like envelope
        A_n = (4 / (n * pi)) * modulation_depth * I_ph_dc
        frequencies.append(f_n)
        amplitudes.append(A_n)

    return frequencies, amplitudes

# Apply receiver chain to each component
V_bp_total = 0
for f, A_iph in zip(frequencies, amplitudes):
    V_sense = A_iph * R_sense
    V_ina = V_sense * G_ina
    H_bpf_at_f = calculate_bpf_response(f)
    V_bp_component = V_ina * H_bpf_at_f
    V_bp_total += V_bp_component**2  # Power sum

V_bp_rms = np.sqrt(V_bp_total)
```

### Why 60× "Worked"
It lumped together:
- Manchester spectral weighting
- BPF frequency response
- Missing optical link gain

### Impact
- Cannot predict SNR vs bit rate
- Cannot design matched filter
- Misses timing jitter effects

---

## 🔴 Issue #4: BER Simulation (Invented SNR Formula)

### Evidence
```python
# Lines 356-363 in kadirvelu_2021.py
def simulate_ber(m_pct, Tbit_us, fsw_khz):
    snr_base = (m_pct/100.0)**2 * (Tbit_us/100.0) * 100  # ← FAKE
    ber = 0.5 * erfc(np.sqrt(snr_base))
```

### Root Cause Physics

BER depends on **signal-to-noise ratio** at the comparator input. The noise has **four independent sources**:

#### 1. Thermal (Johnson-Nyquist) Noise
```python
# From current-sense resistor R_sense
v_n_thermal_density = sqrt(4 * k_B * T * R_sense)  # V/√Hz
# After amplification and filtering:
v_n_thermal_bpf = v_n_thermal_density * sqrt(BW_bpf) * G_ina
```

#### 2. Shot Noise
```python
# From photocurrent
i_n_shot_density = sqrt(2 * q * I_ph_dc)  # A/√Hz
v_n_shot = i_n_shot_density * R_sense * G_ina * sqrt(BW_bpf)
```

#### 3. Switching Noise
```python
# From DC-DC converter (measured in paper Fig. 14)
# Peak at f_sw with harmonics
# After BPF: attenuated by |H_bpf(f_sw)|
v_n_switching = measure_switching_noise(fsw_khz) * |H_bpf(fsw_khz)|
```

#### 4. Ambient Noise
```python
# From fluorescent lights (1/f pink noise per Fig. 14)
# Integrate PSD over BPF passband
v_n_ambient = sqrt(integrate(PSD_pink(f) * |H_bpf(f)|², f_low, f_high))
```

#### Total Noise (RMS)
```python
v_n_total = sqrt(v_thermal² + v_shot² + v_switching² + v_ambient²)
```

#### Signal (Manchester)
```python
# Peak-to-peak voltage at comparator input
v_signal_pp = 2 * m * I_ph_dc * R_sense * G_ina * |H_bpf(f_bit/2)|
```

#### SNR and BER
```python
# Decision threshold at 0V (Manchester has zero DC)
# For OOK-like Manchester with matched filter:
SNR_matched = (v_signal_pp / (2 * v_n_total))² * (T_bit * BW_bpf)

# BER for binary detection
BER = 0.5 * erfc(sqrt(SNR_matched / 2))
```

### Why Fake Formula "Worked"
- `(m/100)²` mimics signal power scaling
- `T_bit/100` mimics integration gain
- But it completely ignores noise sources!

### Impact
- Cannot predict BER vs switching frequency (ignores switching noise filtering)
- Cannot optimize BPF bandwidth
- Cannot design error correction codes

---

## 🔴 Issue #5: Vout vs Modulation (Backward I_ph Scaling)

### Evidence
```python
# Lines 395-396 in kadirvelu_2021.py
'I_ph': 0.025 * (1 - 0.1 * m),  # ← WHY does m reduce I_ph?
```

### Root Cause Physics

Modulation depth **does not change time-averaged photocurrent**.

**For OOK modulation:**
- Low state: P_low = P_avg × (1 - m)
- High state: P_high = P_avg × (1 + m)
- Duty cycle: 50% (for Manchester/balanced data)
- **Average**: P_avg = (P_low + P_high) / 2 = P_avg  ← Unchanged!

**What DOES change:**
The AC component bypasses the DC-DC converter via the current-sense resistor:

```python
# Current split at solar cell output:
#   ┌──[R_sense]──[INA+BPF]─── (data path, AC only)
#   │
# PV├──[C_p]──[Boost]────────── (power path, DC only)
#   │
#   └────────────────────────── (ground)

# Power harvested:
P_dc = I_ph_dc² × R_boost_input

# Power "stolen" by data receiver:
P_ac = (m × I_ph_dc)² × R_sense × G_ina² × ...

# Effective current to converter:
I_eff ≈ I_ph_dc × sqrt(1 - (P_ac / P_available))
```

### Physics-Based Solution

```python
# Base photocurrent (time-average, independent of m)
I_ph_dc = R_lambda * P_rx_avg

# AC power diverted to receiver (depends on m)
P_ac_receiver = compute_ac_power_through_sense_resistor(m, I_ph_dc, R_sense, ...)

# Effective DC power to converter
P_effective_dc = I_ph_dc * V_pv - P_ac_receiver

# Solve for operating point
V_op, I_op = solve_with_effective_power(P_effective_dc, ...)
```

### Why 0.1×m "Worked"
Accidentally mimicked power splitting, but coefficient is arbitrary.

### Impact
- Cannot optimize power-data tradeoff
- Cannot predict crossover point where data starves energy harvesting

---

## 🔴 Issue #6: Power vs Bit Rate (Magic AC Penalty)

### Evidence
```python
# Lines 454, 465 in kadirvelu_2021.py
ac_penalty = 1.0 / (1.0 + (br / 15000.0)**2)  # ← Why 15 kHz? Why squared?
p_harvest = ... * (0.8 + 0.2*ac_penalty)      # ← Why 0.8 and 0.2?
```

### Root Cause Physics

The input capacitor C_p (10 µF, from paper) acts as a **high-pass filter for AC**, diverting AC photocurrent away from the boost inductor.

**Impedance model:**
```python
Z_cap(f) = 1 / (j * 2π * f * C_p)
Z_sense = R_sense

# At frequency f, AC current divider:
I_ac_to_inductor / I_ac_total = Z_cap / (Z_cap + Z_sense)
```

**For Manchester encoding:**
```python
# Spectrum: fundamentalat f_bit/2, harmonics at 3f/2, 5f/2, ...
# Each harmonic has amplitude: A_n = (4/(nπ)) × m × I_ph_dc

# Power diverted to cap (bypassing converter):
P_bypassed = sum over harmonics of:
    A_n² × |Z_cap(f_n) / (Z_cap(f_n) + Z_sense)|²
```

### Physics-Based Solution

```python
def calculate_ac_loss_fraction(bit_rate, m, I_ph_dc, C_p, R_sense):
    # Manchester harmonic frequencies
    harmonics = [1, 3, 5, 7, 9]
    freqs = [n * bit_rate / 2 for n in harmonics]
    amplitudes = [(4 / (n * pi)) * m * I_ph_dc for n in harmonics]

    P_total_ac = 0
    P_bypassed = 0

    for f, A in zip(freqs, amplitudes):
        Z_cap = 1 / (2 * pi * f * C_p)
        Z_sense = R_sense

        # Current divider ratio
        bypass_fraction = abs(Z_cap / (Z_cap + Z_sense))

        P_component = A**2
        P_total_ac += P_component
        P_bypassed += P_component * bypass_fraction**2

    return P_bypassed / P_total_ac

# Apply to harvested power
ac_loss = calculate_ac_loss_fraction(bit_rate, m, I_ph_dc, 10e-6, 1.0)
P_harvest_effective = P_harvest_ideal × (1 - ac_loss)
```

### Why Magic Formula "Worked"
- `1/(1 + (f/15k)²)` resembles 2nd-order low-pass
- 15 kHz ≈ 1/(2π × C_p × R_eq) for typical values
- But ignores Manchester spectrum shape

### Impact
- Cannot optimize C_p value
- Cannot predict power harvesting at high bit rates
- Misses resonance effects

---

## 🔴 Issue #7: UVLO Threshold (Hard-Coded 0.3V)

### Evidence
```python
# Lines 461-462 in kadirvelu_2021.py
if V_op < 0.3:  # ← Datasheet value? Measured? Guessed?
    p_harvest = 0.0
```

### Root Cause Physics

**UVLO (Under-Voltage Lockout)** is a chip feature, not a universal constant.

**Physical requirements:**
1. **MOSFET gate threshold**: V_gs,th ≈ 0.8-1.2V (typical)
2. **Gate driver overhead**: ~0.3V
3. **Chip quiescent current**: 10-100 µA (from PV input)

**Typical UVLO values by chip family:**
- **Ultra-low voltage**: BQ25504 → 0.08V, SPV1050 → 0.1V
- **Low voltage**: LTC3105 → 0.25V, TPS61200 → 0.5V
- **Standard**: TPS61021 → 0.9V

**Plus: UVLO has hysteresis!**
```python
V_uvlo_rising = 0.5   # Chip turns ON
V_uvlo_falling = 0.4  # Chip turns OFF
# Prevents oscillation near threshold
```

### Physics-Based Solution

```python
# From datasheet (example: TPS61200)
V_uvlo_rising = 0.5
V_uvlo_falling = 0.4
I_quiescent = 50e-6  # 50 µA

# State machine with hysteresis
if not hasattr(self, 'converter_enabled'):
    self.converter_enabled = False

if V_op > V_uvlo_rising:
    self.converter_enabled = True
elif V_op < V_uvlo_falling:
    self.converter_enabled = False

if not self.converter_enabled:
    p_harvest = 0.0
else:
    # Subtract quiescent current from available power
    P_quiescent = V_op * I_quiescent
    p_harvest = max(0, (V_op * I_op - P_quiescent) * efficiency)
```

### Why 0.3V "Worked"
Falls within range of ultra-low-voltage chips, but paper doesn't specify which chip was used!

### Impact
- Cannot select optimal boost IC
- Cannot predict startup behavior
- Misses cold-start problems (V < V_uvlo at low light)

---

## Summary Table: Magic Numbers Decoded

| Issue | Magic Number | Units | Root Cause | Should Be |
|-------|--------------|-------|------------|-----------|
| 1 | +10 dB | dB | Missing optical link gain | `20×log10(G_led × G_op)` |
| 2 | a=2.24, b=0.19, c=0.67 | V | PV I-V curve fit | Solve `I_supply(V) = I_demand(V)` |
| 3 | 60× gain | - | Manchester spectrum + BPF | Compute harmonics with `H_bpf(f_n)` |
| 4 | `(m²)(T/100)(100)` | - | Invented SNR | Sum 4 noise sources + signal |
| 5 | `0.025(1 - 0.1m)` | A | Confused power split | AC current via `Z_sense` vs `Z_cap` |
| 6 | `1/(1+(f/15k)²)` | - | Guessed AC bypass | Integrate Manchester × `|Z_cap/Z_total|²` |
| 7 | 0.3 V | V | Unknown chip UVLO | Datasheet + hysteresis model |

---

## Verification Test: Do Magic Numbers Generalize?

**Question**: If we change parameters, do the magic numbers still work?

### Test 1: Double the distance (r = 0.65 m)
- **With magic +10 dB**: Gain stays same (wrong!)
- **With physics**: Gain drops by `20×log10(0.325²/0.65²) = -12 dB` ✓

### Test 2: Use Silicon PV cell (R_λ = 0.6 A/W instead of GaAs 0.457 A/W)
- **With magic I_ph = 0.025**: Wrong photocurrent
- **With physics**: I_ph = 0.6 × P_rx (correct) ✓

### Test 3: Change C_p from 10 µF to 47 µF
- **With magic 15 kHz**: AC penalty wrong
- **With physics**: Corner frequency scales as 1/C_p ✓

### Test 4: Use different boost IC (V_uvlo = 0.8 V)
- **With magic 0.3 V**: System "works" when it shouldn't
- **With physics**: Correctly predicts failure ✓

---

## Recommended Correction Strategy

### Phase 1: Document (1 hour)
Add comments to each magic number:
```python
# MAGIC NUMBER - KNOWN ISSUE #1
# This +10 dB offset compensates for missing optical link gain (G_led × G_op).
# It is a CURVE FIT, not physics. See PHYSICS_ROOT_CAUSE_ANALYSIS.md.
ref_db = ... + 10.0  # ← PLACEHOLDER
```

### Phase 2: Replace One-by-One (1 week)
For each figure, implement physics-based model:
1. Fig 13: Add optical link gain calculation
2. Fig 15: Use existing `solve_operating_point()`
3. Fig 16: Implement Manchester spectrum decomposition
4. Fig 17: Build 4-source noise model
5. Fig 18: Correct power-splitting model
6. Fig 19: Compute AC bypass via impedance
7. UVLO: Add chip datasheet parameters

### Phase 3: Validate (2 days)
- Compare physics-based results to paper measurements
- Discrepancies reveal:
  - Missing parasitics (ESR, trace inductance)
  - Temperature effects (I_0 doubles every 10K)
  - Measurement artifacts (scope bandwidth, probe capacitance)

### Phase 4: Document Findings (1 day)
Write paper:
- "Physics-Based Validation of Kadirvelu et al. 2021 SLIPT System"
- Include all discrepancies and proposed explanations
- Predict performance outside measured range (extrapolation test)

---

## Physics Checklist (Before Adding Any Number)

Before writing `variable = <number>`, ask:

- [ ] **Can I derive this from first principles?**
  - Maxwell's equations (EM waves)
  - Shockley equation (diodes)
  - Kirchhoff's laws (circuits)
  - Shannon's theorem (information)

- [ ] **Can I look this up in a datasheet?**
  - IC UVLO threshold
  - LED radiant efficiency
  - PV cell responsivity
  - Op-amp gain-bandwidth product

- [ ] **Can I measure this independently?**
  - Oscilloscope waveform amplitude
  - Spectrum analyzer noise PSD
  - IV curve tracer
  - Optical power meter

- [ ] **Does it have correct units?**
  - Dimensional analysis
  - Convert everything to SI base units
  - Check that equations balance

- [ ] **Does it scale correctly?**
  - 2× input → what output? (linear, quadratic, sqrt?)
  - Sanity check: "If I double X, what happens to Y?"

**If the answer is "no" to ALL FIVE → It's a magic number.**

---

## Conclusion: Why This Matters

### The Problem with Magic Numbers
1. **No generalization**: Works for one setup, fails for others
2. **No intuition**: Can't explain WHY system behaves this way
3. **No optimization**: Don't know which knob to turn
4. **No debugging**: When it fails, no idea where to look

### The Benefit of Physics-First
1. **Predictive power**: Extrapolate beyond measurements
2. **Design insight**: "Aha! BPF is limiting bandwidth, not PV cell!"
3. **Systematic optimization**: Sensitivity analysis on real parameters
4. **Robust simulation**: Works for MIMO, different cells, outdoor sunlight, etc.

### Next Actions
1. ✅ Read this document
2. ⏳ Choose one figure to fix (recommend: Fig 15, easiest)
3. ⏳ Implement physics model
4. ⏳ Compare to paper measurements
5. ⏳ Document discrepancies
6. ⏳ Iterate through all seven issues

---

**Remember**: "All models are wrong, but some are useful" (George Box).
The goal is not to match the paper's plots exactly, but to **understand the physics** so we can build better systems.

**End of Report**
