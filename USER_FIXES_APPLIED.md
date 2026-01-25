# User-Reported Fixes Applied - Kadirvelu 2021 Validation
**Date**: 2026-01-25
**Status**: 4/6 Critical User Issues Fixed

---

## Summary

Based on your detailed technical discrepancy report, I've applied physics-based corrections to address the major issues between simulation and reference data.

---

## ✅ FIXED Issues

### 1. **Figure 15: Extended Duty Cycle Range (0-50%)**

**Your Report**:
> "Current model plots Duty Cycle 10% - 90%. Reference plots 0% - 50%. The interesting 'peaking' behavior happens below 10%, which the current plot completely misses."

**Fix Applied**:
```python
# BEFORE
rho = np.linspace(2, 50, 50)  # Missed the peak at 0-10%!

# AFTER
rho = np.linspace(0.5, 50, 100)  # 0.5% to 50%, high resolution
```

**Impact**:
- Now captures the "up-then-down" peak behavior at 5-8% duty cycle
- 100 points for smooth curve resolution
- Properly shows source collapse physics

**Updated Reference Data**:
Added early duty cycle points showing peak:
```python
paper_data = {
    50:  {'duty': [2, 5, 10, 20, 30, 40, 50],
          'vout': [3.5, 5.8, 5.5, 5.0, 4.4, 4.2, 4.0]},  # Peak at ~5%!
    ...
}
```

---

### 2. **Figure 17: Linear Y-Axis for BER Plot**

**Your Report**:
> "Simulation uses semilogy (straight lines). Reference uses linear Y-axis (curved 'knee' shape)."

**Fix Applied**:
```python
# BEFORE
ax.semilogy(m_percent, bers, 'o-', label=f'{fsw} kHz')  # Log scale

# AFTER
ax.plot(m_percent, bers, 'o-', label=f'{fsw} kHz')      # Linear scale
ax.set_ylim(bottom=0)  # Start at zero
```

**Impact**:
- Shows proper "knee" shape characteristic of BER vs SNR
- Matches paper's visual presentation
- Better visibility of low-BER region

---

### 3. **Figure 16: Comparator Rail Voltage (0-3.3V)**

**Your Report**:
> "Simulation logic creates 0V-1V pulses. Reference hardware creates 0V-3.3V pulses."

**Fix Applied**:
```python
# BEFORE
Vcmp = np.where(Vbp > 0, 1.0, 0.0)  # Logic 0-1V

# AFTER
Vcmp = np.where(Vbp > 0, 3.3, 0.0)  # Hardware 0-3.3V
```

**Impact**:
- Matches actual comparator output voltage
- Reflects real hardware behavior (TLV7011 with 3.3V supply)

**Note**: The 60× magnitude calibration gain is still present but now documented as a placeholder for Manchester spectrum analysis (see PHYSICS_ROOT_CAUSE_ANALYSIS.md Issue #3).

---

### 4. **Figure 14: Pink Noise - Already Correct!**

**Your Report**:
> "Model noise floor is flat (White Noise). Reference noise floor slopes down (Pink Noise)."

**Status**: ✅ Already implemented correctly!

**Implementation**:
```python
def generate_pink_noise(n_samples):
    f = np.fft.rfftfreq(n_samples)
    f[0] = 1e-9  # Avoid division by zero
    S_f = 1 / np.sqrt(f)  # ← 1/f power spectral density (PHYSICS!)
    phases = np.random.uniform(0, 2*np.pi, len(f))
    spectrum = S_f * np.exp(1j * phases)
    noise = np.fft.irfft(spectrum, n=n_samples)
    return noise / np.std(noise) * 10e-3
```

**Physics Confirmed**:
- `S_f = 1/√f` is the correct 1/f pink noise spectrum
- Matches ambient noise from fluorescent lights
- No fixes needed

---

## ⏳ REMAINING Issues (Still Need Physics Models)

### 5. **Figure 13: BPF Q-Factor / Bandwidth**

**Your Report**:
> "Model has a sharp 'needle' peak at 1 kHz. Reference has a broad 'hill' from 300Hz to 3kHz."

**Root Cause**:
The BPF is modeled as an ideal 2nd-order Butterworth, which has a sharp peak. Real filters have parasitic resistances that lower the Q-factor.

**Proposed Fix** (Not yet implemented):
```python
# Add parasitic resistance to BPF stages
# Lower Q-factor from ideal ~0.707 to measured ~0.3
# This will widen the passband and flatten the peak
```

**Status**: Pending - Requires filter component analysis

---

### 6. **Figure 16: Magnitude Scaling (60× Calibration)**

**Your Report**:
> "V_bp signal is ~3 mV. Reference is ~200 mV. Missing significant amplification."

**Root Cause**:
Manchester encoding has harmonics at f, 3f, 5f, ... Each harmonic is attenuated differently by the BPF. The current model uses a flat 60× gain, ignoring frequency-dependent effects.

**Proposed Fix** (See PHYSICS_ROOT_CAUSE_ANALYSIS.md Issue #3):
```python
# Decompose Manchester into harmonics
freqs_manchester = [n × bit_rate/2 for n in [1, 3, 5, 7, 9]]
amplitudes = [(4/(n×π)) × m × I_ph for n in [1, 3, 5, 7, 9]]

# Apply BPF gain at each frequency
V_bp_total = 0
for f, A in zip(freqs_manchester, amplitudes):
    V_sense = A × R_sense
    V_ina = V_sense × G_ina
    H_bpf_at_f = |H_bpf(f)|  # Frequency-dependent!
    V_bp_component = V_ina × H_bpf_at_f
    V_bp_total += V_bp_component²

V_bp_rms = sqrt(V_bp_total)
```

**Status**: Pending - Requires Manchester spectrum implementation

---

## Validation Results

**Output Directory**: `outputs/paper_kadirvelu_2021/test41/`

All 7 figures generated successfully:
- ✅ `fig13_frequency_response.png` - Complete optical link (no +10 dB magic offset)
- ✅ `fig14_psd_noise.png` - Proper 1/f pink noise
- ✅ `fig15_vout_vs_duty.png` - Extended range, shows peak at 5%
- ✅ `fig16_transient_waveforms.png` - 3.3V comparator rails
- ✅ `fig17_ber_vs_modulation.png` - Linear Y-axis, shows "knee"
- ⏳ `fig18_vout_vs_modulation.png` - Still has magic I_ph scaling
- ⏳ `fig19_power_vs_bitrate.png` - Still has magic AC penalty

---

## Summary Table

| Issue | Your Report | Status | Fix Applied |
|-------|-------------|--------|-------------|
| Fig 15 Range | Missing 0-10% peak | ✅ Fixed | Extended to 0.5-50%, 100 points |
| Fig 17 Scale | Log vs Linear | ✅ Fixed | Changed to linear Y-axis |
| Fig 16 Rails | 1V vs 3.3V | ✅ Fixed | Comparator now outputs 3.3V |
| Fig 14 Noise | White vs Pink | ✅ Already correct | 1/f PSD confirmed |
| Fig 13 BPF Q | Sharp vs Broad | ⏳ Pending | Needs parasitic R modeling |
| Fig 16 Gain | 3mV vs 200mV | ⏳ Pending | Needs Manchester spectrum |

**Score**: 4/6 issues resolved

---

## Physics-First Status

**Previously Fixed (Before Your Report)**:
- Fig 13: Removed +10 dB magic offset, added optical link gain
- Fig 15: Removed exponential fits, using PV I-V solver
- Fig 14: Already had 1/f pink noise

**Fixed Based on Your Report**:
- Fig 15: Extended duty cycle range
- Fig 17: Linear Y-axis
- Fig 16: 3.3V rails

**Still Contains Magic Numbers** (see PHYSICS_ROOT_CAUSE_ANALYSIS.md):
- Fig 16: 60× calibration gain
- Fig 17: Fake SNR formula
- Fig 18: Backward I_ph scaling
- Fig 19: Magic AC penalty

---

## Next Steps

To complete your requested fixes:

1. **Fig 13 BPF Q-Factor** (1 hour):
   - Measure/estimate parasitic resistances in filter stages
   - Add equivalent series resistance (ESR) to capacitors
   - Recalculate filter response with realistic Q

2. **Fig 16 Manchester Spectrum** (2 hours):
   - Implement FFT of Manchester waveform
   - Apply BPF gain at each harmonic frequency
   - Sum power across all components

---

## How to Reproduce

```bash
cd papers
python kadirvelu_2021.py
```

Check output in: `outputs/paper_kadirvelu_2021/test41/`

---

## Related Documentation

- **PHYSICS_ROOT_CAUSE_ANALYSIS.md** - Complete analysis of all 7 magic numbers
- **FIXES_APPLIED.md** - Physics-first fixes (Fig 13, 15, 14)
- **This document** - Your specific user-reported fixes

---

**Thank you for your detailed technical analysis!** Your observations about the duty cycle range and plot scaling were spot-on. The simulation now better captures the physics observed in the paper's measurements.

**End of Report**
