# simulator/dc_dc_converter.py
"""
DC-DC Converter Model for Energy Harvesting.

Implements boost converter with frequency-dependent efficiency
based on Kadirvelu et al. IEEE TGCN 2021 measured data.

Paper efficiency values:
- 67% at fsw = 50 kHz
- 56.4% at fsw = 100 kHz  
- 42% at fsw = 200 kHz

FIXES:
- Safe exponential clipping in solve_operating_point (prevents np.exp overflow)
- Robust root-finding with bracket search fallback
- Safe exp in PV diode equations throughout
"""

import numpy as np
from scipy.optimize import fsolve, brentq
from scipy import optimize


# Safe exponential to prevent overflow everywhere in this module
def _safe_exp(x, limit=500):
    """Clipped exponential: exp(clip(x, -limit, limit))."""
    return np.exp(np.clip(x, -limit, limit))


class DCDCConverter:
    """
    Boost DC-DC converter model for solar cell energy harvesting.
    
    Models voltage boost with switching losses that increase
    with switching frequency (as observed in paper measurements).
    """
    
    # Paper-measured efficiency values
    PAPER_EFFICIENCY_DATA = {
        50: 0.67,    # 67% at 50 kHz
        100: 0.564,  # 56.4% at 100 kHz
        200: 0.42,   # 42% at 200 kHz
    }
    
    def __init__(self, params=None):
        """
        Initialize DC-DC converter.
        
        Args:
            params (dict): Configuration with optional keys:
                - 'fsw_khz': Switching frequency (default 100 kHz)
                - 'duty_cycle': Duty cycle 0-1 (default 0.5)
                - 'efficiency_mode': 'paper' (measured) or 'fixed' (default 'paper')
                - 'fixed_efficiency': If mode='fixed', use this value (default 0.85)
                - 'v_diode_drop': Diode forward voltage (default 0.3V)
                - 'r_on_mohm': MOSFET on-resistance in mΩ (default 100)
        """
        if params is None:
            params = {}
        
        self.fsw_khz = params.get('fsw_khz', 100)
        self.duty_cycle = params.get('duty_cycle', 0.5)
        self.efficiency_mode = params.get('efficiency_mode', 'paper')
        self.fixed_efficiency = params.get('fixed_efficiency', 0.85)
        self.V_diode = params.get('v_diode_drop', 0.3)  # Volts
        self.R_on = params.get('r_on_mohm', 100) * 1e-3  # mΩ to Ω
        
    def get_efficiency(self, fsw_khz=None):
        """
        Get converter efficiency for given switching frequency.
        
        Uses paper-measured values with interpolation for other frequencies.
        
        Args:
            fsw_khz (float): Switching frequency in kHz (or use instance value)
            
        Returns:
            float: Efficiency (0-1)
        """
        if fsw_khz is None:
            fsw_khz = self.fsw_khz
            
        if self.efficiency_mode == 'fixed':
            return self.fixed_efficiency
        
        # Paper data points
        f_points = np.array([50, 100, 200])
        eta_points = np.array([0.67, 0.564, 0.42])
        
        # Interpolate/extrapolate
        if fsw_khz <= f_points[0]:
            return eta_points[0]
        elif fsw_khz >= f_points[-1]:
            slope = (eta_points[-1] - eta_points[-2]) / (f_points[-1] - f_points[-2])
            eta = eta_points[-1] + slope * (fsw_khz - f_points[-1])
            return max(0.1, eta)  # Floor at 10%
        else:
            return float(np.interp(fsw_khz, f_points, eta_points))
    
    def calculate_boost_ratio(self, duty_cycle=None):
        """
        Calculate ideal boost voltage ratio: V_out / V_in = 1 / (1 - D).
        
        Args:
            duty_cycle (float): Duty cycle 0-1 (or use instance value)
            
        Returns:
            float: Voltage boost ratio
        """
        if duty_cycle is None:
            duty_cycle = self.duty_cycle
        duty_cycle = np.clip(duty_cycle, 0.01, 0.99)
        return 1.0 / (1.0 - duty_cycle)
    
    def calculate_output(self, V_pv, I_pv, duty_cycle=None, fsw_khz=None, L_uH=100):
        """
        Calculate converter output voltage and power.
        
        Uses proper boost converter physics with CCM/DCM detection.
        
        Args:
            V_pv (float or ndarray): PV cell voltage in Volts
            I_pv (float or ndarray): PV cell current in Amperes
            duty_cycle (float): Duty cycle 0-1 (optional)
            fsw_khz (float): Switching frequency in kHz (optional)
            L_uH (float): Inductor value in µH for DCM calculation
            
        Returns:
            dict: Output values including V_out, P_out, efficiency
        """
        if duty_cycle is None:
            duty_cycle = self.duty_cycle
        if fsw_khz is None:
            fsw_khz = self.fsw_khz
            
        eta = self.get_efficiency(fsw_khz)
        
        V_pv_avg = float(np.mean(V_pv)) if isinstance(V_pv, np.ndarray) else float(V_pv)
        I_pv_avg = float(np.mean(I_pv)) if isinstance(I_pv, np.ndarray) else float(I_pv)
        
        V_pv_avg = max(0.001, V_pv_avg)
        I_pv_avg = max(1e-9, I_pv_avg)
        
        P_in = V_pv_avg * I_pv_avg
        
        # DCM/CCM Detection
        L_H = L_uH * 1e-6
        fsw_Hz = fsw_khz * 1e3
        D = np.clip(duty_cycle, 0.01, 0.99)
        
        I_boundary = V_pv_avg * D * (1 - D) / (2 * L_H * fsw_Hz)
        
        if I_pv_avg < I_boundary:
            # DCM Mode
            K = max(0.01, I_pv_avg / I_boundary)
            M_dcm = 0.5 * (1 + np.sqrt(1 + 4 * D**2 / K))
            M_dcm = min(M_dcm, 20)
            V_out_ideal = V_pv_avg * M_dcm
            mode = 'DCM'
        else:
            # CCM Mode
            V_out_ideal = V_pv_avg / (1.0 - D)
            mode = 'CCM'
        
        # Apply losses
        V_out = eta * V_out_ideal - self.V_diode
        V_out = max(0, V_out)
        P_out = P_in * eta
        I_out = P_out / V_out if V_out > 0 else 0
        
        return {
            'V_out': V_out,
            'P_out': P_out,
            'P_in': P_in,
            'I_out': I_out,
            'efficiency': eta,
            'boost_ratio': V_out / V_pv_avg if V_pv_avg > 0 else 0,
            'fsw_khz': fsw_khz,
            'duty_cycle': duty_cycle,
            'mode': mode,
            'I_boundary': I_boundary,
        }
    
    def solve_operating_point(self, pv_params, R_load, duty_cycle=None, fsw_khz=None):
        """
        Find the operating point (V_pv, I_pv) where PV source matches converter demand.
        
        Solves: I_supply(V) = I_demand(V) using robust root-finding.
        
        FIXED: Uses safe exponential to prevent overflow for large V/V_T ratios.
        
        Args:
            pv_params (dict): PV cell parameters:
                - 'I_ph': Photocurrent (A)
                - 'I_0': Dark/saturation current (A)
                - 'n': Ideality factor (default 1.0)
                - 'T': Temperature (K, default 300)
                - 'R_sh': Shunt resistance (Ω, default 1e6)
            R_load (float): Output load resistance in Ohms
            duty_cycle (float): Duty cycle (0-1)
            fsw_khz (float): Switching frequency
            
        Returns:
            tuple: (V_pv_operating, I_pv_operating)
        """
        if duty_cycle is None:
            duty_cycle = self.duty_cycle
        if fsw_khz is None:
            fsw_khz = self.fsw_khz
            
        eta = self.get_efficiency(fsw_khz)
        
        # Unpack PV parameters
        I_ph = pv_params.get('I_ph', 0.025)
        I_0 = pv_params.get('I_0', 1e-9)
        n = pv_params.get('n', 1.5)
        T = pv_params.get('T', 300)
        R_sh = pv_params.get('R_sh', 1000)
        
        # Thermal voltage (includes ideality factor)
        k = 1.380649e-23
        q = 1.60217663e-19
        V_T = n * k * T / q
        
        # PV supply current (single diode model)
        def get_pv_current(V):
            return I_ph - I_0 * (_safe_exp(V / V_T) - 1) - V / R_sh
        
        # Converter demand current (power balance)
        def get_demand_current(V):
            if V < 1e-6:
                return 0.0
            denom = eta * ((1 - duty_cycle) ** 2) * R_load
            if denom <= 0:
                return 0.0
            return V / denom
        
        # Objective: supply - demand = 0
        def objective(V):
            return get_pv_current(V) - get_demand_current(V)
        
        # Estimate V_oc for upper bracket (safe)
        V_oc_est = V_T * np.log(I_ph / I_0 + 1) if I_0 > 0 else 1.0
        V_oc_est = min(V_oc_est, 50.0)  # Safety cap
        
        # Try Brent's method first (most robust for bracketed root)
        try:
            # Verify bracket: objective should change sign in [0, V_oc]
            f_low = objective(1e-6)
            f_high = objective(V_oc_est)
            
            if f_low * f_high < 0:
                V_op = brentq(objective, 1e-6, V_oc_est, xtol=1e-8)
            else:
                # No sign change — try fsolve with a reasonable guess
                V_op = fsolve(objective, V_oc_est * 0.5, full_output=False)[0]
                V_op = np.clip(V_op, 0.0, V_oc_est)
        except (ValueError, RuntimeError):
            # Fallback: demand exceeds supply → voltage collapses
            V_op = 0.01
        
        I_op = get_pv_current(V_op)
        return float(V_op), float(I_op)
    
    def sweep_duty_cycle(self, V_pv, I_pv, duty_cycles=None, fsw_khz=None):
        """
        Sweep duty cycle and return output characteristics.
        
        Args:
            V_pv (float): PV cell voltage
            I_pv (float): PV cell current
            duty_cycles (array): Duty cycles to sweep (default 0.05-0.5)
            fsw_khz (float): Switching frequency
            
        Returns:
            dict: Arrays of V_out, P_out, efficiency for each duty cycle
        """
        if duty_cycles is None:
            duty_cycles = np.linspace(0.05, 0.5, 20)
        if fsw_khz is None:
            fsw_khz = self.fsw_khz
            
        results = {
            'duty_cycle': duty_cycles,
            'V_out': [],
            'P_out': [],
            'efficiency': [],
        }
        
        for D in duty_cycles:
            out = self.calculate_output(V_pv, I_pv, D, fsw_khz)
            results['V_out'].append(out['V_out'])
            results['P_out'].append(out['P_out'])
            results['efficiency'].append(out['efficiency'])
        
        results['V_out'] = np.array(results['V_out'])
        results['P_out'] = np.array(results['P_out'])
        results['efficiency'] = np.array(results['efficiency'])
        
        return results
    
    def sweep_modulation_depth(self, V_pv_func, I_pv_func, mod_depths=None, fsw_khz=None):
        """
        Sweep modulation depth and return output characteristics.
        
        Args:
            V_pv_func (callable): Function(mod_depth) -> V_pv
            I_pv_func (callable): Function(mod_depth) -> I_pv
            mod_depths (array): Modulation depths to sweep
            fsw_khz (float): Switching frequency
            
        Returns:
            dict: Arrays of V_out, P_out for each modulation depth
        """
        if mod_depths is None:
            mod_depths = np.linspace(0.1, 1.0, 20)
        if fsw_khz is None:
            fsw_khz = self.fsw_khz
            
        results = {
            'mod_depth': mod_depths,
            'V_out': [],
            'P_out': [],
        }
        
        for m in mod_depths:
            V_pv = V_pv_func(m)
            I_pv = I_pv_func(m)
            out = self.calculate_output(V_pv, I_pv, self.duty_cycle, fsw_khz)
            results['V_out'].append(out['V_out'])
            results['P_out'].append(out['P_out'])
        
        results['V_out'] = np.array(results['V_out'])
        results['P_out'] = np.array(results['P_out'])
        
        return results


class MPPTController:
    """
    Maximum Power Point Tracking controller.
    
    Uses Perturb-and-Observe algorithm to find optimal duty cycle.
    """
    
    def __init__(self, converter, step_size=0.01):
        """
        Initialize MPPT controller.
        
        Args:
            converter (DCDCConverter): Converter to control
            step_size (float): Duty cycle step for P&O algorithm
        """
        self.converter = converter
        self.step_size = step_size
        self.D_current = 0.3
        self.P_prev = 0
        
    def update(self, V_pv, I_pv):
        """
        Update MPPT and return new duty cycle (Perturb-and-Observe).
        
        Args:
            V_pv (float): Current PV voltage
            I_pv (float): Current PV current
            
        Returns:
            float: New optimal duty cycle
        """
        P_current = V_pv * I_pv
        
        if P_current > self.P_prev:
            self.D_current += self.step_size
        else:
            self.step_size = -self.step_size
            self.D_current += self.step_size
        
        self.D_current = np.clip(self.D_current, 0.1, 0.6)
        self.P_prev = P_current
        
        return self.D_current


# ========== TESTS ==========

def test_dcdc_converter():
    """Unit test for DC-DC converter — validates safe exp and operating point solver."""
    
    print("\n" + "="*60)
    print("DC-DC CONVERTER UNIT TEST")
    print("="*60)
    
    dcdc = DCDCConverter({'fsw_khz': 100})
    
    # --- Test 1: Efficiency lookup ---
    print("\n[Test 1] Paper efficiency values...")
    for fsw in [50, 100, 200]:
        eta = dcdc.get_efficiency(fsw)
        expected = DCDCConverter.PAPER_EFFICIENCY_DATA[fsw]
        assert abs(eta - expected) < 0.001, f"[ERROR] fsw={fsw}: {eta} != {expected}"
        print(f"  fsw={fsw} kHz: η={eta*100:.1f}%")
    print("  [OK]")
    
    # --- Test 2: Interpolation ---
    print("\n[Test 2] Interpolated efficiency...")
    for fsw in [75, 150]:
        eta = dcdc.get_efficiency(fsw)
        print(f"  fsw={fsw} kHz: η={eta*100:.1f}%")
    assert dcdc.get_efficiency(75) > dcdc.get_efficiency(150), \
        "[ERROR] 75kHz should be more efficient than 150kHz"
    print("  [OK]")
    
    # --- Test 3: Output calculation ---
    print("\n[Test 3] Output calculation...")
    result = dcdc.calculate_output(V_pv=0.4, I_pv=1e-3, duty_cycle=0.5)
    print(f"  V_out = {result['V_out']:.3f} V")
    print(f"  P_out = {result['P_out']*1e6:.2f} µW")
    print(f"  Mode: {result['mode']}")
    assert result['V_out'] > 0, "[ERROR] V_out should be positive"
    assert result['P_out'] > 0, "[ERROR] P_out should be positive"
    print("  [OK]")
    
    # --- Test 4: Safe exponential (CRITICAL FIX) ---
    print("\n[Test 4] Safe exponential in solve_operating_point...")
    
    # Test with normal parameters
    pv_normal = {'I_ph': 0.025, 'I_0': 1e-9, 'n': 1.5, 'T': 300, 'R_sh': 1000}
    V_op, I_op = dcdc.solve_operating_point(pv_normal, R_load=1000, duty_cycle=0.3)
    print(f"  Normal: V_op={V_op:.4f}V, I_op={I_op*1e3:.3f}mA")
    assert 0 < V_op < 2.0, f"[ERROR] V_op={V_op} out of range"
    assert I_op > 0, f"[ERROR] I_op={I_op} should be positive"
    
    # Test with extreme parameters that would overflow old code
    pv_extreme = {'I_ph': 0.5, 'I_0': 1e-15, 'n': 1.0, 'T': 300, 'R_sh': 1e6}
    try:
        V_op_ext, I_op_ext = dcdc.solve_operating_point(pv_extreme, R_load=100, duty_cycle=0.5)
        print(f"  Extreme: V_op={V_op_ext:.4f}V, I_op={I_op_ext*1e3:.3f}mA")
        assert np.isfinite(V_op_ext), "[ERROR] V_op is not finite"
        assert np.isfinite(I_op_ext), "[ERROR] I_op is not finite"
        print("  [OK] No overflow!")
    except OverflowError:
        print("  [FAIL] OverflowError — safe_exp not working!")
        raise
    
    # Test with very small I_0 (would cause V/V_T >> 500 in old code)
    pv_tiny_I0 = {'I_ph': 0.001, 'I_0': 1e-20, 'n': 1.0, 'T': 300, 'R_sh': 1e8}
    V_op_tiny, I_op_tiny = dcdc.solve_operating_point(pv_tiny_I0, R_load=5000, duty_cycle=0.2)
    print(f"  Tiny I_0: V_op={V_op_tiny:.4f}V, I_op={I_op_tiny*1e6:.3f}µA")
    assert np.isfinite(V_op_tiny), "[ERROR] Overflow with tiny I_0"
    print("  [OK] Safe exp handles tiny I_0")
    
    # --- Test 5: Duty cycle sweep ---
    print("\n[Test 5] Duty cycle sweep...")
    sweep = dcdc.sweep_duty_cycle(V_pv=0.4, I_pv=1e-3, fsw_khz=100)
    assert len(sweep['V_out']) == 20
    assert all(np.isfinite(sweep['V_out']))
    print(f"  D range: {sweep['duty_cycle'][0]:.2f} - {sweep['duty_cycle'][-1]:.2f}")
    print(f"  V_out range: {sweep['V_out'].min():.3f} - {sweep['V_out'].max():.3f} V")
    print("  [OK]")
    
    # --- Test 6: MPPT Controller ---
    print("\n[Test 6] MPPT controller...")
    mppt = MPPTController(dcdc)
    D_history = []
    for _ in range(20):
        D = mppt.update(0.4, 1e-3)
        D_history.append(D)
    assert 0.1 <= D_history[-1] <= 0.6, f"[ERROR] MPPT D={D_history[-1]} out of range"
    print(f"  Final D: {D_history[-1]:.3f}")
    print("  [OK]")
    
    print("\n" + "="*60)
    print("[OK] ALL DC-DC CONVERTER TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_dcdc_converter()
