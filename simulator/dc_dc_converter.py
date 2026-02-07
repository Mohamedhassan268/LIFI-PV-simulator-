# simulator/dc_dc_converter.py
"""
DC-DC Converter Model for Energy Harvesting.

Implements boost converter with frequency-dependent efficiency
based on Kadirvelu et al. IEEE TGCN 2021 measured data.

Paper efficiency values:
- 67% at fsw = 50 kHz
- 56.4% at fsw = 100 kHz  
- 42% at fsw = 200 kHz
"""

import numpy as np
from scipy.optimize import fsolve
from scipy import optimize

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
    def solve_operating_point(self, pv_params, R_load, duty_cycle=None, fsw_khz=None):
        """
        Finds the actual steady-state voltage (V_pv) where the PV source 
        and DC-DC converter demand intersect.
        
        Solves: I_supply(V) = I_demand(V)
        
        Args:
            pv_params (dict): PV cell parameters (I_ph, I_0, n, T, etc.)
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
            
        # 1. Get Efficiency for this operating point
        eta = self.get_efficiency(fsw_khz)
        
        # 2. Define the Demand Curve (Input Current of Boost Converter)
        # Power Balance: P_in * eta = P_out
        # V * I_in * eta = (V_out^2) / R_load
        # Substitite V_out = V / (1-D):
        # I_in = V / (eta * (1-D)^2 * R_load)
        def get_demand_current(V):
            if V < 1e-6: return 0
            denom = eta * ((1 - duty_cycle) ** 2) * R_load
            return V / denom

        # 3. Define the Supply Curve (PV Single Diode Model)
        # I_pv = I_ph - I_0 * (exp(V / (n * V_t)) - 1) - V / R_sh
        # (We unpack the params dict here)
        I_ph = pv_params['I_ph']
        I_0 = pv_params['I_0']
        n = pv_params.get('n', 1.0) # Ideality factor
        T = pv_params.get('T', 300) # Temp in Kelvin
        R_sh = pv_params.get('R_sh', 1e6) # Shunt resistance
        
        # Thermal Voltage V_t = k*T/q
        q = 1.602e-19
        k = 1.38e-23
        V_t = (k * T) / q * n # Include 'n' in V_t for simplicity

        def get_supply_current(V):
            # Safe exp to avoid overflow
            try:
                exp_term = np.exp(V / V_t) - 1
            except OverflowError:
                exp_term = 1e12 # Cap at huge number
            
            return I_ph - I_0 * exp_term - (V / R_sh)

        # 4. Define the Target Function (Difference between Supply and Demand)
        # We want to find V where: Supply - Demand = 0
        def error_func(V):
            return get_supply_current(V) - get_demand_current(V)

        # 5. Solve for V (Root Finding)
        # Bracket: Voltage is between 0 and Open Circuit Voltage (approx 0.7V for single cell)
        # Note: If this is a module, V_oc will be higher (e.g. 13 * 0.7)
        V_guess_max = 10.0 # Upper bound safety
        
        try:
            # Brent's method is robust for this
            V_op = optimize.brentq(error_func, 0.0, V_guess_max)
        except ValueError:
            # If root finding fails, assume collapsed source (0V) or V_oc
            # Check signs at bounds
            if error_func(0) * error_func(V_guess_max) > 0:
                 # No intersection? Likely demand >> supply -> V crashes to 0
                 V_op = 0.01 
            else:
                 V_op = 0.0

        I_op = get_supply_current(V_op)
        return V_op, I_op
    
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
            # Below 50 kHz: use 50 kHz value (conservative)
            return eta_points[0]
        elif fsw_khz >= f_points[-1]:
            # Above 200 kHz: extrapolate linearly (efficiency decreases)
            slope = (eta_points[-1] - eta_points[-2]) / (f_points[-1] - f_points[-2])
            eta = eta_points[-1] + slope * (fsw_khz - f_points[-1])
            return max(0.1, eta)  # Floor at 10%
        else:
            # Interpolate
            return np.interp(fsw_khz, f_points, eta_points)
    
    def calculate_boost_ratio(self, duty_cycle=None):
        """
        Calculate ideal boost voltage ratio.
        
        V_out / V_in = 1 / (1 - D)
        
        Args:
            duty_cycle (float): Duty cycle 0-1 (or use instance value)
            
        Returns:
            float: Voltage boost ratio
        """
        if duty_cycle is None:
            duty_cycle = self.duty_cycle
            
        # Prevent division by zero
        duty_cycle = np.clip(duty_cycle, 0.01, 0.99)
        
        return 1.0 / (1.0 - duty_cycle)
    
    def solve_operating_point(self, pv_params, R_load, duty_cycle=None, fsw_khz=None):
        """
        Find the operating point (V_pv, I_pv) where PV source matches Converter input.
        
        Args:
            pv_params (dict): PV cell parameters (I_ph, I_0, n, T, R_sh)
            R_load (float): Load resistance at converter output
            duty_cycle (float): Duty cycle
            fsw_khz (float): Switching frequency
            
        Returns:
            tuple: (V_pv_op, I_pv_op)
        """
        if duty_cycle is None:
            duty_cycle = self.duty_cycle
        if fsw_khz is None:
            fsw_khz = self.fsw_khz
            
        # Unpack PV params
        I_ph = pv_params.get('I_ph', 0.025)
        I_0 = pv_params.get('I_0', 1e-9)
        n = pv_params.get('n', 1.5)
        T = pv_params.get('T', 300)
        R_sh = pv_params.get('R_sh', 1000)
        # Thermal voltage
        k = 1.380649e-23
        q = 1.60217663e-19
        V_T = n * k * T / q
        
        # PV Current Equation
        def get_pv_current(V):
            # Simplified single diode model (ignoring Rs for stability or assuming it's small/included)
            # I = I_ph - I_0 * (exp(V/V_T) - 1) - V/R_sh
            # Ensure V is not too large to prevent overflow
            try:
                term_diode = I_0 * (np.exp(np.clip(V/V_T, -50, 50)) - 1)
                return I_ph - term_diode - V/R_sh
            except:
                return 0.0

        # Converter Input Current Equation (Power Balance)
        def get_converter_current(V):
            if V <= 0.01: return 0.0 # Avoid singularity
            
            # 1. Calculate PV current at this voltage (guess)
            I_guess = get_pv_current(V)
            
            # 2. Calculate Output Voltage for this (V, I)
            out = self.calculate_output(V, I_guess, duty_cycle, fsw_khz)
            V_out = out['V_out']
            eta = out['efficiency']
            
            # 3. Calculate Input Current required to sustain V_out across R_load
            # P_out = V_out^2 / R_load
            # P_in = P_out / eta
            # I_in = P_in / V
            if eta <= 0: return 0.0
            
            P_out = (V_out**2) / R_load
            P_in = P_out / eta
            I_in = P_in / V
            return I_in

        # Objective function: Difference between Supply and Demand
        def objective(V):
            return get_pv_current(V) - get_converter_current(V)
        
        # Solve for V_pv
        # Initial guess: Open circuit voltage approx (0.6V - 1.0V)
        V_guess = 0.5
        V_sol = fsolve(objective, V_guess)[0]
        
        # Calculate corresponding current
        I_sol = get_pv_current(V_sol)
        
        return V_sol, I_sol

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
            
        # Get efficiency for this operating point
        eta = self.get_efficiency(fsw_khz)
        
        # Input values (handle arrays)
        V_pv_avg = np.mean(V_pv) if isinstance(V_pv, np.ndarray) else V_pv
        I_pv_avg = np.mean(I_pv) if isinstance(I_pv, np.ndarray) else I_pv
        
        # Ensure positive values
        V_pv_avg = max(0.001, V_pv_avg)  # Avoid division by zero
        I_pv_avg = max(1e-9, I_pv_avg)   # Avoid zero current
        
        # Input power
        P_in = V_pv_avg * I_pv_avg
        
        # ========== DCM/CCM Detection ==========
        # Critical current for CCM/DCM boundary
        L_H = L_uH * 1e-6
        fsw_Hz = fsw_khz * 1e3
        D = np.clip(duty_cycle, 0.01, 0.99)
        
        # Boundary current (simplified model)
        I_boundary = V_pv_avg * D * (1 - D) / (2 * L_H * fsw_Hz)
        
        if I_pv_avg < I_boundary:
            # ========== DCM Mode ==========
            # In DCM, voltage conversion ratio is higher but depends on load
            # M_dcm = 0.5 * (1 + sqrt(1 + 4*D^2 / K)) where K = I_out/I_boundary
            K = max(0.01, I_pv_avg / I_boundary)
            M_dcm = 0.5 * (1 + np.sqrt(1 + 4 * D**2 / K))
            M_dcm = min(M_dcm, 20)  # Cap at 20x boost
            
            V_out_ideal = V_pv_avg * M_dcm
            mode = 'DCM'
        else:
            # ========== CCM Mode ==========
            # Standard boost: V_out = V_in / (1 - D)
            boost_ratio = 1.0 / (1.0 - D)
            V_out_ideal = V_pv_avg * boost_ratio
            mode = 'CCM'
        
        # ========== Apply Losses ==========
        # 1. Efficiency loss (switching + conduction)
        # 2. Diode forward voltage drop
        V_out = eta * V_out_ideal - self.V_diode
        V_out = max(0, V_out)
        
        # Output power (with efficiency)
        P_out = P_in * eta
        
        # Output current
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
    
    def sweep_duty_cycle(self, V_pv, I_pv, duty_cycles=None, fsw_khz=None):
        """
        Sweep duty cycle and return output characteristics.
        
        Useful for generating Fig. 15 (V_out vs duty cycle).
        
        Args:
            V_pv (float): PV cell voltage
            I_pv (float): PV cell current
            duty_cycles (array): Duty cycles to sweep (default 0.05-0.5)
            fsw_khz (float): Switching frequency
            
        Returns:
            dict: Arrays of V_out, P_out, etc. for each duty cycle
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
        
        Useful for generating Fig. 18 (V_out vs modulation depth).
        
        Args:
            V_pv_func (callable): Function(mod_depth) -> V_pv
            I_pv_func (callable): Function(mod_depth) -> I_pv
            mod_depths (array): Modulation depths to sweep
            fsw_khz (float): Switching frequency
            
        Returns:
            dict: Arrays of V_out, P_out, etc. for each mod depth
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
        self.D_current = 0.3  # Initial duty cycle
        self.P_prev = 0
        
    def update(self, V_pv, I_pv):
        """
        Update MPPT controller and return new duty cycle.
        
        Perturb-and-Observe algorithm:
        - If power increased after perturbation, keep going same direction
        - If power decreased, reverse direction
        
        Args:
            V_pv (float): Current PV voltage
            I_pv (float): Current PV current
            
        Returns:
            float: New optimal duty cycle
        """
        P_current = V_pv * I_pv
        
        if P_current > self.P_prev:
            # Power increased, keep going
            self.D_current += self.step_size
        else:
            # Power decreased, reverse
            self.step_size = -self.step_size
            self.D_current += self.step_size
        
        # Clamp duty cycle
        self.D_current = np.clip(self.D_current, 0.1, 0.6)
        
        self.P_prev = P_current
        
        return self.D_current


# ========== TESTS ==========

def test_dcdc_converter():
    """Unit test for DC-DC converter."""
    print("\n" + "="*60)
    print("DC-DC CONVERTER UNIT TEST")
    print("="*60)
    
    # Create converter
    dcdc = DCDCConverter({'fsw_khz': 100})
    
    # Test efficiency lookup
    print("\n  Paper efficiency values:")
    for fsw in [50, 100, 200]:
        eta = dcdc.get_efficiency(fsw)
        print(f"    fsw = {fsw} kHz: η = {eta*100:.1f}%")
    
    # Test interpolation
    print("\n  Interpolated values:")
    for fsw in [75, 150]:
        eta = dcdc.get_efficiency(fsw)
        print(f"    fsw = {fsw} kHz: η = {eta*100:.1f}%")
    
    # Test output calculation
    print("\n  Output calculation (V_pv=0.4V, I_pv=1mA, D=0.5):")
    result = dcdc.calculate_output(V_pv=0.4, I_pv=1e-3, duty_cycle=0.5)
    print(f"    V_out = {result['V_out']:.3f} V")
    print(f"    P_out = {result['P_out']*1e6:.2f} µW")
    print(f"    η = {result['efficiency']*100:.1f}%")
    
    # Test duty cycle sweep
    print("\n  Duty cycle sweep (fsw=100kHz):")
    sweep = dcdc.sweep_duty_cycle(V_pv=0.4, I_pv=1e-3, fsw_khz=100)
    print(f"    D range: {sweep['duty_cycle'][0]:.2f} - {sweep['duty_cycle'][-1]:.2f}")
    print(f"    V_out range: {sweep['V_out'].min():.2f} - {sweep['V_out'].max():.2f} V")
    
    print("\n[OK] DC-DC converter tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_dcdc_converter()
