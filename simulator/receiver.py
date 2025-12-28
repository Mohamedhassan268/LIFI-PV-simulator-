# simulator/receiver.py
"""
PV Cell Receiver: Photodiode + Junction Circuit

Equation 6: Photodiode Response
  I_ph = R * P_rx

Equation 7: PV Junction Diode Equation
  I = I_0*(exp(qV/k_B*T) - 1) - I_ph

Equation 8: PV Cell RC Circuit (ODE)
  dV/dt = [I_ph - V/R_sh - I_0*exp(V/V_T)] / C_j

This is a first-order nonlinear ODE solved with Euler integration.
"""

import numpy as np
from scipy import signal
from utils.constants import (
    Q,
    K_B,
    PHOTODIODE_RESPONSIVITY_A_PER_W,
    PHOTODIODE_DARK_CURRENT_NA,
    PHOTODIODE_JUNCTION_CAP_PF,
    PHOTODIODE_SHUNT_RESISTANCE_MOHM,
    ROOM_TEMPERATURE_K,
    thermal_voltage,
)


class PVReceiver:
    """
    PV Cell Receiver with RC circuit dynamics.
    
    This class:
    1. Converts optical power to photocurrent (Equation 6)
    2. Solves PV junction ODE (Equation 8)
    3. Returns voltage waveform V(t)
    """
    def apply_tia(self, I_ph, t, R_tia=50e3, f_3db=3e6):
        """
        Simple transimpedance amplifier (TIA) model.

        Args:
            I_ph (ndarray): Photocurrent versus time (A).
            t (ndarray): Time vector (s), same length as I_ph.
            R_tia (float): Transimpedance gain (ohms), e.g. 50 kΩ.
            f_3db (float): TIA 3 dB bandwidth (Hz), e.g. 3 MHz.

        Returns:
            V_tia (ndarray): TIA output voltage (V).
        """
        # Ideal current-to-voltage conversion: V = R * I
        V_ideal = R_tia * I_ph

        # Enforce finite bandwidth with 1st‑order low-pass
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        nyq = fs / 2.0
        wn = f_3db / nyq  # normalized cutoff (0–1)

        # If cutoff above Nyquist, skip filtering
        if wn >= 1.0:
            return V_ideal

        b, a = signal.butter(1, wn, btype='low')
        V_tia = signal.filtfilt(b, a, V_ideal)
        return V_tia
    
    def __init__(self, params=None):
        """
        Initialize PV receiver.
        
        Args:
            params (dict): Configuration with keys:
                - 'responsivity': R (A/W)
                - 'capacitance': C_j (pF)
                - 'shunt_resistance': R_sh (MOhm)
                - 'dark_current': I_0 (nA)
                - 'temperature': T (K)
        """
        
        if params is None:
            params = {}
        
        # Physical parameters
        self.R = params.get('responsivity', PHOTODIODE_RESPONSIVITY_A_PER_W)  # A/W
        self.C_j = params.get('capacitance', PHOTODIODE_JUNCTION_CAP_PF) * 1e-12  # pF -> F
        self.R_sh = params.get('shunt_resistance', PHOTODIODE_SHUNT_RESISTANCE_MOHM) * 1e6  # MOhm -> Ohm
        self.I_0 = params.get('dark_current', PHOTODIODE_DARK_CURRENT_NA) * 1e-9  # nA -> A
        self.T = params.get('temperature', ROOM_TEMPERATURE_K)  # K
        
        # Derived constants
        self.V_T = thermal_voltage(self.T)  # Thermal voltage (~26 mV at 300K)
        
        # Verify physically reasonable values
        assert 0.3 <= self.R <= 0.8, "Responsivity should be 0.3-0.8 A/W for Si"
        assert 10e-12 <= self.C_j <= 1000e-12, "Capacitance should be 10-1000 pF"
        assert 0.1e6 <= self.R_sh <= 100e6, "Shunt resistance should be 0.1-100 MOhm"
        
        print(f"[OK] PV Receiver initialized:")
        print(f"    R = {self.R:.2f} A/W")
        print(f"    C_j = {self.C_j*1e12:.1f} pF")
        print(f"    R_sh = {self.R_sh*1e-6:.1f} MOhm")
        print(f"    I_0 = {self.I_0*1e9:.1f} nA")
        print(f"    V_T = {self.V_T*1e3:.1f} mV (at {self.T}K)")
        print(f"    C_j = {self.C_j*1e12:.1f} pF")
        print(f"    DEBUG: Loaded from constants = {PHOTODIODE_JUNCTION_CAP_PF} pF")  # ADD THIS

    
    def optical_to_current(self, P_rx):
        """
        Convert optical power to photocurrent.
        
        Equation 6: I_ph = R * P_rx
        
        Args:
            P_rx (array): Received optical power (Watts)
        
        Returns:
            I_ph (array): Photocurrent (Amperes)
        """
        
        I_ph = self.R * P_rx
        
        return I_ph
    
    def solve_pv_circuit(self, I_ph, t, method='euler', verbose=False):
        """
        Solve PV cell RC circuit ODE.
        
        Equation 8: dV/dt = [I_ph(t) - V/R_sh - I_0*exp(V/V_T)] / C_j
        
        This is a first-order nonlinear ODE.
        We use simple Euler integration for clarity.
        
        Args:
            I_ph (array): Photocurrent (Amperes)
            t (array): Time vector (seconds)
            method (str): 'euler' or 'rk45' (Runge-Kutta)
            verbose (bool): Print progress
        
        Returns:
            V (array): Junction voltage (Volts)
        """
        
        dt = t[1] - t[0]  # Time step
        V = np.zeros(len(t))  # Initialize voltage array
        V[0] = 0.0  # Start at V=0
        
        if method == 'euler':
            # ========== EULER INTEGRATION ==========
            for i in range(len(t) - 1):
                # Current balance (Kirchhoff's current law)
                # I_ph = V/R_sh + C_j*dV/dt + I_diode
                
                # Shunt leakage: I_sh = V / R_sh
                I_leak_sh = V[i] / self.R_sh
                
                # Diode leakage: I_diode = I_0 * (exp(V/V_T) - 1)
                # For numerical stability, clip V_norm to avoid overflow
                V_norm = V[i] / self.V_T
                V_norm = np.clip(V_norm, -100, 100)  # Prevent overflow
                I_diode = self.I_0 * (np.exp(V_norm) - 1)
                
                # Differential equation: dV/dt = [I_ph - I_sh - I_diode] / C_j
                dV_dt = (I_ph[i] - I_leak_sh - I_diode) / self.C_j
                
                # Euler step
                V[i+1] = V[i] + dV_dt * dt
                
                # Safety: clip voltage to reasonable range
                V[i+1] = np.clip(V[i+1], -0.5, 0.5)  # +/-500 mV max
        
        elif method == 'rk45':
            # Optional: use scipy RK45 for better accuracy
            # For now, Euler is sufficient
            raise NotImplementedError("RK45 not yet implemented. Use 'euler'.")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if verbose:
            print(f"\nPV Circuit Solution (Euler):")
            print(f"  dt: {dt*1e6:.2f} us")
            print(f"  Time span: {t[0]*1e3:.2f} - {t[-1]*1e3:.2f} ms")
            print(f"  V_min: {V.min()*1e3:.2f} mV")
            print(f"  V_max: {V.max()*1e3:.2f} mV")
            print(f"  V_final: {V[-1]*1e3:.2f} mV")
            print(f"  Steady-state reached: {np.abs(V[-1] - V[-2]) < 1e-4}")
        
        return V
    
    def estimate_open_circuit_voltage(self, I_ph_avg):
        """
        Estimate open-circuit voltage (no load current).
        
        V_oc ~ V_T * ln(I_ph / I_0)
        
        Args:
            I_ph_avg (float): Average photocurrent (Amperes)
        
        Returns:
            V_oc (float): Open-circuit voltage (Volts)
        """
        
        if I_ph_avg <= 0:
            return 0.0
        
        V_oc = self.V_T * np.log(I_ph_avg / self.I_0)
        
        return V_oc


# ========== TESTS ==========

def test_receiver():
    """Unit test for PV receiver."""
    
    print("\n" + "="*60)
    print("PV RECEIVER UNIT TEST")
    print("="*60)
    
    # Create receiver
    rx = PVReceiver()
    
    # Create a simple received power waveform
    # Constant light at 2 uW
    P_rx = np.ones(10000) * 2e-6  # 2 uW
    t = np.arange(10000) / 1e6  # 1 us sample period -> 10 ms duration
    
    # Convert to photocurrent
    I_ph = rx.optical_to_current(P_rx)
    
    print(f"\nPhotocurrent:")
    print(f"  I_ph (2uW): {I_ph[0]*1e6:.2f} uA")
    print(f"  I_ph (avg): {I_ph.mean()*1e6:.2f} uA")
    
    # Estimate open-circuit voltage
    V_oc_est = rx.estimate_open_circuit_voltage(I_ph.mean())
    print(f"  V_oc (est): {V_oc_est*1e3:.1f} mV")
    
    # Solve PV circuit ODE
    V = rx.solve_pv_circuit(I_ph, t, verbose=True)
    
    # Validation
    assert V.min() >= -0.5, "[ERROR] Voltage too negative!"
    assert V.max() <= 0.5, "[ERROR] Voltage too positive!"
    assert len(V) == len(t), "[ERROR] Voltage length mismatch!"
    assert np.isnan(V).sum() == 0, "[ERROR] NaN detected in voltage!"
    
    print("\n[OK] All receiver tests passed!")
    print("="*60)
    
    return V


if __name__ == "__main__":
    test_receiver()
