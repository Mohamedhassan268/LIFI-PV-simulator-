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
    
    # ========== PAPER VALIDATION METHODS (Kadirvelu et al. 2021) ==========
    
    def apply_current_sense(self, I_ph, R_sense=1.0):
        """
        Convert photocurrent to voltage via current-sense resistor.
        
        Per IEEE paper: V_sense = I_ph * R_sense
        
        Args:
            I_ph (ndarray): Photocurrent (A)
            R_sense (float): Current-sense resistor (Ω), default 1Ω
            
        Returns:
            V_sense (ndarray): Voltage across sense resistor (V)
        """
        return I_ph * R_sense
    
    def apply_ina_gain(self, V_sense, gain=100):
        """
        Apply instrumentation amplifier (INA) gain.
        
        Per IEEE paper: INA gain = 100 (40 dB)
        
        Args:
            V_sense (ndarray): Input voltage from current-sense resistor (V)
            gain (float): INA voltage gain, default 100
            
        Returns:
            V_ina (ndarray): Amplified voltage (V)
        """
        return V_sense * gain
    
    def apply_ina_gain_physics(self, V_sense, t, gain_dc=100, gbw_hz=700000):
        """
        Apply INA gain with physical Gain-Bandwidth Product (GBW) limit.
        
        Model: 1st-order low-pass response where f_c = GBW / Gain.
        
        Args:
            V_sense (ndarray): Input voltage
            t (ndarray): Time vector
            gain_dc (float): DC Voltage gain
            gbw_hz (float): Gain-Bandwidth Product in Hz
            
        Returns:
            V_ina (ndarray): Amplified voltage with bandwidth limit
        """
        # Calculate cutoff frequency from GBW
        # f_c = GBW / Gain
        f_c = gbw_hz / gain_dc
        
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        nyq = fs / 2.0
        
        # Normalized cutoff
        wn = f_c / nyq
        
        # 1st order Butterworth low-pass represents the dominant pole
        if wn >= 1.0:
            # Bandwidth is higher than Nyquist, just apply gain
            return V_sense * gain_dc
            
        b, a = signal.butter(1, wn, btype='low')
        
        # Apply gain * filter
        V_ideal = V_sense * gain_dc
        V_ina = signal.lfilter(b, a, V_ideal)
        
        return V_ina

    def apply_bpf_physics(self, signal_in, t, Rhp, Chp, Rlp, Clp):
        """
        Apply Band-Pass Filter based on physical component values.
        
        Structure: High-Pass Stage -> Low-Pass Stage (Cascaded)
        Transfer Function: H(s) = [sRC_hp / (1 + sRC_hp)] * [1 / (1 + sRC_lp)]
        
        Args:
            signal_in (ndarray): Input signal
            t (ndarray): Time vector
            Rhp, Chp: High-pass resistor (Ohm) and capacitor (Farad)
            Rlp, Clp: Low-pass resistor (Ohm) and capacitor (Farad)
            
        Returns:
            signal_out (ndarray): Filtered signal
        """
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        nyq = fs / 2.0
        
        # Time constants
        tau_hp = Rhp * Chp
        tau_lp = Rlp * Clp
        
        # Corner frequencies
        fl = 1.0 / (2 * np.pi * tau_hp)
        fh = 1.0 / (2 * np.pi * tau_lp)
        
        # Digital filter design (bilinear transform equivalent via butterworth)
        # High-pass: fc = fl
        wn_hp = fl / nyq
        
        # Low-pass: fc = fh
        wn_lp = fh / nyq
        
        # Apply filters in series (mimicking stages)
        # Check Nyquist limits
        current_signal = signal_in
        
        # Stage 1: High-pass
        if wn_hp > 0 and wn_hp < 1:
            b_hp, a_hp = signal.butter(1, wn_hp, btype='high')
            current_signal = signal.lfilter(b_hp, a_hp, current_signal)
            
        # Stage 2: Low-pass
        if wn_lp > 0 and wn_lp < 1:
            b_lp, a_lp = signal.butter(1, wn_lp, btype='low')
            current_signal = signal.lfilter(b_lp, a_lp, current_signal)
            
        return current_signal

    def apply_rc_highpass(self, V_in, t, R_ohm, C_f):
        """
        Apply first-order RC High-Pass Filter (AC Coupling).
        
        Transfer Function: H(s) = sRC / (1 + sRC)
        
        Args:
            V_in (array): Input voltage
            t (array): Time vector
            R_ohm (float): Resistance (Ohms)
            C_f (float): Capacitance (Farads)
            
        Returns:
            V_out (array): Filtered voltage (DC removed)
        """
        dt = np.mean(np.diff(t))
        tau = R_ohm * C_f
        
        # Discrete equivalent (Bilinear)
        # H(z) = (2*tau / (2*tau + T)) * (1 - z^-1) / (1 - (2*tau - T)/(2*tau + T) * z^-1)
        
        alpha = (2*tau) / (2*tau + dt)
        beta  = (2*tau - dt) / (2*tau + dt)
        
        # b = [alpha, -alpha]
        # a = [1, -beta]
        # Standard form: a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] - a[1]*y[n-1]
        
        b = [alpha, -alpha]
        a = [1, -beta]
        
        V_out = signal.lfilter(b, a, V_in)
        return V_out

    def apply_voltage_amplifier(self, V_in, gain_linear, supply_voltage=None):
        """
        Apply Voltage Amplifier with optional rail clipping.
        
        Args:
            V_in (array): Input voltage
            gain_linear (float): Voltage gain (V/V)
            supply_voltage (float, optional): Max output voltage (clipping limit)
            
        Returns:
            V_out (array): Amplified voltage
        """
        V_out = V_in * gain_linear
        
        if supply_voltage is not None:
            # Clip to [0, Supply]
            V_out = np.clip(V_out, 0, supply_voltage)
            
        return V_out
    
    def apply_bandpass_filter(self, signal_in, t, f_low=700, f_high=10000, order=2):
        """
        Apply Butterworth bandpass filter (paper: 700 Hz – 10 kHz).
        
        Args:
            signal_in (ndarray): Input signal (V)
            t (ndarray): Time vector (s)
            f_low (float): Lower cutoff frequency (Hz), default 700
            f_high (float): Upper cutoff frequency (Hz), default 10000
            order (int): Filter order (default 2nd order)
            
        Returns:
            signal_out (ndarray): Filtered signal (V)
        """
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        nyq = fs / 2.0
        
        # Normalize frequencies
        low_norm = f_low / nyq
        high_norm = f_high / nyq
        
        # Ensure valid frequency range
        if low_norm >= 1.0 or high_norm >= 1.0:
            print(f"WARNING: BPF cutoffs ({f_low}, {f_high} Hz) exceed Nyquist ({nyq} Hz)")
            return signal_in
        
        if low_norm <= 0:
            low_norm = 0.001
            
        # Design bandpass filter
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        
        # Apply zero-phase filtering
        try:
            signal_out = signal.filtfilt(b, a, signal_in)
        except ValueError as e:
            print(f"WARNING: BPF failed ({e}), returning unfiltered signal")
            signal_out = signal_in
            
        return signal_out
    
    def paper_receiver_chain(self, I_ph, t, R_sense=1.0, ina_gain=100, 
                             f_low=700, f_high=10000, bpf_order=2, verbose=False):
        """
        Complete paper-mode receiver chain: Rsense → INA → BPF.
        
        This implements the Kadirvelu et al. 2021 receiver architecture
        for simultaneous data and power reception.
        
        Args:
            I_ph (ndarray): Photocurrent (A)
            t (ndarray): Time vector (s)
            R_sense (float): Current-sense resistor (Ω)
            ina_gain (float): INA voltage gain
            f_low (float): BPF lower cutoff (Hz)
            f_high (float): BPF upper cutoff (Hz)
            bpf_order (int): BPF filter order
            verbose (bool): Print debug info
            
        Returns:
            dict: Contains V_sense, V_ina, V_bp (filtered output)
        """
        # Step 1: Current-sense resistor
        V_sense = self.apply_current_sense(I_ph, R_sense)
        
        # Step 2: INA amplification
        V_ina = self.apply_ina_gain(V_sense, ina_gain)
        
        # Step 3: Bandpass filter
        V_bp = self.apply_bandpass_filter(V_ina, t, f_low, f_high, bpf_order)
        
        if verbose:
            print(f"\n  Paper Receiver Chain:")
            print(f"    R_sense = {R_sense} Ω")
            print(f"    INA gain = {ina_gain}× ({20*np.log10(ina_gain):.0f} dB)")
            print(f"    BPF: {f_low} Hz – {f_high} Hz (order {bpf_order})")
            print(f"    V_sense: {V_sense.min()*1e6:.2f} to {V_sense.max()*1e6:.2f} µV")
            print(f"    V_ina: {V_ina.min()*1e3:.2f} to {V_ina.max()*1e3:.2f} mV")
            print(f"    V_bp: {V_bp.min()*1e3:.2f} to {V_bp.max()*1e3:.2f} mV")
        
        return {
            'V_sense': V_sense,
            'V_ina': V_ina,
            'V_bp': V_bp,
        }
    
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
                - 'mode': 'photovoltaic' (default) or 'photoconductive'
        """
        
        if params is None:
            params = {}
        
        # Operating mode (Phase 4 enhancement)
        self.mode = params.get('mode', 'photovoltaic')
        assert self.mode in ('photovoltaic', 'photoconductive'), \
            "Mode must be 'photovoltaic' or 'photoconductive'"
        
        # Physical parameters
        self.R = params.get('responsivity', PHOTODIODE_RESPONSIVITY_A_PER_W)  # A/W
        
        # Base capacitance (before mode adjustment)
        base_capacitance_pf = params.get('capacitance', PHOTODIODE_JUNCTION_CAP_PF)
        
        # Photoconductive mode: reverse bias reduces junction capacitance by ~2-5x
        # and increases bandwidth, but eliminates energy harvesting
        if self.mode == 'photoconductive':
            # Typical reduction factor for reverse-biased junction
            capacitance_reduction = params.get('capacitance_reduction', 3.0)
            self.C_j = (base_capacitance_pf / capacitance_reduction) * 1e-12  # pF -> F
            self._mode_note = f"(C_j reduced {capacitance_reduction}x for reverse bias)"
        else:
            self.C_j = base_capacitance_pf * 1e-12  # pF -> F
            self._mode_note = ""
        
        self.R_sh = params.get('shunt_resistance', PHOTODIODE_SHUNT_RESISTANCE_MOHM) * 1e6  # MOhm -> Ohm
        self.I_0 = params.get('dark_current', PHOTODIODE_DARK_CURRENT_NA) * 1e-9  # nA -> A
        self.T = params.get('temperature', ROOM_TEMPERATURE_K)  # K
        
        # Derived constants
        self.V_T = thermal_voltage(self.T)  # Thermal voltage (~26 mV at 300K)
        
        # Physics Parameters for C_j(V)
        self.V_bi = params.get('V_bi', 0.8) # Built-in potential (V)
        self.grading_m = params.get('grading_m', 0.5) # Grading coefficient
        
        # Verify physically reasonable values (relaxed for large-area solar cells)
        assert 0.3 <= self.R <= 0.8, "Responsivity should be 0.3-0.8 A/W for Si/GaAs"
        assert 1e-12 <= self.C_j <= 1000e-6, "Capacitance should be 1 pF - 1000 µF"
        assert 1e3 <= self.R_sh <= 500e6, "Shunt resistance should be 1 kΩ - 500 MΩ"
        
        print(f"[OK] PV Receiver initialized ({self.mode} mode):")
        print(f"    R = {self.R:.2f} A/W")
        print(f"    C_j = {self.C_j*1e12:.1f} pF {self._mode_note}")
        print(f"    R_sh = {self.R_sh*1e-6:.1f} MOhm")
        print(f"    I_0 = {self.I_0*1e9:.1f} nA")
        print(f"    V_T = {self.V_T*1e3:.1f} mV (at {self.T}K)")
    
    def calculate_bandwidth(self, R_load):
        """
        Calculate electrical 3dB bandwidth based on load resistance.
        
        Per Gonzalez-Uriarte 2024 reference:
        f_c = 1 / (2π × R_load × C_eq)
        
        where C_eq is the equivalent capacitance (junction + parasitic).
        
        Measured data from Gonzalez Fig. 2:
        | R_load | BW     |
        |--------|--------|
        | 1 MΩ   | 500 Hz |
        | 10 kΩ  | 1 kHz  |
        | 1 kΩ   | 10 kHz |
        | 220 Ω  | 50 kHz |
        
        Args:
            R_load (float): Load resistance in Ohms
            
        Returns:
            f_3db (float): 3dB bandwidth in Hz
        """
        import numpy as np
        
        # Use parallel combination of R_sh and R_load
        R_eq = (self.R_sh * R_load) / (self.R_sh + R_load) if R_load > 0 else self.R_sh
        
        # 3dB frequency: f_c = 1 / (2π × R × C)
        f_3db = 1.0 / (2 * np.pi * R_eq * self.C_j)
        
        return f_3db
    
    def get_rload_bandwidth_curve(self, R_loads=None):
        """
        Generate R_load vs Bandwidth curve (for plotting/validation).
        
        Args:
            R_loads (array): Load resistances in Ohms (optional)
            
        Returns:
            dict: {'R_load': array, 'bandwidth_hz': array}
        """
        import numpy as np
        
        if R_loads is None:
            R_loads = np.array([100, 220, 1e3, 10e3, 100e3, 1e6])
        
        bandwidths = np.array([self.calculate_bandwidth(r) for r in R_loads])
        
        return {'R_load': R_loads, 'bandwidth_hz': bandwidths}
    
    def get_voltage_vs_rload(self, I_ph_dc, R_loads=None):
        """
        Calculate output voltage vs load resistance (power-bandwidth trade-off).
        
        Higher R_load → Higher voltage but lower bandwidth.
        
        Args:
            I_ph_dc (float): DC photocurrent in Amps
            R_loads (array): Load resistances in Ohms
            
        Returns:
            dict: {'R_load': array, 'voltage_v': array, 'bandwidth_hz': array}
        """
        import numpy as np
        
        if R_loads is None:
            R_loads = np.array([100, 220, 1e3, 10e3, 100e3, 1e6])
        
        voltages = []
        bandwidths = []
        
        for R_load in R_loads:
            # Simple resistive divider approximation for small-signal voltage
            R_eq = (self.R_sh * R_load) / (self.R_sh + R_load)
            V_out = I_ph_dc * R_eq
            voltages.append(V_out)
            bandwidths.append(self.calculate_bandwidth(R_load))
        
        return {
            'R_load': R_loads,
            'voltage_v': np.array(voltages),
            'bandwidth_hz': np.array(bandwidths)
        }
    
    def set_temperature(self, T_kelvin):
        """
        Update the operating temperature dynamically.
        
        This affects:
        - Thermal voltage: V_T = k_B × T / q
        - Dark current increases with temperature
        
        Args:
            T_kelvin (float): Temperature in Kelvin (typically 273-373 K)
        """
        from utils.constants import thermal_voltage
        
        assert 200 <= T_kelvin <= 500, "Temperature should be 200-500 K"
        
        self.T = T_kelvin
        self.V_T = thermal_voltage(self.T)
        
        # Approximate dark current temperature dependence (doubles every ~10K)
        T_ref = 300.0  # Reference at room temp
        I0_ref = 1e-9  # 1 nA at reference
        self.I_0 = I0_ref * (2 ** ((self.T - T_ref) / 10.0))
    
    def sweep_temperature(self, T_range_k=None, I_ph_dc=1e-6, R_load=1000):
        """
        Sweep temperature and calculate V_oc and I_sc characteristics.
        
        Per Kadirvelu paper: V_T = k_B × T / q affects PV cell performance.
        
        Args:
            T_range_k (array): Temperature values in Kelvin
            I_ph_dc (float): DC photocurrent (A)
            R_load (float): Load resistance (Ohms)
            
        Returns:
            dict: {'T_kelvin': array, 'V_T_mV': array, 'V_out_mV': array, 'I_0_nA': array}
        """
        import numpy as np
        from utils.constants import thermal_voltage
        
        if T_range_k is None:
            T_range_k = np.arange(273, 353, 10)  # 0°C to 80°C
        
        V_T_list = []
        V_out_list = []
        I_0_list = []
        
        # Save original state
        T_orig = self.T
        V_T_orig = self.V_T
        I_0_orig = self.I_0
        
        for T in T_range_k:
            self.set_temperature(T)
            V_T_list.append(self.V_T * 1000)  # mV
            I_0_list.append(self.I_0 * 1e9)   # nA
            
            # Approximate open-circuit voltage (simplified)
            # V_oc ≈ V_T × ln(I_ph / I_0 + 1)
            V_oc = self.V_T * np.log(I_ph_dc / self.I_0 + 1)
            V_out_list.append(V_oc * 1000)  # mV
        
        # Restore original state
        self.T = T_orig
        self.V_T = V_T_orig
        self.I_0 = I_0_orig
        
        return {
            'T_kelvin': np.array(T_range_k),
            'T_celsius': np.array(T_range_k) - 273.15,
            'V_T_mV': np.array(V_T_list),
            'V_oc_mV': np.array(V_out_list),
            'I_0_nA': np.array(I_0_list)
        }

    
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
    
    def solve_pv_circuit(self, I_ph, t, R_load=None, method='euler', verbose=False):
        """
        Solve PV cell RC circuit ODE.
        
        Equation 8: dV/dt = [I_ph(t) - V/R_sh - I_0*exp(V/V_T) - V/R_load] / C_j
        
        This is a first-order nonlinear ODE.
        We use simple Euler integration for clarity.
        
        Args:
            I_ph (array): Photocurrent (Amperes)
            t (array): Time vector (seconds)
            R_load (float, optional): Load resistance (Ohms). If None, assumes open circuit.
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
            
            # Pre-compute temperature profile if array
            T_is_array = isinstance(self.T, (np.ndarray, list))
            if T_is_array and len(self.T) != len(t):
                 print(f"Warning: Temperature array length {len(self.T)} != time length {len(t)}. Using T[0].")
                 T_current = self.T[0]
                 T_is_array = False
            elif not T_is_array:
                 T_current = self.T

            # Initial constants
            V_T_current = thermal_voltage(T_current) if not T_is_array else thermal_voltage(self.T[0])
            # Simple I_0 scaling: I_0(T) = I_0_ref * 2^((T-T_ref)/10)
            I_0_ref = 1e-9 # approx base
            T_ref = 300.0
            
            def get_I0(T_val):
                 return I_0_ref * (2.0 ** ((T_val - T_ref) / 10.0))

            I_0_current = get_I0(T_current) if not T_is_array else get_I0(self.T[0])

            for i in range(len(t) - 1):
                # Update Physics if Temperature Changes
                if T_is_array:
                    T_val = self.T[i]
                    V_T_current = thermal_voltage(T_val)
                    I_0_current = get_I0(T_val)
                
                # Voltage Dependent Capacitance (Varicap)
                # C_j(V) = C_j0 / (1 - V/V_bi)^m
                # V_bi ~ 0.8V for Silicon, m ~ 0.5
                V_bi = getattr(self, 'V_bi', 0.8)
                grading_m = getattr(self, 'grading_m', 0.5)
                
                # Protect against singularity at V = V_bi
                # And reverse bias limit
                V_safe_C = np.clip(V[i], -10.0, V_bi - 0.05)
                
                # C_j0 is the zero-bias capacitance (stored in self.C_j)
                # But actually self.C_j usually implies the average or 0V value.
                # Let's assume self.C_j is C_j0.
                C_j_dynamic = self.C_j / ((1 - V_safe_C/V_bi) ** grading_m)
                
                # Current balance (Kirchhoff's current law)
                # I_ph = V/R_sh + C_j*dV/dt + I_diode + V/R_load
                
                # Shunt leakage: I_sh = V / R_sh
                I_leak_sh = V[i] / self.R_sh
                
                # External Load: I_load = V / R_load
                I_load = 0.0
                if R_load is not None and R_load > 0:
                    I_load = V[i] / R_load
                
                # Diode leakage: I_diode = I_0 * (exp(V/V_T) - 1)
                # For numerical stability, clip V_norm to avoid overflow
                V_norm = V[i] / V_T_current
                V_norm = np.clip(V_norm, -100, 100)  # Prevent overflow
                I_diode = I_0_current * (np.exp(V_norm) - 1)
                
                # Differential equation: dV/dt = [I_ph - I_sh - I_diode - I_load] / C_j
                dV_dt = (I_ph[i] - I_leak_sh - I_diode - I_load) / C_j_dynamic
                
                # Euler step
                V[i+1] = V[i] + dV_dt * dt
                
                # Safety: clip voltage to reasonable range (prevent explosion)
                V[i+1] = np.clip(V[i+1], -5.0, 50.0)
        
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


class MIMOPVReceiver:
    """
    MIMO Receiver: A collection of multiple PVReceiver cells.
    
    Handles independent physics simulation for each receiver element.
    """
    
    def __init__(self, num_receivers, params=None):
        """
        Initialize MIMO Receiver Array.
        
        Args:
            num_receivers (int): Number of independent PV cells
            params (dict or list of dicts): 
                - If dict: Applied to all receivers
                - If list: Specific params for each receiver
        """
        self.receivers = []
        
        for i in range(num_receivers):
            p = params
            if isinstance(params, list):
                p = params[i]
                
            self.receivers.append(PVReceiver(p))
            
    def optical_to_current(self, P_rx_mimo):
        """
        Convert matrix of optical powers to matrix of photocurrents.
        
        Args:
            P_rx_mimo (ndarray): Shape (L_rx, N_samples)
            
        Returns:
            I_ph_mimo (ndarray): Shape (L_rx, N_samples)
        """
        I_ph_mimo = np.zeros_like(P_rx_mimo)
        
        for i, rx in enumerate(self.receivers):
            I_ph_mimo[i] = rx.optical_to_current(P_rx_mimo[i])
            
        return I_ph_mimo
        
    def solve_pv_circuit(self, I_ph_mimo, t, R_load=None, verbose=False):
        """
        Solve physics for all receivers in parallel (conceptually).
        
        Args:
            I_ph_mimo (ndarray): Shape (L_rx, N_samples)
            
        Returns:
            V_mimo (ndarray): Shape (L_rx, N_samples)
        """
        V_mimo = np.zeros_like(I_ph_mimo)
        
        for i, rx in enumerate(self.receivers):
            if verbose:
                print(f"Solving PV Cell {i+1}/{len(self.receivers)}...")
            V_mimo[i] = rx.solve_pv_circuit(I_ph_mimo[i], t, R_load=R_load, verbose=False)
            
        return V_mimo


class ReconfigurablePVArray:
    """
    Reconfigurable PV Array (Xu et al. 2024).
    Simulates a switch matrix that changes Series/Parallel connections.
    """
    
    def __init__(self, params=None):
        """
        Initialize Array.
        
        Args:
            params: Dict with 'num_cells', 'cell_params'
        """
        self.num_cells = params.get('num_cells', 16)
        self.cell_params = params.get('cell_params', {})
        
        # Base Unit (Single Cell Physics)
        self.base_cell = PVReceiver(self.cell_params)
        
        # Default Config: All Parallel (Charging) or All Series?
        # Xu paper uses 2s-8p (Charging) and 8s-2p (Comm)
        self.config = {'series': 1, 'parallel': self.num_cells}
        
    def set_configuration(self, num_series, num_parallel):
        """
        Change electrical wiring.
        Must satisfy ns * np = num_cells.
        """
        if num_series * num_parallel != self.num_cells:
            print(f"Warning: Invalid config {num_series}s-{num_parallel}p for {self.num_cells} cells.")
            return
            
        self.config['series'] = num_series
        self.config['parallel'] = num_parallel
        
    def optical_to_current(self, P_rx_total, t):
        """
        Convert Optical Power on Array to Array Current.
        
        Assumption: Uniform illumination.
        P_cell = P_total / num_cells
        I_ph_cell = responsivity * P_cell
        I_ph_array = np * I_ph_cell
        
        Args:
            P_rx_total (ndarray): Total optical power on array area (Watts)
        """
        # Power per cell
        P_rx_cell = P_rx_total / self.num_cells
        
        # Photocurrent per cell
        I_ph_cell = self.base_cell.optical_to_current(P_rx_cell)
        
        # Total Array Photocurrent (Parallel branches sum up)
        I_ph_array = I_ph_cell * self.config['parallel']
        
        return I_ph_array
        
    def solve_circuit(self, I_ph_array, t, V_cap_initial=0.0, C_storage=0.47, R_load_comm=None):
        """
        Solve Array + Load circuit.
        
        Two Modes:
        1. Charging (Capacitor Load): voltage determined by Supercap integ(I).
           V_array = V_cap (ignoring series resistance drop for now).
        2. Communication (Resistive Load): voltage determined by V=IR or PV curve.
        
        Args:
            I_ph_array (ndarray): Source current
            t (ndarray): Time
            V_cap_initial (float): Initial capacitor voltage
            C_storage (float): Supercap value (Farads)
            R_load_comm (float): If set, simulates Resistor load (Comm Mode). 
                                 If None, assumes Capacitor load (Charging Mode).
        
        Returns:
            V_out (ndarray): Output voltage over time
            I_out (ndarray): Output current over time
        """
        dt = t[1] - t[0]
        N_s = self.config['series']
        N_p = self.config['parallel']
        
        # Cell parameters scaling
        # Array behaves like a Super-PV:
        # I_0_array = N_p * I_0_cell
        # V_T_array = N_s * V_T_cell
        # R_sh_array = (N_s / N_p) * R_sh_cell
        # C_j_array = (N_p / N_s) * C_j_cell (Approx)
        
        I_0_arr = N_p * self.base_cell.I_0
        V_T_arr = N_s * self.base_cell.V_T
        
        if R_load_comm is None:
            # CHARGING MODE (Charging Supercap)
            # V_out is constrained by the Capacitor voltage.
            # I_charging = I_ph_array - I_diode(V_cap)
            # dV_cap/dt = I_charging / C_storage
            
            V_out = np.zeros_like(I_ph_array)
            I_out = np.zeros_like(I_ph_array)
            v_cap = V_cap_initial
            
            for i in range(len(t)):
                # Diode current at current capacitor voltage
                if I_0_arr > 0:
                    I_diode = I_0_arr * (np.exp(v_cap / V_T_arr) - 1)
                else:
                    I_diode = 0
                
                # Net charging current
                i_charge = I_ph_array[i] - I_diode
                
                # Update Cap Voltage
                v_cap += (i_charge / C_storage) * dt
                
                V_out[i] = v_cap
                I_out[i] = i_charge
                
            return V_out, I_out
            
        else:
            # COMMUNICATION MODE (Resistive Load)
            # Standard PV solver but with scaled parameters
            # Use base_cell.solve_pv_circuit but with scaled inputs?
            # Or simplified V=IR solution?
            # Let's use the explicit ODE solver from PVReceiver but modify it
            # Or instantiate a temporary "Super PV" and solve.
            
            # Scaled "Super PV" params
            super_params = self.cell_params.copy()
            super_params['dark_current'] = I_0_arr
            super_params['temperature'] = self.base_cell.T # V_T handled by solver logic if we hack it
            
            # Actually, easiest way:
            # The solver uses self.V_T and self.I_0.
            # We can create a temporary PVReceiver with the scaled properties.
            # V_T needs to be overridden.
            
            super_pv = PVReceiver(super_params)
            super_pv.V_T = V_T_arr # Override thermal voltage scalar
            super_pv.I_0 = I_0_arr
            # C_j scaling: Parallel adds, Series divides
            super_pv.C_j = self.base_cell.C_j * (N_p / N_s) 
            
            V_out = super_pv.solve_pv_circuit(I_ph_array, t, R_load=R_load_comm)
            I_out = V_out / R_load_comm
            
            return V_out, I_out


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
