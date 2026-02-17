# simulator/harvesting.py
"""
Energy Harvesting Engine for Optical Wireless / Li-Fi Systems.

Provides models for converting received optical power into usable
electrical energy, including power management and storage.

Components:
  - IlluminancePowerModel: Maps illuminance (lux) to raw harvested power
    using measured calibration data. Any paper's data can be loaded.
  - MPPTModel: Maximum Power Point Tracking efficiency model.
    Supports named ICs (BQ25504) or simple fixed efficiency.
  - SupercapStorage: Supercapacitor energy storage with charge/discharge.
  - EnergyHarvester: Top-level orchestrator combining all components.
  - SimpleDissipativeHarvest: Minimal P=V²/R model (for systems without MPPT).

Physics extracted from:
  - Oliveira 2024: BQ25504 MPPT, 0.1F supercap, Table 1 illuminance data
  - Kadirvelu 2021: Simple V²/R harvesting (existing main.py behavior)
  - Xu 2024: ReconfigurablePVArray charging mode (supercap integration)

Designed for generality — each component is independently configurable
and composable. A paper runner wires them together with its specific params.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# =============================================================================
# ILLUMINANCE-TO-POWER MODEL
# =============================================================================

class IlluminancePowerModel:
    """
    Maps illuminance (lux) to raw harvested electrical power.
    
    Uses log-linear interpolation of measured (lux, power_mw) calibration
    data. Different papers / PV cells will have different calibration curves.
    
    Args:
        calibration_data: List of (lux, power_mw) tuples, sorted by lux.
                          Must have at least 2 points.
        max_power_mw: Optional power cap (measured max from paper)
        n_pds_reference: Number of PDs the calibration was measured with.
                         Power scales linearly with n_pds / n_pds_reference.
    """
    
    def __init__(self, calibration_data: List[Tuple[float, float]],
                 max_power_mw: float = None,
                 n_pds_reference: int = 9):
        
        if len(calibration_data) < 2:
            raise ValueError("Need at least 2 calibration points")
        
        # Sort by lux
        sorted_data = sorted(calibration_data, key=lambda x: x[0])
        self.lux_values = np.array([d[0] for d in sorted_data])
        self.power_values = np.array([d[1] for d in sorted_data])
        self.max_power_mw = max_power_mw
        self.n_pds_reference = n_pds_reference
        
        # Apply power cap to calibration data
        if self.max_power_mw is not None:
            self.power_values = np.minimum(self.power_values, self.max_power_mw)
        
        # Pre-compute log-lux for interpolation
        self._log_lux = np.log10(np.maximum(self.lux_values, 1))
    
    def power_at_lux(self, lux: float, n_pds: int = None) -> float:
        """
        Estimate harvested power at given illuminance.
        
        Args:
            lux: Illuminance in lux
            n_pds: Number of PDs harvesting. If None, uses reference count.
            
        Returns:
            Estimated power in mW
        """
        if n_pds is None:
            n_pds = self.n_pds_reference
        
        log_lux_query = np.log10(max(lux, 1))
        power = np.interp(log_lux_query, self._log_lux, self.power_values)
        
        # Scale by PD count
        power *= (n_pds / self.n_pds_reference)
        
        # Apply cap
        if self.max_power_mw is not None:
            power = min(power, self.max_power_mw * (n_pds / self.n_pds_reference))
        
        return float(power)
    
    def __repr__(self):
        return (f"IlluminancePowerModel("
                f"{len(self.lux_values)} points, "
                f"range={self.lux_values[0]:.0f}-{self.lux_values[-1]:.0f} lux)")


# =============================================================================
# MPPT MODEL
# =============================================================================

class MPPTModel:
    """
    Maximum Power Point Tracking efficiency model.
    
    Models the efficiency of a power management IC that extracts
    maximum power from PV cells.
    
    Supports:
      - Fixed efficiency (η = constant)
      - Named IC models with power-dependent efficiency curves
    
    Args:
        efficiency: Fixed efficiency (0-1), OR
        ic_model: Named IC model ('bq25504', 'ideal', 'none')
    """
    
    def __init__(self, efficiency: float = None, ic_model: str = None):
        if efficiency is not None:
            self.mode = 'fixed'
            self.efficiency = float(efficiency)
        elif ic_model is not None:
            self.mode = 'ic'
            self.ic_model = ic_model.lower()
        else:
            # Default: BQ25504 typical
            self.mode = 'fixed'
            self.efficiency = 0.85
    
    def apply(self, raw_power_mw: float) -> float:
        """
        Apply MPPT efficiency to raw harvested power.
        
        Args:
            raw_power_mw: Raw power from PV cells (mW)
            
        Returns:
            Net power after MPPT (mW)
        """
        if self.mode == 'fixed':
            return raw_power_mw * self.efficiency
        
        elif self.mode == 'ic':
            if self.ic_model == 'bq25504':
                return raw_power_mw * self._bq25504_efficiency(raw_power_mw)
            elif self.ic_model == 'ideal':
                return raw_power_mw  # η = 1.0
            elif self.ic_model == 'none':
                return raw_power_mw  # No MPPT, direct connection
            else:
                raise ValueError(f"Unknown IC model: {self.ic_model}")
        
        return raw_power_mw
    
    def get_efficiency(self, raw_power_mw: float = None) -> float:
        """Return current efficiency value."""
        if self.mode == 'fixed':
            return self.efficiency
        elif self.mode == 'ic' and self.ic_model == 'bq25504':
            return self._bq25504_efficiency(raw_power_mw or 10.0)
        return 1.0
    
    @staticmethod
    def _bq25504_efficiency(power_mw: float) -> float:
        """
        BQ25504 efficiency curve (approximation from datasheet).
        
        Efficiency varies with input power:
          - Very low power (<0.1 mW): ~50% (quiescent losses dominate)
          - Low power (0.1-1 mW): ~70%
          - Medium power (1-50 mW): ~85% (typical operating point)
          - High power (>50 mW): ~80% (switching losses increase)
        """
        if power_mw < 0.01:
            return 0.30
        elif power_mw < 0.1:
            return 0.50
        elif power_mw < 1.0:
            return 0.70
        elif power_mw < 50.0:
            return 0.85
        else:
            return 0.80
    
    def __repr__(self):
        if self.mode == 'fixed':
            return f"MPPTModel(η={self.efficiency:.0%})"
        return f"MPPTModel(ic={self.ic_model})"


# =============================================================================
# SUPERCAPACITOR STORAGE
# =============================================================================

class SupercapStorage:
    """
    Supercapacitor energy storage model.
    
    E = 0.5 * C * V²
    V = sqrt(2 * E / C)
    
    Tracks energy state across charge/discharge cycles.
    
    Args:
        capacitance_f: Capacitance in Farads
        v_max: Maximum voltage (V)
        v_initial: Initial voltage (V), default 0
        leakage_ua: Leakage current (µA), default 1.0 (typical for commercial supercaps)
    """
    
    def __init__(self, capacitance_f: float = 0.1,
                 v_max: float = 5.0,
                 v_initial: float = 0.0,
                 leakage_ua: float = 1.0):
        self.capacitance_f = capacitance_f
        self.v_max = v_max
        self.leakage_ua = leakage_ua
        
        # Maximum energy
        self.max_energy_j = 0.5 * capacitance_f * v_max ** 2
        
        # Initial state
        self.voltage = v_initial
        self.energy_j = 0.5 * capacitance_f * v_initial ** 2
    
    def charge(self, power_mw: float, duration_s: float) -> Dict:
        """
        Add energy to supercap.
        
        Args:
            power_mw: Charging power (mW)
            duration_s: Duration (s)
            
        Returns:
            Dict with updated state
        """
        energy_in = power_mw * 1e-3 * duration_s
        
        # Subtract leakage
        energy_leak = self.leakage_ua * 1e-6 * self.voltage * duration_s
        
        self.energy_j += energy_in - energy_leak
        self.energy_j = np.clip(self.energy_j, 0, self.max_energy_j)
        self.voltage = np.sqrt(2 * self.energy_j / self.capacitance_f)
        
        return self._state()
    
    def discharge(self, power_mw: float, duration_s: float) -> Tuple[bool, Dict]:
        """
        Consume energy from supercap.
        
        Args:
            power_mw: Consumption power (mW)
            duration_s: Duration (s)
            
        Returns:
            (sufficient: bool, state: Dict)
        """
        energy_needed = power_mw * 1e-3 * duration_s
        sufficient = self.energy_j >= energy_needed
        
        if sufficient:
            self.energy_j -= energy_needed
            self.voltage = np.sqrt(2 * self.energy_j / self.capacitance_f)
        
        return sufficient, self._state()
    
    def reset(self, v_initial: float = 0.0):
        """Reset to initial state."""
        self.voltage = v_initial
        self.energy_j = 0.5 * self.capacitance_f * v_initial ** 2
    
    def charge_time_estimate(self, power_mw: float, target_voltage: float = None) -> float:
        """
        Estimate time to reach target voltage from current state.
        
        Args:
            power_mw: Constant charging power (mW)
            target_voltage: Target voltage (default: v_max)
            
        Returns:
            Estimated time in seconds (inf if power ≤ 0)
        """
        if target_voltage is None:
            target_voltage = self.v_max
        
        target_energy = 0.5 * self.capacitance_f * target_voltage ** 2
        energy_deficit = max(0, target_energy - self.energy_j)
        
        if power_mw <= 0:
            return float('inf')
        
        return energy_deficit / (power_mw * 1e-3)
    
    def _state(self) -> Dict:
        """Current state as dict."""
        return {
            'voltage': self.voltage,
            'energy_j': self.energy_j,
            'charge_pct': (self.energy_j / self.max_energy_j * 100)
                          if self.max_energy_j > 0 else 0,
        }
    
    def __repr__(self):
        return (f"SupercapStorage("
                f"C={self.capacitance_f}F, V={self.voltage:.2f}/{self.v_max}V, "
                f"{self.energy_j/self.max_energy_j*100:.1f}% charged)")


# =============================================================================
# ENERGY HARVESTER (TOP-LEVEL ORCHESTRATOR)
# =============================================================================

class EnergyHarvester:
    """
    Complete energy harvesting system.
    
    Combines illuminance model, MPPT, and supercap storage into a single
    interface. A paper runner instantiates this with paper-specific config.
    
    Args:
        illuminance_model: IlluminancePowerModel (or None for direct power input)
        mppt: MPPTModel instance
        storage: SupercapStorage instance (or None for no storage)
    """
    
    def __init__(self, illuminance_model: IlluminancePowerModel = None,
                 mppt: MPPTModel = None,
                 storage: SupercapStorage = None):
        self.illuminance_model = illuminance_model
        self.mppt = mppt or MPPTModel(efficiency=1.0)
        self.storage = storage
    
    def harvest(self, illuminance_lux: float = None,
                raw_power_mw: float = None,
                n_pds: int = None,
                duration_s: float = 1.0) -> Dict:
        """
        Simulate energy harvesting over a time period.
        
        Provide either illuminance_lux (uses illuminance model) or
        raw_power_mw (direct power input).
        
        Args:
            illuminance_lux: Ambient illuminance (lux)
            raw_power_mw: Direct raw power input (mW) — overrides illuminance
            n_pds: Number of harvesting PDs (for illuminance model scaling)
            duration_s: Time duration (s)
            
        Returns:
            Dict with harvesting results
        """
        # Step 1: Get raw power
        if raw_power_mw is not None:
            raw = raw_power_mw
        elif illuminance_lux is not None and self.illuminance_model is not None:
            raw = self.illuminance_model.power_at_lux(illuminance_lux, n_pds)
        else:
            raise ValueError("Provide illuminance_lux (with model) or raw_power_mw")
        
        # Step 2: Apply MPPT
        net_power_mw = self.mppt.apply(raw)
        
        # Step 3: Energy
        energy_j = net_power_mw * 1e-3 * duration_s
        
        # Step 4: Store (if storage available)
        storage_state = None
        if self.storage is not None:
            storage_state = self.storage.charge(net_power_mw, duration_s)
        
        return {
            'raw_power_mw': raw,
            'mppt_efficiency': self.mppt.get_efficiency(raw),
            'net_power_mw': net_power_mw,
            'energy_j': energy_j,
            'storage': storage_state,
        }
    
    def consume(self, power_mw: float, duration_s: float) -> Tuple[bool, Dict]:
        """
        Consume energy from storage.
        
        Args:
            power_mw: Power consumption (mW)
            duration_s: Duration (s)
            
        Returns:
            (sufficient: bool, state: Dict or None)
        """
        if self.storage is None:
            return True, None
        return self.storage.discharge(power_mw, duration_s)
    
    def get_charge_time(self, illuminance_lux: float,
                        n_pds: int = None) -> float:
        """
        Estimate time to fully charge storage.
        
        Args:
            illuminance_lux: Ambient illuminance (lux)
            n_pds: Number of harvesting PDs
            
        Returns:
            Charge time in seconds
        """
        if self.illuminance_model is None or self.storage is None:
            return float('inf')
        
        raw = self.illuminance_model.power_at_lux(illuminance_lux, n_pds)
        net = self.mppt.apply(raw)
        return self.storage.charge_time_estimate(net)
    
    def reset(self):
        """Reset storage to empty."""
        if self.storage is not None:
            self.storage.reset()
    
    def __repr__(self):
        parts = []
        if self.illuminance_model:
            parts.append(f"model={self.illuminance_model}")
        parts.append(f"mppt={self.mppt}")
        if self.storage:
            parts.append(f"storage={self.storage}")
        return f"EnergyHarvester({', '.join(parts)})"


# =============================================================================
# SIMPLE DISSIPATIVE HARVEST (V²/R fallback)
# =============================================================================

class SimpleDissipativeHarvest:
    """
    Minimal P = V² / R_load harvesting model.
    
    For systems without MPPT where harvested power is simply the
    electrical power dissipated across the load resistor.
    This matches the existing main.py behavior.
    
    Args:
        R_load: Load resistance (Ohms)
    """
    
    def __init__(self, R_load: float = 1360.0):
        self.R_load = R_load
    
    def power_from_voltage(self, V_pv):
        """
        Calculate instantaneous harvested power from PV voltage waveform.
        
        P(t) = V(t)² / R_load
        
        Args:
            V_pv: PV voltage waveform (V), scalar or array
            
        Returns:
            Power (W), same shape as V_pv
        """
        return np.asarray(V_pv) ** 2 / self.R_load
    
    def average_power(self, V_pv, t=None):
        """
        Calculate average harvested power.
        
        Args:
            V_pv: PV voltage waveform (V)
            t: Time vector (s) — if provided, uses trapezoidal integration
            
        Returns:
            Average power in Watts
        """
        P_inst = self.power_from_voltage(V_pv)
        if t is not None:
            return float(np.trapezoid(P_inst, t) / (t[-1] - t[0]))
        return float(np.mean(P_inst))
    
    def __repr__(self):
        return f"SimpleDissipativeHarvest(R_load={self.R_load}Ω)"


# =============================================================================
# PRE-BUILT CONFIGURATIONS
# =============================================================================

def oliveira_2024_harvester(n_harvest_pds: int = 8) -> EnergyHarvester:
    """
    Create EnergyHarvester configured for Oliveira 2024.
    
    Uses Table 1 calibration data, BQ25504 MPPT, 0.1F supercap.
    
    Args:
        n_harvest_pds: Number of PDs for harvesting (default 8 in SISO mode)
        
    Returns:
        Configured EnergyHarvester instance
    """
    # Table 1 calibration: (lux, power_mw = Voc * Isc)
    calibration = [
        (14,    2.57 * 0.0078),     # 0.02 mW  — unilluminated
        (1275,  3.82 * 2.75),       # 10.5 mW  — incandescent 60W
        (1852,  3.84 * 14.10),      # 54.1 mW  — shaded sunlight
        (17780, 5.06 * 21.36),      # 108 mW   — focused 75W (will be capped)
        (55900, 4.10 * 21.30),      # 87.3 mW  — direct sunlight
    ]
    
    model = IlluminancePowerModel(
        calibration_data=calibration,
        max_power_mw=87.33,
        n_pds_reference=9,
    )
    
    mppt = MPPTModel(efficiency=0.85)  # BQ25504 typical
    
    storage = SupercapStorage(
        capacitance_f=0.1,
        v_max=5.0,
    )
    
    return EnergyHarvester(
        illuminance_model=model,
        mppt=mppt,
        storage=storage,
    )


def simple_resistive_harvester(R_load: float = 1360.0) -> SimpleDissipativeHarvest:
    """
    Create simple V²/R harvester matching existing main.py behavior.
    
    Args:
        R_load: Load resistance in Ohms (default: Kadirvelu's 1360Ω)
        
    Returns:
        SimpleDissipativeHarvest instance
    """
    return SimpleDissipativeHarvest(R_load=R_load)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Energy Harvesting Module Tests")
    print("=" * 60)
    
    # Test 1: Oliveira harvester
    harvester = oliveira_2024_harvester(n_harvest_pds=8)
    print(f"\n1. Oliveira harvester: {harvester}")
    
    # Test 2: Harvest at known illuminance levels (Table 1 validation)
    print("\n2. Illuminance → Power (Oliveira Table 1):")
    test_lux = [14, 1275, 1852, 17780, 55900]
    for lux in test_lux:
        result = harvester.harvest(illuminance_lux=lux, n_pds=9, duration_s=1.0)
        print(f"   {lux:6d} lux -> raw={result['raw_power_mw']:.2f} mW, "
              f"net={result['net_power_mw']:.2f} mW")
    
    # Test 3: Max power should be ~87.33 mW raw (9 PDs, direct sunlight)
    result_max = harvester.harvest(illuminance_lux=55900, n_pds=9, duration_s=1.0)
    max_ok = abs(result_max['raw_power_mw'] - 87.33) < 1.0
    print(f"\n3. Max power validation: {result_max['raw_power_mw']:.2f} mW "
          f"(target: 87.33 mW) [{'PASS' if max_ok else 'FAIL'}]")
    
    # Test 4: Supercap charging
    harvester.reset()
    print("\n4. Supercap charging (1275 lux, 8 PDs, 60s steps):")
    for i in range(5):
        result = harvester.harvest(illuminance_lux=1275, n_pds=8, duration_s=60.0)
        st = result['storage']
        print(f"   t={60*(i+1):3d}s: V={st['voltage']:.2f}V, "
              f"E={st['energy_j']*1000:.1f}mJ, charge={st['charge_pct']:.1f}%")
    
    # Test 5: Consume energy
    ok, state = harvester.consume(power_mw=43.0, duration_s=1.0)
    print(f"\n5. Consume 43mW for 1s: sufficient={ok}, "
          f"V={state['voltage']:.2f}V" if state else "")
    
    # Test 6: Simple V²/R harvester (matches main.py)
    simple = simple_resistive_harvester(R_load=1360.0)
    V_test = np.array([0.1, 0.2, 0.3])  # Volts
    P_test = simple.power_from_voltage(V_test)
    print(f"\n6. Simple V²/R (R=1360Ω):")
    for v, p in zip(V_test, P_test):
        print(f"   V={v:.1f}V -> P={p*1e6:.1f} µW")
    
    # Test 7: Charge time estimate
    harvester.reset()
    charge_time = harvester.get_charge_time(illuminance_lux=55900, n_pds=9)
    print(f"\n7. Full charge time (direct sunlight, 9 PDs): "
          f"{charge_time/60:.1f} min")
    
    all_pass = max_ok
    print(f"\n{'TASK 2 PASSED' if all_pass else 'TASK 2 FAILED'}")
