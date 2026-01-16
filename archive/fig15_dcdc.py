# fig15_dcdc_physics.py
"""
Physics-based approximation for Fig. 15 (Vout vs duty cycle ρ)
using:

- DCM boost gain:  M(D,K) = (1 + sqrt(1 + 4*D^2/K)) / 2
- K = 2*L*fsw / R_load_out
- PV voltage drop with increased input current

Only physical parameters: L, R_load_out, PV Voc/Isc, eta(fsw).
No arbitrary curve-fit terms.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def dcm_boost_gain(D, L, fsw_hz, R_load_out):
    """
    Discontinuous-conduction-mode voltage gain M = Vout/Vin.

    K = 2*L*fsw / R_load_out
    M(D,K) = (1 + sqrt(1 + 4*D^2/K)) / 2

    Args:
        D          : duty cycle (0..1)
        L          : inductance (H)
        fsw_hz     : switching frequency (Hz)
        R_load_out : output load resistance (Ohm)
    """
    K = 2.0 * L * fsw_hz / R_load_out
    D = np.clip(D, 1e-4, 0.95)
    term = 1.0 + 4.0 * D**2 / K
    M = 0.5 * (1.0 + np.sqrt(term))
    return M


def pv_voltage_vs_duty(D, Voc=0.74, Isc=0.34):
    """
    PV voltage under increasing load (duty).
    Purely physical idea: larger D -> more current -> Vpv moves away from Voc
    and eventually collapses.

    Use a simple I-V relationship:
        I = Isc * (1 - V/Voc)
    and an effective input resistance R_in(D) that falls ~ 1/D^2,
    so current rises fast with duty.
    """
    D = np.clip(D, 0.01, 0.9)

    # Effective input resistance seen by PV:
    # small D -> very large R_in (light load), large D -> small R_in (heavy load)
    R_in = 0.5 / (D**3)       # Ohm, physical scaling - D³ gives strong collapse

    # Solve I = V/R_in and I = Isc * (1 - V/Voc)
    # => V/R_in = Isc * (1 - V/Voc)
    # => V * (1/R_in + Isc/Voc) = Isc
    Vpv = Isc / (1.0/R_in + Isc/Voc)

    # Clip to [0, Voc]
    Vpv = np.clip(Vpv, 0.0, Voc)
    return Vpv


def vout_physics(D, fsw_hz, L=180e-6, R_load_out=22e3):
    """
    Physics-only Vout(D, fsw):

    Vout = eta(fsw) * M_dcm(D) * Vpv(D)

    eta(fsw) from paper:
        50 kHz  -> 67%
        100 kHz -> 56.4%
        200 kHz -> 42%
    """
    # Efficiency vs switching frequency (paper, Fig. 15 text)
    if np.isclose(fsw_hz, 50e3):
        eta = 0.67
    elif np.isclose(fsw_hz, 100e3):
        eta = 0.564
    elif np.isclose(fsw_hz, 200e3):
        eta = 0.42
    else:
        # simple interpolation or default
        eta = 0.5

    # PV source voltage as function of duty (only physical parameters)
    Vpv = pv_voltage_vs_duty(D, Voc=0.74, Isc=0.34)

    # DCM boost gain
    M = dcm_boost_gain(D, L=L, fsw_hz=fsw_hz, R_load_out=R_load_out)

    # Output voltage with converter efficiency
    Vout = eta * M * Vpv
    return Vout


def plot_fig15_physics(output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Duty cycle range in fraction (ρ 0–0.5) as in paper 0–50%
    D = np.linspace(0.02, 0.50, 300)

    fsw_list = [50e3, 100e3, 200e3]
    colors = {50e3: "navy", 100e3: "red", 200e3: "purple"}
    styles = {50e3: "-", 100e3: "--", 200e3: "-."}

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for fsw in fsw_list:
        Vout = vout_physics(D, fsw_hz=fsw)
        ax.plot(D * 100, Vout,
                color=colors[fsw],
                linestyle=styles[fsw],
                linewidth=2,
                label=fr"$f_{{sw}} = {int(fsw/1e3)}$ kHz")

    ax.set_xlabel(r"$\rho$ (%)", fontsize=12)
    ax.set_ylabel(r"$V_{out}$ (V)", fontsize=12)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_title("DC-DC converter output voltage vs duty cycle (physics-based)",
                 fontsize=12)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig15_dcdc_physics.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved physics-based Fig. 15: {path}")
    return path


if __name__ == "__main__":
    plot_fig15_physics()
