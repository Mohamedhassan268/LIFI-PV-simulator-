# fig15_dcdc_vout_fit.py
"""
Reproduce paper Fig. 15 (Vout vs duty cycle ρ) using an analytical fit.

Model for each switching frequency:
    V(ρ) = a * ρ * exp(-b * ρ) + c

ρ is in percent (0–50).
Parameters (a, b, c) were fitted to approximate points read from the paper:
ρ = [2, 4, 5, 8, 10, 20, 30, 40, 50] % for each fsw.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# ---- Fitted parameters from paper figure ----
FITS = {
    50:  {"a": 2.24325955, "b": 0.18985278, "c": 0.66831680},
    100: {"a": 1.91849074, "b": 0.18357597, "c": 0.77402226},
    200: {"a": 1.50728145, "b": 0.17361466, "c": 0.88656267},
}


def vout_model(rho_percent: np.ndarray, fsw_khz: int) -> np.ndarray:
    """
    Vout(ρ) = a * ρ * exp(-b * ρ) + c

    Args:
        rho_percent : duty cycle in percent (0–50)
        fsw_khz     : switching frequency in kHz (50, 100, 200)

    Returns:
        Vout in Volts (same shape as rho_percent)
    """
    p = FITS[fsw_khz]
    rho = np.asarray(rho_percent, dtype=float)
    return p["a"] * rho * np.exp(-p["b"] * rho) + p["c"]


def plot_fig15(output_dir: str = None) -> str:
    """Generate a Fig. 15 style plot using the fitted curves."""
    if output_dir is None:
        from utils.output_manager import get_plots_dir
        output_dir = get_plots_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Dense rho for smooth curves
    rho = np.linspace(2, 50, 400)

    # Compute curves
    v_50 = vout_model(rho, 50)
    v_100 = vout_model(rho, 100)
    v_200 = vout_model(rho, 200)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(rho, v_50,  color="navy",      linestyle="-",  linewidth=2,
            label=r"$f_{sw} = 50$ kHz")
    ax.plot(rho, v_100, color="red",       linestyle="--", linewidth=2,
            label=r"$f_{sw} = 100$ kHz")
    ax.plot(rho, v_200, color="purple",    linestyle="-.", linewidth=2,
            label=r"$f_{sw} = 200$ kHz")

    ax.set_xlabel(r"$\rho$ (%)", fontsize=12)
    ax.set_ylabel(r"$V_{out}$ (V)", fontsize=12)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 7)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title("Output voltage of DC-DC converter vs duty cycle (Fig. 15 fit)",
                 fontsize=12)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig15_dcdc_vout_fit.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print("=" * 60)
    print("Fig. 15 fitted reproduction saved to:")
    print(path)
    print("=" * 60)

    return path


if __name__ == "__main__":
    plot_fig15()
