# Li-Fi + PV Simulator

A comprehensive, physics-based simulator for **Visible Light Communication (VLC)** systems integrated with **Photovoltaic (PV)** energy harvesting. This project models the complete signal chain—from optical transmission to PV reception and demodulation—allowing for accurate analysis of **Simultaneous Lightwave Information and Power Transfer (SLIPT)** systems.

## 🚀 Key Features

- **Physics-Based Modeling**: Uses accurate single-diode circuit models for PV cells (solving ODEs for $dV/dt$) rather than simple linear approximations.
- **Advanced Modulation**: Supports **OOK**, **PWM**, and **Adaptive DCO-OFDM** with bit loading.
- **MIMO Support**: Simulate multi-input multi-output configurations with **Zero-Forcing (ZF)** spatial demultiplexing.
- **Reconfigurable Arrays**: Models dynamic series/parallel switching of PV arrays for optimal power/bandwidth.
- **Paper Validation**: Includes a plugin system to benchmark simulations against published research papers.
- **7-Layer Architecture**: Modular design separating Transmitter, Channel, Receiver, Demodulation, Bit Recovery, Metrics, and Output.

## 📂 Project Structure

| Directory | Description |
|---|---|
| `simulator/` | **Core Physics Engine**. Contains modules for TX, Channel, RX, and Noise. |
| `papers/` | **Validation Plugins**. Scripts that reproduce results from specific research papers. |
| `utils/` | **Helper Tools**. Constants, plotting utilities, and configuration loaders. |
| `outputs/` | **Artifacts**. Stores generated plots, logs, and CSV datasets. |
| `main.py` | **Main CLI**. Primary entry point for general simulations. |
| `run_validation.py` | **Validation Runner**. Helper script to execute paper benchmarks. |

## 🏗️ 7-Layer Architecture

The simulator follows a strict 7-layer layered approach found in `main.py`:

1.  **Layer 1: Transmitter**: LED modulation (OOK, PWM, OFDM) and signal generation.
2.  **Layer 2: Channel**: Geometric optical path loss (Lambertian) and noise addition.
3.  **Layer 3: Receiver**: Dual-path architecture:
    *   **Energy Path**: PV circuit ODE solver (Single Diode Model).
    *   **Data Path**: Transimpedance Amplifier (TIA) simulation.
4.  **Layer 4: Post-Processing**: Filters (HPF/LPF for data, DC extraction for energy).
5.  **Layer 5: Bit Recovery**: Sampling and threshold detection / QAM demapping.
6.  **Layer 6: Metrics**: BER, SNR, Q-Factor, Eye Diagram, and Harvested Power calculations.
7.  **Layer 7: Output**: Data packaging and CSV export.

## ⚙️ Configuration

The simulator uses a Python dictionary for configuration. Default settings in `main.py`:

```python
config = {
    'simulation': {
        'data_rate_bps': 1e6,      # 1 Mbps
        'samples_per_bit': 500,    # High oversampling for analog fidelity
        'n_bits': 1000
    },
    'transmitter': {
        'dc_bias': 200,            # mA
        'modulation_depth': 0.9,
        'led_efficiency': 0.08     # W/A
    },
    'channel': {
        'distance': 1.0,           # Meters
        'beam_angle_half': 30,     # Degrees
        'receiver_area': 1.0       # cm²
    },
    'receiver': {
        'responsivity': 0.457,     # A/W (GaAs)
        'capacitance': 798,        # pF
        'shunt_resistance': 0.1388 # MΩ
    },
    'tia': {
        'R_tia': 50e3,             # 50 kΩ Transimpedance
        'f_3db': 3e6               # 3 MHz Bandwidth
    }
}
```

## 🧪 Paper Validation

The `papers/` directory contains scripts to reproduce results from state-of-the-art literature:

| Module | Paper / Focus |
|---|---|
| `kadirvelu_2021.py` | **Baseline Physics**. Validates harvested power vs switching frequency and bandwidth limitations. |
| `gonzalez_2024.py` | **IoT Application**. Validates OOK decoding on Poly-Si cells for low-power IoT. |
| `correa_2025.py` | **Greenhouse VLC**. Validates PWM-ASK and Analog Front End (AFE) filters in agricultural settings. |
| `xu_2024.py` | **Sunlight-Duo**. Validates Passive VLC and Simultaneous Lightwave Info & Power (SLIPT). |
| `sarwar_2017.py` | **High-Speed**. Validates 15 Mbps OFDM pipeline. |

## 🛠️ Usage

> 📖 **Quick Reference**: See [`COMMANDS.txt`](file:///c:/Users/HP%20OMEN/lifi_pv_simulator/COMMANDS.txt) for a complete list of all CLI arguments with examples.

### 1. Installation
```bash
git clone https://github.com/Mohamedhassan268/LIFI-PV-simulator-.git
cd LIFI-PV-simulator-
pip install -r requirements.txt
```

### 2. Run Paper Benchmarks
Use the validation runner to list and execute benchmarks:

```bash
# List available papers
python run_validation.py --list

# Run a specific benchmark (e.g., Kadirvelu 2021)
python run_validation.py kadirvelu_2021
```

### 3. Run General Simulation
To run the default 1 Mbps simulation with the full 7-layer TIA model:
```bash
python main.py
```

### 4. Advanced MIMO Demo
Test the 4x4 MIMO and Adaptive OFDM features:
```bash
python validate_mimo_ofdm.py
```

### 5. Custom Configuration Examples
```bash
# IoT scenario with Poly-Si cell
python main.py --rx-cell poly-si --sim-rate 10000 --ch-distance 0.5

# High-speed OFDM
python main.py --rx-cell high-speed --tia-r 100e3 --sim-rate 15e6

# Energy-first mode (no TIA)
python main.py --no-tia --sim-rate 10000
```

## 📊 Outputs
Results are generated in the `outputs/` folder:
- **`outputs/csv/`**: Raw waveform data (`waveforms_*.csv`) and summary metrics (`summary_*.csv`).
- **`outputs/plots/`**: Generated figures (BER curves, Eye Diagrams, I-V curves).

## 👨‍💻 Author
**Mohamed Hassan**
