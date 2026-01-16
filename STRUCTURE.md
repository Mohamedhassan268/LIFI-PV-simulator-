# Li-Fi PV Simulator Structure

This project simulates Visible Light Communication (VLC) systems using Photovoltaic (PV) cells as receivers (Li-Fi + Energy Harvesting).

## Root Directory

| File/Folder | Description |
|---|---|
| `run_validation.py` | **Main Entry Point**. Helper script to run paper benchmarks. |
| `validate_mimo_ofdm.py` | **MIMO Benchmark**. Demonstrates 4x4 MIMO and Adaptive OFDM features. |
| `simulator/` | **Core Library**. Physics-based simulation modules. |
| `papers/` | **Plugins**. Validation scripts for specific research papers. |
| `utils/` | **Helpers**. Constants, plotting tools, and configs. |
| `outputs/` | **Results**. Generated plots, logs, and CSVs. |

## 1. Core Library (`simulator/`)

The physics engine is built on these four pillars:

- **`transmitter.py`**:
    - `Transmitter`: Base class for LED modulation (OOK, PWM).
    - `modulate_ofdm`: Adaptive OFDM (DCO-OFDM) with bit loading.
    - `modulate_fsk_passive`: Passive Liquid-Crystal modulation (Sunlight).
- **`channel.py`**:
    - `OpticalChannel`: Geometric path loss calculation.
    - `compute_h_matrix`: MIMO Channel Gain Matrix ($L \times M$).
- **`receiver.py`**:
    - `PVReceiver`: Single-diode model solving the ODE $dV/dt = f(I_{ph}, V)$.
    - `MIMOPVReceiver`: Wrapper for Receiver Arrays.
    - `ReconfigurablePVArray`: Dynamic Series/Parallel switching logic.
- **`demodulator.py`**:
    - `Demodulator`: Signal processing chain.
    - `demodulate_ofdm`: FFT and QAM demapping.
    - `demodulate_mimo`: Zero-Forcing (ZF) spatial demultiplexing.

## 2. Validation Plugins (`papers/`)

Each file corresponds to a research paper benchmark. Run them using `python run_validation.py <name>`.

| Module | Paper / Features |
|---|---|
| `kadirvelu_2021.py` | **Baseline Physics**. Validates bandwidth vs $R_{load}, C_{j}$ and DC-DC efficiency. |
| `gonzalez_2024.py` | **IoT Application**. Validates OOK decoding on Poly-Si cells. |
| `correa_2025.py` | **Greenhouse**. Validates PWM-ASK and Analog Front End (AFE) filters. |
| `xu_2024.py` | **Sunlight-Duo**. Validates Passive VLC and Simultaneous Lightwave Info & Power (SLIPT). |
| `sarwar_2017.py` | **High-Speed**. Validates 15 Mbps OFDM pipeline. |

## 3. Usage

### List Available Benchmarks
```bash
python run_validation.py --list
```

### Run a Benchmark
```bash
python run_validation.py xu_2024
```

### Run Advanced Feature Demo (MIMO)
```bash
python validate_mimo_ofdm.py
```
