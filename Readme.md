# Prelude

This repository contains the main code implementation under the `Prelude_main` directory, focusing on dynamic gated mechanisms for early Website Fingerprinting (WF) classification on streaming traffic.

## Directory Structure Overview

The `Prelude_main` directory is divided into two primary subdirectories:

- **Model/**: Contains the core model architectures, including baseline models, dataloaders, and the dynamic gating mechanisms.
- **Run/**: Contains the execution scripts for training backbones, testing inference, and evaluating the early classification gating system.

## How to Use

### 1. Configuration (`Run/const.py` & `Run/config/`)
Before running any script, make sure that the environment and the datasets are properly configured.
- `Run/const.py`: Defines system constants such as dataset properties (e.g., number of labels, upper bounds for max load time) and base path mappings for your local machine.
- `Run/config/`: Holds `.ini` configuration files containing hyper-parameters specific to different baseline neural networks (e.g., AWF, DF, TikTok).

### 2. Training and Evaluating Backbone Baselines
The backbone extractors and basic baseline networks are trained using `main.py` and evaluated sequentially via `test.py`.

**Using the built-in Bash Script:**
You can directly review and execute `Run/bash_main_process.sh`.
```bash
cd Prelude_main/Run
bash bash_main_process.sh
```
- **Training Step**: The script invokes `main.py` with necessary arguments such as `--dataset CW` and `--config config/DF.ini` to train the targeted model.
- **Testing Step**: The script sequentially calls `test.py`, incrementing parameters like `--load_ratio` to evaluate the targeted baseline’s performance under different fixed proportions of observed traffic.

### 3. Training and Running the Dynamic Gating Mechanism (Prelude)
The project's core proposition is the dynamic early gated inference framework, unified within `gate_main.py`.

**Using the Built-in Bash Script:**
Tailor the specific parameters defined within `Run/bash_gate_process.sh` and execute it:
```bash
cd Prelude_main/Run
bash bash_gate_process.sh
```

**Crucial Parameters in `bash_gate_process.sh`:**
- `dataset`: Select the dataset you wish to experiment on (e.g., `CW` or `OW`).
- `train_epochs` / `batch_size`: Control the training cycle attributes for the internal gate network.
- `checkconf_list`: A list of classification confidence thresholds (e.g., `0.1, 0.2, ..., 1.0` separated by commas) iteratively validated against simulated, continual traffic streams.
- `load_mode`: Defines the increment scale strategy (`ratio` works by percentage increments, `window` increments by a set count of packets).
- `mode`: String designating which stages (e.g., training layout, inference tasks) are to be executed.

### 4. Supplementary Utility Files Context
Throughout execution in the `Run/` directory, several utility files are regularly called upon:
- **`utils_dataset_metric.py`**: Standard toolset for dataset logistics. Automatically handles data matrix initialization, length truncation or padding, converting raw traces to ready-to-run Dataloaders.
- **`utils_porcess.py`** (and related): General pipeline support. Contains arguments parsers, logging managers, and bridges standard system options to parameter registries.
- **`utils_early.py`**: Supplementary mathematical or logical processing blocks intended exclusively to assist early classification operations.
