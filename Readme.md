# Prelude

This repository contains the core code implementation located in the `Prelude_main` directory, mainly focusing on early-stage classification mechanisms for streaming traffic based on dynamic gating mechanisms in Website Fingerprinting (WF) attacks.

## 0. Dependency

```bash
# Prelude Dependency Environment
conda create -n prelude python=3.10.18
conda activate prelude
# pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# other packages
pip install tqdm
pip install pytorch_metric_learning
pip install captum
pip install pandas
pip install timm
pip install natsort
pip install noise
pip install transformers==4.53.2
pip install tabulate
pip install torchinfo

# If you need to compare with CountMamba, please install the following:
# CountMamba Dependency Environment
# Mamba-ssm
# wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# causal-conv1d
# wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 1. Quick Start and Dataset Links

* **Zenodo Repository:** [wf dataset address](https://zenodo.org/records/18173326?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlhZGEwOTk3LTY1ZDEtNDhiYy1iMzJkLWZmMjJlY2E5MTBiOSIsImRhdGEiOnt9LCJyYW5kb20iOiI2NzdlOWE5ODdkNjFhNzczZDZlZTY2ZjViNmY1NGM2ZiJ9.egmMz4UVnRCp9IBsbZXB2i4dMHx6FzkWeLiN6jBdAQdoUbF0bALbvnDK1tpPzI3olJZekXKTMWOM9gPI47YFDQ)
* **Setup:** After downloading, unpack and place the dataset into the `npz_dataset` folder.

### Project File Structure

```plaintext
Prelude
├── /dataset         # Raw traffic files (Raw trace files)
├── /npz_dataset     # Processed .npz files
│   ├── /CW
│   ├── /OW
│   └── ...
├── /data_process    # Conversion and split scripts
├── /defense         # Defense simulation scripts
└── /Prelude_main    # Core models and running scripts
```

## 2. Dataset Preparation

<details>
<summary><b>Expand to view raw dataset processing steps (single stream)</b></summary>

### 2.1 Download Dataset

Prepare the following datasets in the `dataset/` folder:
- **DF:** provided by Tik-Tok
  - CW ([Download Link](https://zenodo.org/records/11631265/files/Undefended.zip?download=1))
  - OW ([Download Link](https://zenodo.org/records/11631265/files/Undefended_OW.zip?download=1))
- **Walkie-Talkie:** [Download Link](https://zenodo.org/records/11631265/files/W_T_Simulated.zip?download=1)
- **k-NN:** [GitHub Repository](https://github.com/kdsec/wangknn-dataset)

### 2.2 Process Raw Dataset

```bash
cd data_process
python concat_cell.py  # Handle k-NN & W_T
python check_format.py  # Manually modify the tail of the illegal file OW/5278340744671043543057
```

### 2.3 Convert to npz Format

```bash
cd data_process
python convert_to_npz.py --dataset CW
python convert_to_npz.py --dataset OW
python convert_to_npz.py --dataset k-NN
python convert_to_npz.py --dataset W_T
```

### 2.4 Dataset Split

```bash
cd data_process
for dataset in CW OW k-NN W_T
do
  python dataset_split.py --dataset ${dataset}
done
```

</details>

## 3. Defense Dataset and Processing

<details>
<summary><b>Expand to view single stream defense methods and processing</b></summary>

### 3.1 Defense Simulations

Navigate to the corresponding defense directory to execute:

- **WTF-PAD:** (Add dummy packets, no latency)
  ```bash
  cd defense/wtfpad
  python main.py --traces_path "../../dataset/CW"
  python main.py --traces_path "../../dataset/OW"
  ```
- **FRONT:** (Add dummy packets with fixed length of 888, no latency)
  ```bash
  cd defense/front
  python main.py --p "../../dataset/CW"
  python main.py --p "../../dataset/OW"
  ```
- **Tamaraw:** (Send packets at constant rate with fixed size)
  ```bash
  cd defense/tamaraw
  python tamaraw.py --traces_path "../../dataset/CW"
  ```
- **RegulaTor:** (Transmit packets in a time-sensitive manner)
  ```bash
  cd defense/regulartor
  python regulator_sim.py --source_path "../../dataset/CW/" --output_path "../results/regulator_CW/"
  python regulator_sim.py --source_path "../../dataset/OW/" --output_path "../results/regulator_OW/"
  ```
- **TrafficSilver:** (Split traffic using different allocation strategies)
  ```bash
  # Round Robin
  cd defense/trafficsilver
  python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_rb_CW/" --s round_robin
  python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_rb_OW/" --s round_robin
  
  # By Direction
  cd defense/trafficsilver
  python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bd_CW/" --s in_and_out
  python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_bd_OW/" --s in_and_out
  
  # Batched Weighted Random
  cd defense/trafficsilver
  python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bwr_CW/" --s batched_weighted_random -r 50,70 -a 1,1,1
  python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_bwr_OW/" --s batched_weighted_random -r 50,70 -a 1,1,1
  ```

### 3.2 Convert to npz Format

All generated single stream defense datasets utilize uniform processing scripts for conversion:

```bash
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW wtfpad_OW front_OW regulator_OW trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python convert_to_npz.py --dataset ${dataset}
done
```

### 3.3 Defense Dataset Split

Split the defense datasets in the exact same manner as standard single streams:

```bash
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW wtfpad_OW front_OW regulator_OW trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python dataset_split.py --dataset ${dataset}
done
```

</details>

## 4. Directory Structure Overview and Usage Instructions

The `Prelude_main` directory mainly consists of the following sections and core code implementations:

- **Model/**: Contains the core model architecture code, such as various baseline networks, custom gate-based dynamic gating mechanisms, and data loading/assembly modules.
- **Run/**: Contains practical launch scripts used for backbone baseline training, inference testing, as well as early detection and interception testing under continuous data streams.

### 4.1 Initialization Configuration and Parameters (`Run/const.py` & `Run/config/`)
Before executing any core scripts, it is recommended to verify the local environment and dataset configuration mapping.
- `Run/const.py`: Defines global constant parameters that affect data behavior and storage location. This includes dataset specifications (such as the number of classification labels, maximum time duration) and baseline file path settings for the local machine.
- `Run/config/`: Stores dedicated `.ini` parameter configuration files for specific baseline models such as AWF, DF, and TikTok.

### 4.2 Baseline Backbone Network Training and Static Evaluation
Various backbone networks and baseline methods complete their standardized training processes through `main.py`, and standard static segment evaluation is handled via the `test.py` script.

**Executing via Bash Scripts:**
You can directly review and execute `Run/bash_main_process.sh` for quick batch runs:
```bash
cd Prelude_main/Run
bash bash_main_process.sh
```
- **Training Process**: Feed training configurations to `main.py` by setting `--dataset CW`, `--config config/DF.ini`, etc.
- **Testing Process**: The script evaluates the model's performance cyclically at different transmission ratios (e.g., using `--load_ratio` from 10 to 100) via `test.py`.

### 4.3 Constructing and Running the Dynamic Gating Mechanism (Core Mechanism)
The core logic addressing streaming traffic is unified within `gate_main.py`. This includes early-stage threshold judgment and multi-step simulation evaluation.

**Executing via Bash Scripts:**
You can adjust and reference the option branches in `Run/bash_gate_process.sh`:
```bash
cd Prelude_main/Run
bash bash_gate_process.sh
```

**Key Parameters in `bash_gate_process.sh`:**
- `dataset`: Specifies the abbreviation of the dataset to be executed (e.g., `CW`).
- `train_epochs` / `batch_size`: Controls the periodic parameters for the small-scale route discrimination Gate network.
- `checkconf_list`: Lists the collective confidence threshold determinations required for the simulated streaming experiments (e.g., comma-separated `0.1,0.2,...,1.0`).
- `load_mode`: Traffic loading mode, either proportional growth format (`ratio`) or fixed-packet-window step format (`window`).
- `mode`: Specifically enables certain experimental branches or controls operational flow variants (e.g., set to `train`).

### 4.4 Auxiliary Module Referencing Mechanism
During execution, various utility files within the `Run/` directory are typically referenced:
- **`utils_dataset_metric.py`**: Core data parser functions. Handles operations on naked data matrices, applies fixed-length Padding and Truncation, and exposes standard Dataloaders arrays for training invocation.
- **`utils_process.py`**: Complements the main pipeline functions; responsible for accepting and parsing command line parameter logic, reading `.ini` combinations, aligning different module parameters, and routing systematic logs.
- **`utils_early.py`**: Specific tools tailored exclusively for extra mathematical measurements, metric evaluations, and validations within the early decision module.
