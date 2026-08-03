# Prelude

[**English**](README.md) | [**中文版**](README_zh.md)

Prelude is a robust early website fingerprinting (WF) attack method. It learns to classify a traffic stream from partial observations through **causality-aware prefix learning**, and uses a lightweight **dynamic gate network** to decide when the observed prefix is reliable enough to make a confident classification. This repository provides the official implementation, including data preparation, defense simulation, and the three-phase training / evaluation pipeline.

## 1. Dependency

```bash
conda create -n prelude python=3.10.18
conda activate prelude
# Install PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# Install other dependencies
pip install -r requirements.txt
```

## 2. Dataset

### 2.1 Download the dataset

**Processed dataset (recommended).** Download the processed `.npz` dataset from [Zenodo](https://zenodo.org/uploads/18173326) and place it into `npz_dataset/`:

```text
npz_dataset/
├── CW/
│   ├── train.npz
│   ├── valid.npz
│   └── test.npz
└── OW/
    ├── train.npz
    ├── valid.npz
    └── test.npz
```

Each `.npz` file contains `X` of shape `(N, 10000, 2)` (timestamp, packet length) and the corresponding labels `y`.

**Raw traces (required for Section 3).** The raw traces are provided by Tik-Tok:

- CW: [Undefended.zip](https://zenodo.org/records/11631265/files/Undefended.zip?download=1)
- OW: [Undefended_OW.zip](https://zenodo.org/records/11631265/files/Undefended_OW.zip?download=1)

Unpack them into `dataset/CW/` and `dataset/OW/`. Each trace file is named `label-id` and contains lines of `timestamp<TAB>packet_length`.

### 2.2 Dataset split

If you obtain the un-split `data.npz` (e.g., produced by `convert_to_npz.py` in Section 3), split it into `train/valid/test`:

```bash
cd data_process
python dataset_split.py --dataset CW
python dataset_split.py --dataset OW
```

## 3. Defense dataset

Defense simulations require the raw traces in `dataset/CW/` and `dataset/OW/` (see Section 2.1).

### 3.1 Defense simulation

- **WTF-PAD** (adds dummy packets, no latency)

  ```bash
  cd defense/wtfpad
  python main.py --traces_path "../../dataset/CW"
  python main.py --traces_path "../../dataset/OW"
  ```

- **FRONT** (adds a fixed number of dummy packets, no latency)

  ```bash
  cd defense/front
  python main.py --p "../../dataset/CW"
  python main.py --p "../../dataset/OW"
  ```

- **Tamaraw** (constant-rate, fixed-size packets)

  ```bash
  cd defense/tamaraw
  python tamaraw.py --traces_path "../../dataset/CW"
  ```

- **RegulaTor** (time-sensitive transmission)

  ```bash
  cd defense/regulartor
  python regulator_sim.py --source_path "../../dataset/CW/" --output_path "../results/regulator_CW/"
  python regulator_sim.py --source_path "../../dataset/OW/" --output_path "../results/regulator_OW/"
  ```

- **TrafficSilver** (traffic splitting)

  ```bash
  cd defense/trafficsilver
  # Round Robin
  python simulator.py --p "../../dataset/CW/" -o "../results/trafficsilver_rb_CW/" --s round_robin
  python simulator.py --p "../../dataset/OW/" -o "../results/trafficsilver_rb_OW/" --s round_robin
  # By Direction
  python simulator.py --p "../../dataset/CW/" -o "../results/trafficsilver_bd_CW/" --s in_and_out
  python simulator.py --p "../../dataset/OW/" -o "../results/trafficsilver_bd_OW/" --s in_and_out
  # Batched Weighted Random
  python simulator.py --p "../../dataset/CW/" -o "../results/trafficsilver_bwr_CW/" --s batched_weighted_random -r 50,70 -a 1,1,1
  python simulator.py --p "../../dataset/OW/" -o "../results/trafficsilver_bwr_OW/" --s batched_weighted_random -r 50,70 -a 1,1,1
  ```

All defense outputs are written under `defense/results/`.

### 3.2 Collect defense results

Copy the defended traces into `dataset/{defense_name}/` (the `--src` can be a prefix; the most recent matching directory is used):

```bash
# WTF-PAD  → defense/results/wtfpad_<timestamp>/
python data_process/collect_defense.py --src defense/results/wtfpad_ --dst dataset/wtfpad_CW
python data_process/collect_defense.py --src defense/results/wtfpad_ --dst dataset/wtfpad_OW
# FRONT    → defense/results/ranpad2_<timestamp>/
python data_process/collect_defense.py --src defense/results/ranpad2_ --dst dataset/front_CW
python data_process/collect_defense.py --src defense/results/ranpad2_ --dst dataset/front_OW
# Tamaraw  → defense/results/tamaraw_<timestamp>/
python data_process/collect_defense.py --src defense/results/tamaraw_ --dst dataset/tamaraw_CW
# RegulaTor
python data_process/collect_defense.py --src defense/results/regulator_CW --dst dataset/regulator_CW
python data_process/collect_defense.py --src defense/results/regulator_OW --dst dataset/regulator_OW
# TrafficSilver
python data_process/collect_defense.py --src defense/results/trafficsilver_rb_CW --dst dataset/trafficsilver_rb_CW
python data_process/collect_defense.py --src defense/results/trafficsilver_bd_CW --dst dataset/trafficsilver_bd_CW
python data_process/collect_defense.py --src defense/results/trafficsilver_bwr_CW --dst dataset/trafficsilver_bwr_CW
python data_process/collect_defense.py --src defense/results/trafficsilver_rb_OW --dst dataset/trafficsilver_rb_OW
python data_process/collect_defense.py --src defense/results/trafficsilver_bd_OW --dst dataset/trafficsilver_bd_OW
python data_process/collect_defense.py --src defense/results/trafficsilver_bwr_OW --dst dataset/trafficsilver_bwr_OW
```

### 3.3 Convert to npz

```bash
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW \
               trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW \
               wtfpad_OW front_OW regulator_OW \
               trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python convert_to_npz.py --dataset ${dataset}
done
```

### 3.4 Dataset split

```bash
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW \
               trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW \
               wtfpad_OW front_OW regulator_OW \
               trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python dataset_split.py --dataset ${dataset}
done
```

## 4. Early-stage Website Fingerprinting attack

All commands below are run from `Prelude_main/Run/`:

```bash
cd Prelude_main/Run
export PYTHONPATH=../..
```

### 4.1 Phase 1: Train the backbone

Train the backbone with the early-stage prefix-learning objective:

```bash
python train_backbone.py --dataset CW --config config/Prelude.ini \
    --checkpoint_path ../../ckp_main --file_base_dir ../../npz_dataset \
    --note base_aug_50 --early_stage True --aug_num 50 \
    --test_flag False --train_epochs 30 --num_workers 16
```

- `--early_stage True`: train on randomly sampled prefixes (the loss is computed on `aug_num` sampled positions).
- `--note`: folder name under `ckp_main/{dataset}/Prelude/` where the checkpoint and logs are saved.
- The best checkpoint is saved by validation F1-score.

### 4.2 Phase 2: Train the GateNet

Train the dynamic gate network on top of the frozen backbone:

```bash
python train_gate.py --dataset CW --backbone_model Prelude \
    --backbone_note base_aug_50 --gatenet_note default \
    --checkpoint_path ../../ckp_main --file_base_dir ../../npz_dataset \
    --test_flag False --train_epochs 30 --num_workers 16
```

The GateNet takes the direction sequence of the traffic as input and learns to predict whether the backbone is correct at a given observed prefix.

### 4.3 Phase 3: Evaluation

**Fixed-ratio evaluation** (accuracy at different load ratios):

```bash
python evaluate.py --dataset CW --backbone_model Prelude --backbone_note base_aug_50 \
    --checkpoint_path ../../ckp_main --file_base_dir ../../npz_dataset \
    --eval_mode_type fixed --check_load_ratio "10,20,30,40,50,60,70,80,90,100" \
    --use_gatenet False --test_flag False
```

**Streaming evaluation with GateNet** (the gate decides when to classify; the load ratio and latency at the decision point are recorded):

```bash
python evaluate.py --dataset CW --backbone_model Prelude --backbone_note base_aug_50 \
    --gatenet_note default \
    --checkpoint_path ../../ckp_main --file_base_dir ../../npz_dataset \
    --eval_mode_type streaming --load_mode ratio --use_gatenet True \
    --checkconf_list "0.5,0.7,0.9,0.95" --test_flag False
```

**Streaming evaluation without GateNet** (the backbone classifies at every step, no early decision):

```bash
python evaluate.py --dataset CW --backbone_model Prelude --backbone_note base_aug_50 \
    --checkpoint_path ../../ckp_main --file_base_dir ../../npz_dataset \
    --eval_mode_type streaming --load_mode ratio --use_gatenet False \
    --checkconf_list "0.5,0.7,0.9,0.95" --test_flag False
```

Results are saved under `ckp_main/{dataset}/Prelude/{backbone_note}/evaluate/`.
