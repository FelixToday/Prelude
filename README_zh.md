# Prelude

[**English**](README.md) | [**中文版**](README_zh.md)

Prelude 是一种鲁棒的早期网站指纹（Website Fingerprinting, WF）攻击方法。它通过**因果感知的前缀学习（Causality-Aware Prefix Learning）**从流量流的局部观测中学习分类，并使用轻量化的**动态门控网络（Dynamic Gate Network）**判断当前观测到的前缀是否足够可靠、从而做出自信的分类决策。本仓库提供了官方实现，包括数据准备、防御模拟以及三阶段训练与评估流程。

## 1. 环境依赖

```bash
conda create -n prelude python=3.10.18
conda activate prelude
# 安装 PyTorch（CUDA 11.8）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# 安装其他依赖
pip install -r requirements.txt
```

## 2. 数据集

### 2.1 下载数据集

**已处理数据集（推荐）。** 从 [Zenodo](https://zenodo.org/uploads/18173326) 下载已处理的 `.npz` 数据集，并放入 `npz_dataset/`：

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

每个 `.npz` 文件包含形状为 `(N, 10000, 2)` 的 `X`（时间戳、包大小）及对应的标签 `y`。

**原始流量（第 3 节需要）。** 原始流量由 Tik-Tok 提供：

- CW: [Undefended.zip](https://zenodo.org/records/11631265/files/Undefended.zip?download=1)
- OW: [Undefended_OW.zip](https://zenodo.org/records/11631265/files/Undefended_OW.zip?download=1)

解压后放入 `dataset/CW/` 和 `dataset/OW/`。每个 trace 文件命名为 `label-id`，内容为 `时间戳<TAB>包大小` 的多行记录。

### 2.2 数据集拆分

如果你获得的是未拆分的 `data.npz`（例如由第 3 节的 `convert_to_npz.py` 生成），可拆分为 `train/valid/test`：

```bash
cd data_process
python dataset_split.py --dataset CW
python dataset_split.py --dataset OW
```

## 3. 防御数据集

防御模拟需要第 2.1 节中的原始流量（位于 `dataset/CW/` 和 `dataset/OW/`）。

### 3.1 防御模拟

- **WTF-PAD**（添加哑包，不增加延迟）

  ```bash
  cd defense/wtfpad
  python main.py --traces_path "../../dataset/CW"
  python main.py --traces_path "../../dataset/OW"
  ```

- **FRONT**（添加固定数量的哑包，不增加延迟）

  ```bash
  cd defense/front
  python main.py --p "../../dataset/CW"
  python main.py --p "../../dataset/OW"
  ```

- **Tamaraw**（恒定速率、固定大小发包）

  ```bash
  cd defense/tamaraw
  python tamaraw.py --traces_path "../../dataset/CW"
  ```

- **RegulaTor**（时间敏感方式传输）

  ```bash
  cd defense/regulartor
  python regulator_sim.py --source_path "../../dataset/CW/" --output_path "../results/regulator_CW/"
  python regulator_sim.py --source_path "../../dataset/OW/" --output_path "../results/regulator_OW/"
  ```

- **TrafficSilver**（流量拆分）

  ```bash
  cd defense/trafficsilver
  # 轮询 (Round Robin)
  python simulator.py --p "../../dataset/CW/" -o "../results/trafficsilver_rb_CW/" --s round_robin
  python simulator.py --p "../../dataset/OW/" -o "../results/trafficsilver_rb_OW/" --s round_robin
  # 按方向 (By Direction)
  python simulator.py --p "../../dataset/CW/" -o "../results/trafficsilver_bd_CW/" --s in_and_out
  python simulator.py --p "../../dataset/OW/" -o "../results/trafficsilver_bd_OW/" --s in_and_out
  # 分批加权随机 (Batched Weighted Random)
  python simulator.py --p "../../dataset/CW/" -o "../results/trafficsilver_bwr_CW/" --s batched_weighted_random -r 50,70 -a 1,1,1
  python simulator.py --p "../../dataset/OW/" -o "../results/trafficsilver_bwr_OW/" --s batched_weighted_random -r 50,70 -a 1,1,1
  ```

所有防御输出统一写入 `defense/results/`。

### 3.2 整理防御结果

将防御后的 trace 复制到 `dataset/{defense_name}/`（`--src` 可以填前缀，脚本会自动选择最近生成的匹配目录）：

```bash
# WTF-PAD  → defense/results/wtfpad_<时间戳>/
python data_process/collect_defense.py --src defense/results/wtfpad_ --dst dataset/wtfpad_CW
python data_process/collect_defense.py --src defense/results/wtfpad_ --dst dataset/wtfpad_OW
# FRONT    → defense/results/ranpad2_<时间戳>/
python data_process/collect_defense.py --src defense/results/ranpad2_ --dst dataset/front_CW
python data_process/collect_defense.py --src defense/results/ranpad2_ --dst dataset/front_OW
# Tamaraw  → defense/results/tamaraw_<时间戳>/
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

### 3.3 转换为 npz 格式

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

### 3.4 数据集拆分

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

## 4. 早期网站指纹攻击

以下命令均在 `Prelude_main/Run/` 目录下执行：

```bash
cd Prelude_main/Run
export PYTHONPATH=../..
```

### 4.1 阶段 1：训练主干网络（Backbone）

使用早期阶段前缀学习目标训练主干网络：

```bash
python train_backbone.py --dataset CW --config config/Prelude.ini \
    --checkpoint_path ../../ckp_main --file_base_dir ../../npz_dataset \
    --note base_aug_50 --early_stage True --aug_num 50 \
    --test_flag False --train_epochs 30 --num_workers 16
```

- `--early_stage True`：在随机采样的前缀上训练（损失在 `aug_num` 个采样位置计算）。
- `--note`：checkpoint 与日志保存到 `ckp_main/{dataset}/Prelude/` 下的文件夹名。
- 按验证集 F1-score 保存最优 checkpoint。

### 4.2 阶段 2：训练门控网络（GateNet）

在冻结的 backbone 之上训练动态门控网络：

```bash
python train_gate.py --dataset CW --backbone_model Prelude \
    --backbone_note base_aug_50 --gatenet_note default \
    --checkpoint_path ../../ckp_main --file_base_dir ../../npz_dataset \
    --test_flag False --train_epochs 30 --num_workers 16
```

GateNet 以流量的方向序列为输入，学习预测在给定的观测前缀下 backbone 的分类是否正确。

### 4.3 阶段 3：评估

**固定比例评估**（不同加载比例下的准确率）：

```bash
python evaluate.py --dataset CW --backbone_model Prelude --backbone_note base_aug_50 \
    --checkpoint_path ../../ckp_main --file_base_dir ../../npz_dataset \
    --eval_mode_type fixed --check_load_ratio "10,20,30,40,50,60,70,80,90,100" \
    --use_gatenet False --test_flag False
```

**带 GateNet 的流式评估**（由门控网络决定何时分类，记录决策时的加载比例与延迟）：

```bash
python evaluate.py --dataset CW --backbone_model Prelude --backbone_note base_aug_50 \
    --gatenet_note default \
    --checkpoint_path ../../ckp_main --file_base_dir ../../npz_dataset \
    --eval_mode_type streaming --load_mode ratio --use_gatenet True \
    --checkconf_list "0.5,0.7,0.9,0.95" --test_flag False
```

**不带 GateNet 的流式评估**（backbone 每步直接分类，无提前决策）：

```bash
python evaluate.py --dataset CW --backbone_model Prelude --backbone_note base_aug_50 \
    --checkpoint_path ../../ckp_main --file_base_dir ../../npz_dataset \
    --eval_mode_type streaming --load_mode ratio --use_gatenet False \
    --checkconf_list "0.5,0.7,0.9,0.95" --test_flag False
```

结果保存在 `ckp_main/{dataset}/Prelude/{backbone_note}/evaluate/`。
