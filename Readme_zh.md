# Prelude

本项目包含了位于 `Prelude_main` 目录下的核心代码实现，主要聚焦于网站指纹（Website Fingerprinting, WF）攻击中基于动态门控机制流式流量的早期分类机制。

## 0. 环境配置

```bash
# Prelude 安装环境
conda create -n prelude python=3.10.18
conda activate prelude
# pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# 其他依赖包
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

# 如果需要与 CountMamba 进行对比，请安装以下内容
# CountMamba 安装环境
# Mamba-ssm
# wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# causal-conv1d
# wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 1. 快速开始与数据集链接

* **Zenodo 仓库:** [https://zenodo.org/records/14195051](https://zenodo.org/records/14195051)
* **设置:** 下载后，解压并将数据集放入 `npz_dataset` 文件夹中。

### 项目文件结构

```plaintext
Prelude
├── /dataset         # 原始流量文件 (Raw trace files)
├── /npz_dataset     # 处理后的 .npz 文件
│   ├── /CW
│   ├── /OW
│   └── ...
├── /data_process    # 转换与拆分脚本
├── /defense         # 防御模拟脚本
└── /Prelude_main    # 核心模型与运行脚本
```

## 2. 数据集准备与处理

<details>
<summary><b>展开查看原始数据处理步骤（单流）</b></summary>

### 2.1 下载原始数据集

在 `dataset/` 文件夹中准备以下内容：
- **DF:** 由 Tik-Tok 提供
  - CW ([下载链接](https://zenodo.org/records/11631265/files/Undefended.zip?download=1))
  - OW ([下载链接](https://zenodo.org/records/11631265/files/Undefended_OW.zip?download=1))
- **Walkie-Talkie:** [下载链接](https://zenodo.org/records/11631265/files/W_T_Simulated.zip?download=1)
- **k-NN:** [GitHub 仓库](https://github.com/kdsec/wangknn-dataset)

### 2.2 处理原始数据集

```bash
cd data_process
python concat_cell.py  # 处理 k-NN & W_T
python check_format.py  # 手动修复非法文件的末尾: OW/5278340744671043543057
```

### 2.3 转换为 npz 格式

```bash
cd data_process
python convert_to_npz.py --dataset CW
python convert_to_npz.py --dataset OW
python convert_to_npz.py --dataset k-NN
python convert_to_npz.py --dataset W_T
```

### 2.4 数据集拆分

```bash
cd data_process
for dataset in CW OW k-NN W_T
do
  python dataset_split.py --dataset ${dataset}
done
```

</details>

## 3. 防御数据集构建与处理

<details>
<summary><b>展开查看防御方法及单流处理方式</b></summary>

### 3.1 运行防御模拟

进入相应的防御目录执行：

- **WTF-PAD:** (添加哑包，且不增加延迟)
  ```bash
  cd defense/wtfpad
  python main.py --traces_path "../../dataset/CW"
  python main.py --traces_path "../../dataset/OW"
  ```
- **FRONT:** (添加固定长度为 888 的哑包，无延迟)
  ```bash
  cd defense/front
  python main.py --p "../../dataset/CW"
  python main.py --p "../../dataset/OW"
  ```
- **Tamaraw:** (以恒定速率和固定大小发送数据包)
  ```bash
  cd defense/tamaraw
  python tamaraw.py --traces_path "../../dataset/CW"
  ```
- **RegulaTor:** (以时间敏感的方式传输数据包)
  ```bash
  cd defense/regulartor
  python regulator_sim.py --source_path "../../dataset/CW/" --output_path "../results/regulator_CW/"
  python regulator_sim.py --source_path "../../dataset/OW/" --output_path "../results/regulator_OW/"
  ```
- **TrafficSilver:** (使用不同的分配策略拆分流量)
  ```bash
  # 轮询 (Round Robin)
  cd defense/trafficsilver
  python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_rb_CW/" --s round_robin
  python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_rb_OW/" --s round_robin
  
  # 按方向 (By Direction)
  cd defense/trafficsilver
  python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bd_CW/" --s in_and_out
  python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_bd_OW/" --s in_and_out
  
  # 分批加权随机 (Batched Weighted Random)
  cd defense/trafficsilver
  python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bwr_CW/" --s batched_weighted_random -r 50,70 -a 1,1,1
  python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_bwr_OW/" --s batched_weighted_random -r 50,70 -a 1,1,1
  ```

### 3.2 转换为 npz 格式

所有防御手段生成的单流数据经过统一的方式进行格式处理：

```bash
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW wtfpad_OW front_OW regulator_OW trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python convert_to_npz.py --dataset ${dataset}
done
```

### 3.3 防御数据集拆分

与标准单流一致的拆分方式：

```bash
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW wtfpad_OW front_OW regulator_OW trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python dataset_split.py --dataset ${dataset}
done
```

</details>

## 4. 目录结构总览与使用指南

`Prelude_main` 目录主要分为以下几个部分以及核心代码说明：

- **Model/**: 包含核心的模型架构代码，如各基线网络、自定义基于 Gate 的动态门控机制以及数据形态加载组装模块。
- **Run/**: 包含用于骨干基线训练、推理测试，以及在连续数据流下进行早期检测和拦截测试的实际启动脚本。

### 4.1 初始化配置与参数 (`Run/const.py` & `Run/config/`)
在执行任何核心脚本前，建议先核对本地环境与数据集的配置映射。
- `Run/const.py`: 定义了影响数据行为和存放位置的全局常量参数。包含数据集规范（如分类标签数、最大时间等）以及本地电脑的基准文件路径设置。
- `Run/config/`: 此目录下存放诸如 AWF、DF、TikTok 等特定基线模型的专用 `.ini` 参数配置文件。

### 4.2 基线骨干网络训练与静态评估
各类骨干网络及基线方法通过 `main.py` 完成标准化训练流程，其在固定分段下的评估经由 `test.py` 脚本处理。

**通过 Bash 脚本执行：**
可直接跳转并修改 `Run/bash_main_process.sh` 实现快速批量跑通：
```bash
cd Prelude_main/Run
bash bash_main_process.sh
```
- **训练流程**：通过设置 `--dataset CW`, `--config config/DF.ini` 向 `main.py` 喂入训练指令。
- **测试流程**：脚本将利用 `test.py` 并在不同的传输比例（如使用 `--load_ratio` 参数由 10 增至 100）下循环跑测评估模型效果。

### 4.3 构建与运行动态门控机制模块 (核心机制)
用于应对流式数据的核心逻辑统一位于 `gate_main.py` 之中。涵盖早起阈值研判和多段步进模拟评估。

**通过 Bash 脚本执行：**
可调整并参考 `Run/bash_gate_process.sh` 内各个选项：
```bash
cd Prelude_main/Run
bash bash_gate_process.sh
```

**`bash_gate_process.sh` 中的核心参数说明：**
- `dataset`: 指定用于执行的数据集缩写（如：`CW`）。
- `train_epochs` / `batch_size`: 控制小规模选路判别 Gate 网络的周期参数集。
- `checkconf_list`: 列出流式环境测试中所需实验的置信度阈值判定线合集（例如以逗号分隔的 `0.1,0.2,...,1.0`）。
- `load_mode`: 流量装载加载模式，分比例增长（`ratio`）和固定包窗口增长（`window`）模式。
- `mode`: 用于精准开启某些实验特定分支或控制流（例如指示为 `train`）。

### 4.4 辅助模块机制参考
在执行期 `Run/` 目录中通常会引入以下工具类：
- **`utils_dataset_metric.py`**: 数据集的核心解析模块。承接裸数据矩阵的处理、应用定长填充（Padding）及截断（Truncation），并暴露标准化的 Dataloaders 以供训练调用。
- **`utils_porcess.py`**: 主功能管线补充文件，负责接受和处理命令行入参逻辑，读入 `.ini` 预设组合、对齐不同模块参数以及完成系统日志文件系统的指路。
- **`utils_early.py`**: 存放特定为早期早决断模块进行额外数学测算、指标辅助评估与验证的相关工具。
