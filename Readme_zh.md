# Prelude

本项目包含了位于 `Prelude_main` 目录下的核心代码实现，主要聚焦于网站指纹（Website Fingerprinting, WF）攻击中基于动态门控机制流式流量的早期分类机制。

## 目录结构总览

`Prelude_main` 目录主要分为两个主要文件夹：

- **Model/**: 包含核心的模型架构代码，如各基线网络、自定义基于 Gate 的动态门控机制以及数据形态加载组装模块。
- **Run/**: 包含用于骨干基线训练、推理测试，以及在连续数据流下进行早期检测和拦截测试的实际启动脚本。

## 使用指南

### 1. 初始化配置与参数 (`Run/const.py` & `Run/config/`)
在执行任何核心脚本前，建议先核对本地环境与数据集的配置映射。
- `Run/const.py`: 定义了影响数据行为和存放位置的全局常量参数。包含数据集规范（如分类标签数、最大时间等）以及本地电脑的基准文件路径设置。
- `Run/config/`: 此目录下存放诸如 AWF、DF、TikTok 等特定基线模型的专用 `.ini` 参数配置文件。

### 2. 基线骨干网络训练与静态评估
各类骨干网络及基线方法通过 `main.py` 完成标准化训练流程，其在固定分段下的评估经由 `test.py` 脚本处理。

**通过 Bash 脚本执行：**
可直接跳转并修改 `Run/bash_main_process.sh` 实现快速批量跑通：
```bash
cd Prelude_main/Run
bash bash_main_process.sh
```
- **训练流程**：通过设置 `--dataset CW`, `--config config/DF.ini` 向 `main.py` 喂入训练指令。
- **测试流程**：脚本将利用 `test.py` 并在不同的传输比例（如使用 `--load_ratio` 参数由 10 增至 100）下循环跑测评估模型效果。

### 3. 构建与运行动态门控机制模块 (核心机制)
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

### 4. 辅助模块机制参考
在执行期 `Run/` 目录中通常会引入以下工具类：
- **`utils_dataset_metric.py`**: 数据集的核心解析模块。承接裸数据矩阵的处理、应用定长填充（Padding）及截断（Truncation），并暴露标准化的 Dataloaders 以供训练调用。
- **`utils_porcess.py`**: 主功能管线补充文件，负责接受和处理命令行入参逻辑，读入 `.ini` 预设组合、对齐不同模块参数以及完成系统日志文件系统的指路。
- **`utils_early.py`**: 存放特定为早期早决断模块进行额外数学测算、指标辅助评估与验证的相关工具。
