#!/bin/bash

# ==========================================================
# Poirot 网站指纹攻击 - 动态门控机制运行脚本 (完整参数版)
# ==========================================================

# 0. 脚本名称
# 1. 基础配置
for dataset in CW
do
#dataset="OW"                     # 数据集名称 (例如: OW, CW)
TAM_type="ED1"
test_flag="True"
gatenet_note="baseline_same"          # GateNet 备注
backbone_note="baseline_same"    # Backbone 备注

checkpoint_path="../../checkpoints_test"
# 2. 训练超参数

train_epochs=15                  # 训练轮数
batch_size=10                    # 批次大小
# 3. 边界与数据增强
lb=0.0                           # 裁剪下界 (Lower bound)
ub=1.0                           # 裁剪上界 (Upper bound)
aug_num=20                       # 数据增强版本数量
# 4. 门控与测试参数
threshold=0.5                    # 训练时的门控阈值
checkconf_list="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"  # 真实流模拟评估时的阈值列表
delta=0.03                       # 步进参数 (Delta parameter for Poirot)
# 5. 效率与模式控制
train_sample_num=3000              # 训练集使用的样本数量 (-1 表示使用全部)
valid_sample_num=-1              # 验证集使用的样本数量 (-1 表示使用全部)
test_sample_num=-1               # 测试集(真实流模拟)使用的样本数量
num_workers=16                   # 数据加载线程数
load_mode="ratio"                # 加载模式：ratio (按固定比例增加) 或 window (按固定窗口/包数量增加)
mode="train"          # 需要运行的模块组合，空格隔开

echo "=========================================================="
echo "🚀 开始运行 Prelude 实验"
echo "📂 数据集: $dataset"
echo "⚙️  加载模式: $load_mode"
echo "📊 测试阈值列表: $checkconf_list"
echo "🔄 执行流程: $mode"
echo "=========================================================="

# 运行 Python 脚本并传入所有参数
python gate_main.py \
    --dataset $dataset \
    --checkpoint_path $checkpoint_path \
    --gatenet_note $gatenet_note \
    --backbone_note $backbone_note \
    --train_epochs $train_epochs \
    --batch_size $batch_size \
    --lb $lb \
    --ub $ub \
    --aug_num $aug_num \
    --threshold $threshold \
    --checkconf_list $checkconf_list \
    --delta $delta \
    --train_sample_num $train_sample_num \
    --valid_sample_num $valid_sample_num \
    --test_sample_num $test_sample_num \
    --mode "$mode" \
    --num_workers $num_workers \
    --load_mode $load_mode \
    --TAM_type $TAM_type \
    --test_flag $test_flag
echo "=========================================================="
echo "✅ 实验运行完毕！"
echo "=========================================================="
done