#!/bin/bash

# ====================================================
# 配置部分
# ====================================================

# 颜色与时间函数
get_bj_time() {
    local format=$1
    local utc_sec=$(date +%s)
    local bj_sec=$((utc_sec + 28800))
    date -d "@$bj_sec" +"$format"
}

RED='\033[1;91m'; GREEN='\033[1;92m'; YELLOW='\033[1;93m'; BLUE='\033[1;94m';
MAGENTA='\033[1;95m'; CYAN='\033[1;96m'; WHITE='\033[1;97m'; NC='\033[0m'

get_duration() {
    local start=$1; local end=$2; local diff=$((end - start))
    echo "$((diff/60))分 $((diff%60))秒"
}

# ====================================================
# 执行逻辑
# ====================================================

train_holmes_epochs=30
datasets=("trafficsilver_rb_CW" "regulator_CW")
note="baseline_same"
valid_name="aug_valid"

for dataset in "${datasets[@]}"
do
    echo -e "${BLUE}##################################################${NC}"
    echo -e "${BLUE}# 开始处理数据集: ${WHITE}${dataset}${NC}"
    echo -e "${BLUE}# 当前时间: $(get_bj_time '%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BLUE}##################################################${NC}"

    # 1. 环境清理
    python -u clean_cache.py --dataset $dataset --clean_aug_data False --file_base_dir auto

    d_start=$(date +%s)

    # ----------------------------------------------------
    # Phase 1: 基础特征提取与归因分析
    # ----------------------------------------------------
    p1_start=$(date +%s)
    echo -e "${CYAN}=== Phase 1: Temporal Baseline & Feature Attribution ===${NC}"

    # Step 1: 时间特征提取
    for filename in train valid; do
        python -u temporal_extractor.py --dataset ${dataset} --in_file ${filename} --file_base_dir auto
    done

    # Step 2: 训练随机森林基准
    python -u train_RF.py --dataset ${dataset} --train_epochs 30 --checkpoints ../checkpoints/ --file_base_dir auto

    # Step 3: 特征重要性分析
    python -u feature_attr.py --dataset ${dataset} --checkpoints ../checkpoints/ --file_base_dir auto
    echo -e "${GREEN}Phase 1 完成，耗时: $(get_duration $p1_start $(date +%s))${NC}"

    # ----------------------------------------------------
    # Phase 2: 数据增强
    # ----------------------------------------------------
    p2_start=$(date +%s)
    echo -e "${CYAN}=== Phase 2: Data Augmentation ===${NC}"

    # Step 4: 数据增强
    for filename in train valid; do
        python -u data_augmentation.py --dataset ${dataset} --in_file ${filename} --checkpoints ../checkpoints/ --file_base_dir auto
    done
    echo -e "${GREEN}Phase 2 完成，耗时: $(get_duration $p2_start $(date +%s))${NC}"

    # ----------------------------------------------------
    # Phase 3: 核心模型训练
    # ----------------------------------------------------
    p3_start=$(date +%s)
    echo -e "${CYAN}=== Phase 3: Core Model Training ===${NC}"

    # Step 6: Holmes 模型训练
    python -u train.py --dataset ${dataset} --train_epochs $train_holmes_epochs --checkpoints ../checkpoints/ --file_base_dir auto --valid_name $valid_name --note $note

    # Step 7: 空间分析
    python -u spatial_analysis.py --dataset ${dataset} --checkpoints ../checkpoints/ --file_base_dir auto --note $note
    echo -e "${GREEN}Phase 3 完成，耗时: $(get_duration $p3_start $(date +%s))${NC}"

    # ----------------------------------------------------
    # Phase 4: 鲁棒性测试 (多载荷比例)
    # ----------------------------------------------------
    p4_start=$(date +%s)
    echo -e "${CYAN}=== Phase 4: Robustness Testing ===${NC}"

    for load_ratio in {10..100..10}; do
        echo -e "${YELLOW}    Testing Ratio: ${load_ratio}%${NC}"
        python -u test.py --dataset ${dataset} --checkpoints ../checkpoints/ --file_base_dir auto --load_ratio $load_ratio --note $note
    done

    d_end=$(date +%s)
    echo -e "${GREEN}>>> 数据集 ${dataset} 全部流程结束！${NC}"
    echo -e "${GREEN}>>> 总耗时: $(get_duration $d_start $d_end)${NC}"
    echo -e "${BLUE}-------------------------------------------${NC}"

done

echo -e "${MAGENTA}=== 所有任务执行完毕！ ===${NC}"