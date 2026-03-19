#!/bin/bash

# 颜色定义
RED='\033[1;91m'        # 更鲜艳的红色
GREEN='\033[1;92m'      # 更鲜艳的绿色
YELLOW='\033[1;93m'     # 更鲜艳的黄色
BLUE='\033[1;94m'       # 更鲜艳的蓝色
MAGENTA='\033[1;95m'    # 更鲜艳的洋红色
CYAN='\033[1;96m'       # 更鲜艳的青色
WHITE='\033[1;97m'      # 更鲜艳的白色
NC='\033[0m' # No Color

# 函数定义：带颜色的输出
color_echo() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Phase 0: Global Environment Setup
# 定义数据集名称为 CW
# CW OW trafficsilver_rb_CW wtfpad_CW front_CW regulator_CW
for dataset in wtfpad_CW front_CW regulator_CW
# Phase 1: Temporal Baseline & Feature Attribution
do
  color_echo "${CYAN}" "=== Phase 1: Temporal Baseline & Feature Attribution ==="

  # Step 1: Extract Temporal Features
  color_echo "${GREEN}" "Step 1: Extracting temporal features..."
  for filename in train valid
  do
      color_echo "${WHITE}" "  Processing ${filename}..."
      python -u temporal_extractor.py --dataset ${dataset} --in_file ${filename} --file_base_dir auto
  done

  # Step 2: Baseline Model Training (RF)
  color_echo "${GREEN}" "Step 2: Training RF baseline model..."
  python -u train_RF.py --dataset ${dataset} --train_epochs 30 --checkpoints ../checkpoints/ --file_base_dir auto

  # Step 3: Feature Attribution Analysis
  color_echo "${GREEN}" "Step 3: Performing feature attribution analysis..."
  python -u feature_attr.py --dataset ${dataset} --checkpoints ../checkpoints/ --file_base_dir auto

  # Phase 2: Data Augmentation & TAF Generation
  color_echo "${CYAN}" "=== Phase 2: Data Augmentation & TAF Generation ==="

  # Step 4: Execute Data Augmentation
  color_echo "${GREEN}" "Step 4: Executing data augmentation..."
  for filename in train valid
  do
      color_echo "${WHITE}" "  Augmenting ${filename}..."
      python -u data_augmentation.py --dataset ${dataset} --in_file ${filename} --checkpoints ../checkpoints/ --file_base_dir auto
  done

  # Step 5: Generate TAF Features
  color_echo "${GREEN}" "Step 5: Generating TAF features..."
  for filename in aug_train aug_valid test
  do
      color_echo "${WHITE}" "  Generating TAF for ${filename}..."
      python -u gen_taf.py --dataset ${dataset} --in_file ${filename} --file_base_dir auto
  done

  # Phase 3: Core Model Training & Analysis
  color_echo "${CYAN}" "=== Phase 3: Core Model Training & Analysis ==="

  # Step 6: Final Model Training (Holmes)
  color_echo "${GREEN}" "Step 6: Training Holmes model..."
  python -u train.py --dataset ${dataset} --train_epochs 30 --checkpoints ../checkpoints/ --file_base_dir auto

  # Step 7: Spatial Analysis
  color_echo "${GREEN}" "Step 7: Performing spatial analysis..."
  python -u spatial_analysis.py --dataset ${dataset} --checkpoints ../checkpoints/ --file_base_dir auto

  # Phase 4: Robustness Testing (Multi-percentile)
  color_echo "${CYAN}" "=== Phase 4: Robustness Testing (Multi-percentile) ==="

  # Step 8: Iterative Test & Evaluation
  color_echo "${GREEN}" "Step 8: Running robustness tests..."
  for load_ratio in {10..100..10}
  do
      color_echo "${YELLOW}" "  Testing with ${load_ratio}% data..."
      # A: Generate TAF for specific test percentile
      python -u gen_taf.py --dataset ${dataset} --in_file test --file_base_dir auto --load_ratio ${load_ratio}

      # B: Evaluate Holmes model performance
      python -u test.py --dataset ${dataset} --test_file taf_test_p${load_ratio} --result_file test_p${load_ratio} --checkpoints ../checkpoints/ --file_base_dir auto
  done

  color_echo "${MAGENTA}" "=== All experiments completed! ==="
done
