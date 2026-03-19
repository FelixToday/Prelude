import os
import argparse
import numpy as np
from tqdm import tqdm
import random


parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, default="trafficsilver_rb_CW", help="Dataset name")
parser.add_argument("--in_file", type=str, default="train", help="Input file name")
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="Directory to save model checkpoints")
parser.add_argument('--file_base_dir', default="auto",help="数据集存储路径")
from lxj_holmes_utils import adjust_args
from lxj_utils_sys import parse_args, print_config_info
args, args_help = parse_args(parser, is_print_help=False)
args = adjust_args(args)
print_config_info(args, args_help, sorted_keys=True)

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

in_path = os.path.join(args['file_base_dir'], args['dataset'])

if (os.path.exists(os.path.join(in_path, "aug_train.npz")) and
        os.path.exists(os.path.join(in_path, "aug_valid.npz"))):
    # 打印形状信息
    print("aug_train.npz 和 aug_valid.npz 文件存在，data_augmentation 提前结束")
    # 提前退出
    exit(0)  # 或者使用 return、sys.exit() 等，具体取决于上下文

data = np.load(os.path.join(in_path, f"{args['in_file']}.npz"), mmap_mode="r")

temporal_data = np.load(os.path.join(args['checkpoints'], args['dataset'], "RF_IS", f"attr_DeepLiftShap.npz"))["attr_values"]


# Calculate effective ranges for each class based on the temporal attribution data
effective_ranges = {}
for web in range(temporal_data.shape[0]):
    cur_temporal = np.cumsum(temporal_data[web])
    cur_temporal /= cur_temporal.max()
    cur_lower = np.searchsorted(cur_temporal, 0.3, side="right") * 100 // temporal_data.shape[1]
    cur_upper = np.searchsorted(cur_temporal, 0.6, side="right") * 100 // temporal_data.shape[1]
    effective_ranges[web] = (cur_lower, cur_upper)

# Construct the output file path for the augmented data
out_file = os.path.join(in_path, f"aug_{args['in_file']}.npz")


def gen_augment(data, num_aug, effective_ranges, out_file):
    X = data["X"]
    y = data["y"]

    new_X = []
    new_y = []
    abs_X = np.absolute(X)
    feat_length = X.shape[1]

    # Loop through each sample in the dataset
    for index in tqdm(range(abs_X.shape[0])):
        cur_abs_X = abs_X[index, :, 0]
        cur_web = y[index]
        loading_time = cur_abs_X.max()

        # Generate augmentations for each sample
        for ii in range(num_aug):
            p = np.random.randint(effective_ranges[cur_web][0], effective_ranges[cur_web][1])
            threshold = loading_time * p / 100

            valid_X = np.trim_zeros(cur_abs_X, "b")
            valid_index = np.where(valid_X <= threshold)

            valid_X = X[index, valid_index, :]
            valid_length = valid_X.shape[1]

            pad_width = ((0, 0), (0, feat_length - valid_length), (0, 0))
            new_X.append(
                np.pad(valid_X, pad_width, "constant", constant_values=(0, 0)))
            new_y.append(cur_web)

        # Add the original sample
        new_X.append(X[index, None])
        new_y.append(cur_web)

    new_X = np.concatenate(new_X, axis=0)
    new_y = np.array(new_y)

    # Save the augmented data to the specified output file
    print("Generate complete, prepare saving...")
    np.savez(out_file, X=new_X, y=new_y)
    print(f"Generate {out_file} done.")
    return True


import numpy as np
from tqdm import tqdm

import numpy as np
from tqdm import tqdm


def gen_augment_batched(data, num_aug, effective_ranges, out_file, batch_size=2000):
    X = data["X"]
    y = data["y"]
    N, L, C = X.shape

    final_X_list = []
    final_y_list = []

    # 预先将 effective_ranges 转换为数组，方便索引
    # 假设 y 的值是连续的整数索引，如果是字符串请先映射
    range_lookup = np.array([effective_ranges[i] for i in range(len(effective_ranges))])

    # 分批处理
    for i in tqdm(range(0, N, batch_size), desc="Processing Batches"):
        end = min(i + batch_size, N)
        batch_X = X[i:end]
        batch_y = y[i:end]
        curr_batch_size = end - i

        # 1. 计算当前 Batch 的阈值 (curr_batch_size, num_aug)
        abs_X_max = np.absolute(batch_X[:, :, 0]).max(axis=1)
        ranges = range_lookup[batch_y]  # (curr_batch_size, 2)
        p = np.random.randint(ranges[:, 0:1], ranges[:, 1:2], size=(curr_batch_size, num_aug))
        thresholds = (abs_X_max[:, np.newaxis] * p / 100.0)

        # 2. 增强数据向量化 (处理当前 Batch)
        # 形状: (curr_batch_size, num_aug, L, C)
        X_expanded = np.repeat(batch_X[:, np.newaxis, :, :], num_aug, axis=1)
        mask = np.absolute(X_expanded[:, :, :, 0]) <= thresholds[:, :, np.newaxis]

        # 排序技巧：将满足条件的元素推到左侧
        idx = np.argsort(~mask, axis=2, kind='stable')
        X_aug = np.take_along_axis(X_expanded, idx[:, :, :, np.newaxis], axis=2)

        # 清除排序带过来的多余数据
        mask_sorted = np.take_along_axis(mask, idx, axis=2)
        X_aug *= mask_sorted[:, :, :, np.newaxis]

        # 3. 整理结果：合并原始数据和增强数据
        # 变形为 (curr_batch_size * (num_aug + 1), L, C)
        # 先把增强后的合并
        X_aug_reshaped = X_aug.reshape(-1, L, C)
        y_aug_reshaped = np.repeat(batch_y, num_aug)

        # 加入原始数据
        final_X_list.append(X_aug_reshaped)
        final_X_list.append(batch_X)
        final_y_list.append(y_aug_reshaped)
        final_y_list.append(batch_y)

        # 显式清理内存（可选）
        del X_expanded, mask, idx, X_aug

    # 4. 最终合并
    print("Concatenating all batches...")
    new_X = np.concatenate(final_X_list, axis=0)
    new_y = np.concatenate(final_y_list, axis=0)

    print("Saving to file...")
    np.savez(out_file, X=new_X, y=new_y)
    print(f"Done! Saved {len(new_y)} samples.")
    return True

complete_flag = gen_augment_batched(data, 2, effective_ranges, out_file)
print("Data augmentation completed.")