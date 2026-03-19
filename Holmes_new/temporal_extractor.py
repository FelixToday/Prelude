import numpy as np
import os
import argparse
from tqdm import tqdm

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description='Temporal feature extraction of Holmes')

# Define command-line arguments
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--in_file", type=str, default="valid", help="Input file name")
parser.add_argument('--file_base_dir', default="auto",help="数据集存储路径")
# Parse command-line arguments
from lxj_holmes_utils import adjust_args
from lxj_utils_sys import parse_args, print_config_info
args, args_help = parse_args(parser, is_print_help=False)
args = adjust_args(args)
print_config_info(args, args_help, sorted_keys=True)


# Construct the input path for the dataset
in_path = os.path.join(args['file_base_dir'], args['dataset'])

# Construct the output file path
out_file = os.path.join(in_path, f"temporal_{args['in_file']}.npz")


if os.path.exists(os.path.join("../checkpoints", args['dataset'], 'RF_IS', 'attr_DeepLiftShap.npz')):
    print("已存在RF_IS特征，跳过生成")
    exit(0)

# 检查文件是否存在
if os.path.exists(out_file):
    # 文件存在，加载数据
    data = np.load(out_file)
    temporal_X = data['X']
    y = data['y']
    # 打印形状信息
    print("Shape of temporal_X:", temporal_X.shape)
    # 提前退出
    exit(0)  # 或者使用 return、sys.exit() 等，具体取决于上下文


# Load data from the specified input file
data = np.load(os.path.join(in_path, f"{args['in_file']}.npz"))
X = data["X"]
y = data["y"]


def extract_temporal_feature(X, feat_length=1000):
    abs_X = np.absolute(X)
    new_X = []

    for idx in tqdm(range(X.shape[0])):
        temporal_array = np.zeros((2, feat_length))
        loading_time = abs_X[idx].max()
        interval = 1.0 * loading_time / feat_length

        for packet in X[idx]:
            if packet == 0:
                break
            elif packet > 0:
                order = int(packet / interval)
                if order >= feat_length:
                    order = feat_length - 1
                temporal_array[0][order] += 1
            else:
                order = int(-packet / interval)
                if order >= feat_length:
                    order = feat_length - 1
                temporal_array[1][order] += 1
        new_X.append(temporal_array)
    new_X = np.array(new_X)
    return new_X


# Extract temporal features from the input data
timestamp = X[:, :, 0]
sign = np.sign(X[:, :, 1])
X = timestamp * sign

temporal_X = extract_temporal_feature(X)

# Print the shape of the extracted temporal features
print("Shape of temporal_X:", temporal_X.shape)

# Save the extracted features and labels to the output file
np.savez(out_file, X=temporal_X, y=y)
