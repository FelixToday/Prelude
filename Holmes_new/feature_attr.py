import os
import torch
import random
import argparse
import numpy as np
from captum import attr
from tqdm import tqdm
import warnings

from RF_model import RF


warnings.filterwarnings("ignore", category=UserWarning, module='captum')

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="Feature attribution")
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="Location of model checkpoints")
parser.add_argument('--file_base_dir', default="auto",help="数据集存储路径")
from lxj_holmes_utils import adjust_args
from lxj_utils_sys import parse_args, print_config_info
args, args_help = parse_args(parser, is_print_help=False)
args = adjust_args(args)
print_config_info(args, args_help, sorted_keys=True)

in_path = os.path.join(args['file_base_dir'], args['dataset'])
ckp_path = os.path.join(args['checkpoints'], args['dataset'], "RF_IS")
out_file = os.path.join(args['checkpoints'], args['dataset'], "RF_IS", f"attr_DeepLiftShap.npz")

if os.path.exists(os.path.join("../checkpoints", args['dataset'], 'RF_IS', 'attr_DeepLiftShap.npz')):
    print("已存在RF_IS特征，跳过生成")
    exit(0)

if (os.path.exists(os.path.join(in_path, "aug_train.npz")) and
        os.path.exists(os.path.join(in_path, "aug_valid.npz"))):
    # 打印形状信息
    print("aug_train.npz 和 aug_valid.npz 文件存在，feature_attr 提前结束")
    # 提前退出
    exit(0)  # 或者使用 return、sys.exit() 等，具体取决于上下文

def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    X = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    return X, y


valid_X, valid_y = load_data(os.path.join(in_path, f"temporal_valid.npz"))
num_classes = len(np.unique(valid_y))

# Print dataset information
print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
print(f"num_classes: {num_classes}")

model = RF(num_classes)
model.load_state_dict(torch.load(os.path.join(ckp_path, f"max_f1.pth"), map_location="cpu"))

model.eval()
attr_model = attr.DeepLiftShap(model)

bg_traffic = []
test_traffic = {}
for web in range(num_classes):
    bg_test_X = valid_X[valid_y == web]
    assert bg_test_X.shape[0] >= 12
    bg_traffic.append(bg_test_X[0:2])  # Use the first 2 samples as background
    test_traffic[web] = bg_test_X[2:12]  # Use the next 10 samples for testing

# Concatenate all background traffic into a single tensor
bg_traffic = torch.concat(bg_traffic, axis=0)

attr_values = []
# Iterate over each class to calculate attribution values
for web in tqdm(range(num_classes)):
    # Calculate attributions for the test samples using the background samples
    attr_result = attr_model.attribute(test_traffic[web], bg_traffic, target=web)
    # Aggregate the attribution results
    attr_result = attr_result.detach().numpy().squeeze().sum(axis=0).sum(axis=0)
    attr_values.append(attr_result)

attr_values = np.array(attr_values)

# Print the shape of the attribution values
print("shape of attr_values:", attr_values.shape)

# Save the attribution values to a file
np.savez(out_file, attr_values=attr_values)
