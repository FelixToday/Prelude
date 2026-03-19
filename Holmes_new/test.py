import numpy as np
import random
import torch
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
from lxj_holmes_utils import measurement

from Holmes_model import Holmes
from lxj_utils_sys import BaseLogger
torch.multiprocessing.set_sharing_strategy('file_system')

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="Location of model checkpoints")
parser.add_argument('--file_base_dir', default="auto", help="数据集存储路径")
parser.add_argument('--load_ratio', default=100, type=int, help="加载比例")
parser.add_argument('--note', default="test_exp", help="运行保存文件夹")
from lxj_holmes_utils import adjust_args
from lxj_utils_sys import parse_args, print_config_info
args, args_help = parse_args(parser, is_print_help=False)
args = adjust_args(args)
print_config_info(args, args_help, sorted_keys=True)

device = torch.device("cuda")

in_path = os.path.join(args['file_base_dir'], args['dataset'])
ckp_path = os.path.join(args['checkpoints'], args['dataset'], "Holmes", args['note'])

logger = BaseLogger(json_save_path=os.path.join(ckp_path, "test", f"test_p{args['load_ratio']}", "result.json"),
                        log_save_path=os.path.join(ckp_path, "test", f"test_p{args['load_ratio']}", "log.txt"))
from lxj_holmes_utils import TrafficDataset
test_dataset = TrafficDataset(os.path.join(in_path, f"test.npz"), drop_extra_time=True, load_time=80, load_ratio=args['load_ratio'])

num_classes = len(np.unique(test_dataset.y))
print(f"num_classes: {num_classes}")

test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False, drop_last=False, num_workers=2)

model = Holmes(num_classes)
model.load_state_dict(torch.load(os.path.join(ckp_path, f"max_f1.pth"), map_location="cpu"))
model.to(device)

open_threshold = 1e-2
spatial_dist_file = os.path.join(ckp_path, "spatial_distribution.npz")
spatial_data = np.load(spatial_dist_file)
webs_centroid = spatial_data["centroid"]
webs_radius = spatial_data["radius"]





with torch.no_grad():
    model.eval()
    y_pred = []
    y_true = []

    for index, cur_data in enumerate(test_iter):
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        embs = model(cur_X).cpu().numpy()
        cur_y = cur_y.cpu().numpy()

        all_sims = 1 - cosine_similarity(embs, webs_centroid)
        all_sims -= webs_radius
        outs = np.argmin(all_sims, axis=1)

        # if scenario == "Open-world":
        #     outs_d = np.min(all_sims, axis=1)
        #     open_indices = np.where(outs_d > open_threshold)[0]
        #     outs[open_indices] = num_classes - 1

        y_pred.append(outs)
        y_true.append(cur_y)
    y_pred = np.concatenate(y_pred).flatten()
    y_true = np.concatenate(y_true).flatten()

    result = measurement(y_true, y_pred)
    print(result)
    logger.log("test.metrics", result, unzip_dict=True)

