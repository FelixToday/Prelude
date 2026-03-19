# -*- coding: utf-8 -*-
import math
# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2026/1/12 上午8:53
import numpy as np
import os
import argparse
import random
import torch
from tqdm import tqdm
import multiprocessing as mp

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description='Feature extraction')
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--in_file", type=str, default="valid", help="input file")
parser.add_argument('--file_base_dir', default="auto",help="数据集存储路径")
parser.add_argument('--load_ratio', default=-1, type=int, help="加载比例")
from lxj_holmes_utils import adjust_args
from lxj_utils_sys import parse_args, print_config_info
args, args_help = parse_args(parser, is_print_help=False)
args = adjust_args(args)
print_config_info(args, args_help, sorted_keys=True)

in_path = os.path.join(args['file_base_dir'], args['dataset'])
if args['load_ratio'] == -1:
    out_file = os.path.join(in_path, f"taf_{args['in_file']}.npz")
    args['load_ratio'] = 100
else:
    out_file = os.path.join(in_path, f"taf_{args['in_file']}_p{args['load_ratio']}.npz")

if os.path.exists(out_file):
    data = np.load(out_file)
    X = data['X']
    y = data['y']
    # 打印形状信息
    print("Shape of X:", X.shape)
    print("文件已存在提前结束")
    # 提前退出
    exit(0)  # 或者使用 return、sys.exit() 等，具体取决于上下文


from Explore.ExploreRun.utils_dataset_metric import load_data, load_partial_page
X, y = load_data(os.path.join(in_path, f"{args['in_file']}.npz"), drop_extra_time=True, load_time=80)
if args['load_ratio'] != 100:
    X, y = load_partial_page(X, y, args['load_ratio'])

print("X.shape:", X.shape)

def fast_count_burst(arr):
    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0]
    segment_starts = np.insert(change_indices + 1, 0, 0)
    segment_ends = np.append(change_indices, len(arr) - 1)
    segment_lengths = segment_ends - segment_starts + 1
    segment_signs = np.sign(arr[segment_starts])
    adjusted_lengths = segment_lengths * segment_signs

    return adjusted_lengths

def agg_interval(packets):
    features = []
    features.append([np.sum(packets > 0), np.sum(packets < 0)])

    dirs = np.sign(packets)
    assert not np.any(dir == 0), "Array contains zero!"
    bursts = fast_count_burst(dirs)
    features.append([np.sum(bursts > 0), np.sum(bursts < 0)])
    features.append([0, 0])
    #
    # pos_bursts = bursts[bursts > 0]
    # neg_bursts = np.abs(bursts[bursts < 0])
    # vals = []
    # if len(pos_bursts) == 0:
    #     vals.append(0)
    # else:
    #     vals.append(np.mean(pos_bursts))
    # if len(neg_bursts) == 0:
    #     vals.append(0)
    # else:
    #     vals.append(np.mean(neg_bursts))
    # features.append(vals)

    return np.array(features, dtype=np.float32)

def process_TAF(index, sequence, interval, max_len):
    timestamp = sequence[:, 0]
    sign = np.sign(sequence[:, 1])
    sequence = timestamp * sign

    interval = interval * 1e-3

    packets = np.trim_zeros(sequence, "fb")
    abs_packets = np.abs(packets)
    st_time = abs_packets[0]
    st_pos = 0
    TAF = np.zeros((3, 2, max_len))

    for interval_idx in range(max_len):
        ed_time = (interval_idx + 1) * interval
        if interval_idx == max_len - 1:
            ed_pos = abs_packets.shape[0]
        else:
            ed_pos = np.searchsorted(abs_packets, st_time + ed_time)

        assert ed_pos >= st_pos, f"{index}: st:{st_pos} -> ed:{ed_pos}"
        if st_pos < ed_pos:
            cur_packets = packets[st_pos:ed_pos]
            TAF[:, :, interval_idx] = agg_interval(cur_packets)
        st_pos = ed_pos

    return index, TAF

def task_generator(sequences, interval, max_len):
    for i, seq in enumerate(sequences):
        yield (i, seq, interval, max_len)


def extract_TAF1(sequences):
    interval = 40
    max_len = 2000
    num_sequences = sequences.shape[0]
    # 初始化结果矩阵
    TAF = np.zeros((num_sequences, 3, 2, max_len))
    num_workers = 8
    print(f"Using {num_workers} workers for feature extraction...")
    with mp.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(
            process_TAM_RF_wrapper,
            task_generator(sequences, interval, max_len),
            chunksize=20
        )
        del sequences  # 现在这一步“更有意义”

        with tqdm(total=num_sequences, desc="Extracting TAF") as pbar:
            for index, result in iterator:
                TAF[index] = result
                pbar.update(1)

    return TAF

def extract_TAF(sequences):
    interval = 40
    max_len = 2000
    num_sequences = sequences.shape[0]
    # 初始化结果矩阵
    TAF = np.zeros((num_sequences, 3, 2, max_len))

    # 构造任务列表：将 index 和对应的 sequence 打包
    tasks = [(i, sequences[i], interval, max_len) for i in range(num_sequences)]

    del sequences
    num_workers = max(1, mp.cpu_count() - 2)  # 留2个核，防止卡死
    #num_workers = 16

    print(f"Using {num_workers} workers for feature extraction...")

    # 使用 mp.Pool 和 imap
    with mp.Pool(processes=num_workers) as pool:
        # imap 会按顺序或通过迭代器返回结果，chunksize 可以根据数据量调整
        # 如果单个任务很快，建议增加 chunksize（例如 10-100）以减少通信开销
        iterator = pool.imap_unordered(process_TAM_RF_wrapper, tasks, chunksize=20)
        with tqdm(total=num_sequences, desc="Extracting TAF") as pbar:
            for index, result in iterator:
                TAF[index] = result
                pbar.update(1)
    return TAF
def process_TAM_RF_wrapper(args):
    return process_TAF(*args)

X = extract_TAF(X)
# Print processing information
print(f"{args['in_file']} -- {args['load_ratio']} process done: X = {X.shape}, y = {y.shape}")
# Save the processed data into a new .npz file
np.savez(out_file, X=X, y=y)