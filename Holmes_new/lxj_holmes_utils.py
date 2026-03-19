# -*- coding: utf-8 -*-
import numpy as np
# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2026/1/12 上午10:24
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import Dataset
import os

def adjust_args(args):

    """
    调整函数参数，根据主机名设置文件基础目录

    参数:
        args (dict): 包含参数的字典，需要包含'file_base_dir'键

    返回:
        dict: 调整后的参数字典

    如果'file_base_dir'值为"auto"，则根据当前主机名从预定义的目录映射中获取对应的文件基础目录
    """
    if args['file_base_dir'] == "auto":  # 检查是否需要自动设置文件基础目录
        from Explore.ExploreRun.const import filebase_dir_dict  # 导入主机名与目录的映射字典
        import socket  # 导入socket模块用于获取主机名
        args['file_base_dir'] = filebase_dir_dict[socket.gethostname()]  # 根据主机名获取对应的文件基础目录
    return args  # 返回调整后的参数字典


def measurement(y_true, y_pred):
    results = {}
    results["Accuracy"] = round(accuracy_score(y_true, y_pred) * 100, 2)
    results["Precision"] = round(precision_score(y_true, y_pred, average="macro") * 100, 2)
    results["Recall"] = round(recall_score(y_true, y_pred, average="macro") * 100, 2)
    results["F1-score"] = round(f1_score(y_true, y_pred, average="macro") * 100, 2)
    return results


from Explore.ExploreRun.utils_dataset_metric import load_data, load_partial_page
import torch
from torch.utils.data import Dataset
import os
import tqdm  # 建议加一个进度条，因为预处理可能耗时

import torch
from torch.utils.data import Dataset
import os
import tqdm
from multiprocessing import Pool, cpu_count


class TrafficDataset(Dataset):
    def __init__(self, data_path, drop_extra_time=False, load_time=80, load_ratio=100, trace_len = -1):
        if isinstance(data_path, str):
            self.X, self.y = load_data(data_path, drop_extra_time=drop_extra_time, load_time=load_time)
        else:
            self.X, self.y = data_path
        if trace_len == -1:
            pass
        else:
            self.X = self.X[:, :trace_len, :]
            print(f"最大数据长度：{len(trace_len)}")

        print(f"数据条数：{len(self.y)}，加载比例：{load_ratio} %")
        self.y =torch.tensor(self.y, dtype=torch.int64)
        self.load_ratio = load_ratio

        if self.load_ratio != 100:
            self.X, self.y = load_partial_page(self.X, self.y, self.load_ratio)

        self.interval = 40
        self.max_len = 2000

        # 调用多进程预处理
        self.pre_gen_TAF_parallel()

    def pre_gen_TAF_parallel(self):
        print(f"开始多进程预生成 (使用 {cpu_count()} 个核心)...")

        # 1. 准备参数列表：因为 process_TAF 需要多个参数，我们要打包它们
        # 如果 process_TAF 只需要 X[i]，可以简化
        task_args = [(item, self.interval, self.max_len) for item in self.X]

        # 2. 释放原始 X 的引用以节省空间（可选，取决于 load_data 返回的类型）
        raw_X = self.X
        self.X = []

        # 3. 使用进程池进行并行计算
        # starmap 可以自动解包 task_args 中的元组并传给函数
        with Pool(processes=cpu_count()) as pool:
            # chunksize 设置大一点可以减少进程间通信的开销
            self.X = list(tqdm.tqdm(
                pool.starmap(process_TAF, task_args),
                total=len(task_args)
            ))

        print("多进程预生成完毕。")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


import torch
import numpy as np
from torch.utils.data import Dataset


class TrafficDataset_Online(Dataset):
    def __init__(self, data_path, drop_extra_time=False, load_time=80, load_ratio=100, trace_len=-1):
        # 1. 仅加载原始数据，不进行任何裁剪或预处理
        if isinstance(data_path, str):
            # 假设 load_data 返回的是原始的 X (特征) 和 y (标签)
            self.raw_X, self.y = load_data(data_path, drop_extra_time=False)
        else:
            self.raw_X, self.y = data_path

        # 基础参数配置
        self.loaded_ratio = load_ratio
        self.drop_extra_time = drop_extra_time
        self.max_load_time_limit = load_time  # 对应你参考代码中的 maximum_load_time
        self.trace_len = trace_len

        # TAF/TAM 处理相关的超参数
        self.interval = 40
        self.max_len = 2000

        print(f"Dataset 初始化完成。总数据量: {len(self.y)}, 默认加载比例: {self.loaded_ratio}%")

    def __len__(self):
        return len(self.raw_X)

    def __getitem__(self, idx):
        # --- 步骤 1: 获取原始样本 ---
        data = self.raw_X[idx]
        label = self.y[idx]

        # --- 步骤 2: 动态计算时间截断 (Online Load Ratio) ---
        # 假设 data 的第一列 [:, 0] 是时间戳 timestamp
        timestamp = data[:, 0]
        max_timestamp = timestamp.max()

        # 计算当前比例下的时间阈值
        if self.drop_extra_time:
            # 参考你提供的逻辑：比例截断 vs 固定时间上限 取最小值
            threshold = min(max_timestamp * self.loaded_ratio / 100, self.max_load_time_limit)
        else:
            threshold = max_timestamp * self.loaded_ratio / 100

        # 根据阈值筛选合法的数据点
        valid_mask = timestamp <= threshold
        data = data[valid_mask, :]

        # --- 步骤 3: 限制数据点个数 (Trace Len) ---
        if self.trace_len != -1:
            data = data[:self.trace_len, :]

        # --- 步骤 4: 执行核心计算 (process_TAF) ---
        # 这里调用你原来的在线处理函数
        processed_x = process_TAF(data, self.interval, self.max_len)

        # 确保输出为 Tensor
        if not isinstance(processed_x, torch.Tensor):
            processed_x = torch.from_numpy(processed_x).float()

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.int64)

        return processed_x, label


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

def process_TAF(sequence, interval, max_len):
    timestamp = sequence[:, 0]
    sign = np.sign(sequence[:, 1])
    sequence = timestamp * sign

    interval = interval * 1e-3

    packets = np.trim_zeros(sequence, "fb")
    abs_packets = np.abs(packets)
    st_time = abs_packets[0]
    st_pos = 0
    TAF = np.zeros((3, 2, max_len), dtype=np.float32)

    for interval_idx in range(max_len):
        ed_time = (interval_idx + 1) * interval
        if interval_idx == max_len - 1:
            ed_pos = abs_packets.shape[0]
        else:
            ed_pos = np.searchsorted(abs_packets, st_time + ed_time)

        assert ed_pos >= st_pos, f"st:{st_pos} -> ed:{ed_pos}"
        if st_pos < ed_pos:
            cur_packets = packets[st_pos:ed_pos]
            TAF[:, :, interval_idx] = agg_interval(cur_packets)
        st_pos = ed_pos

    return TAF

