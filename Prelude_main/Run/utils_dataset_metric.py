

# -*- coding: utf-8 -*-
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np

from torch.utils.data import DataLoader
from Prelude_main.Model import get_model, EDdataset, RandomEarlyTruncationWrapper

from lxj_utils_sys import print_colored


def autodict(**kwargs):
    return kwargs


def get_model_and_dataloader(X1, y1, X2, y2, num_classes, config: dict, args: dict):
    """根据配置创建 Prelude 模型和数据加载器"""
    num_workers = args['num_workers']

    dataset_config = autodict(loaded_ratio=args['load_ratio'], seq_len=config['seq_len'], is_idx=config['use_idx'],
                              TAM_type=args['TAM_type'], BAPM=None, maximum_cell_number=config['maximum_cell_number'],
                              max_matrix_len=config['max_matrix_len'], log_transform=config['log_transform'],
                              maximum_load_time=args['maximum_load_time'],
                              time_interval_threshold=config['time_interval_threshold'],
                              drop_extra_time=args['drop_extra_time'])
    set1 = EDdataset(X1, y1, **dataset_config)
    set2 = EDdataset(X2, y2, **dataset_config)

    # ========== early_augment 包装 ==========
    if args.get('early_augment', False):
        aug_lb = args.get('aug_lb', 0.01)
        aug_ub = args.get('aug_ub', 1.0)
        aug_num = args.get('aug_num', 20)
        set1 = RandomEarlyTruncationWrapper([set1], X1, y1, lb=aug_lb, ub=aug_ub, aug_num=aug_num, info="训练集")
    else:
        print_colored("=" * 10 + " 不需要增强 " + "=" * 10, "red")
    # =========================================

    try:
        patch_size = next(iter(set1))[0].shape[1]
    except:
        patch_size = next(iter(set1))[0][0].shape[1]

    loader1 = DataLoader(set1, batch_size=int(config['batch_size']),
                         shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    loader2 = DataLoader(set2, batch_size=int(config['batch_size']),
                         shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    # 模型实例化
    model = get_model(patch_size=patch_size, num_classes=num_classes, num_tabs=args['num_tabs'],
                      drop_path_rate=config['drop_path_rate'], depth=config['depth'],
                      embed_dim=config['embed_dim'], max_matrix_len=config['max_matrix_len'],
                      early_stage=args.get('early_stage', False), fine_predict=config['fine_predict'],
                      overlap_ratio=args['overlap_ratio'])

    return model, loader1, loader2


def load_data(data_path, drop_extra_time=False, load_time=None):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]
    # 时间负数调整
    X[:, :, 0] = np.abs(X[:, :, 0])
    # 去除大小信息
    #X[:, :, 1] = np.sign(X[:, :, 1])
    if drop_extra_time and load_time is not None:
        print(f"丢弃额外时间，时间上限：{load_time}")
        invalid_ind = X[:, :, 0]>load_time
        X[invalid_ind, :] = 0
    else:
        print("加载完整流量!")
    return X, y


# def load_partial_page(X, y, load_ratio):
#     """
#     根据加载比例截断流量数据。
#
#     参数:
#     - X: np.array, 形状为 (N, L, C)，其中 C=0 通常是时间戳
#     - y: np.array, 形状为 (N,)
#     - load_ratio: float, 加载比例 (0 到 100 之间)
#
#     返回:
#     - processed_X: 处理后并补齐长度的 X
#     - processed_y: 对应的标签 y
#     """
#
#     N, feat_length, C = X.shape
#     processed_X = np.zeros_like(X)  # 预分配空间，初始全为 0 (自动完成 padding)
#
#     # 提取时间戳列 (假设第一列是时间)
#     abs_X = X[:, :, 0]
#     print("提取比例数据...")
#     for i in tqdm(range(N)):
#         # 1. 获取当前样本的有效报文时间
#         current_sample_time = abs_X[i]
#         # 去掉末尾补零的部分，找到真正的加载结束时间
#         valid_times = np.trim_zeros(current_sample_time, 'b')
#         if len(valid_times) == 0:
#             continue
#
#         loading_time = valid_times.max()
#         threshold = loading_time * (load_ratio / 100.0)
#
#         # 2. 找到符合时间条件的索引
#         indices = np.where(current_sample_time <= threshold)[0]
#
#         # 3. 提取特征并填入预分配的数组中
#         # 注意：这里直接填入前部分，后面自然保持为 0，实现了 padding
#         selected_data = X[i, indices, :]
#         processed_X[i, :len(indices), :] = selected_data
#
#     return processed_X, np.array(y)

#
# def parse_value(config):
#     """尝试将字符串转换成 int/float/bool，失败则保持原样"""
#     def parse_value(value):
#         value = value.strip()
#         try:
#             return int(value)
#         except ValueError:
#             try:
#                 return float(value)
#             except ValueError:
#                 if value.lower() in ('true', 'false'):
#                     return value.lower() == 'true'
#                 return value
#
#     config_dict = {k: parse_value(v) for k, v in config['config'].items()}
#     return config_dict
#
# # gen_one_hot 暂时保留在文件中以防万一，但 compute_metric 不再调用它处理多标签
# def gen_one_hot(arr, num_classes):
#     binary = np.zeros((arr.shape[0], num_classes))
#     for i in range(arr.shape[0]):
#         binary[i, arr[i]] = 1
#     return binary


