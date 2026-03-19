

# -*- coding: utf-8 -*-

# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/1 下午8:22
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np


from torch.utils.data import DataLoader
from CountMambaModel import CountMambaModel
from Prelude_main.Model.baseline import *
from Prelude_main.Model import get_model, EDdataset
import torch.nn.functional as F

def autodict(**kwargs):
    return kwargs


def get_model_and_dataloader(X1, y1, X2, y2, num_classes, config: dict, args: dict):
    """
    根据配置创建模型和数据加载器
    """
    num_workers = args['num_workers']

    # 移除 MultiTab 相关的逻辑判断，统一数据集加载方式
    if config['model'] in ["AWF", "DF", "TF", "TMWF", "ARES", "TikTok", "VarCNN", "RF"]:
        dataset_config = autodict(loaded_ratio=args['load_ratio'], length=int(config['seq_len']))
        if config['model'] in ["AWF", "DF", "TF", "TMWF", "ARES"]:
            dataset_str = 'DirectionDataset'
        elif config['model'] in ["TikTok"]:
            dataset_str = 'DTDataset'
        elif config['model'] in ["VarCNN"]:
            dataset_str = 'DT2Dataset'
        elif config['model'] in ["RF"]:
            dataset_str = 'RFDataset'
        else:
            # 这里的 MultiTabRF 已经移除，默认为 RFDataset
            dataset_str = 'RFDataset'

        set1 = eval(dataset_str)(X1, y1, **dataset_config)
        set2 = eval(dataset_str)(X2, y2, **dataset_config)
    else:
        # 非传统模型（Mamba, ExploreModel 等）
        TAM_type = "Mamba" if config['model'] == "CountMamba" else args['TAM_type']

        dataset_config = autodict(loaded_ratio=args['load_ratio'], seq_len=config['seq_len'], is_idx=args['use_idx'],
                                  TAM_type=TAM_type, BAPM=None, maximum_cell_number=config['maximum_cell_number'],
                                  max_matrix_len=config['max_matrix_len'], log_transform=config['log_transform'],
                                  maximum_load_time=args['maximum_load_time'],
                                  time_interval_threshold=config['time_interval_threshold'],
                                  drop_extra_time=args['drop_extra_time'])
        set1 = EDdataset(X1, y1, **dataset_config)
        set2 = EDdataset(X2, y2, **dataset_config)

    try:
        patch_size = next(iter(set1))[0].shape[1]
    except:
        patch_size = next(iter(set1))[0][0].shape[1]

    loader1 = DataLoader(set1, batch_size=int(config['batch_size']),
                         shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    loader2 = DataLoader(set2, batch_size=int(config['batch_size']),
                         shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    # 模型实例化
    if config['model'] == "AWF":
        model = AWF(num_classes=num_classes)
    elif config['model'] == "DF":
        model = DF(num_classes=num_classes)
    elif config['model'] == "TikTok":
        model = TikTok(num_classes=num_classes)
    elif config['model'] == "VarCNN":
        model = VarCNN(num_classes=num_classes)
    elif config['model'] == "TF":
        model = TF(num_classes=num_classes)
    elif config['model'] == "TMWF":
        model = TMWF(num_classes=num_classes)
    elif config['model'] == "ARES":
        model = ARES(num_classes=num_classes)
    elif config['model'] == "RF":
        model = RF(num_classes=num_classes)
    elif config['model'] == "CountMamba":
        # 即使是单标签，CountMamba 可能仍需要 num_tabs 参数来配置输入层或头，故保留传参
        model = CountMambaModel(num_classes=num_classes, drop_path_rate=config['drop_path_rate'], depth=config['depth'],
                                embed_dim=config['embed_dim'], patch_size=patch_size,
                                max_matrix_len=config['max_matrix_len'],
                                early_stage=config['early_stage'], num_tabs=args['num_tabs'],
                                fine_predict=config['fine_predict'])
    elif config['model'] == 'Prelude':
        model = get_model( patch_size=patch_size, num_classes=num_classes, num_tabs=args['num_tabs'],
                           model_name=args['Model_name'],
                           drop_path_rate=config['drop_path_rate'], depth=config['depth'],
                           embed_dim=config['embed_dim'], max_matrix_len=config['max_matrix_len'],
                           early_stage=config['early_stage'], fine_predict=config['fine_predict'],
                           overlap_ratio=args['overlap_ratio'])
    else:
        raise Exception(f'模型名称错误: {config["model"]}')

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


def knn_monitor(net, device, memory_data_loader, test_data_loader, num_classes, k=200, t=0.1):
    """
    Perform k-Nearest Neighbors (kNN) monitoring.

    Parameters:
    net (nn.Module): The neural network model.
    device (torch.device): The device to run the computations on.
    memory_data_loader (DataLoader): DataLoader for the memory bank.
    test_data_loader (DataLoader): DataLoader for the test data.
    num_classes (int): Number of classes.
    k (int): Number of nearest neighbors to use.
    t (float): Temperature parameter for scaling.

    Returns:
    tuple: True labels and predicted labels.
    """
    net.eval()
    total_num = 0
    feature_bank, feature_labels = [], []
    y_pred = []
    y_true = []

    with torch.no_grad():
        # Generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(device)
        feature_labels = torch.cat(feature_labels, dim=0).t().contiguous().to(device)

        # Loop through test data to predict the label by weighted kNN search
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, k, t)
            total_num += data.size(0)
            y_pred.append(pred_labels[:, 0].cpu().numpy())
            y_true.append(target.cpu().numpy())

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    return y_true, y_pred


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """
    Predict labels using k-Nearest Neighbors (kNN) with cosine similarity.

    Parameters:
    feature (Tensor): Feature tensor.
    feature_bank (Tensor): Feature bank tensor.
    feature_labels (Tensor): Labels corresponding to the feature bank.
    classes (int): Number of classes.
    knn_k (int): Number of nearest neighbors to use.
    knn_t (float): Temperature parameter for scaling.

    Returns:
    Tensor: Predicted labels.
    """
    feature_labels = feature_labels.long()

    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    return pred_labels