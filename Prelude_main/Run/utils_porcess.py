# -*- coding: utf-8 -*-
# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/11 下午7:25

from tqdm import tqdm
import argparse
import os
import torch
import numpy as np
import configparser
from lxj_utils_sys import parse_args, str_to_bool, print_colored, measurement
from Prelude_main.Run.utils_dataset_metric import load_data, knn_monitor
from Prelude_main.Run.const import dataset_lib, get_filebase_dir, get_machine_name


param_description = {
    "model": "模型名称",
    "device": "训练设备",
    "seq_len": "输入序列的最大长度",
    "train_epochs": "训练的总轮数",
    "batch_size": "每个批次的样本数量",
    "learning_rate": "学习率",
    "optimizer": "优化器",
    "eval_metrics": "评估指标",
    "save_metric": "选择最佳模型的指标",
    "max_matrix_len": "特征矩阵的最大长度",
    "log_transform": "是否对特征进行对数变换",
    "embed_dim": "嵌入向量的维度",
    "time_interval_threshold": "判断簇的百分比",
    "maximum_cell_number": "数据包cell分级数量",
    "num_heads": "注意力头的数量",
    "r_of_lina": "Lina中的关键参数 `r`",
    "atten_type": "使用的注意力机制类型",
    "num_tabs": "标签页数量",
    "sample_num": "样本数量设置",
}


def get_parser(mode="train"):
    """
    根据模式创建参数解析器

    Args:
        mode (str): 模式选择，可选 "train" 或 "test"

    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(description=f"参数配置 - {mode}模式")

    # ========== 公共参数 ==========
    # 数据集相关
    parser.add_argument('--machine_name', type=str, default=get_machine_name(),
                        help='服务器名称')
    parser.add_argument('--config', default="config/AWF.ini",
                        help="模型配置文件路径")
    parser.add_argument('--checkpoint_path', default="../../checkpoints",
                        help="运行结果存储路径")
    parser.add_argument('--file_base_dir', default="auto",
                        help="数据集存储路径")
    parser.add_argument('--dataset', default="CW",
                        help="训练和评估使用的数据集名称")
    parser.add_argument('--load_ratio', type=float, default=100,
                        help="数据加载比例（百分比）")

    # 设备与运行配置
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="训练设备，如：cuda:0, cpu")
    parser.add_argument('--note', type=str, default='test',
                        help="保存运行结果的文件夹")

    # 数据加载配置
    parser.add_argument('--maximum_load_time', type=float, default=80,
                        help="最大加载时间（秒）")
    parser.add_argument('--drop_extra_time', type=str_to_bool, default=True,
                        help="是否丢弃超出最大加载时间的数据")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="数据加载的工作进程数")

    # 模型相关
    parser.add_argument('--TAM_type', type=str, default='none',
                        help="提取特征的TAM方法")
    parser.add_argument('--Model_name', type=str, default='none',
                        help="模型名称")

    parser.add_argument('--use_idx', type=str, default='False',
                        help="是否使用idx作为模型输入")

    parser.add_argument('--overlap_ratio', type=float, default=0,
                        help="重叠率")

    # 测试模式相关
    parser.add_argument('--is_Sen', type=str_to_bool, default=False,
                        help="是否为参数敏感性测试模式")
    parser.add_argument("--Sen_num_aug", type=int, default=30)
    parser.add_argument("--Sen_max_matrix_len", type=int, default=1800)






    parser.add_argument('--test_flag', type=str_to_bool, default=True,
                        help="是否打开测试模式")
    # ========== 模式特定参数 ==========
    if mode == "train":
        # 训练模式特有参数
        parser.add_argument('--train_epochs', type=int, default=30,
                            help="训练的总轮次")
        # 优化器相关
        parser.add_argument('--optim', type=str_to_bool, default=False,
                            help="是否使用优化配置参数")
        parser.add_argument('--weight_decay', type=float, default=0.05,
                            help="权重衰减系数，用于正则化")
        parser.add_argument('--min_lr', type=float, default=1e-6,
                            help="学习率的最小值")
        parser.add_argument('--warmup_epochs', type=int, default=5,
                            help="学习率预热轮数")
        parser.add_argument('--stag_epochs', type=int, default=20,
                            help="最大停滞次数")
        parser.add_argument('--valid_name', type=str, default='valid',
                            help="验证数据集名称")
    elif mode == "test":
        parser.add_argument('--is_pr_auc', type=str_to_bool, default=True,
                            help=print_colored("是否计算pr曲线值", "green", is_print=False))
    else:
        raise ValueError(f"不支持的mode参数: {mode}，请使用 'train' 或 'test'")

    return parser

def model_forward(model, method, cur_X, idx=None):
    """统一模型前向传播接口"""
    if method in ['CountMamba', 'ExploreModel_test']:
        return model(cur_X, idx)
    else:
        return model(cur_X)


def prepare_batch_data(cur_data, device):
    """准备批次数据，处理可能的tuple/list结构"""
    if isinstance(cur_data[0], (list, tuple)):
        cur_X, cur_y = cur_data[0][0].to(device), cur_data[1].to(device)
        idx = cur_data[0][1].to(device)
    else:
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        idx = None
    return cur_X, cur_y.long(), idx


def adjust_system_args(args, config):
    """调整系统参数"""
    # 自动设置tab数量和最大加载时长
    args["num_tabs"] = dataset_lib[args['dataset']]['num_tabs']
    args["maximum_load_time"] = dataset_lib[args['dataset']]['maximum_load_time']

    if args['is_Sen']:
        config['max_matrix_len'] = args['Sen_max_matrix_len']
        config['num_aug'] = args['Sen_num_aug']

    # 自动设置文件基础目录
    if args['file_base_dir'] == "auto":
        args['file_base_dir'] = get_filebase_dir()

    # 如果是优化模式，需要调整一些参数
    if args.get('optim', False):
        args['train_epochs'] = 100

    # 如果是测试模式，就调整较小的epoch和数据量
    if args['test_flag']:
        print_colored(">>> [TEST] 模式运行", "yellow")
        args['train_epochs'] = 3
        args['sample_num'] = 800
    else:
        args['sample_num'] = -1

    # 修改点：移除 if args['num_tabs'] > 1 的判断，业务已不存在
    # 如果有特定模型需要特殊配置，建议直接根据 config_name 判断，不再关联 num_tabs
    return args, config


def load_config_file(config_path):
    """加载配置文件"""
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path)
    else:
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    return parse_args(config)[0]


# def get_num_classes(test_y, args):
#     """确定类别数量"""
#     # 修改点：直接返回类别数，移除多标签 shape[1] 的逻辑
#     return len(np.unique(test_y))


def load_dataset_data(args, dataname, shuffle=True):
    """加载训练和验证数据"""
    train_path = os.path.join(args['file_base_dir'], args['dataset'], f"{dataname[0]}.npz")
    valid_path = os.path.join(args['file_base_dir'], args['dataset'], f"{dataname[1]}.npz")

    X1, y1 = load_data(train_path, drop_extra_time=args['drop_extra_time'], load_time=args['maximum_load_time'])
    X2, y2 = load_data(valid_path, drop_extra_time=args['drop_extra_time'], load_time=args['maximum_load_time'])

    if args['sample_num'] > 0:
        if shuffle:
            idx1 = np.random.permutation(range(len(y1)))
            X1, y1 = X1[idx1], y1[idx1]
            idx2 = np.random.permutation(range(len(y2)))
            X2, y2 = X2[idx2], y2[idx2]
        X1, y1 = X1[:args['sample_num']], y1[:args['sample_num']]
        X2, y2 = X2[:args['sample_num']], y2[:args['sample_num']]

    print(f"{dataname[0]} 数据集样本数: {len(y1)}, {dataname[1]} 数据集样本数: {len(y2)}")
    return X1, y1, X2, y2


def evaluate_one_epoch(model, val_loader, device, config, args, num_classes):
    """执行单个验证轮次"""
    model.eval()

    # 修改点：彻底移除 if args['num_tabs'] > 1 及其内部的所有多标签评价逻辑
    # 统一使用单标签分类验证逻辑
    if config['model'] == "TF":
        valid_true, valid_pred = knn_monitor(model, device, val_loader, val_loader, num_classes, 10)
    else:
        valid_pred = []
        valid_true = []

        with torch.no_grad():
            for cur_data in tqdm(val_loader):
                cur_X, cur_y, idx = prepare_batch_data(cur_data, device)
                outs = model_forward(model, config, cur_X, idx)
                # 获取概率最大的类别索引
                cur_pred = torch.argsort(outs, dim=1, descending=True)[:, 0]

                valid_pred.append(cur_pred.cpu().numpy())
                valid_true.append(cur_y.cpu().numpy())

        valid_pred = np.concatenate(valid_pred)
        valid_true = np.concatenate(valid_true)

    # 计算单标签指标 (F1, Precision, Recall 等)
    valid_result = measurement(valid_true, valid_pred, config['eval_metrics'])

    # 兼容前面的代码，返回结果字典和主指标（F1-score）
    return valid_result, valid_result.get("F1-score", 0)
