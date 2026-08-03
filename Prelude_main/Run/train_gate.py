# -*- coding: utf-8 -*-
"""
Phase 2: GateNet 训练脚本
替代原 gate_main.py，为任意 backbone 训练统一的轻量门控网络。
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

from lxj_utils_sys import BaseLogger_v2, ModelCheckpoint, same_seed, IncrementalMeanCalculator
from lxj_utils_sys import print_colored, print_config_info, parse_args
from utils_dataset_metric import get_model_and_dataloader, load_data
from utils_porcess import *
from Prelude_main.Run.const import get_filebase_dir, dataset_lib
from Prelude_main.Model import RandomEarlyTruncationWrapper, RawTrafficDataset, UnifiedGateNet
from torch.utils.data import DataLoader
import argparse

warnings.filterwarnings("ignore")
same_seed(2025)


def balanced_sample(X, y, n_total, seed=2025):
    """
    平衡采样：每类均匀选取样本，总数接近 n_total。
    
    Args:
        X: numpy array, shape (N, ...)
        y: numpy array, shape (N,)
        n_total: int — 目标总样本数
        seed: int — 随机种子，默认 2025（与之前固定种子一致），传入不同种子可实现不同采样
    
    Returns:
        X_sampled, y_sampled
    """
    n_total = min(n_total, len(y))
    classes = np.unique(y)
    n_classes = len(classes)
    n_per_class = n_total // n_classes
    remainder = n_total % n_classes
    
    indices = []
    rng = np.random.RandomState(seed)
    for i, cls in enumerate(classes):
        cls_indices = np.where(y == cls)[0]
        # 前 remainder 个类多分一个
        n_take = n_per_class + (1 if i < remainder else 0)
        n_take = min(n_take, len(cls_indices))
        if n_take <= 0:
            continue
        chosen = rng.choice(cls_indices, size=n_take, replace=False)
        indices.append(chosen)
    
    if len(indices) == 0:
        return X, y
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


class WeightedBCELoss(nn.Module):
    """加权二分类交叉熵损失"""
    def __init__(self, pos_weight=1.0, neg_weight=10.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, gate_prob, target):
        loss_raw = F.binary_cross_entropy(gate_prob, target, reduction='none')
        weights = target * self.pos_weight + (1 - target) * self.neg_weight
        loss = (loss_raw * weights).mean()
        return loss


# ==========================================
# 训练门控网络
# ==========================================
def train_gate_mechanism(backbone, gate_net, dataloader, device, gatenet_checkpoint, logger,
                         threshold=0.5, epochs=20, lr=1e-3, alpha=0.1, beta=1.0,
                         create_loader_fn=None):
    """
    训练 GateNet：backbone 冻结，GateNet 可训练。
    输入数据来自 RandomEarlyTruncationWrapper（Phase 2 模式：2 个 dataset）。
    
    Args:
        create_loader_fn: 若不为 None，每轮开始时调用此函数获取新的 DataLoader，实现每轮重新采样。
    """
    print_colored("\n[Phase 2] Training GateNet...", "blue")

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    gate_net.train()
    criterion = WeightedBCELoss(pos_weight=alpha, neg_weight=beta)
    optimizer = optim.AdamW(gate_net.parameters(), lr=lr)

    for epoch in range(epochs):
        # 每轮开始：若提供了 create_loader_fn，则重新创建 DataLoader（实现每轮不同样本）
        if create_loader_fn is not None:
            dataloader = create_loader_fn(epoch)
            actual_samples = len(dataloader.dataset)
            print_colored(f"  每轮重新采样: 已创建第 {epoch + 1} 轮的新 DataLoader"
                          f" (实际样本数: {actual_samples})", "cyan")
        tracker_gate_loss = IncrementalMeanCalculator()
        tracker_gate_acc = IncrementalMeanCalculator()
        tracker_model_acc = IncrementalMeanCalculator()
        tracker_pass_ratio = IncrementalMeanCalculator()
        tracker_load_ratio = IncrementalMeanCalculator()
        calc_accept_ratio = IncrementalMeanCalculator()
        calc_reject_ratio = IncrementalMeanCalculator()

        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            # Phase 2: batch_data = (x1, y1, x2, y2, ratios)
            # x1: backbone input (B, C, H, W) — k 维已在 wrapper 中展平
            # x2: raw traffic for GateNet (B, 2, L)
            # y: labels (B,)
            # ratios: 截断比例 (B,)
            x1, y1, x2, y2, ratios = batch_data

            # 数据已是扁平格式，无需 Rearrange 展平
            aug_x1 = x1.to(device)
            aug_x2 = x2.to(device)
            aug_labels = y1.to(device)

            # Backbone 推理（冻结）
            with torch.no_grad():
                logits_m = backbone(aug_x1)
                preds_m = torch.argmax(logits_m, dim=1)
                backbone_is_correct = (preds_m == aug_labels).float().view(-1, 1)

            # GateNet 前向传播
            gate_prob = gate_net(aug_x2)

            # Loss 计算
            loss = criterion(gate_prob, backbone_is_correct)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            gate_decision = (gate_prob >= threshold).float()
            gate_prob_np = torch.squeeze(gate_prob).cpu().detach().numpy()
            ratios_np = ratios.cpu().numpy() if isinstance(ratios, torch.Tensor) else np.array(ratios)
            # 处理 gate_prob 为标量时 squeeze 后为 0-d 数组的情况
            if gate_prob_np.ndim == 0:
                gate_prob_np = np.array([gate_prob_np])
            accept_ratios = ratios_np[gate_prob_np >= threshold]
            reject_ratios = ratios_np[gate_prob_np < threshold]

            tracker_gate_loss.add(loss.item())
            tracker_gate_acc.add((gate_decision == backbone_is_correct).float().tolist())
            tracker_model_acc.add(backbone_is_correct.tolist())
            tracker_pass_ratio.add((gate_decision == 1).float().tolist())
            tracker_load_ratio.add(ratios_np)
            calc_accept_ratio.add(accept_ratios)
            calc_reject_ratio.add(reject_ratios)

        print(f"Epoch [{epoch + 1}/{epochs}]"
              f"\n - Loss: {tracker_gate_loss.get():.4f} "
              f"\n - Gate Acc: {tracker_gate_acc.get() * 100:.2f}% "
              f"\n - Model Acc: {tracker_model_acc.get() * 100:.2f}% "
              f"\n - Pass Ratio: {tracker_pass_ratio.get() * 100:.2f}% "
              f"\n - Load Ratio: {tracker_load_ratio.get() * 100:.2f}% "
              f"\n - Accept Ratio: {calc_accept_ratio.get() * 100:.2f}% "
              f"\n - Reject Ratio: {calc_reject_ratio.get() * 100:.2f}%")
        
        logger.record("train", {
            "epoch": epoch + 1,
            "loss": tracker_gate_loss.get(),
            "gate_acc": tracker_gate_acc.get() * 100,
            "model_acc": tracker_model_acc.get() * 100,
            "pass_ratio": tracker_pass_ratio.get() * 100,
            "load_ratio": tracker_load_ratio.get(2),
            "accept_ratio": calc_accept_ratio.get(2),
            "reject_ratio": calc_reject_ratio.get(2)
        }, unpack_dict=True)
        
        gatenet_checkpoint.save(tracker_gate_acc.get() * 100, gate_net, epoch + 1)

    print_colored("[Info] GateNet training completed.", "green")


def main():
    parser = get_gate_parser()
    run_args, args_help = parse_args(parser, is_print_help=False)

    if run_args['test_flag']:
        print_colored(">>> [TEST] 模式运行", "yellow")
        run_args['train_epochs'] = 3
        run_args['train_sample_num'] = 200

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. 加载 backbone 配置 ---
    if run_args['file_base_dir'] == "auto":
        run_args['file_base_dir'] = get_filebase_dir()

    checkpoint_dir = os.path.join(run_args['checkpoint_path'], run_args['dataset'],
                                  run_args['backbone_model'], run_args['backbone_note']).rstrip('/')
    result_json_path = os.path.join(checkpoint_dir, "result.json")
    if not os.path.exists(result_json_path):
        raise FileNotFoundError(
            f"Backbone 训练结果文件不存在: {result_json_path}\n"
            f"请先运行 Phase 1 (train_backbone.py) 训练 {run_args['backbone_model']} 模型。"
        )
    backbone_logger = BaseLogger_v2(json_path=result_json_path)
    backbone_logger.load()
    if 'config' not in backbone_logger.data or 'args' not in backbone_logger.data.get('config', {}):
        raise KeyError(
            f"Backbone 结果文件缺少 config 配置信息: {result_json_path}\n"
            f"文件可能由旧版本代码生成，请重新运行 Phase 1 (train_backbone.py)。\n"
            f"当前 data keys: {list(backbone_logger.data.keys())}"
        )
    exp_config = backbone_logger.data['config']['args']
    exp_config['file_base_dir'] = run_args['file_base_dir']
    exp_config['num_workers'] = run_args['num_workers']
    model_config = backbone_logger.data['config']['config']
    
    num_classes = dataset_lib[run_args['dataset']]['num_classes']
    print_colored("加载backbone模型配置成功", "green")

    # --- 2. 路径与日志 ---
    ckp_gate_path = os.path.join(str(checkpoint_dir), "gatenet", run_args['gatenet_note'])
    if not os.path.exists(ckp_gate_path):
        os.makedirs(ckp_gate_path)

    train_logger = BaseLogger_v2(
        json_path=os.path.join(ckp_gate_path, "gating_train_result.json"),
        log_path=os.path.join(ckp_gate_path, "train_log.txt"))

    # --- 3. 数据加载 ---
    train_X_full, train_y_full = None, None  # 仅 resample_each_epoch 时使用
    if run_args['resample_each_epoch']:
        # 每轮重新采样模式：先加载全量数据，保留完整副本供每轮重新采样
        exp_config_save = exp_config.get('sample_num', -1)
        exp_config['sample_num'] = -1
        train_X_full, train_y_full, valid_X, valid_y = load_dataset_data(exp_config, ['train', 'valid'])
        exp_config['sample_num'] = exp_config_save
        # 首次采样，用于创建初始 DataLoader
        if run_args['train_sample_num'] > 0:
            if run_args['sample_mode'] == 'balanced':
                train_X, train_y = balanced_sample(train_X_full, train_y_full, run_args['train_sample_num'])
                print_colored(f"首次平衡采样: 目标 {run_args['train_sample_num']} 样本, "
                              f"实际 {len(train_y)} 样本, {len(np.unique(train_y))} 类", "green")
            else:
                idx = np.random.choice(len(train_y_full), run_args['train_sample_num'], replace=False)
                train_X, train_y = train_X_full[idx], train_y_full[idx]
        else:
            train_X, train_y = train_X_full, train_y_full
    elif run_args['train_sample_num'] > 0:
        if run_args['sample_mode'] == 'balanced':
            # 平衡采样：先加载全部数据，再按类别均匀选取
            exp_config['sample_num'] = -1
            train_X, train_y, valid_X, valid_y = load_dataset_data(exp_config, ['train', 'valid'])
            train_X, train_y = balanced_sample(train_X, train_y, run_args['train_sample_num'])
            print_colored(f"平衡采样: 目标 {run_args['train_sample_num']} 样本, "
                          f"实际 {len(train_y)} 样本, {len(np.unique(train_y))} 类", "green")
        else:
            # 随机采样：通过 load_dataset_data 内部 shuffle + 截断
            exp_config['sample_num'] = run_args['train_sample_num']
            train_X, train_y, valid_X, valid_y = load_dataset_data(exp_config, ['train', 'valid'])
    else:
        train_X, train_y, valid_X, valid_y = load_dataset_data(exp_config, ['train', 'valid'])

    # 强制关闭增强包装（Phase 2 自己包 RandomEarlyTruncationWrapper）
    exp_config['early_augment'] = False
    exp_config['early_stage'] = False

    # 复用 get_model_and_dataloader：自动匹配模型类型创建正确的 dataset + backbone 模型
    model_backbone, train_loader_ref, _ = get_model_and_dataloader(
        train_X, train_y, valid_X, valid_y, num_classes, model_config, exp_config)

    # 从 loader 中取出 backbone_dataset
    backbone_dataset = train_loader_ref.dataset

    # 创建 gate dataset（原始流量数据）
    gate_dataset = RawTrafficDataset(train_X, train_y)

    # 使用 RandomEarlyTruncationWrapper 包装（Phase 2 模式：2 datasets，共享截断点）
    train_dataset = RandomEarlyTruncationWrapper(
        [backbone_dataset, gate_dataset], train_X, train_y,
        lb=run_args['lb'], ub=run_args['ub'], aug_num=run_args['aug_num'],
        info="gate 的 train[backbone gatenet] 数据集", return_ratio=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=run_args['batch_size'],
                              shuffle=True,
                              num_workers=run_args['num_workers'],
                              drop_last=True)
    print_colored("训练数据集加载完毕", "green")

    # --- 4. 加载预训练 Backbone ---
    print_colored(f"Loading Pre-trained Backbone: {checkpoint_dir}", 'yellow')

    backbone_checkpoint = ModelCheckpoint(filename=os.path.join(checkpoint_dir, f"model.pth"),
                                          mode='max', metric_name="f1")
    model_backbone = backbone_checkpoint.load(model_backbone, device)[0]

    # --- 5. 初始化 GateNet ---
    gate_net = UnifiedGateNet(seq_len=model_config['seq_len']).to(device)
    gatenet_checkpoint = ModelCheckpoint(filename=os.path.join(ckp_gate_path, f"gatenet_model.pth"),
                                         mode='max', metric_name='gate_acc')

    print_colored(f"Current Execution Mode: [TRAIN]", "cyan")

    # --- 6. 准备训练（每轮重新采样模式需构造 create_loader_fn） ---
    create_loader_fn = None
    if run_args['resample_each_epoch'] and train_X_full is not None:
        def _create_loader(epoch):
            """每轮重新采样并重建 DataLoader"""
            nonlocal train_X_full, train_y_full
            # 重新采样（使用 epoch 相关的 seed，每轮得到不同的子集）
            if run_args['sample_mode'] == 'balanced':
                epoch_X, epoch_y = balanced_sample(train_X_full, train_y_full,
                                                   run_args['train_sample_num'],
                                                   seed=2025 + epoch)
            else:
                rng = np.random.RandomState(2025 + epoch)
                idx = rng.choice(len(train_y_full), run_args['train_sample_num'], replace=False)
                epoch_X, epoch_y = train_X_full[idx], train_y_full[idx]

            # 重建 backbone dataset
            _, loader_ref, _ = get_model_and_dataloader(
                epoch_X, epoch_y, valid_X, valid_y, num_classes, model_config, exp_config)
            bb_dataset = loader_ref.dataset

            # 重建 gate dataset + wrapper + loader
            gate_dataset = RawTrafficDataset(epoch_X, epoch_y)
            train_dataset = RandomEarlyTruncationWrapper(
                [bb_dataset, gate_dataset], epoch_X, epoch_y,
                lb=run_args['lb'], ub=run_args['ub'], aug_num=run_args['aug_num'],
                info="gate 的 train[backbone gatenet] 数据集", return_ratio=True)
            return DataLoader(train_dataset,
                              batch_size=run_args['batch_size'],
                              shuffle=True,
                              num_workers=run_args['num_workers'],
                              drop_last=True)

        create_loader_fn = _create_loader

    # --- 7. 训练 GateNet ---
    train_logger.record("config.run_config", run_args)
    train_logger.start_timer("训练")
    train_gate_mechanism(model_backbone, gate_net, train_loader, device, gatenet_checkpoint,
                         threshold=run_args['threshold'], epochs=run_args['train_epochs'],
                         logger=train_logger, alpha=run_args['alpha'], beta=run_args['beta'],
                         create_loader_fn=create_loader_fn)
    train_logger.stop_timer("训练")
    print_colored(">>> [TRAIN] 模式运行结束", "green")


def get_gate_parser():
    parser = argparse.ArgumentParser(description="GateNet 训练流程")
    parser.add_argument('--machine_name', type=str, default=get_machine_name(), help='Machine name')
    parser.add_argument('--checkpoint_path', default="../../checkpoints", help="运行结果存储路径")
    parser.add_argument('--file_base_dir', default="auto", help="数据集存储路径")
    parser.add_argument('--dataset', type=str, default='CW', help='Dataset for GateNet')
    parser.add_argument('--gatenet_note', type=str, default='default', help='Notes for GateNet')
    parser.add_argument('--backbone_note', type=str, default='default', help='Notes for backbone network')
    parser.add_argument('--backbone_model', type=str, default='Prelude', help='Backbone model name (default: Prelude)')

    parser.add_argument('--train_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')

    parser.add_argument('--lb', type=float, default=0.0, help='Lower bound (lb)')
    parser.add_argument('--ub', type=float, default=1.0, help='Upper bound (ub)')
    parser.add_argument('--aug_num', type=int, default=20, help='Number of augmentation versions')

    parser.add_argument('--threshold', type=float, default=0.5, help='Gating threshold for Training')
    parser.add_argument('--train_sample_num', type=int, default=-1, help='Number of samples for training')
    parser.add_argument('--sample_mode', type=str, default='random', choices=['random', 'balanced'],
                        help='采样模式: random=随机采样, balanced=每类均匀采样')
    parser.add_argument('--resample_each_epoch', type=str_to_bool, default=False,
                        help='每轮重新采样: True=每轮从全量数据中重新采 train_sample_num 个样本, False=固定使用同一批样本')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--alpha', type=float, default=0.1, help='Pos weight for BCE')
    parser.add_argument('--beta', type=float, default=1, help='Neg weight for BCE')
    parser.add_argument('--TAM_type', type=str, default='ED1', help="提取特征的TAM方法")
    parser.add_argument('--test_flag', type=str_to_bool, default=True, help="是否打开测试模式")

    return parser


if __name__ == "__main__":
    main()
