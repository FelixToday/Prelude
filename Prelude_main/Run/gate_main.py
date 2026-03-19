# -*- coding: utf-8 -*-

# @Author: Xianjun Li
# @Description: Streaming Traffic Fingerprinting with Dynamic Gating Mechanism
# @Strategy: Frozen Backbone M + Trainable Auxiliary Gating Network A
import multiprocessing as mp
import os
import warnings
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops.layers.torch import Rearrange
from tqdm import tqdm
import torch.nn.functional as F
from tabulate import tabulate
# 导入工具库
from lxj_utils_sys import BaseLogger, ModelCheckpoint, same_seed, IncrementalMeanCalculator, BaseLogger_v2
from lxj_utils_sys import print_colored, print_config_info, parse_args
from utils_dataset_metric import get_model_and_dataloader
from utils_porcess import *
# 导入模型和数据处理
from Prelude_main.Run.const import get_machine_name
from Prelude_main.Model import GateNet, CountDataset_RandomEarly
from torch.utils.data import DataLoader
import argparse

warnings.filterwarnings("ignore")
same_seed(2025)


class WeightedBCELoss(nn.Module):
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
# 1. 辅助函数
# ==========================================
def fetch_sample(args):
    dataset, idx, lb, ub = args
    load_ratio_value = (lb + (ub - lb) * np.random.rand()) * 100
    data, label = dataset.__getitem__(idx, load_ratio=load_ratio_value)
    return torch.tensor(data[0]), torch.tensor(label)


# ==========================================
# 2. 训练门控网络的函数 (Train Phase)
# ==========================================
def train_gate_mechanism(backbone, gate_net, dataloader, device, gatenet_checkpoint, logger, threshold=0.5, epochs=20,
                         lr=1e-3, alpha=0.1, beta=1.0):
    print_colored("\n[Phase 1] Training Auxiliary Gating Network A...", "blue")

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    gate_net.train()
    criterion = WeightedBCELoss(pos_weight=alpha, neg_weight=beta)
    optimizer = optim.AdamW(gate_net.parameters(), lr=lr)

    for epoch in range(epochs):
        tracker_load_ratio = IncrementalMeanCalculator()
        tracker_gate_acc = IncrementalMeanCalculator()
        tracker_gate_loss = IncrementalMeanCalculator()
        tracker_model_acc = IncrementalMeanCalculator()
        tracker_pass_ratio = IncrementalMeanCalculator()
        calc_accept_ratio = IncrementalMeanCalculator()
        calc_reject_ratio = IncrementalMeanCalculator()
        for data, labels, ratios in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            aug_data = Rearrange('b s c d t -> (b s) c d t')(data).to(device)
            aug_labels = Rearrange('b s -> (b s)')(labels).to(device)
            ratios = Rearrange('b s -> (b s)')(ratios).numpy()

            with torch.no_grad():
                logits_m = backbone(aug_data)
                preds_m = torch.argmax(logits_m, dim=1)
                backbone_is_correct = (preds_m == aug_labels).float().view(-1, 1)

            optimizer.zero_grad()
            gate_prob = gate_net(aug_data)

            loss = criterion(gate_prob, backbone_is_correct)
            loss.backward()
            optimizer.step()

            gate_decision = (gate_prob >= threshold).float()
            gate_prob_np = torch.squeeze(gate_prob).cpu().detach().numpy()
            accept_ratios = ratios[gate_prob_np >= threshold]
            reject_ratios = ratios[gate_prob_np < threshold]

            tracker_gate_loss.add(loss.item())
            tracker_gate_acc.add((gate_decision == backbone_is_correct).float().tolist())
            tracker_model_acc.add(backbone_is_correct.tolist())
            tracker_pass_ratio.add((gate_decision == 1).float().tolist())
            tracker_load_ratio.add(ratios)
            calc_accept_ratio.add(accept_ratios)
            calc_reject_ratio.add(reject_ratios)

        print(f"Epoch [{epoch + 1}/{epochs}]"
              f"\n - Loss: {tracker_gate_loss.get():.4f} "
              f"\n - Gate Accuracy: {tracker_gate_acc.get() * 100:.2f} % "
              f"\n - Model Accuracy: {tracker_model_acc.get() * 100:.2f} % "
              f"\n - 通过率: {tracker_pass_ratio.get() * 100:.2f} % "
              f"\n - 加载比例：{tracker_load_ratio.get():.2f} %"
              f"\n - Accept Ratio: {calc_accept_ratio.get():.2f} %"
              f"\n - Reject Ratio: {calc_reject_ratio.get():.2f} %")
        logger.record("train", {"epoch": epoch + 1,
                                "loss": tracker_gate_loss.get(),
                                "gate_acc": tracker_gate_acc.get() * 100,
                                "model_acc": tracker_model_acc.get() * 100,
                                "pass_ratio": tracker_pass_ratio.get() * 100,
                                "load_ratio": tracker_load_ratio.get(2),
                                "accept_ratio": calc_accept_ratio.get(2),
                                "reject_ratio": calc_reject_ratio.get(2)
                                }, unpack_dict=True)
        gatenet_checkpoint.save(tracker_gate_acc.get() * 100, gate_net, epoch + 1)

    print_colored("[Info] Gating Network training completed.", "green")


# ==========================================
# 3. 评估函数 (Eval Phase)
# ==========================================
def evaluate_gatenet(model_gateNet, model_backbone, dataloader, logger, device='cuda',
                     threshold=0.5):
    model_gateNet.eval()
    model_backbone.eval()

    dataset = dataloader.dataset

    tracker_gate_acc = IncrementalMeanCalculator()
    tracker_model_acc = IncrementalMeanCalculator()
    tracker_pass_ratio = IncrementalMeanCalculator()
    cal_load_ratio = IncrementalMeanCalculator()
    calc_accept_ratio = IncrementalMeanCalculator()
    calc_reject_ratio = IncrementalMeanCalculator()
    print(f"Start Evaluation on {len(dataset)} traces...")

    with torch.no_grad():
        for data, labels, ratios in tqdm(dataloader, desc="Evaluating"):
            aug_data = Rearrange('b s c d t -> (b s) c d t')(data).to(device)
            aug_labels = Rearrange('b s -> (b s)')(labels).to(device)
            ratios = Rearrange('b s -> (b s)')(ratios).numpy()
            logits_m = model_backbone(aug_data)
            preds_m = torch.argmax(logits_m, dim=1)

            backbone_is_correct = (preds_m == aug_labels).float().view(-1, 1)

            gate_prob = model_gateNet(aug_data)

            tracker_gate_acc.add((((gate_prob >= threshold).float()) == backbone_is_correct).float().tolist())
            tracker_model_acc.add((preds_m == aug_labels).float().tolist())
            tracker_pass_ratio.add((((gate_prob >= threshold).float()) == 1).float().tolist())
            cal_load_ratio.add(ratios)

            gate_prob_np = torch.squeeze(gate_prob).cpu().detach().numpy()
            accept_ratios = ratios[gate_prob_np >= threshold]
            reject_ratios = ratios[gate_prob_np < threshold]
            calc_accept_ratio.add(accept_ratios)
            calc_reject_ratio.add(reject_ratios)

    final_gate_acc = tracker_gate_acc.get()
    final_model_acc = tracker_model_acc.get()
    final_pass_ratio = tracker_pass_ratio.get()

    print(f"\nEvaluation Result:"
          f"\n - Gate Accuracy:  {final_gate_acc * 100:.2f} %"
          f"\n - Model Accuracy: {final_model_acc * 100:.2f} %"
          f"\n - 通过率:      {final_pass_ratio * 100:.2f} %"
          f"\n - 加载比例: {cal_load_ratio.get():.2f} %"
          f"\n - Accept Ratio: {calc_accept_ratio.get():.2f} %"
          f"\n - Reject Ratio: {calc_reject_ratio.get():.2f} %")
    logger.record("eval", {"gate_acc": final_gate_acc * 100,
                           "model_acc": final_model_acc * 100,
                           "pass_ratio": final_pass_ratio * 100,
                           "load ratio": cal_load_ratio.get(2),
                           "accept_ratio": calc_accept_ratio.get(2),
                           "reject_ratio": calc_reject_ratio.get(2)
                           }, unpack_dict=True)
    return final_gate_acc, final_model_acc


# ==========================================
# 4. 真实流模拟评估 (Real Phase)
# ==========================================
def evaluate_real_stream(model_gateNet, model_backbone, dataloader, logger,
                         delta=0.001, checkvalues=[0.5], device='cuda', mode='ratio'):
    model_gateNet.eval()
    model_backbone.eval()
    dataset = dataloader.dataset

    metrics = {
        "load": {th: IncrementalMeanCalculator() for th in checkvalues},
        "model_acc": {th: IncrementalMeanCalculator() for th in checkvalues},
        "gate_acc": {th: IncrementalMeanCalculator() for th in checkvalues},
        "trigger_ratio": {th: IncrementalMeanCalculator() for th in checkvalues},
        "load_latency": {th: IncrementalMeanCalculator() for th in checkvalues}
    }
    time_metrics = {
        "time_construct": IncrementalMeanCalculator(),
        "time_backbone": IncrementalMeanCalculator(),
        "time_gate": IncrementalMeanCalculator()
    }

    print(f"Start Real-stream Evaluation ({mode.upper()} mode, thresholds={checkvalues})...")
    pbar = tqdm(range(len(dataset)), desc=f"Eval {mode}")

    W_CONST = 80 / 1800
    WIN_STEP = 6

    with torch.no_grad():
        for idx in pbar:
            current_ratio = delta if mode == 'ratio' else 0
            win_index = 0
            recorded_thresholds = set()
            trigger_info = {}
            max_trace_time = float(np.max(dataset.X[idx][:, 0]))

            while current_ratio <= 1.0 + 1e-5:
                if mode == 'window':
                    win_index += WIN_STEP
                    current_ratio = (win_index * W_CONST) / max_trace_time

                start_time = time.time()
                data, labels = dataset.__getitem__(idx, min(current_ratio * 100, 100.0))
                input_tensor = torch.tensor(data[0]).unsqueeze(0).to(device)
                label_tensor = torch.tensor(labels).unsqueeze(0).to(device)
                time_metrics["time_construct"].add(time.time() - start_time)

                start_time = time.time()
                gate_prob = model_gateNet(input_tensor)
                time_metrics["time_gate"].add(time.time() - start_time)

                for th in checkvalues:
                    if th not in recorded_thresholds:
                        metrics["trigger_ratio"][th].add(1.0 if gate_prob.item() >= th else 0.0)
                        if gate_prob.item() >= th:
                            recorded_thresholds.add(th)
                            trigger_info[th] = {
                                "data": input_tensor,
                                "labels": label_tensor,
                                "ratio": current_ratio,
                                "latency": max_trace_time * current_ratio,
                                "gate_prob": gate_prob.item(),
                            }

                if len(recorded_thresholds) == len(checkvalues):
                    break

                if mode == 'ratio':
                    current_ratio += delta

                if current_ratio >= 1.0 and len(recorded_thresholds) < len(checkvalues):
                    current_ratio = 1.0
                    data, labels = dataset.__getitem__(idx, 100.0)
                    input_tensor = torch.tensor(data[0]).unsqueeze(0).to(device)
                    label_tensor = torch.tensor(labels).unsqueeze(0).to(device)
                    gate_prob = model_gateNet(input_tensor)

                    for th in checkvalues:
                        if th not in recorded_thresholds:
                            metrics["trigger_ratio"][th].add(1.0 if gate_prob.item() >= th else 0.0)
                            recorded_thresholds.add(th)
                            trigger_info[th] = {
                                "data": input_tensor,
                                "labels": label_tensor,
                                "ratio": current_ratio,
                                "latency": max_trace_time * current_ratio,
                                "gate_prob": gate_prob.item(),
                            }

                    break

            for th in checkvalues:
                info = trigger_info[th]
                final_data = info["data"]
                final_labels = info["labels"]

                start_time = time.time()
                logits_m = model_backbone(final_data)
                time_metrics["time_backbone"].add(time.time() - start_time)

                preds_m = torch.argmax(logits_m, dim=1)
                is_backbone_correct = (preds_m == final_labels).float().item()

                metrics["model_acc"][th].add(is_backbone_correct)
                metrics["load"][th].add(info["ratio"])
                metrics["load_latency"][th].add(info["latency"])

                gate_pred_class = (info["gate_prob"] >= th)
                is_gate_correct = float(gate_pred_class == is_backbone_correct)
                metrics["gate_acc"][th].add(is_gate_correct)

            one_th = checkvalues[0]
            pbar.set_postfix({
                f'Load({one_th})': f"{metrics['load'][one_th].get() * 100:.1f}%",
                f'Pass({one_th})': f"{metrics['trigger_ratio'][one_th].get() * 100:.1f}%",
                f'M_Acc({one_th})': f"{metrics['model_acc'][one_th].get() * 100:.1f}%",
                f'G_Acc({one_th})': f"{metrics['gate_acc'][one_th].get() * 100:.1f}%",
            })

    logger.print(f"\n======== 真实流模拟评估 (Real-Stream) Mode: {mode.upper()} ========", save_to_file=True)
    table_data = []
    for th in checkvalues:
        avg_load = metrics["load"][th].get() * 100
        avg_m_acc = metrics["model_acc"][th].get() * 100
        avg_g_acc = metrics["gate_acc"][th].get() * 100
        trigger_rate = metrics["trigger_ratio"][th].get() * 100
        avg_latency = metrics["load_latency"][th].get() * 1000

        table_data.append([
            f"{th:.2f}",
            f"{avg_load:.2f}%",
            f"{avg_m_acc:.2f}%",
            f"{avg_g_acc:.2f}%",
            f"{trigger_rate:.2f}%",
            f"{avg_latency:.2f} ms"
        ])

        logger.record(f"result", {
            "load": avg_load,
            "model_acc": avg_m_acc,
            "gate_acc": avg_g_acc,
            "pass_ratio": trigger_rate,
            "time_construct": time_metrics["time_construct"].get(),
            "time_backbone_forward": time_metrics["time_backbone"].get(),
            "time_gatenet_forward": time_metrics["time_gate"].get(),
            "avg_load_latency": avg_latency
        }, unpack_dict=True)

    headers = ["Threshold", "Avg Load Ratio", "Model Acc", "Gate Acc", "Pass Ratio", "Avg Latency"]
    logger.print(tabulate(table_data, headers=headers, tablefmt="simple"),save_to_file=True)
    logger.print("====================================================\n",save_to_file=True)

    return metrics["load"][checkvalues[0]].get(), metrics["model_acc"][checkvalues[0]].get(), metrics["gate_acc"][
        checkvalues[0]].get()


def main(run_args, current_mode="eval"):
    # -----------------------------------------------------------
    # [CONTROL] 运行模式选择
    # -----------------------------------------------------------

    gatenet_note = run_args['gatenet_note']
    backbone_note = run_args['backbone_note']
    train_epochs = run_args['train_epochs']
    num_classes = dataset_lib[run_args['dataset']]['num_classes']
    lb, ub = run_args['lb'], run_args['ub']
    aug_num = run_args['aug_num']
    delta = run_args['delta']
    batch_size = run_args['batch_size']
    num_workers = run_args['num_workers']
    GATE_THRESHOLD = run_args['threshold']

    checkconf_list = sorted([float(x) for x in run_args['checkconf_list'].split(',')])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = os.path.join(run_args['checkpoint_path'], run_args['dataset'], "Prelude", backbone_note).rstrip('/')
    backbone_logger = BaseLogger_v2(json_path=os.path.join(checkpoint_dir, "result.json"))
    backbone_logger.load()
    exp_config = backbone_logger.data['config']['args']
    exp_config['file_base_dir'] = get_filebase_dir()

    model_config = backbone_logger.data['config']['config']
    print_colored("加载backbone模型配置成功", "green")

    # --- 2. 路径与日志设置 ---
    ckp_gate_path = os.path.join(str(checkpoint_dir), "gatenet", gatenet_note)
    if not os.path.exists(ckp_gate_path): os.makedirs(ckp_gate_path)

    train_logger = BaseLogger_v2(json_path=os.path.join(ckp_gate_path, "test_gating", "gating_train_result.json"))
    eval_logger = BaseLogger_v2(json_path=os.path.join(ckp_gate_path, "test_gating", "gating_eval_result.json"))
    real_logger = BaseLogger_v2(json_path=os.path.join(ckp_gate_path, "test_gating", "gating_real_result.json"),
                                log_path=os.path.join(ckp_gate_path, "test_gating", "real_log.txt"))

    # --- 3. 数据加载 ---
    # 动态匹配对应模式的样本数量
    if current_mode == "train":
        exp_config['sample_num'] = run_args['train_sample_num']
    elif current_mode == "eval":
        exp_config['sample_num'] = run_args['valid_sample_num']
    elif current_mode == "real":
        exp_config['sample_num'] = run_args['test_sample_num']

    if current_mode in ["train", "all"]:
        train_X, train_y, valid_X, valid_y = load_dataset_data(exp_config, ['train', 'valid'])
        train_dataset = CountDataset_RandomEarly(train_X, train_y, loaded_ratio=100, TAM_type=run_args['TAM_type'],
                                                 is_idx=True, maximum_load_time=80, drop_extra_time=True,
                                                 max_matrix_len=run_args['max_matrix_len'], seq_len=5000, lb=lb, ub=ub, aug_num=aug_num)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
        print_colored("训练/验证数据集加载完毕", "green")

    if current_mode in ["eval", "real", "all"]:
        valid_X, valid_y, test_X, test_y = load_dataset_data(exp_config, ['valid', 'test'])
        model_backbone, _, test_loader = get_model_and_dataloader(test_X, test_y, test_X, test_y,
                                                                  num_classes, model_config, exp_config)
        valid_dataset = CountDataset_RandomEarly(valid_X, valid_y, loaded_ratio=100, TAM_type="ED1",
                                                 is_idx=True, maximum_load_time=80, drop_extra_time=True,
                                                 max_matrix_len=run_args['max_matrix_len'], seq_len=5000, lb=lb, ub=ub, aug_num=aug_num)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
        print_colored("验证/真实流模拟测试集加载完毕", "green")
    else:
        import numpy as np
        dummy_x, dummy_y = np.zeros((1, 1, 5000)), np.zeros((1,))
        model_backbone, _, _ = get_model_and_dataloader(dummy_x, dummy_y, dummy_x, dummy_y,
                                                        num_classes, model_config, exp_config)

    # --- 4. 加载预训练 Backbone M ---
    print_colored(f"Loading Pre-trained Backbone: {checkpoint_dir}", 'yellow')
    backbone_checkpoint = ModelCheckpoint(filename=os.path.join(checkpoint_dir, f"model.pth"),
                                          mode='max',
                                          metric_name="f1" if exp_config['num_tabs'] == 1 else "map")
    model_backbone = backbone_checkpoint.load(model_backbone, device)[0]

    # --- 5. 初始化 GateNet 与 Checkpoint Manager ---
    gate_net = GateNet().to(device)
    gatenet_checkpoint = ModelCheckpoint(filename=os.path.join(ckp_gate_path, f"gatenet_model.pth"),
                                         mode='max',
                                         metric_name='gate_acc')

    print_colored(f"Current Execution Mode: [{current_mode.upper()}]", "cyan")

    if current_mode in ["train", "all"]:
        train_logger.record("config.run_config", run_args)
        train_logger.start_timer("训练")
        train_gate_mechanism(model_backbone, gate_net, train_loader, device, gatenet_checkpoint,
                             threshold=GATE_THRESHOLD, epochs=train_epochs, logger=train_logger,
                             alpha=run_args['alpha'], beta=run_args['beta'])
        train_logger.stop_timer("训练")
        print_colored(">>> [TRAIN] 模式运行结束", "green")

    if current_mode in ["eval", "real", "all"]:
        print_colored("Loading Best GateNet for Evaluation...", "yellow")
        try:
            gate_net = gatenet_checkpoint.load(gate_net, device)[0]
        except Exception as e:
            print_colored(f"Error loading checkpoint: {e}. Ensure you have trained the model first.", "red")
            return

    if current_mode in ["eval", "all"]:
        print_colored("\n[Phase 2] Standard Evaluation...", "blue")
        eval_logger.record("config.run_config", run_args)
        eval_logger.start_timer("验证")
        evaluate_gatenet(gate_net, model_backbone, valid_loader, logger=eval_logger, device='cuda',
                         threshold=GATE_THRESHOLD)
        eval_logger.stop_timer("验证")
        print_colored(">>> [EVAL] 模式运行结束", "green")

    if current_mode in ["real", "all"]:
        print_colored("\n[Phase 3] Real-Stream Simulation...", "blue")
        real_logger.record("config.run_config", run_args)
        real_logger.start_timer("真实流")
        assert run_args['load_mode'] in ['ratio', 'window'], '加载模式错误'

        evaluate_real_stream(gate_net, model_backbone, test_loader, logger=real_logger, delta=delta,
                             checkvalues=checkconf_list,
                             device=device, mode=run_args["load_mode"])

        real_logger.stop_timer("真实流")
        print_colored(">>> [REAL] 模式运行结束", "green")


def get_args():
    parser = argparse.ArgumentParser(
        description="门控网络train valid test流程")
    parser.add_argument('--machine_name', type=str, default=get_machine_name(), help='Machine name')

    parser.add_argument('--checkpoint_path', default="../../checkpoints",help="运行结果存储路径")
    parser.add_argument('--dataset', type=str, default='OW', help='Dataset for GateNet')
    parser.add_argument('--gatenet_note', type=str, default='baseline_same', help='Notes for GateNet')
    parser.add_argument('--backbone_note', type=str, default='baseline_same', help='Notes for backbone network')

    parser.add_argument('--train_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training/testing')

    parser.add_argument('--lb', type=float, default=0.0, help='Lower bound (lb)')
    parser.add_argument('--ub', type=float, default=1.0, help='Upper bound (ub)')
    parser.add_argument('--aug_num', type=int, default=20, help='Number of augmentation versions')

    parser.add_argument('--threshold', type=float, default=0.5, help='Gating threshold for Training')
    parser.add_argument('--checkconf_list', type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0",
                        help='List of gating thresholds to evaluate in real stream')
    parser.add_argument('--delta', type=float, default=0.03, help='Delta parameter for Poirot')

    # 将原有的 sample_num 拆分为三个，并添加 mode 字段
    parser.add_argument('--train_sample_num', type=int, default=-1, help='Number of samples to use for training')
    parser.add_argument('--valid_sample_num', type=int, default=-1, help='Number of samples to use for validation')
    parser.add_argument('--test_sample_num', type=int, default=-1, help='Number of samples to use for testing')
    parser.add_argument('--mode', type=str, default='train valid test', help='Execution modes, separated by space')

    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')

    parser.add_argument('--load_mode', type=str, default='ratio', help='Mode: ratio/window')

    parser.add_argument('--max_matrix_len', type=int, default=1800, help='特征矩阵维度')
    parser.add_argument('--alpha', type=float, default=0.1, help='第1项权重')
    parser.add_argument('--beta', type=float, default=1, help='第2项权重')

    parser.add_argument('--TAM_type', type=str, default='ED1', help="提取特征的TAM方法")
    parser.add_argument('--test_flag', type=str_to_bool, default=True, help="是否打开测试模式")

    return parser


if __name__ == "__main__":
    parser = get_args()
    run_args, args_help = parse_args(parser, is_print_help=False)
    if run_args['test_flag']:
        print_colored(">>> [TEST] 模式运行", "yellow")
        run_args['train_epochs'] = 3
        run_args['train_sample_num'] = 200
        run_args['valid_sample_num'] = 200
        run_args['test_sample_num'] = 200
        #run_args['mode'] = "train valid test"
        run_args['checkconf_list'] = "0.1,0.2,0.3"

    # 根据传入的模式列表执行
    modes = run_args['mode'].split()

    if "train" in modes:
        print_config_info(run_args, args_help)
        main(run_args, current_mode="train")

    if "valid" in modes:
        print_config_info(run_args, args_help)
        main(run_args, current_mode="eval")

    if "test" in modes:
        print_config_info(run_args, args_help)
        main(run_args, current_mode="real")