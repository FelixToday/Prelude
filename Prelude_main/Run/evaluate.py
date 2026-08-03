# -*- coding: utf-8 -*-
"""
Phase 3: 测试脚本
整合固定比例测试和流式推理测试。
"""

import re
import time
import json
import torch
import numpy as np
import warnings
from tqdm import tqdm
import os
import argparse

from lxj_utils_sys import BaseLogger_v2, ModelCheckpoint, same_seed
from lxj_utils_sys import print_colored, print_config_info, IncrementalMeanCalculator
from lxj_utils_sys import measurement, str_to_bool

from utils_dataset_metric import get_model_and_dataloader, load_data
from utils_porcess import *
from Prelude_main.Run.const import get_filebase_dir, dataset_lib
from Prelude_main.Model import EDdataset, UnifiedGateNet, GateTrafficDataset
from tabulate import tabulate
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from Prelude_main.Run.stream_utils import (
    apply_window_offset_to_trace,
    prepare_streaming_sample,
    _stream_worker_initializer,
)
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
same_seed(2025)


def parse_float_list(value, default=0.0):
    if value is None:
        return [float(default)]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(parse_float_list(item, default=default))
        return values or [float(default)]

    tokens = [x for x in re.split(r"[,;\s]+", str(value).strip()) if x]
    return [float(x) for x in tokens] if tokens else [float(default)]


def format_float_token(value):
    token = f"{float(value):g}"
    return token.replace("-", "m").replace(".", "p")


def add_offset_to_note(base_note, offset_windows, force=False):
    if not force and abs(float(offset_windows)) < 1e-12:
        return base_note
    return f"{base_note}_offset_{format_float_token(offset_windows)}"


def build_offset_test_data(test_X, offset_windows, w_const):
    if float(offset_windows) <= 0:
        return test_X

    shifted_X = np.zeros_like(test_X)
    for i in range(len(test_X)):
        offset_result = apply_window_offset_to_trace(test_X[i], offset_windows, w_const)
        if not offset_result["empty"]:
            shifted_X[i] = offset_result["data"]
    return shifted_X


# ==========================================
# 测试模式 A: 固定比例测试
# ==========================================
def evaluate_fixed_ratio(model, test_loader, device, config, args, num_classes,
                         gate_net=None, test_X_raw=None, compute_pr_auc=False):
    """
    固定比例测试 —— 支持多加载率循环测试。
    依次使用 args['check_load_ratio'] 中的每个加载率进行测试并输出结果，
    返回两个列表：
      - all_results:  每个加载率对应的完整 metric 字典
      - all_f1scores: 每个加载率对应的 F1-score

    reliability_threshold: 若不为 None，仅统计置信度 > 该阈值的样本。
    reliability_type: 置信度分数类型
        - "softmax": backbone softmax 最大概率（默认）
        - "gatenet": GateNet 输出（需传入 gate_net + test_X_raw）
        - "auto": 有 GateNet 则用 gatenet，否则 softmax
    compute_pr_auc: 若为 True，计算 PR 曲线（macro 平均，插值到 100 点），
        结果存入各加载率的 dict。
    """
    # 获取待检查的加载率列表
    check_ratios = args.get('check_load_ratio', [])
    print("待检查的加载率", check_ratios)
    assert check_ratios, "请指定 check_ratio"

    reliability_threshold = args.get('reliability_threshold', None)
    reliability_type = args.get('reliability_type', 'auto')

    # 解析 auto → 实际类型
    if reliability_type == 'auto':
        reliability_type = 'gatenet' if (gate_net is not None) else 'softmax'

    if reliability_threshold is not None:
        print(f"  置信度类型: {reliability_type}, 阈值: {reliability_threshold}")

    # 使用 GateNet 置信度时校验 + 创建 GateNet DataLoader
    if reliability_type == 'gatenet':
        if gate_net is None:
            raise ValueError("reliability_type='gatenet' 但 GateNet 未加载，请设置 --use_gatenet True")
        if test_X_raw is None:
            raise ValueError("reliability_type='gatenet' 需要传入 test_X_raw")
        gate_net.eval()

        # === 创建 GateNet DataLoader：与主干网络一样，通过设置 loaded_ratio 控制截断比例 ===
        gate_dataset = GateTrafficDataset(
            test_X_raw, np.zeros(len(test_X_raw), dtype=int),
            seq_len=config.get('seq_len', 5000))
        gate_loader = DataLoader(
            gate_dataset,
            batch_size=int(config.get('batch_size', 20)),
            shuffle=False, drop_last=False, num_workers=16)

    all_results = []
    all_f1scores = []

    for ratio in check_ratios:
        # 修改 DataLoader 中 dataset 的加载率
        if hasattr(test_loader.dataset, 'loaded_ratio'):
            test_loader.dataset.loaded_ratio = ratio

        model.eval()
        valid_pred = []
        valid_true = []
        valid_reliability = []
        valid_prob = []  # 仅 compute_pr_auc=True 时收集每类 softmax 概率

        with torch.no_grad():
            for cur_data in tqdm(test_loader, desc=f"Fixed Ratio Eval ({ratio:.0f}%)"):
                cur_X, cur_y, idx = prepare_batch_data(cur_data, device)
                outs = model_forward(model, config, cur_X, idx)
                cur_pred = torch.argsort(outs, dim=1, descending=True)[:, 0]
                valid_pred.append(cur_pred.cpu().numpy())
                valid_true.append(cur_y.cpu().numpy())
                if compute_pr_auc:
                    valid_prob.append(torch.softmax(outs, dim=1).cpu().numpy())

                if reliability_threshold is not None:
                    if reliability_type == 'softmax':
                        cur_prob = torch.softmax(outs, dim=1)
                        cur_reliability = cur_prob.max(dim=1).values
                        valid_reliability.append(cur_reliability.cpu().numpy())

        valid_pred = np.concatenate(valid_pred)
        valid_true = np.concatenate(valid_true)
        if compute_pr_auc:
            valid_prob = np.concatenate(valid_prob)
        total_samples = len(valid_true)
        # --- GateNet 置信度：通过 DataLoader 逐 batch 推理（与主干网络模式一致）---
        if reliability_threshold is not None and reliability_type == 'gatenet':
            gate_loader.dataset.loaded_ratio = ratio
            raw_reliability = []
            for batch_data, _ in tqdm(gate_loader, desc=f"GateNet Reliability ({ratio:.0f}%)"):
                batch_t = batch_data.to(device)
                batch_conf = gate_net(batch_t).squeeze(-1).detach().cpu().numpy()
                raw_reliability.extend(batch_conf.tolist())
            valid_reliability = [np.array(raw_reliability)]
        # --- 可靠性过滤 ---
        if reliability_threshold is not None and len(valid_reliability) > 0:
            valid_reliability = np.concatenate(valid_reliability)
            mask = valid_reliability > reliability_threshold
            n_passed = int(mask.sum())
            valid_pred = valid_pred[mask]
            valid_true = valid_true[mask]
            if compute_pr_auc:
                valid_prob = valid_prob[mask]
            pass_rate = n_passed / total_samples if total_samples > 0 else 0.0
            print(f"  可靠性过滤 (type={reliability_type}, threshold={reliability_threshold}): "
                  f"{n_passed}/{total_samples} ({pass_rate:.1%}) 通过")
        else:
            pass_rate = 1.0

        valid_result = measurement(valid_true, valid_pred, config['eval_metrics'])
        valid_result['pass_rate'] = pass_rate
        valid_result['n_total'] = total_samples
        valid_result['n_passed'] = int(total_samples * pass_rate)

        # --- PR 曲线计算（基于 backbone softmax 每类概率，与置信度来源无关）---
        if compute_pr_auc and len(valid_true) > 0:
            from sklearn.metrics import precision_recall_curve, auc

            n_classes = num_classes
            y_onehot = np.eye(n_classes, dtype=np.float64)[valid_true.astype(int)]

            precisions, recalls, aucs = [], [], []
            for i in range(n_classes):
                if y_onehot[:, i].sum() == 0:
                    continue
                p, r, t = precision_recall_curve(y_onehot[:, i], valid_prob[:, i])
                precisions.append(p)
                recalls.append(r)
                aucs.append(auc(r, p))

            if len(aucs) > 0:
                n_active = len(aucs)
                # 插值到固定 100 个 recall 点做宏平均
                mean_recall = np.linspace(0, 1, 100)
                mean_precision = np.zeros(100)
                for p, r in zip(precisions, recalls):
                    mean_precision += np.interp(mean_recall, r[::-1], p[::-1])
                mean_precision /= n_active

                valid_result['pr_auc_macro'] = float(np.mean(aucs))
                valid_result['pr_precision'] = mean_precision.tolist()
                valid_result['pr_recall'] = mean_recall.tolist()
                valid_result['pr_num_classes'] = n_active
            else:
                valid_result['pr_auc_macro'] = 0.0
                valid_result['pr_precision'] = []
                valid_result['pr_recall'] = []
                valid_result['pr_num_classes'] = 0

        all_results.append(valid_result)
        all_f1scores.append(valid_result.get("F1-score", 0))

    return all_results, all_f1scores


# ==========================================
# 测试模式 B: 流式推理测试
# ==========================================
def evaluate_streaming(model_backbone, gate_net, test_X, test_y, args, config, num_classes,
                       device, out_dir, result_note="default"):
    """
    流式推理测试 —— 操作原始 numpy 数据，不依赖 dataset。

    步进模式:
      - ratio: 按固定比例递增 delta=0.03
      - window: 按固定时间窗口递增 WIN_STEP=6

    评估方式:
      - pass_only: 只统计触发阈值的样本
      - all_fallback: 全样本+回退（未触发用100% backbone推理）
    """
    from Prelude_main.Model.dataset import get_TAM_ED1, pad_sequence

    mode = args.get('load_mode', 'ratio')
    delta = args.get('delta', 0.03)
    checkconf_list = sorted([float(x) for x in args.get('checkconf_list', "0.5,0.7,0.9").split(',')])
    use_gatenet = args.get('use_gatenet', True) and gate_net is not None
    eval_mode = args.get('eval_mode', 'all_fallback')
    win_step = args.get('win_step', 6)
    offset_windows = args.get('offset_windows', 0)

    seq_len = config['seq_len']
    max_matrix_len = config.get('max_matrix_len', 1800)
    maximum_load_time = args.get('maximum_load_time', 80)
    maximum_cell_number = config.get('maximum_cell_number', 2)
    time_interval_threshold = config.get('time_interval_threshold', 0.1)
    log_transform = config.get('log_transform', False)

    W_CONST = maximum_load_time / max_matrix_len

    # 选择 TAM 函数
    get_TAM_fn = get_TAM_ED1

    tam_args = {
        "seq_len": seq_len,
        "max_matrix_len": max_matrix_len,
        "maximum_load_time": maximum_load_time,
        "maximum_cell_number": maximum_cell_number,
        "time_interval_threshold": time_interval_threshold,
    }

    if use_gatenet:
        gate_net.eval()
    model_backbone.eval()

    metrics = {}
    for th in checkconf_list:
        metrics[th] = {
            "loading_ratio": IncrementalMeanCalculator(),
            "confidence": IncrementalMeanCalculator(),
            "load_latency": IncrementalMeanCalculator(),
            "trigger_count": 0,
            "not_triggered": 0,
            "pred": [],
            "label": [],
        }

    with torch.no_grad():
        loop = tqdm(range(len(test_X)), ncols=150)
        for num in loop:
            raw_data = test_X[num]
            label = test_y[num]

            offset_result = apply_window_offset_to_trace(raw_data, offset_windows, W_CONST)
            if offset_result["empty"]:
                # 空流量样本：跳过，所有阈值标记为未触发
                for th in checkconf_list:
                    metrics[th]["not_triggered"] += 1
                    metrics[th]["trigger_count"] += 1
                    metrics[th]["loading_ratio"].add(1.0)
                    metrics[th]["load_latency"].add(offset_result["original_max_trace_time"])
                    metrics[th]["confidence"].add(1.0)
                continue

            raw_data = offset_result["data"]
            timestamp = offset_result["timestamp"]
            original_max_trace_time = offset_result["original_max_trace_time"]
            observed_start_time = offset_result["observed_start_time"]
            max_trace_time = float(timestamp[-1])


            # 逐步推进
            current_ratio = delta if mode == 'ratio' else 0.0
            win_index = 0
            triggered_thresholds = set()
            trigger_info = {}

            while current_ratio <= 1.0 + 1e-8:
                if mode == 'window':
                    win_index += win_step
                    current_ratio = (win_index * W_CONST) / max(max_trace_time, 1e-8)

                actual_ratio = min(current_ratio, 1.0)
                threshold_time = actual_ratio * max_trace_time
                cutoff_idx = np.searchsorted(timestamp, threshold_time, side='right')

                # --- 截断数据 ---
                truncated = raw_data.copy()
                truncated[cutoff_idx:, :] = 0.0

                # --- 计算 backbone 输入 ---
                time_arr = truncated[:, 0]
                pkt_arr = truncated[:, 1]
                time_arr_pad = pad_sequence(time_arr, seq_len)
                pkt_arr_pad = pad_sequence(pkt_arr, seq_len)
                TAM, _, _ = get_TAM_fn(pkt_arr_pad, time_arr_pad, args=tam_args, bapm=None)
                TAM = TAM.reshape((1, -1, max_matrix_len)).astype(np.float32)
                if log_transform:
                    TAM = np.log1p(TAM)
                backbone_input_t = torch.tensor(TAM).unsqueeze(0).to(device)

                # --- 置信度计算 ---
                if use_gatenet:
                    # GateNet 输入: 原始流量 (1, 2, L)
                    raw_seq = torch.tensor(truncated.T, dtype=torch.float32).unsqueeze(0).to(device)
                    confidence = gate_net(raw_seq).item()
                else:
                    # 使用 backbone 自身置信度作为置信度
                    cur_logits = model_backbone(backbone_input_t)
                    cur_probs = torch.softmax(cur_logits, dim=1)
                    confidence = cur_probs.max().item()
                    cur_pred = torch.argmax(cur_logits, dim=1).item()

                # 检查触发
                for th in checkconf_list:
                    if th not in triggered_thresholds and confidence >= th:
                        triggered_thresholds.add(th)
                        info = {
                            "backbone_input": backbone_input_t.clone(),
                            "label": label,
                            "ratio": min(
                                (observed_start_time + actual_ratio * max_trace_time)
                                / max(original_max_trace_time, 1e-8),
                                1.0),
                            "latency": observed_start_time + actual_ratio * max_trace_time,
                            "confidence": confidence,
                        }
                        if not use_gatenet:
                            info["pred"] = cur_pred
                        trigger_info[th] = info

                if len(triggered_thresholds) == len(checkconf_list):
                    break

                if mode == 'ratio':
                    current_ratio += delta

                # 强制100%回退
                if current_ratio >= 1.0 + 1e-8 and len(triggered_thresholds) < len(checkconf_list):
                    full_truncated = raw_data.copy()

                    full_time = full_truncated[:, 0]
                    full_pkt = full_truncated[:, 1]
                    full_time_pad = pad_sequence(full_time, seq_len)
                    full_pkt_pad = pad_sequence(full_pkt, seq_len)
                    full_TAM, _, _ = get_TAM_fn(full_pkt_pad, full_time_pad, args=tam_args, bapm=None)
                    full_TAM = full_TAM.reshape((1, -1, max_matrix_len)).astype(np.float32)
                    if log_transform:
                        full_TAM = np.log1p(full_TAM)
                    full_input_t = torch.tensor(full_TAM).unsqueeze(0).to(device)

                    if use_gatenet:
                        raw_full = torch.tensor(full_truncated.T, dtype=torch.float32).unsqueeze(0).to(device)
                        confidence_full = gate_net(raw_full).item()
                    else:
                        # 100% 全量数据使用 backbone 自身置信度
                        full_logits = model_backbone(full_input_t)
                        full_probs = torch.softmax(full_logits, dim=1)
                        confidence_full = full_probs.max().item()
                        full_pred = torch.argmax(full_logits, dim=1).item()

                    for th in checkconf_list:
                        if th not in triggered_thresholds:
                            triggered_thresholds.add(th)
                            info = {
                                "backbone_input": full_input_t.clone(),
                                "label": label,
                                "ratio": 1.0,
                                "latency": original_max_trace_time,
                                "confidence": confidence_full,
                            }
                            if not use_gatenet:
                                info["pred"] = full_pred
                            trigger_info[th] = info
                    break

            # 记录每个阈值的 backbone 预测
            for th in checkconf_list:
                info = trigger_info[th]
                true_label = info["label"]

                if use_gatenet:
                    # GateNet 模式下，需运行 backbone 获取 pred
                    bb_input = info["backbone_input"]
                    pred_scores = model_backbone(bb_input)
                    pred = torch.argmax(pred_scores, dim=1).item()
                else:
                    # 无 GateNet 模式下，pred 已在 stepping 循环中计算
                    pred = info["pred"]

                metrics[th]["loading_ratio"].add(info["ratio"])
                metrics[th]["load_latency"].add(info["latency"])
                metrics[th]["confidence"].add(info["confidence"])
                metrics[th]["trigger_count"] += 1

                if eval_mode == 'pass_only':
                    if info["ratio"] < 1.0:
                        metrics[th]["pred"].append(pred)
                        metrics[th]["label"].append(true_label)
                    else:
                        metrics[th]["not_triggered"] += 1
                else:
                    metrics[th]["pred"].append(pred)
                    metrics[th]["label"].append(true_label)

            first_th = checkconf_list[0]
            if first_th in trigger_info:
                loop.set_postfix({
                    f'Ratio': f"{trigger_info[first_th]['ratio'] * 100:.1f}%",
                    f'Conf': f"{trigger_info[first_th]['confidence']:.2f}",
                })

    # ========== 汇总结果 ==========
    early_logger = BaseLogger_v2(
        json_path=os.path.join(out_dir, f"stream_{result_note}.json"),
        log_path=os.path.join(out_dir, f"stream_{result_note}.txt"))

    early_logger.start_timer("流式推理")
    early_logger.record("config", {
        "load_mode": mode,
        "delta": delta,
        "offset_windows": offset_windows,
        "checkconf_list": checkconf_list,
        "use_gatenet": use_gatenet,
        "eval_mode": eval_mode,
        "win_step": win_step,
    })

    table_data = []
    for th in checkconf_list:
        m = metrics[th]
        if len(m["pred"]) > 0:
            preds = np.array(m["pred"])
            labels = np.array(m["label"])
            result = measurement(labels, preds, config['eval_metrics'])
        else:
            result = {k: 0.0 for k in config['eval_metrics']}

        avg_load = m["loading_ratio"].get() * 100
        avg_latency = m["load_latency"].get() * 1000
        avg_confidence = m["confidence"].get()
        total_samples = len(test_X)
        pass_ratio = m["trigger_count"] / total_samples * 100

        table_data.append([
            f"{th:.2f}",
            f"{avg_load:.2f}%",
            f"{result.get('Accuracy', 0):.2f}%",
            f"{result.get('Precision', 0):.2f}%",
            f"{result.get('Recall', 0):.2f}%",
            f"{result.get('F1-score', 0):.4f}",
            f"{avg_confidence:.4f}",
            f"{avg_latency:.2f}ms",
            f"{pass_ratio:.1f}%",
        ])

        early_logger.record(f"th_{th:.2f}", {
            "avg_load_ratio": avg_load,
            "avg_latency_ms": avg_latency,
            "avg_confidence": avg_confidence,
            "pass_ratio": pass_ratio,
            "trigger_count": m["trigger_count"],
            "not_triggered": m["not_triggered"],
            **{f"metric_{k}": v for k, v in result.items()},
        })

    headers = ["Threshold", "Avg Load", "Accuracy", "Precision", "Recall", "F1", "Confidence", "Avg Latency", "Pass Ratio"]
    early_logger.print(tabulate(table_data, headers=headers, tablefmt="simple"), save_to_file=True)
    early_logger.stop_timer("流式推理")

    print_colored(f"\n流式推理完成，结果保存至: {out_dir}", "green")


# ==========================================
# 测试模式 B2: 快速流式推理（批量 GPU 推理）
# ==========================================
def evaluate_streaming_fast(model_backbone, gate_net, test_X, test_y, args, config, num_classes,
                             device, out_dir, result_note="default"):
    """
    快速流式推理 —— CPU 并行预处理 + 批量 GPU 推理。
    每个样本的所有截断步合并为一次 GPU forward，避免逐步推理的 kernel launch 开销。
    与 evaluate_streaming 输出完全一致，速度提升 10~30 倍。
    """
    mode = args.get('load_mode', 'ratio')
    delta = args.get('delta', 0.03)
    checkconf_list = sorted([float(x) for x in args.get('checkconf_list', "0.5,0.7,0.9").split(',')])
    use_gatenet = args.get('use_gatenet', True) and gate_net is not None
    eval_mode = args.get('eval_mode', 'all_fallback')
    win_step = args.get('win_step', 6)
    offset_windows = args.get('offset_windows', 0)

    seq_len = config['seq_len']
    max_matrix_len = config.get('max_matrix_len', 1800)
    maximum_load_time = args.get('maximum_load_time', 80)
    maximum_cell_number = config.get('maximum_cell_number', 2)
    time_interval_threshold = config.get('time_interval_threshold', 0.1)
    log_transform = config.get('log_transform', False)

    W_CONST = maximum_load_time / max_matrix_len
    TAM_type = "ED1"
    tam_args = {
        "seq_len": seq_len, "max_matrix_len": max_matrix_len,
        "maximum_load_time": maximum_load_time, "maximum_cell_number": maximum_cell_number,
        "time_interval_threshold": time_interval_threshold,
    }

    if use_gatenet:
        gate_net.eval()
    model_backbone.eval()

    # ---- 并行 workers 配置（默认多进程）----
    stream_workers = int(args.get('stream_workers') or args.get('num_workers', 16) or 1)
    stream_workers = max(1, stream_workers)
    parallel_backend = args.get('stream_parallel_backend', 'process')
    process_start_method = args.get('stream_process_start_method', 'auto')

    if parallel_backend not in ('thread', 'process'):
        parallel_backend = 'process'
    if process_start_method == 'auto':
        process_start_method = 'spawn' if device.type == 'cuda' else None

    stream_prefetch = args.get('stream_prefetch', None)
    if stream_prefetch is None:
        max_pending = stream_workers if parallel_backend == 'process' else stream_workers * 2
    else:
        max_pending = int(stream_prefetch)
    max_pending = max(stream_workers, max_pending)

    if stream_workers > 1:
        print_colored(
            f"快速流式推理使用 {parallel_backend} 并行: workers={stream_workers}, prefetch={max_pending}",
            "cyan")

    # ---- 指标初始化 ----
    metrics = {}
    for th in checkconf_list:
        metrics[th] = {
            "loading_ratio": IncrementalMeanCalculator(),
            "confidence": IncrementalMeanCalculator(),
            "load_latency": IncrementalMeanCalculator(),
            "trigger_count": 0, "not_triggered": 0,
            "pred": [], "label": [],
        }

    # ---- task 构建 ----
    def _make_task(num):
        return (
            num, test_X[num], int(test_y[num]), mode, delta, win_step, W_CONST,
            config, seq_len, max_matrix_len,
            log_transform, tam_args, use_gatenet, TAM_type, offset_windows,
        )

    # ---- GPU 推理 + 阈值评估（主线程串行，确保线程安全）----
    def _infer_and_evaluate(prepared):
        label = prepared["label"]
        max_trace_time = prepared["max_trace_time"]
        original_max_trace_time = prepared.get("original_max_trace_time", max_trace_time)
        observed_start_time = prepared.get("observed_start_time", 0.0)

        if prepared["empty"]:
            for th in checkconf_list:
                metrics[th]["not_triggered"] += 1
                metrics[th]["trigger_count"] += 1
                metrics[th]["loading_ratio"].add(1.0)
                metrics[th]["load_latency"].add(original_max_trace_time)
                metrics[th]["confidence"].add(1.0)
            return

        ratio_list = prepared["ratio_list"]
        n_steps = len(ratio_list)

        # === 批量 GPU 推理（一次 forward 处理所有截断步）===
        batch_inp = torch.as_tensor(prepared["model_input"], dtype=torch.float32).to(device)

        all_logits = model_backbone(batch_inp)
        all_preds = torch.argmax(all_logits, dim=1)
        if use_gatenet:
            gate_batch = torch.as_tensor(prepared["gate_input"], dtype=torch.float32).to(device)
            all_scores = gate_net(gate_batch).squeeze(-1)
        else:
            all_probs = torch.softmax(all_logits, dim=1)
            all_scores = all_probs.max(dim=1).values

        # 转为 numpy 做向量化阈值判断
        scores_np = all_scores.cpu().numpy()
        preds_np = all_preds.cpu().numpy()
        ratio_arr = np.array(ratio_list)

        # === 对每个阈值找首次触发位置 ===
        for th in checkconf_list:
            hit = np.argmax(scores_np >= th)
            if scores_np[hit] >= th:
                # 触发了
                trig_ratio = float(ratio_arr[hit])
                trig_pred = int(preds_np[hit])
                trig_conf = float(scores_np[hit])
                trig_latency = observed_start_time + trig_ratio * max_trace_time
                trig_ratio = min(trig_latency / max(original_max_trace_time, 1e-8), 1.0)
                is_triggered = True
            else:
                # 未触发 → fallback
                last_idx = n_steps - 1
                trig_pred = int(preds_np[last_idx])
                trig_conf = float(scores_np[last_idx])
                if eval_mode == 'all_fallback':
                    trig_ratio = 1.0
                    trig_latency = original_max_trace_time
                else:
                    trig_ratio = float(ratio_arr[last_idx])
                    trig_latency = observed_start_time + trig_ratio * max_trace_time
                    trig_ratio = min(trig_latency / max(original_max_trace_time, 1e-8), 1.0)
                is_triggered = False

            metrics[th]["loading_ratio"].add(trig_ratio)
            metrics[th]["load_latency"].add(trig_latency)
            metrics[th]["confidence"].add(trig_conf)
            metrics[th]["trigger_count"] += 1

            if eval_mode == 'pass_only':
                if is_triggered and trig_ratio < 1.0:
                    metrics[th]["pred"].append(trig_pred)
                    metrics[th]["label"].append(label)
                else:
                    metrics[th]["not_triggered"] += 1
            else:
                metrics[th]["pred"].append(trig_pred)
                metrics[th]["label"].append(label)

    # ---- 主循环 ----
    with torch.no_grad():
        if stream_workers <= 1:
            loop = tqdm(range(len(test_X)), desc="Fast Stream Eval (batched)", ncols=150)
            for num in loop:
                prepared = prepare_streaming_sample(_make_task(num))
                _infer_and_evaluate(prepared)
        else:
            # 并行 CPU 预处理 + 主线程 GPU 推理
            if parallel_backend == 'process':
                executor_kwargs = {
                    "max_workers": stream_workers,
                    "initializer": _stream_worker_initializer,
                    "initargs": (args.get('stream_worker_torch_threads', 1),),
                }
                if process_start_method:
                    executor_kwargs["mp_context"] = mp.get_context(process_start_method)
                executor = ProcessPoolExecutor(**executor_kwargs)
            else:
                executor = ThreadPoolExecutor(max_workers=stream_workers)

            with executor:
                pending = set()
                next_num = 0

                def _submit_until_full():
                    nonlocal next_num
                    while next_num < len(test_X) and len(pending) < max_pending:
                        pending.add(executor.submit(prepare_streaming_sample, _make_task(next_num)))
                        next_num += 1

                _submit_until_full()
                pbar_desc = f"Fast Stream Eval ({parallel_backend}x{stream_workers})"
                with tqdm(total=len(test_X), desc=pbar_desc, ncols=150) as pbar:
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            prepared = future.result()
                            _infer_and_evaluate(prepared)
                            pbar.update(1)
                        _submit_until_full()

    # ========== 汇总结果（与旧函数完全一致）==========
    early_logger = BaseLogger_v2(
        json_path=os.path.join(out_dir, f"stream_{result_note}.json"),
        log_path=os.path.join(out_dir, f"stream_{result_note}.txt"))

    early_logger.start_timer("快速流式推理")
    early_logger.record("config", {
        "load_mode": mode, "delta": delta, "checkconf_list": checkconf_list,
        "offset_windows": offset_windows,
        "use_gatenet": use_gatenet, "eval_mode": eval_mode, "win_step": win_step,
        "stream_fast": True,
    })

    table_data = []
    for th in checkconf_list:
        m = metrics[th]
        if len(m["pred"]) > 0:
            preds = np.array(m["pred"])
            labels = np.array(m["label"])
            result = measurement(labels, preds, config['eval_metrics'])
        else:
            result = {k: 0.0 for k in config['eval_metrics']}

        avg_load = m["loading_ratio"].get() * 100
        avg_latency = m["load_latency"].get() * 1000
        avg_confidence = m["confidence"].get()
        total_samples = len(test_X)
        pass_ratio = m["trigger_count"] / total_samples * 100

        table_data.append([
            f"{th:.2f}", f"{avg_load:.2f}%",
            f"{result.get('Accuracy', 0):.2f}%", f"{result.get('Precision', 0):.2f}%",
            f"{result.get('Recall', 0):.2f}%", f"{result.get('F1-score', 0):.4f}",
            f"{avg_confidence:.4f}", f"{avg_latency:.2f}ms", f"{pass_ratio:.1f}%",
        ])

        early_logger.record(f"th_{th:.2f}", {
            "avg_load_ratio": avg_load, "avg_latency_ms": avg_latency,
            "avg_confidence": avg_confidence, "pass_ratio": pass_ratio,
            "trigger_count": m["trigger_count"], "not_triggered": m["not_triggered"],
            **{f"metric_{k}": v for k, v in result.items()},
        })

    headers = ["Threshold", "Avg Load", "Accuracy", "Precision", "Recall", "F1", "Confidence", "Avg Latency", "Pass Ratio"]
    early_logger.print(tabulate(table_data, headers=headers, tablefmt="simple"), save_to_file=True)
    early_logger.stop_timer("快速流式推理")

    print_colored(f"\n快速流式推理完成，结果保存至: {out_dir}", "green")


# ==========================================
# 主函数
# ==========================================
def main():
    parser = get_evaluate_parser()
    args, args_help = parse_args(parser, is_print_help=False)

    if args['file_base_dir'] == "auto":
        args['file_base_dir'] = get_filebase_dir()

    device = torch.device(args['device']) if torch.cuda.is_available() else torch.device('cpu')

    # --- 加载 backbone 配置 ---
    checkpoint_dir = os.path.join(args['checkpoint_path'], args['dataset'],
                                  args['backbone_model'] if args.get('backbone_model') else 'Prelude',
                                  args['backbone_note']).rstrip('/')

    logger = BaseLogger_v2(json_path=os.path.join(checkpoint_dir, "result.json"))
    if not os.path.exists(os.path.join(checkpoint_dir, "result.json")):
        raise FileNotFoundError(
            f"Backbone 训练结果文件不存在: {os.path.join(checkpoint_dir, 'result.json')}\n"
            f"请先运行 Phase 1 (train_backbone.py) 训练模型。"
        )
    logger.load()
    if 'config' not in logger.data or 'args' not in logger.data.get('config', {}):
        raise KeyError(
            f"Backbone 结果文件缺少 config 配置信息: {os.path.join(checkpoint_dir, 'result.json')}\n"
            f"文件可能由旧版本代码生成，请重新运行 Phase 1 (train_backbone.py)。\n"
            f"当前 data keys: {list(logger.data.keys())}"
        )
    config = logger.data['config']['config']
    exp_config = logger.data['config']['args']
    exp_config['file_base_dir'] = args['file_base_dir']
    exp_config['load_ratio'] = 100
    # ---- 可靠性阈值处理：支持 auto / 数值 / None ----
    raw_threshold = args.get('reliability_threshold', None)
    if raw_threshold is not None and str(raw_threshold).lower() == 'auto':
        # auto 模式：从阈值搜索结果文件自动加载
        search_note = args.get('threshold_search_note', 'pre_greater_than_80')
        threshold_search_path = os.path.join(checkpoint_dir, "evaluate",
                                             f"threshold_search_{search_note}.json")
        if os.path.exists(threshold_search_path):
            with open(threshold_search_path, 'r', encoding='utf-8') as f:
                search_data = json.load(f)
            loaded = search_data.get('selected_threshold', None)
            if loaded is not None:
                reliability_threshold = float(loaded)
                print_colored(
                    f" ★ 自动加载阈值: reliability_threshold = {reliability_threshold:.4f}"
                    f" (来自 {threshold_search_path})",
                    "cyan")
            else:
                reliability_threshold = 0.5
                print_colored(
                    f" ★ 阈值文件缺少 selected_threshold 字段，回退到默认值: {reliability_threshold}",
                    "yellow")
        else:
            reliability_threshold = 0.5
            print_colored(
                f" ★ 阈值搜索文件不存在，回退到默认值: reliability_threshold = {reliability_threshold}"
                f" (期望路径: {threshold_search_path})",
                "yellow")
    elif raw_threshold is not None:
        # 数值模式：字符串 → float
        try:
            reliability_threshold = float(raw_threshold)
        except (ValueError, TypeError):
            reliability_threshold = None
            print_colored(
                f" ★ 无法解析 reliability_threshold = {raw_threshold}，已禁用过滤",
                "yellow")
    else:
        reliability_threshold = None

    # 同步到 args 和 exp_config
    args['reliability_threshold'] = reliability_threshold
    exp_config['reliability_threshold'] = reliability_threshold
    exp_config['reliability_type'] = args.get('reliability_type', 'auto')

    # ---- checkconf_list 处理：支持 auto（加载搜索出的最优阈值）----
    raw_checkconf = args.get('checkconf_list', "0.5,0.7,0.9,0.95")
    if str(raw_checkconf).lower() == 'auto':
        # auto 模式：从同一份阈值搜索文件加载 selected_threshold 作为单一阈值
        search_note = args.get('threshold_search_note', 'pre_greater_than_80')
        threshold_search_path = os.path.join(checkpoint_dir, "evaluate",
                                             f"threshold_search_{search_note}.json")
        if os.path.exists(threshold_search_path):
            with open(threshold_search_path, 'r', encoding='utf-8') as f:
                search_data = json.load(f)
            loaded = search_data.get('selected_threshold', None)
            if loaded is not None:
                checkconf_list_str = f"{float(loaded):.6f}"
                print_colored(
                    f" ★ 自动加载 checkconf_list: [{checkconf_list_str}]"
                    f" (来自 {threshold_search_path})",
                    "cyan")
            else:
                checkconf_list_str = "0.5"
                print_colored(
                    f" ★ 阈值文件缺少 selected_threshold，checkconf_list 回退到默认: {checkconf_list_str}",
                    "yellow")
        else:
            checkconf_list_str = "0.5"
            print_colored(
                f" ★ 阈值搜索文件不存在，checkconf_list 回退到默认值: {checkconf_list_str}"
                f" (期望路径: {threshold_search_path})",
                "yellow")
        args['checkconf_list'] = checkconf_list_str
    else:
        args['checkconf_list'] = raw_checkconf

    offset_window_values = parse_float_list(args.get('offset_windows', 0), default=0.0)
    if any(v < 0 for v in offset_window_values):
        raise ValueError("--offset_windows must be non-negative")
    multi_offset = len(offset_window_values) > 1
    W_CONST = args.get('maximum_load_time', 80) / config.get('max_matrix_len', 1800)
    check_ratio_str = args.get('check_load_ratio', '').strip()
    if check_ratio_str:
        exp_config['check_load_ratio'] = [float(x) for x in re.split(r'[,，\s]+', check_ratio_str) if x]
    else:
        exp_config['check_load_ratio'] = [100]
    num_classes = dataset_lib[args['dataset']]['num_classes']

    # --- 数据加载 ---
    test_X, test_y = load_data(
        os.path.join(args['file_base_dir'], args['dataset'], "test.npz"),
        drop_extra_time=True, load_time=args.get('maximum_load_time', 80))

    if args.get('test_flag', False):
        n_samples = 200
        rand_idx = np.random.permutation(len(test_X))[:n_samples]
        test_X, test_y = test_X[rand_idx], test_y[rand_idx]
    elif args.get('sample_num', -1) > 0:
        n_samples = args['sample_num']
        rand_idx = np.random.permutation(len(test_X))[:n_samples]
        test_X, test_y = test_X[rand_idx], test_y[rand_idx]


    # --- 模型加载 ---
    model, _, test_loader = get_model_and_dataloader(
        test_X, test_y, test_X, test_y, num_classes, config, exp_config)

    mode = 'max'
    metric_name = "f1"
    modelsaver = ModelCheckpoint(filename=os.path.join(checkpoint_dir, f"model.pth"),
                                 mode=mode, metric_name=metric_name)
    model = modelsaver.load(model, device)[0]
    print_colored(f"Backbone 模型加载成功: {checkpoint_dir}", "green")

    # --- GateNet 加载（可选；fixed 评估的 gatenet 置信度也需要）---
    gate_net = None
    use_gatenet = args.get('use_gatenet', False)
    reliability_type = args.get('reliability_type', 'auto')
    reliability_threshold = exp_config.get('reliability_threshold', None)
    need_gatenet = use_gatenet or (
        reliability_threshold is not None
        and float(reliability_threshold) > 0
        and args.get('eval_mode_type', 'streaming') == 'fixed'
        and reliability_type == 'gatenet'
    )
    if need_gatenet:
        gate_dir = os.path.join(checkpoint_dir, "gatenet", args.get('gatenet_note', 'default'))
        try:
            gate_net = UnifiedGateNet(seq_len=config['seq_len']).to(device)
            gatesaver = ModelCheckpoint(filename=os.path.join(gate_dir, "gatenet_model.pth"),
                                        mode='max', metric_name='gate_acc')
            gate_net = gatesaver.load(gate_net, device)[0]
            gate_net.eval()
            print_colored(f"GateNet 模型加载成功: {gate_dir}", "green")
        except Exception as e:
            print_colored(f"GateNet 加载失败: {e}，回退为无 GateNet 模式", "yellow")
            use_gatenet = False
            gate_net = None

    # --- 输出目录 ---
    out_dir = os.path.join(checkpoint_dir, "evaluate")
    os.makedirs(out_dir, exist_ok=True)

    # --- 执行测试 ---
    for offset_windows_val in offset_window_values:
        cur_args = {**args, 'offset_windows': float(offset_windows_val)}
        cur_exp_config = {**exp_config, 'offset_windows': float(offset_windows_val)}
        offset_suffix_note = add_offset_to_note(args.get('threshold_search_note', 'default'), offset_windows_val, force=multi_offset)
        file_suffix = add_offset_to_note('', offset_windows_val, force=multi_offset)

        if args.get('eval_mode_type', 'streaming') == 'fixed':
            print_colored(f"\n=== Fixed Ratio Eval (offset_windows={offset_windows_val:g}) ===", "blue")
            cur_test_X = build_offset_test_data(test_X, offset_windows_val, W_CONST)
            _, _, cur_test_loader = get_model_and_dataloader(
                cur_test_X, test_y, cur_test_X, test_y, num_classes, config, cur_exp_config)
            all_results, all_f1scores = evaluate_fixed_ratio(
                model, cur_test_loader, device, config, cur_exp_config, num_classes,
                gate_net=gate_net, test_X_raw=cur_test_X,
                compute_pr_auc=args.get('is_pr_auc', False))

            check_ratios = cur_exp_config.get('check_load_ratio', [100])
            reliability_threshold = cur_exp_config.get('reliability_threshold', None)
            reliability_type = cur_exp_config.get('reliability_type', 'auto')
            fixed_name = f"fixed_result{file_suffix}"

            logger_v2 = BaseLogger_v2(
                json_path=os.path.join(out_dir, f"{fixed_name}.json"),
                log_path=os.path.join(out_dir, f"{fixed_name}.txt"))
            logger_v2.record("check_ratios", check_ratios)
            logger_v2.record("offset_windows", float(offset_windows_val))
            logger_v2.record("offset_time", float(offset_windows_val) * W_CONST)
            if reliability_threshold is not None:
                logger_v2.record("reliability_threshold", reliability_threshold)
                logger_v2.record("reliability_type", reliability_type)

            table_data = []
            for ratio, result, f1 in zip(check_ratios, all_results, all_f1scores):
                logger_v2.record(f"ratio_{ratio:.0f}%", {
                    "load_ratio": ratio,
                    **{f"metric_{k}": v for k, v in result.items()},
                })
                row = [
                    f"{ratio:.0f}%",
                    f"{result.get('Accuracy', 0):.2f}%",
                    f"{result.get('Precision', 0):.2f}%",
                    f"{result.get('Recall', 0):.2f}%",
                    f"{result.get('F1-score', 0):.4f}",
                ]
                if reliability_threshold is not None:
                    n_passed = result.get('n_passed', 0)
                    n_total = result.get('n_total', 1)
                    row.append(f"{n_passed}/{n_total} ({result.get('pass_rate', 0):.1%})")
                table_data.append(row)

            headers = ["Load Ratio", "Accuracy", "Precision", "Recall", "F1"]
            if reliability_threshold is not None:
                headers.append("Pass (n/N)")
            logger_v2.print(tabulate(table_data, headers=headers, tablefmt="pretty"), save_to_file=True)
            print_colored("Fixed ratio eval done", "green")

        elif args.get('eval_mode_type', 'streaming') == 'streaming':
            streaming_args = {**cur_args}
            streaming_args['use_gatenet'] = use_gatenet
            streaming_args['maximum_load_time'] = args.get('maximum_load_time', 80)

            if args.get('stream_fast', True):
                print_colored(
                    f"\n=== Fast Stream Eval (mode={args['load_mode']}, offset_windows={offset_windows_val:g}, batched GPU) ===",
                    "blue")
                evaluate_streaming_fast(model, gate_net, test_X, test_y,
                                        streaming_args, config, num_classes, device,
                                        out_dir, offset_suffix_note)
            else:
                print_colored(
                    f"\n=== Stream Eval (mode={args['load_mode']}, offset_windows={offset_windows_val:g}, step-by-step) ===",
                    "blue")
                evaluate_streaming(model, gate_net, test_X, test_y,
                                  streaming_args, config, num_classes, device,
                                  out_dir, offset_suffix_note)

        else:
            print_colored(f"Unknown eval mode: {args['eval_mode_type']}", "red")


def get_evaluate_parser():
    parser = argparse.ArgumentParser(description="Phase 3: 测试评估")
    parser.add_argument('--checkpoint_path', default="../../checkpoints", help="运行结果存储路径")
    parser.add_argument('--file_base_dir', default="auto", help="数据集存储路径")
    parser.add_argument('--dataset', default="CW", help="数据集名称")
    parser.add_argument('--backbone_model', type=str, default='Prelude', help='Backbone 模型名称')
    parser.add_argument('--backbone_note', type=str, default='default', help='Backbone 训练 notes')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')

    # 评估模式
    parser.add_argument('--eval_mode_type', type=str, default='streaming',
                        choices=['fixed', 'streaming'], help='评估模式类型')

    # 固定比例测试参数
    parser.add_argument('--check_load_ratio', type=str, default="100", help='固定加载比例')
    parser.add_argument('--reliability_threshold', type=str, default=None,
                        help='可靠性分数阈值。设为 auto 将自动从 threshold_search 结果加载；'
                             '设为数值（如 "0.5"）则直接使用该值；None=不启用过滤')
    parser.add_argument('--reliability_type', type=str, default='auto',
                        choices=['auto', 'softmax', 'gatenet'],
                        help='置信度分数类型: auto(有GateNet则用GateNet否则softmax) / softmax / gatenet')
    parser.add_argument('--is_pr_auc', type=str_to_bool, default=False,
                        help='是否计算 PR 曲线（固定比例测试时，基于 backbone softmax 各类概率）')

    # 流式推理测试参数
    parser.add_argument('--load_mode', type=str, default='ratio', choices=['ratio', 'window'],
                        help='步进模式: ratio/window')
    parser.add_argument('--delta', type=float, default=0.03, help='ratio 模式步进增量')
    parser.add_argument('--win_step', type=int, default=6, help='window 模式步进步长')
    parser.add_argument('--offset_windows', type=str, default="0",
                        help='窗口偏移量，去除前 n 个窗口的原始数据再开始流式推理。默认 0=不偏移。每窗口长度=W_CONST')
    parser.add_argument('--checkconf_list', type=str, default="0.5,0.7,0.9,0.95",
                        help='逗号分隔的多个置信度阈值')
    parser.add_argument('--use_gatenet', type=str_to_bool, default=False,
                        help='是否使用 GateNet 门控')
    parser.add_argument('--gatenet_note', type=str, default='default',
                        help='GateNet 训练 notes')
    parser.add_argument('--eval_mode', type=str, default='all_fallback',
                        choices=['pass_only', 'all_fallback'],
                        help='评估方式: pass_only 仅通过样本 / all_fallback 全样本+回退')
    parser.add_argument('--threshold_search_note', type=str, default='default', help='阈值搜索结果标识，对应 threshold_search_<note>.json')
    parser.add_argument('--maximum_load_time', type=float, default=80, help='最大加载时间')
    parser.add_argument('--check_ratio', type=str, default='', help='加载率序列，支持中英文逗号或空格分隔，如 "10,20,30"')
    parser.add_argument('--stream_fast', type=str_to_bool, default=True,
                        help='是否使用快速批量 GPU 推理（默认 True，False 回退到逐步推理）')
    parser.add_argument('--stream_workers', type=int, default=None,
                        help='快速模式 CPU 预处理并行 worker 数（默认复用 num_workers）')
    parser.add_argument('--stream_parallel_backend', type=str,
                        default='thread' if os.name == 'nt' else 'process',
                        choices=['thread', 'process'],
                        help='快速模式并行后端: thread / process（Windows 默认 thread，Linux 默认 process）')
    parser.add_argument('--stream_process_start_method', type=str, default='auto',
                        choices=['auto', 'spawn', 'forkserver', 'fork'],
                        help='process 后端启动方式；auto 在 CUDA 推理时使用 spawn')
    parser.add_argument('--stream_worker_torch_threads', type=int, default=1,
                        help='process worker 内 torch CPU 线程数')
    parser.add_argument('--stream_prefetch', type=int, default=None,
                        help='快速模式预取任务数')

    # 通用参数
    parser.add_argument('--sample_num', type=int, default=-1, help='测试样本数 (-1=全部)')
    parser.add_argument('--test_flag', type=str_to_bool, default=True, help='是否测试模式')
    parser.add_argument('--num_workers', type=int, default=16, help='DataLoader workers')

    return parser


if __name__ == "__main__":
    main()
    print_colored("evaluate 全部运行结束", "green")
