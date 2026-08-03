# -*- coding: utf-8 -*-
"""
流式推理公共工具模块。
提供 CPU 端的输入构建函数，供 evaluate.py 和 threshold_search.py 共用。

导出:
    - _stream_worker_initializer: ProcessPoolExecutor worker 初始化函数
    - prepare_streaming_sample: CPU worker, 为单个样本构建所有截断步的输入
"""

import numpy as np


def _stream_worker_initializer(torch_num_threads=1):
    """Keep each process worker single-threaded to avoid CPU oversubscription."""
    try:
        import torch
        torch.set_num_threads(int(torch_num_threads))
    except Exception:
        pass


def apply_window_offset_to_trace(raw_data, offset_windows, w_const):
    """Drop the first offset windows while keeping padded rows as real zeros."""
    offset_windows = float(offset_windows)
    timestamp = np.trim_zeros(raw_data[:, 0], "b")
    if len(timestamp) == 0:
        return {
            "data": None,
            "timestamp": np.array([], dtype=raw_data.dtype),
            "empty": True,
            "original_max_trace_time": 0.0,
            "observed_start_time": 0.0,
            "offset_time": 0.0,
        }

    original_max_trace_time = float(timestamp[-1])
    offset_time = max(offset_windows, 0.0) * float(w_const)
    if offset_time >= original_max_trace_time:
        return {
            "data": None,
            "timestamp": np.array([], dtype=raw_data.dtype),
            "empty": True,
            "original_max_trace_time": original_max_trace_time,
            "observed_start_time": original_max_trace_time,
            "offset_time": offset_time,
        }

    valid_data = raw_data[:len(timestamp)]
    offset_idx = np.searchsorted(timestamp, offset_time, side="right") if offset_time > 0 else 0
    observed = valid_data[offset_idx:].copy()
    if len(observed) == 0:
        return {
            "data": None,
            "timestamp": np.array([], dtype=raw_data.dtype),
            "empty": True,
            "original_max_trace_time": original_max_trace_time,
            "observed_start_time": original_max_trace_time,
            "offset_time": offset_time,
        }

    observed_start_time = float(observed[0, 0]) if offset_idx > 0 else 0.0
    observed[:, 0] -= observed_start_time

    shifted = np.zeros_like(raw_data)
    shifted[:len(observed)] = observed
    shifted_timestamp = shifted[:len(observed), 0]

    return {
        "data": shifted,
        "timestamp": shifted_timestamp,
        "empty": False,
        "original_max_trace_time": original_max_trace_time,
        "observed_start_time": observed_start_time,
        "offset_time": offset_time,
    }


def _build_backbone_input(truncated, config, seq_len, max_matrix_len, log_transform,
                           get_TAM_fn, tam_args):
    """
    纯 CPU 构建 backbone 输入张量，不涉及 GPU 推理。
    将输入构建与 forward 分离，便于批量堆叠后单次 GPU 推理。
    
    返回 numpy.float32，避免进程模式下通过 PyTorch IPC 回传大 Tensor。
    
    Args:
        truncated: (N, 2) numpy array, 列0=时间戳, 列1=包大小/方向
        config: 模型配置 dict
        seq_len: 序列长度
        max_matrix_len: TAM 矩阵最大长度
        log_transform: 是否对 TAM 做 log1p 变换
        get_TAM_fn: TAM 生成函数 (get_TAM_ED1)
        tam_args: TAM 参数字典
    
    Returns:
        inp: numpy array, shape (1, 1, feature_dim, max_matrix_len)
    """
    from Prelude_main.Model.dataset import pad_sequence

    time_arr = truncated[:, 0]
    pkt_arr = truncated[:, 1]

    # Prelude: TAM 特征路径
    time_arr_pad = pad_sequence(time_arr, seq_len)
    pkt_arr_pad = pad_sequence(pkt_arr, seq_len)
    TAM, _, _ = get_TAM_fn(pkt_arr_pad, time_arr_pad, args=tam_args, bapm=None)
    TAM = TAM.reshape((1, -1, max_matrix_len)).astype(np.float32)
    if log_transform:
        TAM = np.log1p(TAM)
    inp = np.expand_dims(TAM, axis=0).astype(np.float32, copy=False)

    return inp


def prepare_streaming_sample(task):
    """
    CPU-only worker: build all truncated inputs for one sample.
    GPU inference stays in the parent process/thread so the model is not duplicated.
    
    Args:
        task: tuple of (
            num, raw_data, label, mode, delta, win_step, w_const,
            config, seq_len, max_matrix_len, log_transform,
            tam_args, use_gatenet, tam_type, offset_windows
        )
    
    Returns:
        dict with keys:
            num, label, max_trace_time, ratio_list, model_input, gate_input, empty
    """
    (num, raw_data, label, mode, delta, win_step, w_const,
     config, seq_len, max_matrix_len, log_transform,
     tam_args, use_gatenet, tam_type, offset_windows) = task

    offset_result = apply_window_offset_to_trace(raw_data, offset_windows, w_const)
    if offset_result["empty"]:
        return {
            "num": num,
            "label": int(label),
            "max_trace_time": 0.0,
            "original_max_trace_time": offset_result["original_max_trace_time"],
            "observed_start_time": offset_result["observed_start_time"],
            "ratio_list": [1.0],
            "model_input": None,
            "gate_input": None,
            "empty": True,
        }

    raw_data = offset_result["data"]
    timestamp = offset_result["timestamp"]

    max_trace_time = float(timestamp[-1])
    ratio_list = []
    truncated_list = []

    current_ratio = delta if mode == 'ratio' else 0.0
    win_index = 0

    while current_ratio <= 1.0 + 1e-8:
        if mode == 'window':
            win_index += win_step
            current_ratio = (win_index * w_const) / max(max_trace_time, 1e-8)

        actual_ratio = min(current_ratio, 1.0)
        threshold_time = actual_ratio * max_trace_time
        cutoff_idx = np.searchsorted(timestamp, threshold_time, side='right')

        truncated = raw_data.copy()
        truncated[cutoff_idx:, :] = 0.0

        ratio_list.append(float(actual_ratio))
        truncated_list.append(truncated)

        if actual_ratio >= 1.0:
            break
        if mode == 'ratio':
            current_ratio += delta

    from Prelude_main.Model.dataset import get_TAM_ED1

    get_TAM_fn = get_TAM_ED1
    backbone_inputs = [
        _build_backbone_input(t, config, seq_len, max_matrix_len,
                              log_transform, get_TAM_fn, tam_args)
        for t in truncated_list
    ]
    model_input = np.concatenate(backbone_inputs, axis=0).astype(np.float32, copy=False)

    gate_input = None
    if use_gatenet:
        gate_inputs = [np.asarray(t.T, dtype=np.float32)
                       for t in truncated_list]
        gate_input = np.stack(gate_inputs).astype(np.float32, copy=False)

    return {
        "num": num,
        "label": int(label),
        "max_trace_time": max_trace_time,
        "original_max_trace_time": offset_result["original_max_trace_time"],
        "observed_start_time": offset_result["observed_start_time"],
        "ratio_list": ratio_list,
        "model_input": model_input,
        "gate_input": gate_input,
        "empty": False,
    }
