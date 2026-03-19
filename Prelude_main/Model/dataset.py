import torch
from torch.utils.data import Dataset
import math
import numpy as np

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)
    return sequence


class CountDataset(Dataset):
    def __init__(self, X, labels, loaded_ratio=100, BAPM=None, is_idx=False, TAM_type='Mamba',
                 seq_len=5000,
                 maximum_cell_number=2,
                 max_matrix_len=1800,
                 log_transform=False,
                 maximum_load_time=80,
                 time_interval_threshold=0.1,
                 drop_extra_time = False,
                 ):
        self.X = X
        self.labels = labels
        self.loaded_ratio = loaded_ratio
        self.BAPM = BAPM
        self.is_idx = is_idx
        self.TAM = TAM_type
        self.drop_extra_time = drop_extra_time

        self.args = {
            "seq_len" : seq_len,
            "maximum_cell_number" : maximum_cell_number,
            "max_matrix_len" : max_matrix_len,
            "log_transform" : log_transform,
            "maximum_load_time" : maximum_load_time,
            "time_interval_threshold" : time_interval_threshold,
            "minimum_packet_number" : 10,
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index, load_ratio=None):
        data = self.X[index]
        label = self.labels[index]

        # 逻辑：如果输入了 load_ratio 就用输入的，否则用类自身的 self.loaded_ratio
        current_ratio = load_ratio if load_ratio is not None else self.loaded_ratio
        current_ratio = min(current_ratio, 100)
        timestamp = data[:, 0]
        loading_time = timestamp.max()

        if self.drop_extra_time:
            # 使用 current_ratio 计算阈值
            threshold = min(loading_time * current_ratio / 100, self.args['maximum_load_time'])
        else:
            threshold = loading_time * current_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)[0]
        data = data[valid_index, :]

        if self.BAPM is not None:
            bapm = self.BAPM[index]
            return self.process_data(data, bapm=bapm), label
        else:
            return self.process_data(data), label

    def process_data(self, data, bapm=None):
        time = data[:, 0]
        packet_length = data[:, 1]

        packet_length = pad_sequence(packet_length, self.args["seq_len"])
        time = pad_sequence(time, self.args["seq_len"])
        # get_TAM_Mamba get_TAM_TF
        TAM, current_index, bapm_labels = eval(f"get_TAM_{self.TAM}")(packet_length, time, args=self.args, bapm=bapm)
        TAM = TAM.reshape((1, -1, self.args["max_matrix_len"]))
        if self.args["log_transform"]:
            TAM = np.log1p(TAM)
        if self.is_idx:
            if bapm is not None:
                return TAM.astype(np.float32), current_index, bapm_labels
            else:
                return TAM.astype(np.float32), current_index
        else:
            if bapm is not None:
                return TAM.astype(np.float32), bapm_labels
            else:
                return TAM.astype(np.float32)


class CountDataset_RandomEarly(Dataset):
    def __init__(self, X, labels, loaded_ratio=100, BAPM=None, is_idx=False, TAM_type='Mamba',
                 seq_len=5000,
                 maximum_cell_number=2,
                 max_matrix_len=1800,
                 log_transform=False,
                 maximum_load_time=80,
                 time_interval_threshold=0.1,
                 drop_extra_time = False,
                 lb = 0.01,
                 ub = 0.5,
                 aug_num = 20,
                 ):
        self.X = X
        self.labels = labels
        self.loaded_ratio = loaded_ratio
        self.BAPM = BAPM
        self.is_idx = is_idx
        self.TAM = TAM_type
        self.drop_extra_time = drop_extra_time

        self.args = {
            "seq_len" : seq_len,
            "maximum_cell_number" : maximum_cell_number,
            "max_matrix_len" : max_matrix_len,
            "log_transform" : log_transform,
            "maximum_load_time" : maximum_load_time,
            "time_interval_threshold" : time_interval_threshold,
            "minimum_packet_number" : 10,
        }

        self.lb = lb
        self.ub = ub
        self.n_samples = aug_num
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x_output, y_output = [], []
        load_ratio_list = []
        for i in range(self.n_samples):
            load_ratio = self.lb + (self.ub - self.lb) * np.random.rand(1)
            x, y = self.__getitem_inner__(index, load_ratio*100)
            x_output.append(x[0])
            y_output.append(y)
            load_ratio_list.append(float(load_ratio*100))
        return np.stack(x_output,axis=0), np.stack(y_output,axis=0), np.array(load_ratio_list)
    def __getitem4__(self, index):
        x_output, y_output, load_ratio_list = [], [], []

        # 提前准备好参数，避免在提交时做太多计算
        ratios = [(self.lb + (self.ub - self.lb) * np.random.rand(1)) * 100 for _ in range(self.n_samples)]

        with ProcessPoolExecutor() as executor:
            # 提交任务
            future_to_ratio = {executor.submit(self.__getitem_inner__, index, r): r for r in ratios}

            for future in as_completed(future_to_ratio):
                ratio = future_to_ratio[future]
                x, y = future.result()
                x_output.append(x[0])
                y_output.append(y)
                load_ratio_list.append(float(ratio))

        return np.stack(x_output, axis=0), np.stack(y_output, axis=0), np.array(load_ratio_list)

    def __getitem3__(self, index):
        x_output, y_output, load_ratio_list = [], [], []

        with ThreadPoolExecutor() as executor:
            # 提交任务，并将 Future 对象与对应的 ratio 绑定
            future_to_ratio = {
                executor.submit(self.__getitem_inner__, index,
                                (self.lb + (self.ub - self.lb) * np.random.rand(1)) * 100):
                    (self.lb + (self.ub - self.lb) * np.random.rand(1)) * 100
                for _ in range(self.n_samples)
            }

            # 谁先完成就先处理谁
            for future in as_completed(future_to_ratio):
                ratio = future_to_ratio[future]
                try:
                    x, y = future.result()
                    x_output.append(x[0])
                    y_output.append(y)
                    load_ratio_list.append(float(ratio))
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        return np.stack(x_output, axis=0), np.stack(y_output, axis=0), np.array(load_ratio_list)

    def __getitem1__(self, index):
        x_output, y_output = [], []
        load_ratio_list = []

        # 准备随机生成的参数列表
        ratios = [self.lb + (self.ub - self.lb) * np.random.rand(1) for _ in range(self.n_samples)]

        # 使用线程池
        with ThreadPoolExecutor() as executor:
            # 使用 map 保持顺序，或者使用 submit 配合 list
            # 这里为了演示简洁使用 map，注意需要传参
            futures = [executor.submit(self.__getitem_inner__, index, r * 100) for r in ratios]

            for i, future in enumerate(futures):
                x, y = future.result()
                x_output.append(x[0])
                y_output.append(y)
                load_ratio_list.append(float(ratios[i] * 100))

        return np.stack(x_output, axis=0), np.stack(y_output, axis=0), np.array(load_ratio_list)

    def __getitem2__(self, index):
        x_output, y_output = [], []
        load_ratio_list = []
        # 预先生成 ratio 以便传递
        ratios = [(self.lb + (self.ub - self.lb) * np.random.rand(1)) * 100 for _ in range(self.n_samples)]

        # 使用进程池
        with ProcessPoolExecutor() as executor:
            # submit 会将任务分发到不同 CPU 核心
            futures = [executor.submit(self.__getitem_inner__, index, r) for r in ratios]

            for i, future in enumerate(futures):
                x, y = future.result()
                x_output.append(x[0])
                y_output.append(y)
                load_ratio_list.append(float(ratios[i]))

        return np.stack(x_output, axis=0), np.stack(y_output, axis=0), np.array(ratios)

    def __getitem_inner__(self, index, load_ratio=None):
        data = self.X[index]
        label = self.labels[index]

        # 逻辑：如果输入了 load_ratio 就用输入的，否则用类自身的 self.loaded_ratio
        current_ratio = load_ratio if load_ratio is not None else self.loaded_ratio

        timestamp = data[:, 0]
        loading_time = timestamp.max()

        if self.drop_extra_time:
            # 使用 current_ratio 计算阈值
            threshold = min(loading_time * current_ratio / 100, self.args['maximum_load_time'])
        else:
            threshold = loading_time * current_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)[0]
        data = data[valid_index, :]

        if self.BAPM is not None:
            bapm = self.BAPM[index]
            return self.process_data(data, bapm=bapm), label
        else:
            return self.process_data(data), label

    def process_data(self, data, bapm=None):
        time = data[:, 0]
        packet_length = data[:, 1]

        packet_length = pad_sequence(packet_length, self.args["seq_len"])
        time = pad_sequence(time, self.args["seq_len"])
        # get_TAM_Mamba get_TAM_TF
        TAM, current_index, bapm_labels = eval(f"get_TAM_{self.TAM}")(packet_length, time, args=self.args, bapm=bapm)
        TAM = TAM.reshape((1, -1, self.args["max_matrix_len"]))
        if self.args["log_transform"]:
            TAM = np.log1p(TAM)
        if self.is_idx:
            if bapm is not None:
                return TAM.astype(np.float32), current_index, bapm_labels
            else:
                return TAM.astype(np.float32), current_index
        else:
            if bapm is not None:
                return TAM.astype(np.float32), bapm_labels
            else:
                return TAM.astype(np.float32)


def get_TAM_RTA_seq(packet_length, time, args, bapm):
    max_matrix_len = args["max_matrix_len"]
    sequence = np.sign(packet_length) * time
    maximum_load_time = args["maximum_load_time"]

    # 4行：
    # 0: 上行频数
    # 1: 下行频数
    # 2: 上行数据包大小之和
    # 3: 下行数据包大小之和
    feature = np.zeros((4, max_matrix_len))

    count = 0
    for i, pack in enumerate(sequence):
        count += 1
        if pack == 0:
            count -= 1
            break

        pkt_size = abs(packet_length[i])  # 当前包大小

        if pack > 0:  # 上行
            if pack >= maximum_load_time:
                idx = -1
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)

            feature[0, idx] += 1           # 频数
            feature[2, idx] += pkt_size   # 大小之和

        else:  # 下行
            pack = abs(pack)
            if pack >= maximum_load_time:
                idx = -1
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)

            feature[1, idx] += 1           # 频数
            feature[3, idx] += pkt_size   # 大小之和

    return feature, count, None

def get_TAM_RF(packet_length, time, args, bapm):
    max_matrix_len = args["max_matrix_len"]
    sequence = np.sign(packet_length) * time
    maximum_load_time = args["maximum_load_time"]

    feature = np.zeros((2, max_matrix_len))  # Initialize feature matrix

    count = 0
    for pack in sequence:
        count += 1
        if pack == 0:
            count -= 1
            break  # End of sequence
        elif pack > 0:
            if pack >= maximum_load_time:
                feature[0, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[0, idx] += 1
        else:
            pack = np.abs(pack)
            if pack >= maximum_load_time:
                feature[1, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[1, idx] += 1
    return feature, count, None


def get_TAM_Mamba(packet_length, time, args, bapm):
    # 统计窗口长度 返回datalen
    args['feature_dim'] = 2 * (args["maximum_cell_number"] + 2)
    feature = np.zeros((args['feature_dim'], args["max_matrix_len"]))
    w = args["maximum_load_time"] / args["max_matrix_len"]
    time_interval = w * args["time_interval_threshold"]

    if bapm is not None:
        bapm_time = np.trim_zeros(time, "b")

        bapm_first_start = min(len(bapm_time) - 1, bapm[0])
        bapm_first_end = min(len(bapm_time) - 1, bapm[0] + bapm[1])
        bapm_second_start = min(len(bapm_time) - 1, bapm[2])
        bapm_second_end = min(len(bapm_time) - 1, bapm[2] + bapm[3])

        bapm_first_start_time = bapm_time[bapm_first_start]
        bapm_first_end_time = bapm_time[bapm_first_end]
        bapm_second_start_time = bapm_time[bapm_second_start]
        bapm_second_end_time = bapm_time[bapm_second_end]

        bapm_first_start_position = min(math.floor(bapm_first_start_time / w), args["max_matrix_len"] - 1)
        bapm_first_end_position = min(math.floor(bapm_first_end_time / w), args["max_matrix_len"] - 1)
        bapm_second_start_position = min(math.floor(bapm_second_start_time / w), args["max_matrix_len"] - 1)
        bapm_second_end_position = min(math.floor(bapm_second_end_time / w), args["max_matrix_len"] - 1)

        bapm_labels = np.full((2, args["max_matrix_len"]), -1)
        bapm_labels[0, bapm_first_start_position:bapm_first_end_position + 1] = bapm[-2]
        bapm_labels[1, bapm_second_start_position:bapm_second_end_position + 1] = bapm[-1]
    else:
        bapm_labels = None

    current_index = 0
    current_timestamps = []
    data_len = []
    for l_k, t_k in zip(packet_length, time):
        if t_k == 0 and l_k == 0:
            break  # End of sequence

        d_k = int(np.sign(l_k))
        c_k = min(int(np.abs(l_k) // 512), args["maximum_cell_number"])  # [0, C]

        fragment = 0 if d_k < 0 else 1
        i = 2 * c_k + fragment  # [0, 2C + 1]
        j = min(math.floor(t_k / w), args["max_matrix_len"] - 1)
        j = max(j, 0)
        feature[i, j] += 1

        if j != current_index:
            feature[2 * args["maximum_cell_number"] + 2, j] = max(j - current_index, 0)
            data_len.append(len(current_timestamps))
            delta_t = np.diff(current_timestamps)
            cluster_count = np.sum(delta_t > time_interval) + 1
            feature[2 * args["maximum_cell_number"] + 3, current_index] = cluster_count



            current_index = j
            current_timestamps = [t_k]
        else:
            current_timestamps.append(t_k)

    delta_t = np.diff(current_timestamps)
    data_len.append(len(current_timestamps))
    cluster_count = np.sum(delta_t > time_interval) + 1
    feature[2 * args["maximum_cell_number"] + 3, current_index] = cluster_count

    return feature, current_index, bapm_labels

# def get_TAM_Mamba1(packet_length, time, args, bapm):
#     # 原始的mamba 返回current_index
#     args['feature_dim'] = 2 * (args["maximum_cell_number"] + 2)
#     feature = np.zeros((args['feature_dim'], args["max_matrix_len"]))
#     w = args["maximum_load_time"] / args["max_matrix_len"]
#     time_interval = w * args["time_interval_threshold"]
#
#     if bapm is not None:
#         bapm_time = np.trim_zeros(time, "b")
#
#         bapm_first_start = min(len(bapm_time) - 1, bapm[0])
#         bapm_first_end = min(len(bapm_time) - 1, bapm[0] + bapm[1])
#         bapm_second_start = min(len(bapm_time) - 1, bapm[2])
#         bapm_second_end = min(len(bapm_time) - 1, bapm[2] + bapm[3])
#
#         bapm_first_start_time = bapm_time[bapm_first_start]
#         bapm_first_end_time = bapm_time[bapm_first_end]
#         bapm_second_start_time = bapm_time[bapm_second_start]
#         bapm_second_end_time = bapm_time[bapm_second_end]
#
#         bapm_first_start_position = min(math.floor(bapm_first_start_time / w), args["max_matrix_len"] - 1)
#         bapm_first_end_position = min(math.floor(bapm_first_end_time / w), args["max_matrix_len"] - 1)
#         bapm_second_start_position = min(math.floor(bapm_second_start_time / w), args["max_matrix_len"] - 1)
#         bapm_second_end_position = min(math.floor(bapm_second_end_time / w), args["max_matrix_len"] - 1)
#
#         bapm_labels = np.full((2, args["max_matrix_len"]), -1)
#         bapm_labels[0, bapm_first_start_position:bapm_first_end_position + 1] = bapm[-2]
#         bapm_labels[1, bapm_second_start_position:bapm_second_end_position + 1] = bapm[-1]
#     else:
#         bapm_labels = None
#
#     current_index = 0
#     current_timestamps = []
#     for l_k, t_k in zip(packet_length, time):
#         if t_k == 0 and l_k == 0:
#             break  # End of sequence
#
#         d_k = int(np.sign(l_k))
#         c_k = min(int(np.abs(l_k) // 512), args["maximum_cell_number"])  # [0, C]
#
#         fragment = 0 if d_k < 0 else 1
#         i = 2 * c_k + fragment  # [0, 2C + 1]
#         j = min(math.floor(t_k / w), args["max_matrix_len"] - 1)
#         j = max(j, 0)
#         feature[i, j] += 1
#
#         if j != current_index:
#             feature[2 * args["maximum_cell_number"] + 2, j] = max(j - current_index, 0)
#
#             delta_t = np.diff(current_timestamps)
#             cluster_count = np.sum(delta_t > time_interval) + 1
#             feature[2 * args["maximum_cell_number"] + 3, current_index] = cluster_count
#
#             current_index = j
#             current_timestamps = [t_k]
#         else:
#             current_timestamps.append(t_k)
#
#     delta_t = np.diff(current_timestamps)
#     cluster_count = np.sum(delta_t > time_interval) + 1
#     feature[2 * args["maximum_cell_number"] + 3, current_index] = cluster_count
#
#     return feature, current_index, bapm_labels

def get_TAM_ED1(packet_length, time, args, bapm):
    # 加上窗口内包大小特征共5个
    feature = extract_features_5f(time, packet_length, args["maximum_load_time"], args["max_matrix_len"])
    data_len = get_actual_length(feature)
    return feature,data_len, None

# def get_TAM_ED1(packet_length, time, args, bapm):
#     # 加上窗口内包大小特征共5个
#     feature = extract_features_5f(time, packet_length, args["maximum_load_time"], args["max_matrix_len"])
#     data_len = get_actual_length(feature)
#     return feature,data_len, None
#
# def get_TAM_ED2(packet_length, time, args, bapm):
#     # 加上窗口内包大小特征共4个(没有包关联)
#     feature = extract_features_4f(time, packet_length, args["maximum_load_time"], args["max_matrix_len"])
#     data_len = get_actual_length(feature)
#     return feature,data_len, None
#
# def get_TAM_ED5(packet_length, time, args, bapm):
#     # 加上窗口内包大小特征共4个(没有包关联)
#     feature = extract_features_rf(time, packet_length, args["maximum_load_time"], args["max_matrix_len"])
#     data_len = get_actual_length(feature)
#     return feature,data_len, None
#
# def get_TAM_ED3(packet_length, time, args, bapm):
#     # 在ED1的5个特征的基础上，添加了burst和avg burst特征
#     feature = extract_features_9f(time, packet_length, args["maximum_load_time"], args["max_matrix_len"])
#     data_len = get_actual_length(feature)
#     return feature,data_len, None
#
# def get_TAM_ED4(packet_length, time, args, bapm):
#     # 统计窗口长度 返回datalen
#     w = args["maximum_load_time"] / args["max_matrix_len"]
#     time_interval = w * args["time_interval_threshold"]
#
#     feature = np.zeros((9, args["max_matrix_len"]))
#     current_index = 0
#     current_timestamps = []
#     current_packs = []
#     for l_k, t_k in zip(packet_length, time):
#         if t_k == 0 and l_k == 0:
#             break  # End of sequence
#
#         j = min(math.floor(t_k / w), args["max_matrix_len"] - 1)
#         j = max(j, 0)
#
#         if j != current_index:
#             local_feat = agg_interval(current_packs)
#             local_feat.append(max(j - current_index, 0))
#             feature[:, current_index] = np.array(local_feat, dtype=np.float32)
#             current_index = j
#             current_timestamps = [t_k]
#             current_packs = [l_k]
#         else:
#             current_timestamps.append(t_k)
#             current_packs.append(l_k)
#     if len(current_packs) != 0:
#         local_feat = agg_interval(current_packs)
#         local_feat.append(max(j - current_index, 0))
#         feature[:, current_index] = np.array(local_feat, dtype=np.float32)
#
#     return feature, current_index, None
#






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
    if isinstance(packets, list):
        packets = np.array(packets, dtype=np.float32)
    features = []
    features += [np.sum(packets[packets > 0]), np.sum(packets[packets < 0])]
    features += [np.sum(packets > 0), np.sum(packets < 0)]

    dirs = np.sign(packets)
    assert not np.any(dir == 0), "Array contains zero!"
    bursts = fast_count_burst(dirs)
    features += [np.sum(bursts > 0), np.sum(bursts < 0)]

    pos_bursts = bursts[bursts > 0]
    neg_bursts = np.abs(bursts[bursts < 0])
    vals = []
    if len(pos_bursts) == 0:
        vals.append(0)
    else:
        vals.append(np.mean(pos_bursts))
    if len(neg_bursts) == 0:
        vals.append(0)
    else:
        vals.append(np.mean(neg_bursts))
    features += vals
    return features




def get_actual_length(feature):
    """
    计算feature的实际时序长度
    feature: 矩阵，每一列是一个时序点
    返回：实际有效时序长度
    """
    # 检查每一列是否全为零
    # 如果某一列不全为零，返回True
    non_zero_cols = np.any(feature != 0, axis=0)

    # 找到最后一个不全为零的列的索引
    # np.where返回满足条件的索引，取最后一个
    non_zero_indices = np.where(non_zero_cols)[0]

    if len(non_zero_indices) == 0:
        return 0  # 全部都是零
    else:
        # 索引从0开始，长度需要+1
        return non_zero_indices[-1] + 1

def extract_features_5f(T, L, Tmax, max_column):
    """
    优化版本特征提取函数，保留五个特征：
    0: 上行包数量
    1: 下行包数量
    2: 包间隔（窗口索引差）
    3: 上行包大小和（除以512）
    4: 下行包大小和（除以512）
    """
    # 初始化特征矩阵 (5行 x max_column列)
    feature = np.zeros((5, max_column))

    # 截断T==0之后的数据
    indices = np.flatnonzero(T)
    ind = indices[-1]+1 if indices.size > 0 else len(T)
    T = T[:ind]
    L = L[:ind]

    # 无有效数据时返回空特征
    if len(T) == 0:
        return feature

    # 计算所有窗口索引
    all_windows = np.floor(T / Tmax * (max_column - 1)).astype(int)
    all_windows = np.clip(all_windows, 0, max_column - 1)

    # 特征0&1: 分别计算上行/下行包数量
    up_mask = L > 0
    down_mask = L < 0
    if np.any(up_mask):
        feature[0] = np.bincount(all_windows[up_mask], minlength=max_column)
    if np.any(down_mask):
        feature[1] = np.bincount(all_windows[down_mask], minlength=max_column)

    # 特征3&4: 分别计算上行/下行包大小和（除以512）
    if np.any(up_mask):
        # 上行包大小和：直接使用L[up_mask]（正数），然后除以512
        up_size_sum = np.bincount(all_windows[up_mask], weights=L[up_mask], minlength=max_column)
        feature[3] = up_size_sum / 512.0  # 使用浮点除法避免整数截断
    if np.any(down_mask):
        # 下行包大小和：取-L[down_mask]（绝对值），然后除以512
        down_size_sum = np.bincount(all_windows[down_mask], weights=-L[down_mask], minlength=max_column)
        feature[4] = down_size_sum / 512.0

    # 特征2: 包间隔（窗口索引差）
    unique_windows = np.unique(all_windows)
    if unique_windows.size > 1:
        window_gaps = np.diff(unique_windows)
        feature[2, unique_windows[1:]] = window_gaps

    return np.log(1+feature)

#
# def extract_features_4f(T, L, Tmax, max_column):
#     """
#     修改后的特征提取函数，保留四个特征：
#     0: 上行包数量
#     1: 下行包数量
#     2: 上行包大小和（除以512）
#     3: 下行包大小和（除以512）
#     """
#     # 初始化特征矩阵 (4行 x max_column列)
#     feature = np.zeros((4, max_column))
#
#     # 截断T==0之后的数据
#     indices = np.flatnonzero(T)
#     ind = indices[-1] + 1 if indices.size > 0 else len(T)
#     T = T[:ind]
#     L = L[:ind]
#
#     # 无有效数据时返回全零特征
#     if len(T) == 0:
#         return feature
#
#     # 计算所有窗口索引
#     all_windows = np.floor(T / Tmax * (max_column - 1)).astype(int)
#     all_windows = np.clip(all_windows, 0, max_column - 1)
#
#     # 区分上下行掩码
#     up_mask = L > 0
#     down_mask = L < 0
#
#     # 特征0 & 1: 上行/下行包数量统计
#     if np.any(up_mask):
#         feature[0] = np.bincount(all_windows[up_mask], minlength=max_column)
#     if np.any(down_mask):
#         feature[1] = np.bincount(all_windows[down_mask], minlength=max_column)
#
#     # 特征2 & 3: 上行/下行包大小和（除以512）
#     if np.any(up_mask):
#         # 权重设为包大小，bincount 自动求和
#         up_size_sum = np.bincount(all_windows[up_mask], weights=L[up_mask], minlength=max_column)
#         feature[2] = up_size_sum / 512.0
#
#     if np.any(down_mask):
#         # 下行包大小取绝对值进行求和
#         down_size_sum = np.bincount(all_windows[down_mask], weights=-L[down_mask], minlength=max_column)
#         feature[3] = down_size_sum / 512.0
#
#     # 返回 log 缩放后的特征
#     return np.log1p(feature)
#
# def extract_features_rf(T, L, Tmax, max_column):
#     """
#     修改后的特征提取函数，保留四个特征：
#     0: 上行包数量
#     1: 下行包数量
#     2: 上行包大小和（除以512）
#     3: 下行包大小和（除以512）
#     """
#     # 初始化特征矩阵 (4行 x max_column列)
#     feature = np.zeros((2, max_column))
#
#     # 截断T==0之后的数据
#     indices = np.flatnonzero(T)
#     ind = indices[-1] + 1 if indices.size > 0 else len(T)
#     T = T[:ind]
#     L = L[:ind]
#
#     # 无有效数据时返回全零特征
#     if len(T) == 0:
#         return feature
#
#     # 计算所有窗口索引
#     all_windows = np.floor(T / Tmax * (max_column - 1)).astype(int)
#     all_windows = np.clip(all_windows, 0, max_column - 1)
#
#     # 区分上下行掩码
#     up_mask = L > 0
#     down_mask = L < 0
#
#     # 特征0 & 1: 上行/下行包数量统计
#     if np.any(up_mask):
#         feature[0] = np.bincount(all_windows[up_mask], minlength=max_column)
#     if np.any(down_mask):
#         feature[1] = np.bincount(all_windows[down_mask], minlength=max_column)
#
#     return np.log1p(feature)
#
# def extract_features_9f(T, L, Tmax, max_column):
#     feature = np.zeros((9, max_column))
#
#     # --- 1. 数据预处理 ---
#     indices = np.flatnonzero(T)
#     ind = indices[-1] + 1 if indices.size > 0 else len(T)
#     T, L = T[:ind], L[:ind]
#     if len(T) == 0: return feature
#
#     # 计算窗口索引 (0 到 max_column-1)
#     all_windows = np.floor(T / Tmax * max_column).astype(int)
#     all_windows = np.clip(all_windows, 0, max_column - 1)
#
#     # --- 2. 基础特征 (0, 1, 3, 4) ---
#     up_mask = L > 0
#     down_mask = L < 0
#
#     if np.any(up_mask):
#         feature[0] = np.bincount(all_windows[up_mask], minlength=max_column)
#         feature[3] = np.bincount(all_windows[up_mask], weights=L[up_mask], minlength=max_column) / 512.0
#
#     if np.any(down_mask):
#         feature[1] = np.bincount(all_windows[down_mask], minlength=max_column)
#         feature[4] = np.bincount(all_windows[down_mask], weights=-L[down_mask], minlength=max_column) / 512.0
#
#     # --- 3. 突发特征计算 (核心) ---
#     direction = np.where(L > 0, 1, -1)
#     # diff_dir != 0 表示方向发生了切换
#     diff_dir = np.concatenate(([1], np.diff(direction)))
#
#     up_burst_starts = (direction == 1) & (diff_dir != 0)
#     down_burst_starts = (direction == -1) & (diff_dir != 0)
#
#     if np.any(up_burst_starts):
#         feature[5] = np.bincount(all_windows[up_burst_starts], minlength=max_column)
#     if np.any(down_burst_starts):
#         feature[6] = np.bincount(all_windows[down_burst_starts], minlength=max_column)
#
#     # 特征 7 & 8: 平均突发长度 (Total Packets / Burst Count)
#     feature[7] = np.divide(feature[0], feature[5], out=np.zeros_like(feature[0]), where=feature[5] != 0)
#     feature[8] = np.divide(feature[1], feature[6], out=np.zeros_like(feature[1]), where=feature[6] != 0)
#
#     # --- 4. 间隔特征 ---
#     unique_windows = np.unique(all_windows)
#     if unique_windows.size > 1:
#         feature[2, unique_windows[1:]] = np.diff(unique_windows)
#
#     # 注意：为了方便测试观察，这里先不加 np.log(1+feature)，直接返回原始值
#     return np.log(1+feature)
#
# test_num = 6
# from Prelude_env.ExploreRun.const import get_filebase_dir
# if __name__ == '__main__':
#
#
#     def load_data(data_path, drop_extra_time=False, load_time=None):
#         data = np.load(data_path)
#         X = data["X"]
#         y = data["y"]
#         # 时间负数调整
#         X[:, :, 0] = np.abs(X[:, :, 0])
#         if drop_extra_time and load_time is not None:
#             print(f"丢弃额外时间，时间上限：{load_time}")
#             invalid_ind = X[:, :, 0] > load_time
#             X[invalid_ind, :] = 0
#         return X, y
#     if test_num == 1:
#         import os
#         dataset = "CW"
#         base_dir = f"/root/autodl-tmp/dataset/wfa/npz_dataset/{dataset}"  # CW trafficsilver_bwr_CW Closed_5tab regulator_Closed_2tab
#         data_path = os.path.join(base_dir, "valid.npz")
#         X, y = load_data(data_path,drop_extra_time=False,load_time=80)
#         dataset = CountDataset(X, y, loaded_ratio=100, TAM_type="Mamba",
#                                is_idx=True,
#                                maximum_load_time=80,
#                                drop_extra_time=True,
#                                max_matrix_len=1800,
#                                seq_len=5000)
#         x, y = dataset[0]
#     elif test_num == 2:
#         import time
#         import os
#
#         def test_process(TAM_type, load_ratio=100, N=500, max_matrix_len=1800, seq_len=5000):
#             dataset = CountDataset(X, y, loaded_ratio=load_ratio, TAM_type=TAM_type,
#                                    is_idx=True,
#                                    maximum_load_time=120,
#                                    drop_extra_time=True,
#                                    max_matrix_len=max_matrix_len,
#                                    seq_len=seq_len)
#             tic = time.time()
#             if N == -1:
#                 N = len(dataset)
#             for i in range(N):
#                 dataset[i]
#             toc = time.time()
#             time_duration = round(toc - tic,2)
#             print(f'dataset shape : {dataset[i][0][0].shape}')
#             print(f"{TAM_type}({load_ratio}%) - 运行时间:{time_duration:.2f}s\n")
#             return dataset, time_duration
#
#
#         for dataset in ['CW']:# , 'Closed_3tab', 'Closed_4tab', 'Closed_5tab'
#             base_dir = f"/root/lixianjun/dataset/wfa/npz_dataset/{dataset}"  # CW trafficsilver_bwr_CW Closed_5tab regulator_Closed_2tab
#             data_path = os.path.join(base_dir, "valid.npz")
#             X, y = load_data(data_path,drop_extra_time=False,load_time=80)
#             # 时间负数调整
#             X[:, :, 0] = np.abs(X[:, :, 0])
#             n = 1000
#             seq_len = 10000
#             test_process(TAM_type='ED1', load_ratio=100, N=n, seq_len=seq_len)
#             test_process(TAM_type='RF', load_ratio=100, N=n, seq_len=seq_len)
#             test_process(TAM_type='Mamba', load_ratio=100, N=n, seq_len=seq_len)
#     elif test_num == 3:
#         import os
#         for dataset in ['CW', 'OW', 'Closed_5tab']:
#             base_dir = f"/root/lixianjun/dataset/wfa/npz_dataset/{dataset}"
#             data_path = os.path.join(base_dir, "train.npz")
#             X, y = load_data(data_path,drop_extra_time=False,load_time=80)
#             print(f"{dataset} -- {X.shape}")
#     elif test_num == 4:
#         import time
#         import os
#
#         def test_process(TAM_type, load_ratio=100, N=500, max_matrix_len=1800, seq_len=5000):
#             dataset = CountDataset(X, y, loaded_ratio=load_ratio, TAM_type=TAM_type,
#                                    is_idx=True,
#                                    maximum_load_time=120,
#                                    drop_extra_time=True,
#                                    max_matrix_len=max_matrix_len,
#                                    seq_len=seq_len)
#             tic = time.time()
#             if N == -1:
#                 N = len(dataset)
#             for i in range(N):
#                 dataset[i]
#             toc = time.time()
#             time_duration = round(toc - tic, 2)
#             print(f'dataset shape : {dataset[i][0][0].shape}')
#             print(f"{TAM_type}({load_ratio}%) - 运行时间:{time_duration:.2f}s\n")
#             return dataset, time_duration
#
#
#         for dataset in ['CW']:  # , 'Closed_3tab', 'Closed_4tab', 'Closed_5tab'
#             base_dir = os.path.join(get_filebase_dir(), dataset)
#             data_path = os.path.join(base_dir, "valid.npz")
#             X, y = load_data(data_path, drop_extra_time=False, load_time=80)
#             # 时间负数调整
#             X[:, :, 0] = np.abs(X[:, :, 0])
#             n = 1000
#             seq_len = 10000
#             test_process(TAM_type='ED4', load_ratio=100, N=n, seq_len=seq_len)
#     elif test_num == 5:
#         import os
#
#         base_dir = os.path.join(get_filebase_dir(), "CW")
#         data_path = os.path.join(base_dir, "valid.npz")
#         X, y = load_data(data_path, drop_extra_time=False, load_time=80)
#         dataset = CountDataset_RandomEarly(X, y, loaded_ratio=100, TAM_type="ED1",
#                                is_idx=True,
#                                maximum_load_time=120,
#                                drop_extra_time=True,
#                                max_matrix_len=1800,
#                                seq_len=5000)
#         x, y = dataset[0]
#         print(x.shape)
#     elif test_num == 6:
#         # 测试多种数据提取方式对应的latency
#         import os
#         import time
#         from tqdm import tqdm
#         base_dir = os.path.join(get_filebase_dir(), "regulator_CW")
#         data_path = os.path.join(base_dir, "test.npz")
#         X, y = load_data(data_path, drop_extra_time=False, load_time=80)
#         for loaded_ratio in [10,20,30]:
#             for TAM_type in ['ED5', 'ED2', 'Mamba']:# ['RF', 'RTA', 'Mamba']:# ['ED5', 'ED2', 'Mamba']
#                 dataset = CountDataset(X, y, loaded_ratio=loaded_ratio, TAM_type=TAM_type,
#                                                    is_idx=True,
#                                                    maximum_load_time=120,
#                                                    drop_extra_time=True,
#                                                    max_matrix_len=1800,
#                                                    seq_len=10000)
#                 tic = time.time()
#                 n_sample = len(dataset)# 1000# len(dataset)
#                 for i in tqdm(range(n_sample)):
#                     dataset[i]
#                 toc = time.time()
#                 time_duration = round((toc - tic)/n_sample*1000, 2)
#                 print(f"{TAM_type}({loaded_ratio}%) - 运行时间:{time_duration:.2f} ms\n")