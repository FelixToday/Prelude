import numpy as np
from torch.utils.data import Dataset

def pad_along_axis(array: np.ndarray,
                               target_length: int,
                               axis: int = 0,
                               pad_value: float = 0.0) -> np.ndarray:
    """
    沿指定维度padding或截断数组

    Args:
        array: 输入numpy数组
        target_length: 目标长度
        axis: 需要操作的维度 (默认0)
        pad_value: 填充值 (默认0.0)

    Returns:
        调整长度后的数组
    """
    current_length = array.shape[axis]

    if current_length < target_length:
        # 需要padding的情况
        pad_size = target_length - current_length
        pad_width = [(0, 0)] * array.ndim  # 初始化所有维度不padding
        pad_width[axis] = (0, pad_size)  # 只padding指定维度
        return np.pad(array, pad_width=pad_width, mode='constant', constant_values=pad_value)
    elif current_length > target_length:
        # 需要截断的情况
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(0, target_length)
        return array[tuple(slices)]
    else:
        # 长度正好，直接返回
        return array

def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)

    return sequence

def process_TAM(sequence, maximum_load_time, max_matrix_len):
    feature = np.zeros((2, max_matrix_len))  # Initialize feature matrix

    for pack in sequence:
        if pack == 0:
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
    return feature

class RFDataset(Dataset):
    def __init__(self, X, Y, length, loaded_ratio=100):
        self.X = X
        self.Y = Y
        self.length = length
        self.loaded_ratio = loaded_ratio

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.Y[index]

        timestamp = data[:, 0]
        sign = np.sign(data[:, 1])
        X = timestamp * sign

        loading_time = timestamp.max()
        threshold = loading_time * self.loaded_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)

        X = X[valid_index]

        X = pad_sequence(X, self.length)

        return self.process_data(X), label

    def process_data(self, data):
        TAM = process_TAM(data, maximum_load_time=80, max_matrix_len=1800)
        TAM = TAM.reshape(1, 2, 1800)
        return TAM.astype(np.float32)


class DT2Dataset(Dataset):
    def __init__(self, X, Y, length, loaded_ratio=100):
        self.X = X
        self.Y = Y
        self.length = length
        self.loaded_ratio = loaded_ratio

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.Y[index]

        timestamp = data[:, 0]
        sign = np.sign(data[:, 1])

        loading_time = timestamp.max()
        threshold = loading_time * self.loaded_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)

        timestamp = timestamp[valid_index]
        sign = sign[valid_index]

        timestamp = pad_sequence(timestamp, self.length)
        sign = pad_sequence(sign, self.length)

        return self.process_data(timestamp, sign), label

    def process_data(self, X_time, X_dir):
        X_time = np.diff(X_time)
        X_time[X_time < 0] = 0
        X_time = np.insert(X_time, 0, 0)

        X_dir = X_dir.reshape(1, -1)
        X_time = X_time.reshape(1, -1)
        data = np.concatenate([X_dir, X_time], axis=0)

        return data.astype(np.float32)


class DTDataset(Dataset):
    def __init__(self, X, Y, length, loaded_ratio=100):
        self.X = X
        self.Y = Y
        self.length = length
        self.loaded_ratio = loaded_ratio

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.Y[index]

        timestamp = data[:, 0]
        sign = np.sign(data[:, 1])
        dt = timestamp * sign

        loading_time = timestamp.max()
        threshold = loading_time * self.loaded_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)

        dt = dt[valid_index]

        return self.process_data(dt), label

    def process_data(self, dt):
        dt = pad_sequence(dt, self.length)

        return dt.reshape(1, -1).astype(np.float32)


class DirectionDataset(Dataset):
    def __init__(self, X, Y, length, loaded_ratio=100):
        self.X = X
        self.Y = Y
        self.length = length
        self.loaded_ratio = loaded_ratio

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.Y[index]

        timestamp = data[:, 0]
        sign = np.sign(data[:, 1])

        loading_time = timestamp.max()
        threshold = loading_time * self.loaded_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)

        sign = sign[valid_index]

        return self.process_data(sign), label

    def process_data(self, direction):
        direction = pad_sequence(direction, self.length)

        return direction.reshape(1, -1).astype(np.float32)

def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]
    return X, y

if __name__ == "__main__":
    from MyModel import CountDataset, default_args

    import os

    default_args['max_matrix_len'] = 1500

    valid_X, valid_y = load_data(os.path.join("../npz_dataset/CW", f"train.npz"))

    val_set1 = CountDataset(valid_X, valid_y, args=default_args, is_idx=False, TAM='Mamba')
    val_set2 = CountDataset(valid_X, valid_y, args=default_args, is_idx=False, TAM='TF')
    val_set3 = FFTDataset(valid_X, valid_y, 5000, L_max=1500)

    import time
    n=1000
    tic = time.time()
    for i in range(n):
        val_set1[i][0]
    toc = time.time()
    print(f"运行时间：{toc - tic}")
    tic = time.time()
    for i in range(n):
        val_set2[i][0]
    toc = time.time()
    print(f"运行时间：{toc - tic}")
    # tic = time.time()
    # for i in range(n):
    #     val_set3[i][0]
    # toc = time.time()
    # print(f"运行时间：{toc - tic}")
# if __name__ == "__main__":
#     import os
#     valid_X, valid_y = load_data(os.path.join("../npz_dataset/CW", f"train.npz"))
#     #val_set = RFDataset(valid_X, valid_y, 5000)
#     val_set = FFTDataset(valid_X, valid_y, 5000)
#
#     tamaraw_CW_valid_X, tamaraw_CW_valid_y = load_data(os.path.join("../npz_dataset/tamaraw_CW", f"train.npz"))
#     # val_set = RFDataset(valid_X, valid_y, 5000)
#     tamaraw_CW_val_set = FFTDataset(tamaraw_CW_valid_X, tamaraw_CW_valid_y, 5000)
#
#     from lxj_utils_user import IncrementalMeanCalculator
#     ca = IncrementalMeanCalculator()
#     cb = IncrementalMeanCalculator()
#     import time
#     tic=time.time()
#     for i in range(100):
#         ca.add(val_set[i][0][1])
#
#     toc = time.time()
#     print(f"运行时间：{toc-tic}, 平均长度{ca.get()}")
#
#     tic = time.time()
#     for i in range(100):
#         cb.add(tamaraw_CW_val_set[i][0][1])
#     toc = time.time()
#     print(f"运行时间：{toc - tic}, 平均长度{cb.get()}")