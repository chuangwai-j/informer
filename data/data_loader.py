# data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime


class AircraftTrajectoryDataset(Dataset):
    def __init__(self, data_path, seq_len=96, label_len=48, pred_len=24, mode='train', scale=True):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.mode = mode

        # 加载数据
        self.data = np.load(data_path, allow_pickle=True)
        self.lengths = np.load(data_path.replace('.npy', '_lengths.npy'), allow_pickle=True)

        # 划分训练集、验证集、测试集
        self._split_data(mode)

        # 过滤掉长度不足的序列
        self._filter_short_sequences()

        # 数据标准化
        if scale:
            self._standardize()

        print(f"{mode}数据集: {len(self.used_data)} 个序列")

    def _split_data(self, mode):
        """划分数据集"""
        total_samples = len(self.data)
        train_ratio, val_ratio = 0.7, 0.2

        if mode == 'train':
            self.used_data = self.data[:int(total_samples * train_ratio)]
        elif mode == 'val':
            self.used_data = self.data[int(total_samples * train_ratio):int(total_samples * (train_ratio + val_ratio))]
        else:  # test
            self.used_data = self.data[int(total_samples * (train_ratio + val_ratio)):]

    def _filter_short_sequences(self):
        """过滤掉长度不足 seq_len + pred_len 的序列"""
        min_required_length = self.seq_len + self.pred_len
        filtered_data = []

        for trajectory in self.used_data:
            if len(trajectory) >= min_required_length:
                filtered_data.append(trajectory)

        self.used_data = filtered_data

        if len(self.used_data) == 0:
            raise ValueError(f"没有足够长的序列。需要至少 {min_required_length} 个时间步，但所有序列都更短。")

    def _standardize(self):
        """数据标准化"""
        # 收集所有数据计算均值和标准差
        all_data = []
        for trajectory in self.used_data:
            all_data.append(trajectory[:, 1:])  # 排除时间列

        all_data = np.vstack(all_data)
        self.data_mean = np.mean(all_data, axis=0)
        self.data_std = np.std(all_data, axis=0)

        # 避免除零
        self.data_std[self.data_std == 0] = 1.0

        print(f"数据均值: {self.data_mean}")
        print(f"数据标准差: {self.data_std}")

    def _extract_relative_time_features(self, timestamps):
        """使用相对时间特征"""
        if len(timestamps) == 0:
            return np.zeros((0, 4))  # 改为4

        # 计算相对时间
        min_time = timestamps.min()
        max_time = timestamps.max()
        time_range = max_time - min_time

        if time_range == 0:
            # 所有时间戳相同
            return np.zeros((len(timestamps), 4))  # 改为4

        # 归一化时间
        normalized_time = (timestamps - min_time) / time_range

        time_features = []
        for norm_t in normalized_time:
            # 创建4个基于相对时间的特征
            t_sin = np.sin(2 * np.pi * norm_t)
            t_cos = np.cos(2 * np.pi * norm_t)
            t_sin2 = np.sin(4 * np.pi * norm_t)  # 高频成分
            t_cos2 = np.cos(4 * np.pi * norm_t)

            time_feature = [t_sin, t_cos, t_sin2, t_cos2]  # 只保留4个
            time_features.append(time_feature)

        return np.array(time_features)

    def __len__(self):
        return len(self.used_data)

    def __getitem__(self, index):
        trajectory = self.used_data[index]

        # 标准化（排除时间列）
        trajectory_std = trajectory.copy()
        trajectory_std[:, 1:] = (trajectory[:, 1:] - self.data_mean) / self.data_std

        # 确保序列足够长
        if len(trajectory) < self.seq_len + self.pred_len:
            # 如果序列不够长，使用零填充
            padded_trajectory = np.zeros((self.seq_len + self.pred_len, trajectory.shape[1]))
            actual_length = len(trajectory)
            padded_trajectory[:actual_length] = trajectory_std
            trajectory_std = padded_trajectory

        # 随机选择序列起始点
        max_start = len(trajectory_std) - self.seq_len - self.pred_len
        if max_start > 0:
            start_idx = np.random.randint(0, max_start + 1)
        else:
            start_idx = 0

        # 提取序列
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = trajectory_std[s_begin:s_end]  # 输入序列
        seq_y = trajectory_std[r_begin:r_end]  # 目标序列

        # 提取时间戳并创建时间特征
        time_x = trajectory[s_begin:s_end, 0]  # 原始时间戳
        time_y = trajectory[r_begin:r_end, 0]  # 原始时间戳

        # 创建6维时间特征
        seq_x_mark = self._extract_relative_time_features(time_x)
        seq_y_mark = self._extract_relative_time_features(time_y)

        # 转换为torch tensor
        seq_x = torch.FloatTensor(seq_x[:, 1:])  # 4个空间特征
        seq_y = torch.FloatTensor(seq_y[:, 1:])  # 4个空间特征
        seq_x_mark = torch.FloatTensor(seq_x_mark)  # 6个时间特征
        seq_y_mark = torch.FloatTensor(seq_y_mark)  # 6个时间特征

        return seq_x, seq_y, seq_x_mark, seq_y_mark


def custom_collate_fn(batch):
    """自定义collate函数"""
    batch_x, batch_y, batch_x_mark, batch_y_mark = zip(*batch)

    # 转换为张量
    batch_x = torch.stack(batch_x)
    batch_y = torch.stack(batch_y)
    batch_x_mark = torch.stack(batch_x_mark)
    batch_y_mark = torch.stack(batch_y_mark)

    return batch_x, batch_y, batch_x_mark, batch_y_mark