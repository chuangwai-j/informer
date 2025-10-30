# data/aircraft/data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.signal import medfilt
import scipy.spatial
import os
import time
import math


def custom_collate_fn(batch):
    """自定义collate函数"""
    batch_x, batch_y, batch_x_mark, batch_y_mark = zip(*batch)

    # 转换为张量
    batch_x = torch.stack(batch_x)
    batch_y = torch.stack(batch_y)
    batch_x_mark = torch.stack(batch_x_mark)
    batch_y_mark = torch.stack(batch_y_mark)

    return batch_x, batch_y, batch_x_mark, batch_y_mark


class AircraftTrajectoryDataset(Dataset):
    def __init__(self, csv_path, seq_len=96, label_len=48, pred_len=24, mode='train', scale=True, speed_min=50,
                 speed_max=300):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.mode = mode
        self.scale = scale

        # 1. 加载和清洗数据
        self.all_sequences = self._clean_adsb_data(csv_path, speed_min, speed_max)

        if len(self.all_sequences) == 0:
            raise ValueError(f"从 CSV 文件 {csv_path} 中未找到任何有效序列。")

        # 2. 划分训练集、验证集、测试集 (操作于所有序列)
        self._split_data(mode)

        # 3. 过滤掉长度不足的序列
        self._filter_short_sequences()

        # 4. 数据标准化
        if self.scale:
            self._standardize()
        else:
            self.data_mean = np.zeros(self.all_sequences[0].shape[1] - 1)
            self.data_std = np.ones(self.all_sequences[0].shape[1] - 1)

        print(f"{mode}数据集: {len(self.used_data)} 个序列")

    # --- 数据预处理辅助函数 (来自 data_preprocess.py) ---

    def _median_filter(self, z, window):
        """中值滤波函数"""
        if window % 2 == 0:
            window += 1

        if len(z) <= window:
            return z.copy()

        # 添加反射边界
        pad_len = math.ceil(window / 2)
        # 简化边界处理，直接使用最近点的中值填充
        lb = np.full(pad_len, np.median(z[:min(3, len(z))]))
        rb = np.full(pad_len, np.median(z[-min(3, len(z)):]))

        final = np.concatenate([lb, z, rb])
        filtered = medfilt(final, window)

        # 返回原始长度的数据
        return filtered[pad_len:-pad_len] if pad_len > 0 else filtered

    def _filter_speedlimit(self, lat, lon, t, speed_min, speed_max, verbose=False):
        '''返回布尔向量，给出符合速度限制的最长点序列'''
        n = len(lat)
        if n <= 1:
            return np.ones(n, dtype=bool)

        R = 6371000  # 地球半径（米）
        speeds = np.zeros(n - 1)

        # 计算相邻点之间的速度
        for i in range(n - 1):
            time_diff = t[i + 1] - t[i]
            if time_diff <= 0:
                speeds[i] = 0
                continue

            # Haversine距离公式在这里可能更准确, 但为保持一致性使用 data_preprocess.py 中的欧氏距离近似
            dx = R * np.cos(np.radians(lat[i])) * np.cos(np.radians(lon[i])) - \
                 R * np.cos(np.radians(lat[i + 1])) * np.cos(np.radians(lon[i + 1]))
            dy = R * np.cos(np.radians(lat[i])) * np.sin(np.radians(lon[i])) - \
                 R * np.cos(np.radians(lat[i + 1])) * np.sin(np.radians(lon[i + 1]))
            dz = R * np.sin(np.radians(lat[i])) - R * np.sin(np.radians(lat[i + 1]))

            distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            speeds[i] = distance / time_diff

        # 找到符合速度限制的最长连续序列
        max_length = 0
        best_start = 0
        current_start = 0
        current_length = 1

        for i in range(n - 1):
            if speed_min <= speeds[i] <= speed_max:
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    best_start = current_start
                current_start = i + 1
                current_length = 1

        if current_length > max_length:
            max_length = current_length
            best_start = current_start

        res = np.zeros(n, dtype=bool)
        if max_length > 0:
            res[best_start:best_start + max_length] = True

        return res

    def _clean_adsb_data(self, csv_file, speed_min, speed_max):
        """清洗ADS-B数据并返回序列列表"""
        print(f"读取并清洗CSV文件: {csv_file}...")
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {csv_file}")
            return []

        aircraft_groups = df.groupby('aircraft')
        all_sequences = []

        for aircraft_id, aircraft_data in aircraft_groups:
            # 按时间排序
            aircraft_data = aircraft_data.sort_values('timeAtServer')

            # 提取所需字段
            times = aircraft_data['timeAtServer'].values
            lats = aircraft_data['latitude'].values
            lons = aircraft_data['longitude'].values
            geo_alts = aircraft_data['geoAltitude'].values
            baro_alts = aircraft_data['baroAltitude'].values

            # 应用速度过滤器
            valid_mask = self._filter_speedlimit(lats, lons, times, speed_min, speed_max, verbose=False)

            if np.sum(valid_mask) == 0:
                continue

            # 应用中值滤波平滑高度数据
            if np.sum(valid_mask) > 5:
                try:
                    filtered_geo_alts = self._median_filter(geo_alts[valid_mask], 5)
                    filtered_baro_alts = self._median_filter(baro_alts[valid_mask], 5)
                except Exception as e:
                    print(f"飞行器 {aircraft_id} 滤波时出错: {e}, 使用原始数据")
                    filtered_geo_alts = geo_alts[valid_mask]
                    filtered_baro_alts = baro_alts[valid_mask]
            else:
                filtered_geo_alts = geo_alts[valid_mask]
                filtered_baro_alts = baro_alts[valid_mask]

            # 创建有效数据序列
            valid_times = times[valid_mask]
            valid_lats = lats[valid_mask]
            valid_lons = lons[valid_mask]

            min_length = min(len(valid_times), len(valid_lats), len(valid_lons),
                             len(filtered_geo_alts), len(filtered_baro_alts))

            # 创建特征矩阵 [时间, 纬度, 经度, 几何高度, 气压高度]
            sequence = np.column_stack([
                valid_times[:min_length],
                valid_lats[:min_length],
                valid_lons[:min_length],
                filtered_geo_alts[:min_length],
                filtered_baro_alts[:min_length]
            ])

            if len(sequence) >= 10:  # 至少10个数据点
                all_sequences.append(sequence)

        print(f"清洗完成。总共得到 {len(all_sequences)} 个有效轨迹序列。")
        return all_sequences

    # --- 现有数据加载逻辑 (来自 data_loader.py) ---

    def _split_data(self, mode):
        """划分数据集"""
        total_samples = len(self.all_sequences)
        train_ratio, val_ratio = 0.7, 0.2

        if mode == 'train':
            self.used_data = self.all_sequences[:int(total_samples * train_ratio)]
        elif mode == 'val':
            self.used_data = self.all_sequences[
                int(total_samples * train_ratio):int(total_samples * (train_ratio + val_ratio))]
        else:  # test
            self.used_data = self.all_sequences[int(total_samples * (train_ratio + val_ratio)):]

    def _filter_short_sequences(self):
        """过滤掉长度不足 seq_len + pred_len 的序列"""
        min_required_length = self.seq_len + self.pred_len
        filtered_data = []

        for trajectory in self.used_data:
            if len(trajectory) >= min_required_length:
                filtered_data.append(trajectory)

        self.used_data = filtered_data

        if len(self.used_data) == 0 and self.mode != 'train':
            # 允许 val/test 为空，如果 train 为空在 _clean_adsb_data 或 __init__ 中已抛出
            pass
        elif len(self.used_data) == 0 and self.mode == 'train':
            raise ValueError(f"没有足够长的序列。需要至少 {min_required_length} 个时间步，但所有序列都更短。")

    def _standardize(self):
        """数据标准化"""
        # 收集所有数据计算均值和标准差
        all_data = []
        for trajectory in self.used_data:
            all_data.append(trajectory[:, 1:])  # 排除时间列

        if not all_data:
            print(f"Warning: {self.mode} dataset is empty, cannot compute statistics. Using zeros/ones.")
            self.data_mean = np.zeros(4)
            self.data_std = np.ones(4)
            return

        all_data = np.vstack(all_data)
        self.data_mean = np.mean(all_data, axis=0)
        self.data_std = np.std(all_data, axis=0)

        # 避免除零
        self.data_std[self.data_std == 0] = 1.0

        print(f"数据均值: {self.data_mean}")
        print(f"数据标准差: {self.data_std}")

    def _extract_relative_time_features(self, timestamps):
        """使用相对时间特征，生成4个特征"""
        if len(timestamps) == 0:
            return np.zeros((0, 4))

        min_time = timestamps.min()
        max_time = timestamps.max()
        time_range = max_time - min_time

        if time_range == 0:
            return np.zeros((len(timestamps), 4))

        normalized_time = (timestamps - min_time) / time_range

        time_features = []
        for norm_t in normalized_time:
            t_sin = np.sin(2 * np.pi * norm_t)
            t_cos = np.cos(2 * np.pi * norm_t)
            t_sin2 = np.sin(4 * np.pi * norm_t)  # 高频成分
            t_cos2 = np.cos(4 * np.pi * norm_t)

            time_feature = [t_sin, t_cos, t_sin2, t_cos2]  # 4个相对时间特征
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
        if len(trajectory_std) < self.seq_len + self.pred_len:
            # 理论上已经被 _filter_short_sequences 过滤掉，这里作为安全检查
            raise IndexError("Sequence is too short. This should not happen.")

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

        # 创建4维时间特征
        seq_x_mark = self._extract_relative_time_features(time_x)
        seq_y_mark = self._extract_relative_time_features(time_y)

        # 转换为torch tensor
        seq_x = torch.FloatTensor(seq_x[:, 1:])  # 4个空间特征
        seq_y = torch.FloatTensor(seq_y[:, 1:])  # 4个空间特征
        seq_x_mark = torch.FloatTensor(seq_x_mark)  # 4个时间特征
        seq_y_mark = torch.FloatTensor(seq_y_mark)  # 4个时间特征

        return seq_x, seq_y, seq_x_mark, seq_y_mark