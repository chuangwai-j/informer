import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
from typing import List, Tuple, Optional

class LargeScaleAircraftDataset(Dataset):
    """
    大规模飞机轨迹数据集加载器
    适用于几千个样本，每个样本上百个序列的场景
    """

    def __init__(self, data_paths: List[str], seq_len: int = 96, label_len: int = 48,
                 pred_len: int = 24, mode: str = 'train', scale: bool = True,
                 speed_min: float = 50, speed_max: float = 300,
                 cache_dir: str = './ai/cache', use_cache: bool = True):
        """
        Args:
            data_paths: 数据文件路径列表
            seq_len: 输入序列长度
            label_len: 标签长度
            pred_len: 预测长度
            mode: 'train', 'val', 'test'
            scale: 是否标准化
            speed_min: 最小速度限制
            speed_max: 最大速度限制
            cache_dir: 缓存目录
            use_cache: 是否使用缓存
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.mode = mode
        self.scale = scale
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 生成缓存文件名
        cache_file = f"{cache_dir}/large_scale_dataset_{mode}_{seq_len}_{label_len}_{pred_len}.pkl"

        # 尝试加载缓存
        if use_cache and os.path.exists(cache_file):
            print(f"从缓存加载数据集: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.all_sequences = cached_data['sequences']
                self.data_mean = cached_data['mean']
                self.data_std = cached_data['std']
        else:
            # 处理所有数据文件
            print("开始处理大规模数据集...")
            self.all_sequences = []
            for data_path in tqdm(data_paths, desc="处理数据文件"):
                sequences = self._process_single_file(data_path)
                self.all_sequences.extend(sequences)

            print(f"总共处理了 {len(self.all_sequences)} 个序列")

            # 数据集划分
            self._split_dataset()

            # 数据清洗和标准化
            self._clean_and_standardize()

            # 保存缓存
            if use_cache:
                print(f"保存数据集到缓存: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'sequences': self.used_data,
                        'mean': self.data_mean,
                        'std': self.data_std
                    }, f)

        # 过滤短序列
        self._filter_short_sequences()

        print(f"{mode}数据集: {len(self.used_data)} 个序列")

    def _process_single_file(self, data_path: str) -> List[np.ndarray]:
        """处理单个数据文件，返回轨迹序列列表"""
        try:
            df = pd.read_csv(data_path)

            # 假设数据格式: [timestamp, latitude, longitude, geo_altitude, baro_altitude]
            required_columns = ['timestamp', 'latitude', 'longitude', 'geo_altitude', 'baro_altitude']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"缺少必要列: {col}")

            # 按样本分组 (假设有sample_id列)
            if 'sample_id' in df.columns:
                groups = df.groupby('sample_id')
            else:
                # 如果没有sample_id，按固定长度分割
                sequence_length = 200  # 假设每个序列200个点
                groups = [(i, df[i*sequence_length:(i+1)*sequence_length])
                         for i in range(len(df) // sequence_length)]

            sequences = []
            for sample_id, group in groups:
                if len(group) < self.seq_len + self.pred_len:
                    continue

                # 数据清洗
                trajectory = self._clean_trajectory(group)
                if trajectory is not None and len(trajectory) >= self.seq_len + self.pred_len:
                    sequences.append(trajectory)

            return sequences

        except Exception as e:
            print(f"处理文件 {data_path} 时出错: {e}")
            return []

    def _clean_trajectory(self, df_group: pd.DataFrame) -> Optional[np.ndarray]:
        """清洗单个轨迹数据"""
        # 转换为numpy数组
        trajectory = df_group[['timestamp', 'latitude', 'longitude', 'geo_altitude', 'baro_altitude']].values

        # 计算速度并过滤
        if len(trajectory) > 1:
            coords = trajectory[:, 1:4]  # lat, lon, geo_alt
            time_diffs = np.diff(trajectory[:, 0])

            # 计算距离 (简化版本)
            lat_diff = np.diff(coords[:, 0])
            lon_diff = np.diff(coords[:, 1])
            alt_diff = np.diff(coords[:, 2])

            # 简化速度计算 (实际应该用大圆距离)
            distances = np.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)
            speeds = distances / np.maximum(time_diffs, 1e-6)

            # 速度过滤
            valid_speeds = (speeds >= self.speed_min) & (speeds <= self.speed_max)
            valid_indices = np.concatenate([[True], valid_speeds])

            if np.sum(valid_indices) < self.seq_len + self.pred_len:
                return None

            trajectory = trajectory[valid_indices]

        return trajectory

    def _split_dataset(self):
        """数据集划分"""
        total_samples = len(self.all_sequences)
        np.random.shuffle(self.all_sequences)

        train_ratio, val_ratio = 0.8, 0.1

        if self.mode == 'train':
            self.used_data = self.all_sequences[:int(total_samples * train_ratio)]
        elif self.mode == 'val':
            self.used_data = self.all_sequences[
                int(total_samples * train_ratio):int(total_samples * (train_ratio + val_ratio))]
        else:  # test
            self.used_data = self.all_sequences[int(total_samples * (train_ratio + val_ratio)):]

    def _clean_and_standardize(self):
        """数据清洗和标准化"""
        # 计算全局统计量
        all_data = np.concatenate([seq[:, 1:] for seq in self.used_data], axis=0)
        self.data_mean = np.mean(all_data, axis=0)
        self.data_std = np.std(all_data, axis=0)

        # 避免除零
        self.data_std = np.where(self.data_std == 0, 1, self.data_std)

        print(f"数据统计 - 均值: {self.data_mean}")
        print(f"数据统计 - 标准差: {self.data_std}")

    def _filter_short_sequences(self):
        """过滤短序列"""
        min_length = self.seq_len + self.pred_len
        self.used_data = [seq for seq in self.used_data if len(seq) >= min_length]
        print(f"过滤后{self.mode}数据集: {len(self.used_data)} 个序列")

    def _extract_time_features(self, timestamps: np.ndarray) -> np.ndarray:
        """提取时间特征"""
        features = []
        base_time = timestamps[0]

        for ts in timestamps:
            rel_time = (ts - base_time) / 3600  # 转换为小时
            features.append([
                np.sin(2 * np.pi * rel_time / 24),  # 小时周期性
                np.cos(2 * np.pi * rel_time / 24),
                rel_time / 24,  # 相对天数
                rel_time / 168  # 相对周数
            ])

        return np.array(features)

    def __len__(self):
        return len(self.used_data)

    def __getitem__(self, index):
        trajectory = self.used_data[index]

        # 标准化
        trajectory_std = trajectory.copy()
        trajectory_std[:, 1:] = (trajectory[:, 1:] - self.data_mean) / self.data_std

        # 随机选择起始点
        max_start = len(trajectory_std) - self.seq_len - self.pred_len
        start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        # 提取序列
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = trajectory_std[s_begin:s_end, 1:]  # 输入特征
        seq_y = trajectory_std[r_begin:r_end, 1:]  # 目标特征

        # 时间特征
        time_x = trajectory[s_begin:s_end, 0]
        time_y = trajectory[r_begin:r_end, 0]
        seq_x_mark = self._extract_time_features(time_x)
        seq_y_mark = self._extract_time_features(time_y)

        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y),
            torch.FloatTensor(seq_x_mark),
            torch.FloatTensor(seq_y_mark)
        )


def create_large_scale_dataloaders(data_paths: List[str], batch_size: int = 32,
                                  num_workers: int = 4, cache_dir: str = './ai/cache') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建大规模数据集的数据加载器"""

    print("创建大规模数据集加载器...")

    # 创建数据集
    train_dataset = LargeScaleAircraftDataset(
        data_paths=data_paths,
        mode='train',
        cache_dir=cache_dir,
        use_cache=True
    )

    val_dataset = LargeScaleAircraftDataset(
        data_paths=data_paths,
        mode='val',
        cache_dir=cache_dir,
        use_cache=True
    )

    test_dataset = LargeScaleAircraftDataset(
        data_paths=data_paths,
        mode='test',
        cache_dir=cache_dir,
        use_cache=True
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 序列")
    print(f"  验证集: {len(val_dataset)} 序列")
    print(f"  测试集: {len(test_dataset)} 序列")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 示例使用
    data_paths = [
        "/mnt/d/model/wind_datas/large_data_part1.csv",
        "/mnt/d/model/wind_datas/large_data_part2.csv",
        "/mnt/d/model/wind_datas/large_data_part3.csv",
    ]

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_large_scale_dataloaders(
        data_paths=data_paths,
        batch_size=64,
        num_workers=8
    )

    # 测试数据加载
    print("\n测试数据加载...")
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        print(f"批次 {i}:")
        print(f"  输入形状: {batch_x.shape}")
        print(f"  目标形状: {batch_y.shape}")
        print(f"  输入时间特征形状: {batch_x_mark.shape}")
        print(f"  目标时间特征形状: {batch_y_mark.shape}")

        if i >= 2:  # 只测试前几个批次
            break