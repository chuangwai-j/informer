import numpy as np
import pandas as pd
from scipy.signal import medfilt
import scipy.spatial
import os
import time


def precompute_distance(lat, lon, t, speedmin, speedmax):
    '''计算矩阵，判断点i和j是否在速度限制内可达'''
    # 地球半径（米）
    R = 6371000

    # 将经纬度转换为笛卡尔坐标（近似处理，不考虑地球曲率）
    x = R * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = R * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = R * np.sin(np.radians(lat))

    xyz = np.stack([x, y, z], axis=-1)
    d = scipy.spatial.distance_matrix(xyz, xyz)

    t = np.transpose(np.array([t]))
    dt = scipy.spatial.distance_matrix(t, t)

    return np.maximum(d - speedmax * dt, 0) + np.maximum(speedmin * dt - d, 0)


def filter_speedlimit(lat, lon, t, speedmin, speedmax, verbose=True):
    '''返回布尔向量，给出符合速度限制的最长点序列'''
    if verbose:
        print("过滤轨迹以保留符合速度限制的最长点序列")

    # 地球半径（米）
    R = 6371000

    n = len(lat)
    if n <= 1:
        return np.ones(n, dtype=bool)

    # 计算相邻点之间的速度
    speeds = np.zeros(n - 1)
    for i in range(n - 1):
        time_diff = t[i + 1] - t[i]
        if time_diff <= 0:
            speeds[i] = 0
            continue

        # 计算点i和i+1之间的距离
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
    current_length = 1  # 第一个点总是有效的

    for i in range(n - 1):
        if speedmin <= speeds[i] <= speedmax:
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                best_start = current_start
            current_start = i + 1
            current_length = 1

    # 检查最后一个序列
    if current_length > max_length:
        max_length = current_length
        best_start = current_start

    # 创建结果布尔数组
    res = np.zeros(n, dtype=bool)
    if max_length > 0:
        res[best_start:best_start + max_length] = True

    if verbose:
        print(f"初始点数: {n}")
        print(f"最长序列点数: {max_length}")

    return res


def median_filter(z, window):
    """中值滤波函数"""
    # 处理窗口大小
    if window % 2 == 0:
        window += 1  # 确保窗口大小为奇数

    # 如果数据点太少，直接返回原数据
    if len(z) <= window:
        return z.copy()

    # 添加反射边界
    lb = 2 * np.median(z[:min(3, len(z))]) - z[min(3, len(z)):min(window + 3, len(z))][::-1]
    rb = 2 * np.median(z[-min(3, len(z)):]) - z[-min(window + 3, len(z)):-min(3, len(z))][::-1]

    # 组合并应用中值滤波
    final = np.concatenate([lb, z, rb])
    filtered = medfilt(final, window)

    return filtered[len(lb):len(lb) + len(z)]


def clean_adsb_data(csv_file, output_file="result.npy", speed_min=50, speed_max=300):
    """
    清洗ADS-B数据并转换为适合Transformer模型训练的格式

    参数:
    csv_file: 输入CSV文件路径
    output_file: 输出npy文件路径
    speed_min: 最小合理速度(m/s)
    speed_max: 最大合理速度(m/s)
    """
    # 读取CSV文件
    print("读取CSV文件...")
    df = pd.read_csv(csv_file)

    # 按aircraft分组处理每个飞行器的数据
    aircraft_groups = df.groupby('aircraft')
    all_sequences = []
    sequence_lengths = []

    for aircraft_id, aircraft_data in aircraft_groups:
        print(f"处理飞行器 {aircraft_id}...")

        # 按时间排序
        aircraft_data = aircraft_data.sort_values('timeAtServer')

        # 提取所需字段
        times = aircraft_data['timeAtServer'].values
        lats = aircraft_data['latitude'].values
        lons = aircraft_data['longitude'].values
        geo_alts = aircraft_data['geoAltitude'].values
        baro_alts = aircraft_data['baroAltitude'].values

        # 应用速度过滤器
        valid_mask = filter_speedlimit(lats, lons, times, speed_min, speed_max)

        # 如果没有有效点，跳过这个飞行器
        if np.sum(valid_mask) == 0:
            print(f"飞行器 {aircraft_id} 没有有效数据点，跳过")
            continue

        # 应用中值滤波平滑高度数据
        if np.sum(valid_mask) > 5:  # 确保有足够的数据点进行滤波
            try:
                filtered_geo_alts = median_filter(geo_alts[valid_mask], 5)
                filtered_baro_alts = median_filter(baro_alts[valid_mask], 5)
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

        # 确保所有数组长度一致
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

        # 只保留足够长的序列
        if len(sequence) >= 10:  # 至少10个数据点
            all_sequences.append(sequence)
            sequence_lengths.append(len(sequence))
            print(f"飞行器 {aircraft_id} 的有效序列长度: {len(sequence)}")
        else:
            print(f"飞行器 {aircraft_id} 的有效序列太短({len(sequence)}个点)，跳过")

    # 将所有序列保存为numpy数组
    print(f"保存结果到 {output_file}...")

    # 创建一个对象数组来存储不同长度的序列
    all_sequences_array = np.empty(len(all_sequences), dtype=object)
    for i, seq in enumerate(all_sequences):
        all_sequences_array[i] = seq

    np.save(output_file, all_sequences_array)

    # 保存序列长度信息
    lengths_file = output_file.replace('.npy', '_lengths.npy')
    np.save(lengths_file, np.array(sequence_lengths))

    print(f"数据已保存到 {output_file}")
    print(f"序列长度信息已保存到 {lengths_file}")
    print(f"总共保存了 {len(all_sequences)} 个飞行器的轨迹数据")

    return all_sequences


if __name__ == "__main__":
    # 设置文件路径
    csv_file = "data.csv"  # 假设CSV文件在同一目录下
    output_file = "../../result.npy"

    # 清洗数据
    clean_adsb_data(csv_file, output_file)