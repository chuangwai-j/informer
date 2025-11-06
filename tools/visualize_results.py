# tools/visualize.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
from easydict import EasyDict as edict

# 导入数据加载器以获取标准化参数
from data.aircraft.data_loader import AircraftTrajectoryDataset


def load_config(config_path):
    """加载YAML配置文件并转换为易于访问的edict对象 (与train.py保持一致)"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        args = edict()
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    setattr(args, sub_key, sub_value)
            else:
                setattr(args, key, value)
        return args
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit()
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit()


def get_normalization_params(args):
    """通过实例化 Dataset 来获取 data_mean 和 data_std"""
    # 注意: 这会重新执行数据清洗和测试集参数计算，以确保参数正确。
    try:
        print("--- 正在加载数据集以获取标准化参数 (将执行一次数据清洗和标准化) ---")
        temp_dataset = AircraftTrajectoryDataset(
            csv_path=args.data_csv_path,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            mode='test',
            scale=True,
            speed_min=args.speed_min,
            speed_max=args.speed_max
        )
        return temp_dataset.data_mean, temp_dataset.data_std
    except Exception as e:
        print(f"Error loading normalization parameters: {e}")
        return None, None


def visualize_predictions(preds_file, trues_file, output_path, config_args):
    """
    加载预测和真实值数据，并进行轨迹可视化（包含 3D 轨迹和关键特征时间序列）

    :param preds_file: 预测值文件路径 (predictions.pt)
    :param trues_file: 真实值文件路径 (ground_truth.pt)
    :param output_path: 图像保存路径 (prediction_results.png)
    :param config_args: 配置参数 (用于获取 seq_len 和 pred_len)
    """
    if not os.path.exists(preds_file) or not os.path.exists(trues_file):
        print(f"错误：找不到预测结果文件。请先运行 tools/train.py 或 tools/test.py 来生成 {preds_file} 和 {trues_file}")
        return

    try:
        # 加载标准化后的预测结果 (列表转换为 NumPy 数组)
        preds_normalized = torch.cat(torch.load(preds_file), dim=0).numpy()
        trues_normalized = torch.cat(torch.load(trues_file), dim=0).numpy()
    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        return

    # 获取反标准化参数
    data_mean, data_std = get_normalization_params(config_args)
    if data_mean is None:
        print("无法进行可视化：未能获取标准化参数。")
        return

    # 1. 反标准化数据
    preds_denorm = preds_normalized * data_std + data_mean
    trues_denorm = trues_normalized * data_std + data_mean

    print(f"加载预测数据形状: {preds_normalized.shape}")
    print(f"加载真实数据形状: {trues_normalized.shape}")

    pred_len = config_args.pred_len
    # 假设特征顺序: [纬度, 经度, 几何高度, 气压高度]
    feature_names = ['Latitude (°)', 'Longitude (°)', 'Geo Altitude (m)', 'Baro Altitude (m)']

    # 计算 RMSE (使用标准化后的数据，与 train.py 保持一致)
    rmse = np.sqrt(np.mean((preds_normalized - trues_normalized) ** 2))

    print(f"\n--- 整体性能指标 ---")
    print(f"Normalized RMSE: {rmse:.4f}")
    print("--------------------")

    # --- 绘图逻辑 ---

    # 随机选取 1 个样本进行详细可视化
    N_samples = preds_denorm.shape[0]
    sample_idx = np.random.randint(0, N_samples)

    # 设置 2x2 子图布局
    fig = plt.figure(figsize=(15, 12))
    plt.rcParams.update({'font.size': 10})

    plt.suptitle(
        f"Informer Trajectory Prediction (Sample {sample_idx + 1}) | Input L={config_args.seq_len}, Pred L={pred_len}\nNormalized RMSE: {rmse:.4f}",
        y=1.00, fontsize=16, fontweight='bold')

    # 准备数据
    lat_true = trues_denorm[sample_idx, :, 0]
    lon_true = trues_denorm[sample_idx, :, 1]
    geo_alt_true = trues_denorm[sample_idx, :, 2]
    baro_alt_true = trues_denorm[sample_idx, :, 3]

    lat_pred = preds_denorm[sample_idx, :, 0]
    lon_pred = preds_denorm[sample_idx, :, 1]
    geo_alt_pred = preds_denorm[sample_idx, :, 2]
    baro_alt_pred = preds_denorm[sample_idx, :, 3]

    time_steps = np.arange(pred_len)

    # --- Subplot 1: 3D Trajectory (Lon, Lat, Geo Alt) ---
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')

    # 绘制真实轨迹
    ax_3d.plot(lon_true, lat_true, geo_alt_true,
               label='Ground Truth', marker='o', linestyle='-', color='blue', alpha=0.7)
    # 绘制预测轨迹
    ax_3d.plot(lon_pred, lat_pred, geo_alt_pred,
               label='Prediction', marker='x', linestyle='--', color='red', alpha=0.8)

    # 突出起点
    ax_3d.scatter(lon_true[0], lat_true[0], geo_alt_true[0],
                  marker='s', color='green', s=100, label='Start Point')

    ax_3d.set_title('3D Trajectory Prediction (Geo Altitude)', fontsize=14)
    ax_3d.set_xlabel(feature_names[1])  # Longitude
    ax_3d.set_ylabel(feature_names[0])  # Latitude
    ax_3d.set_zlabel(feature_names[2])  # Geo Altitude
    ax_3d.legend(loc='lower left')

    # --- Subplot 2: 2D Trajectory (Lon vs. Lat) ---
    ax_2d = fig.add_subplot(2, 2, 2)
    ax_2d.plot(lon_true, lat_true, label='Ground Truth', marker='o', linestyle='-', color='blue', alpha=0.7)
    ax_2d.plot(lon_pred, lat_pred, label='Prediction', marker='x', linestyle='--', color='red', alpha=0.8)
    ax_2d.plot(lon_true[0], lat_true[0], marker='s', color='green', markersize=8, label='Start Point')

    ax_2d.set_title(f'2D Trajectory Prediction (Lon vs. Lat)', fontsize=14)
    ax_2d.set_xlabel(feature_names[1])
    ax_2d.set_ylabel(feature_names[0])
    ax_2d.legend()
    ax_2d.grid(True, linestyle='--', alpha=0.6)
    ax_2d.set_aspect('equal', adjustable='box')

    # --- Subplot 3: Geo Altitude Time-Series ---
    ax_geo_alt = fig.add_subplot(2, 2, 3)
    # 绘制 Geo Altitude (index 2)
    ax_geo_alt.plot(time_steps, geo_alt_true, label='Ground Truth', marker='o', linestyle='-', color='blue', alpha=0.7)
    ax_geo_alt.plot(time_steps, geo_alt_pred, label='Prediction', marker='x', linestyle='--', color='red', alpha=0.8)
    ax_geo_alt.set_title(f'{feature_names[2]} Time-Series Forecast', fontsize=12)
    ax_geo_alt.set_xlabel('Time Steps (Future)')
    ax_geo_alt.set_ylabel(feature_names[2])
    ax_geo_alt.legend()
    ax_geo_alt.grid(True, linestyle='--', alpha=0.6)

    # --- Subplot 4: Baro Altitude Time-Series ---
    ax_baro_alt = fig.add_subplot(2, 2, 4)
    # 绘制 Baro Altitude (index 3)
    ax_baro_alt.plot(time_steps, baro_alt_true, label='Ground Truth', marker='o', linestyle='-', color='blue',
                     alpha=0.7)
    ax_baro_alt.plot(time_steps, baro_alt_pred, label='Prediction', marker='x', linestyle='--', color='red', alpha=0.8)
    ax_baro_alt.set_title(f'{feature_names[3]} Time-Series Forecast', fontsize=12)
    ax_baro_alt.set_xlabel('Time Steps (Future)')
    ax_baro_alt.set_ylabel(feature_names[3])
    ax_baro_alt.legend()
    ax_baro_alt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以适应标题

    # 确保保存路径存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_path)
    print(f"\n可视化图表已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Informer Prediction Results')
    parser.add_argument('--config', type=str, default='./config/aircraft.yaml', help='Configuration file path')
    parser.add_argument('--preds', type=str, default='predictions.pt',
                        help='Predictions file name (default: predictions.pt)')
    parser.add_argument('--trues', type=str, default='ground_truth.pt',
                        help='Ground Truth file name (default: ground_truth.pt)')
    parser.add_argument('--output', type=str, default='tools/prediction_results.png',
                        help='Output visualization file path')

    cfg_args = parser.parse_args()

    # 1. 加载配置
    args = load_config(cfg_args.config)

    # 2. 执行可视化
    visualize_predictions(
        preds_file=cfg_args.preds,
        trues_file=cfg_args.trues,
        output_path=cfg_args.output,
        config_args=args
    )


if __name__ == '__main__':
    # 确保可以找到 yaml 和 easydict
    try:
        import yaml
        from easydict import EasyDict as edict
    except ImportError:
        print("请安装依赖: pip install pyyaml easydict")
        exit()

    # 确保安装了绘图库
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("请安装绘图库: pip install matplotlib")
        exit()

    main()
