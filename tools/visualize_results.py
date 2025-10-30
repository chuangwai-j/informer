# tools/visualize.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
from easydict import EasyDict as edict


# 假设 matplotlib, numpy, torch, pyyaml, easydict 已经安装

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


def visualize_predictions(preds_file, trues_file, output_path, config_args):
    """
    加载预测和真实值数据，并进行可视化

    :param preds_file: 预测值文件路径 (predictions.pt)
    :param trues_file: 真实值文件路径 (ground_truth.pt)
    :param output_path: 图像保存路径 (prediction_results.png)
    :param config_args: 配置参数 (用于获取 seq_len 和 pred_len)
    """
    if not os.path.exists(preds_file) or not os.path.exists(trues_file):
        print(f"错误：找不到预测结果文件。请先运行 tools/train.py 或 tools/test.py 来生成 {preds_file} 和 {trues_file}")
        return

    try:
        preds = torch.load(preds_file)
        trues = torch.load(trues_file)
    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        return

    # 将列表转换为 NumPy 数组
    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    print(f"加载预测数据形状: {preds.shape}")
    print(f"加载真实数据形状: {trues.shape}")

    # 结果分析
    # preds, trues 的形状为 (N_samples, pred_len, C_out=4)
    pred_len = preds.shape[1]

    # 假设预测值包含：[纬度, 经度, 几何高度, 气压高度]
    feature_names = ['Latitude', 'Longitude', 'Geo Altitude', 'Baro Altitude']

    # 计算整体性能指标（这里简单计算一下，详细指标应在test.py中进行）
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    print(f"\n--- 整体性能指标 ---")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    print("--------------------")

    # --- 绘图逻辑 ---

    # 随机选取 3 个样本进行可视化
    N_samples = preds.shape[0]
    num_plots = min(N_samples, 3)

    fig, axes = plt.subplots(num_plots, preds.shape[2], figsize=(18, 5 * num_plots))
    if num_plots == 1:
        # 当只有一行图时，axes需要特殊处理
        axes = [axes]

    for i in range(num_plots):
        sample_idx = np.random.randint(0, N_samples)

        for j in range(preds.shape[2]):
            # 第 i 个样本的第 j 个特征
            ax = axes[i][j]

            # 预测值 (仅 pred_len 长度)
            pred_sequence = preds[sample_idx, :, j]
            # 真实值 (仅 pred_len 长度)
            true_sequence = trues[sample_idx, :, j]

            time_steps = np.arange(pred_len)

            ax.plot(time_steps, true_sequence, label='Ground Truth', marker='o', linestyle='-', color='blue', alpha=0.7)
            ax.plot(time_steps, pred_sequence, label='Prediction', marker='x', linestyle='--', color='red', alpha=0.8)

            ax.set_title(f'Sample {sample_idx + 1} | {feature_names[j]} Forecast (L_y={pred_len})', fontsize=12)
            ax.set_xlabel('Time Steps (Future)')
            ax.set_ylabel(f'{feature_names[j]} (Normalized)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.suptitle(
        f"Informer Trajectory Prediction Results (L_x={config_args.seq_len}, L_y={config_args.pred_len})\nRMSE: {rmse:.4f}",
        y=1.02, fontsize=16, fontweight='bold')

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