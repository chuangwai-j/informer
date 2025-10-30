# visualize_results.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import AircraftTrajectoryDataset


def load_and_visualize():
    # 加载测试数据集用于反标准化
    test_dataset = AircraftTrajectoryDataset('result_filtered.npy',
                                             seq_len=48,
                                             label_len=24,
                                             pred_len=12,
                                             mode='test',
                                             scale=True)

    # 加载预测结果
    preds = torch.load('predictions.pt')
    trues = torch.load('ground_truth.pt')

    print(f"预测结果数量: {len(preds)}")
    print(f"真实值数量: {len(trues)}")

    # 可视化几个样本
    visualize_samples(preds, trues, test_dataset, num_samples=3)


def visualize_samples(preds, trues, dataset, num_samples=3):
    """可视化几个预测样本"""

    data_mean = dataset.data_mean
    data_std = dataset.data_std

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    feature_names = ['Latitude', 'Longitude', 'Geo Altitude', 'Baro Altitude']

    for sample_idx in range(num_samples):
        if sample_idx >= len(preds):
            break

        # 获取第一个batch的第一个样本
        pred_sample = preds[sample_idx][0].numpy()  # [pred_len, 4]
        true_sample = trues[sample_idx][0].numpy()  # [pred_len, 4]

        # 反标准化
        pred_denorm = pred_sample * data_std + data_mean
        true_denorm = true_sample * data_std + data_mean

        # 时间步
        time_steps = np.arange(len(pred_denorm))

        # 绘制四个特征
        for feat_idx in range(4):
            ax = axes[sample_idx, feat_idx]

            ax.plot(time_steps, true_denorm[:, feat_idx], 'b-', label='True', linewidth=2, marker='o')
            ax.plot(time_steps, pred_denorm[:, feat_idx], 'r--', label='Predicted', linewidth=2, marker='x')
            ax.set_xlabel('Time Step')
            ax.set_ylabel(feature_names[feat_idx])
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 计算该特征的指标
            mae = np.mean(np.abs(pred_denorm[:, feat_idx] - true_denorm[:, feat_idx]))
            rmse = np.sqrt(np.mean((pred_denorm[:, feat_idx] - true_denorm[:, feat_idx]) ** 2))

            ax.set_title(f'{feature_names[feat_idx]}\nMAE: {mae:.2f}, RMSE: {rmse:.2f}')

    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印总体指标
    print("\n总体预测指标:")
    total_mae, total_mse = 0, 0
    total_samples = 0

    for pred_batch, true_batch in zip(preds, trues):
        pred_np = pred_batch.numpy()
        true_np = true_batch.numpy()

        batch_mae = np.mean(np.abs(pred_np - true_np))
        batch_mse = np.mean((pred_np - true_np) ** 2)

        total_mae += batch_mae * len(pred_batch)
        total_mse += batch_mse * len(pred_batch)
        total_samples += len(pred_batch)

    avg_mae = total_mae / total_samples
    avg_rmse = np.sqrt(total_mse / total_samples)

    print(f"平均 MAE: {avg_mae:.4f}")
    print(f"平均 RMSE: {avg_rmse:.4f}")


if __name__ == "__main__":
    load_and_visualize()