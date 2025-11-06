import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Add parent directory to Python path for importing Informer modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.model import Informer
from data.aircraft.data_loader import AircraftTrajectoryDataset
from utils.metrics import metric


class TrajectoryVisualizer:
    """轨迹可视化器"""

    def __init__(self, config_path='../config/aircraft.yaml'):
        self.config_path = config_path
        self.args = None
        self.model = None
        self.device = None

    def load_config_and_model(self):
        """加载配置和模型"""
        import yaml
        from easydict import EasyDict as edict

        # 加载配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        args = edict()
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    setattr(args, sub_key, sub_value)
            else:
                setattr(args, key, value)

        # 配置设备
        if args.use_gpu and torch.cuda.is_available():
            args.device = torch.device(f'cuda:{args.gpu}')
        else:
            args.device = torch.device('cpu')

        self.args = args
        self.device = args.device

        # 加载模型
        model = Informer(
            args.enc_in, args.dec_in, args.c_out,
            args.seq_len, args.label_len, args.pred_len,
            factor=args.factor, d_model=args.d_model, n_heads=args.n_heads,
            e_layers=args.e_layers, d_layers=args.d_layers, d_ff=args.d_ff,
            dropout=args.dropout, attn=args.attn, embed=args.embed,
            freq=args.freq, activation=args.activation,
            output_attention=args.output_attention, distil=args.distil,
            mix=args.mix, device=args.device
        ).float().to(args.device)

        model.load_state_dict(torch.load('../checkpoints/checkpoint.pth', map_location=args.device))
        model.eval()

        self.model = model

    def generate_predictions(self, num_samples=5):
        """生成预测样本用于可视化"""
        if not self.model:
            self.load_config_and_model()

        # 加载测试数据
        test_data = AircraftTrajectoryDataset(
            csv_path=self.args.data_csv_path,
            seq_len=self.args.seq_len,
            label_len=self.args.label_len,
            pred_len=self.args.pred_len,
            mode='test',
            scale=True,
            speed_min=self.args.speed_min,
            speed_max=self.args.speed_max
        )

        samples = []
        for i in range(min(num_samples, len(test_data))):
            batch_x, batch_y, batch_x_mark, batch_y_mark = test_data[i]

            # 转换为tensor并移到设备
            batch_x = batch_x.unsqueeze(0).float().to(self.device)
            batch_y = batch_y.unsqueeze(0).float().to(self.device)
            batch_x_mark = batch_x_mark.unsqueeze(0).float().to(self.device)
            batch_y_mark = batch_y_mark.unsqueeze(0).float().to(self.device)

            # 准备解码器输入
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

            with torch.no_grad():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :].cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, :].cpu().numpy()

            # 反标准化
            data_mean = test_data.data_mean
            data_std = test_data.data_std

            outputs_denorm = outputs * data_std + data_mean
            batch_y_denorm = batch_y * data_std + data_mean

            samples.append({
                'prediction': outputs_denorm[0],
                'ground_truth': batch_y_denorm[0],
                'input_seq': batch_x[0].cpu().numpy() * data_std + data_mean
            })

        return samples

    def create_3d_matplotlib(self, samples, save_path='django_informer/static/trajectory_3d.png'):
        """使用matplotlib创建3D轨迹图"""
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('飞机轨迹预测 3D 可视化', fontsize=16, fontweight='bold')

        for i, sample in enumerate(samples):
            ax = fig.add_subplot(2, 3, i+1, projection='3d')

            # 提取坐标 (纬度, 经度, 几何高度)
            input_seq = sample['input_seq']
            pred_seq = sample['prediction']
            true_seq = sample['ground_truth']

            # 时间轴
            input_time = np.arange(len(input_seq))
            pred_time = np.arange(len(input_seq), len(input_seq) + len(pred_seq))
            true_time = np.arange(len(input_seq), len(input_seq) + len(true_seq))

            # 绘制输入序列
            ax.plot(input_seq[:, 0], input_seq[:, 1], input_seq[:, 2],
                   label='输入序列', marker='o', markersize=2, color='blue', alpha=0.6)

            # 绘制预测序列
            ax.plot(pred_seq[:, 0], pred_seq[:, 1], pred_seq[:, 2],
                   label='预测', marker='s', markersize=3, color='red', alpha=0.8)

            # 绘制真实序列
            ax.plot(true_seq[:, 0], true_seq[:, 1], true_seq[:, 2],
                   label='真实', marker='^', markersize=3, color='green', alpha=0.8)

            # 标记起点和终点
            ax.scatter(input_seq[0, 0], input_seq[0, 1], input_seq[0, 2],
                      marker='*', s=100, color='black', label='起点')
            ax.scatter(true_seq[-1, 0], true_seq[-1, 1], true_seq[-1, 2],
                      marker='D', s=80, color='orange', label='终点')

            ax.set_xlabel('纬度')
            ax.set_ylabel('经度')
            ax.set_zlabel('高度 (m)')
            ax.set_title(f'样本 {i+1}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def create_3d_plotly(self, samples, save_path='django_informer/static/trajectory_3d_interactive.html'):
        """使用plotly创建交互式3D轨迹图"""
        fig = go.Figure()

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, sample in enumerate(samples):
            input_seq = sample['input_seq']
            pred_seq = sample['prediction']
            true_seq = sample['ground_truth']

            # 时间轴
            input_time = np.arange(len(input_seq))
            pred_time = np.arange(len(input_seq), len(input_seq) + len(pred_seq))
            true_time = np.arange(len(input_seq), len(input_seq) + len(true_seq))

            color = colors[i % len(colors)]

            # 输入序列
            fig.add_trace(go.Scatter3d(
                x=input_seq[:, 0],
                y=input_seq[:, 1],
                z=input_seq[:, 2],
                mode='lines+markers',
                name=f'样本{i+1} 输入',
                line=dict(color=color, width=3),
                marker=dict(size=4),
                showlegend=True
            ))

            # 预测序列
            fig.add_trace(go.Scatter3d(
                x=pred_seq[:, 0],
                y=pred_seq[:, 1],
                z=pred_seq[:, 2],
                mode='lines+markers',
                name=f'样本{i+1} 预测',
                line=dict(color='red', width=4, dash='dash'),
                marker=dict(size=5, symbol='square'),
                showlegend=True
            ))

            # 真实序列
            fig.add_trace(go.Scatter3d(
                x=true_seq[:, 0],
                y=true_seq[:, 1],
                z=true_seq[:, 2],
                mode='lines+markers',
                name=f'样本{i+1} 真实',
                line=dict(color='green', width=4),
                marker=dict(size=5, symbol='diamond'),
                showlegend=True
            ))

        fig.update_layout(
            title='飞机轨迹预测 3D 交互式可视化',
            scene=dict(
                xaxis_title='纬度',
                yaxis_title='经度',
                zaxis_title='高度 (m)',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8)
                )
            ),
            width=1000,
            height=700
        )

        # 保存为HTML文件
        fig.write_html(save_path)
        return save_path

    def create_2d_projections(self, samples, save_path='django_informer/static/trajectory_2d.png'):
        """创建2D投影图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('飞机轨迹预测 2D 投影', fontsize=16, fontweight='bold')

        # 定义投影组合
        projections = [
            ('纬度', '经度', 0, 1, '水平轨迹'),
            ('纬度', '高度(m)', 0, 2, '垂直剖面(纬度-高度)'),
            ('经度', '高度(m)', 1, 2, '垂直剖面(经度-高度)'),
            ('时间步', '高度(m)', None, 2, '高度变化趋势')
        ]

        for idx, (sample) in enumerate(samples[:1]):  # 只显示第一个样本
            input_seq = sample['input_seq']
            pred_seq = sample['prediction']
            true_seq = sample['ground_truth']

            for proj_idx, (xlabel, ylabel, x_idx, y_idx, title) in enumerate(projections):
                ax = axes[proj_idx // 2, proj_idx % 2]

                if x_idx is not None:  # 空间投影
                    # 输入序列
                    ax.plot(input_seq[:, x_idx], input_seq[:, y_idx],
                           'o-', label='输入序列', color='blue', markersize=2, alpha=0.6)

                    # 预测序列
                    ax.plot(pred_seq[:, x_idx], pred_seq[:, y_idx],
                           's--', label='预测', color='red', markersize=3, alpha=0.8)

                    # 真实序列
                    ax.plot(true_seq[:, x_idx], true_seq[:, y_idx],
                           '^-', label='真实', color='green', markersize=3, alpha=0.8)

                    # 标记起点终点
                    ax.scatter(input_seq[0, x_idx], input_seq[0, y_idx],
                              marker='*', s=100, color='black', label='起点', zorder=5)
                    ax.scatter(true_seq[-1, x_idx], true_seq[-1, y_idx],
                              marker='D', s=80, color='orange', label='终点', zorder=5)

                else:  # 时间序列
                    time_input = np.arange(len(input_seq))
                    time_pred = np.arange(len(input_seq), len(input_seq) + len(pred_seq))
                    time_true = np.arange(len(input_seq), len(input_seq) + len(true_seq))

                    ax.plot(time_input, input_seq[:, y_idx],
                           'o-', label='输入序列', color='blue', markersize=2, alpha=0.6)
                    ax.plot(time_pred, pred_seq[:, y_idx],
                           's--', label='预测', color='red', markersize=3, alpha=0.8)
                    ax.plot(time_true, true_seq[:, y_idx],
                           '^-', label='真实', color='green', markersize=3, alpha=0.8)

                    ax.set_xlabel('时间步')

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def calculate_error_metrics(self, samples):
        """计算误差指标"""
        all_errors = []
        for sample in samples:
            pred = sample['prediction']
            true = sample['ground_truth']

            # 计算每个点的欧几里得距离
            errors = np.sqrt(np.sum((pred - true) ** 2, axis=1))
            all_errors.extend(errors)

        all_errors = np.array(all_errors)

        metrics = {
            'mean_error': np.mean(all_errors),
            'max_error': np.max(all_errors),
            'min_error': np.min(all_errors),
            'std_error': np.std(all_errors),
            'median_error': np.median(all_errors)
        }

        return metrics

    def generate_visualization_report(self, num_samples=5):
        """生成完整的可视化报告"""
        print("正在生成预测样本...")
        samples = self.generate_predictions(num_samples)

        print("正在创建3D可视化...")
        matplotlib_3d_path = self.create_3d_matplotlib(samples)
        plotly_3d_path = self.create_3d_plotly(samples)

        print("正在创建2D投影...")
        matplotlib_2d_path = self.create_2d_projections(samples)

        print("正在计算误差指标...")
        error_metrics = self.calculate_error_metrics(samples)

        return {
            'matplotlib_3d': matplotlib_3d_path,
            'plotly_3d': plotly_3d_path,
            'matplotlib_2d': matplotlib_2d_path,
            'error_metrics': error_metrics,
            'num_samples': len(samples)
        }