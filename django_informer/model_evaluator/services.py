import sys
import os
import torch
import numpy as np
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import logging
from datetime import datetime

# Add parent directory to Python path for importing Informer modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import Informer
from data.aircraft.data_loader import AircraftTrajectoryDataset, custom_collate_fn
from utils.metrics import metric

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器 - 基于test.py的逻辑封装"""

    def __init__(self, config_path=None, checkpoint_path=None):
        self.config_path = config_path or '../config/aircraft.yaml'
        self.checkpoint_path = checkpoint_path or '../checkpoints/checkpoint.pth'
        self.args = None
        self.model = None
        self.device = None

    def load_config(self):
        """加载YAML配置文件"""
        try:
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
            logger.info(f"配置加载成功，设备: {args.device}")
            return args

        except FileNotFoundError:
            logger.error(f"配置文件未找到: {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise

    def load_model(self):
        """加载训练好的模型"""
        if not self.args:
            self.load_config()

        try:
            model = Informer(
                self.args.enc_in, self.args.dec_in, self.args.c_out,
                self.args.seq_len, self.args.label_len, self.args.pred_len,
                factor=self.args.factor, d_model=self.args.d_model, n_heads=self.args.n_heads,
                e_layers=self.args.e_layers, d_layers=self.args.d_layers, d_ff=self.args.d_ff,
                dropout=self.args.dropout, attn=self.args.attn, embed=self.args.embed,
                freq=self.args.freq, activation=self.args.activation,
                output_attention=self.args.output_attention, distil=self.args.distil,
                mix=self.args.mix, device=self.args.device
            ).float().to(self.args.device)

            model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.args.device))
            model.eval()

            self.model = model
            self.device = self.args.device
            logger.info(f"模型加载成功，设备: {self.args.device}")
            return model

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def get_test_dataloader(self):
        """获取测试数据加载器"""
        if not self.args:
            self.load_config()

        try:
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

            test_loader = DataLoader(
                test_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=custom_collate_fn
            )

            logger.info(f"测试数据加载器创建成功，批次数量: {len(test_loader)}")
            return test_loader

        except Exception as e:
            logger.error(f"测试数据加载失败: {e}")
            raise

    def evaluate_model(self, test_loader=None):
        """评估模型并返回指标"""
        if not self.model:
            self.load_model()

        if test_loader is None:
            test_loader = self.get_test_dataloader()

        self.model.eval()
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                all_preds.append(outputs.detach().cpu())
                all_trues.append(batch_y.detach().cpu())

                if batch_idx >= 10:  # 限制评估批次以加快速度
                    break

        # 计算指标
        total_mae, total_mse, total_rmse = 0, 0, 0
        num_batches = len(all_preds)

        if num_batches == 0:
            logger.warning("测试集中没有足够长的序列可用于评估")
            return None

        for pred, true in zip(all_preds, all_trues):
            pred_np = pred.numpy()
            true_np = true.numpy()

            mae = np.mean(np.abs(pred_np - true_np))
            mse = np.mean((pred_np - true_np) ** 2)
            rmse = np.sqrt(mse)

            total_mae += mae
            total_mse += mse
            total_rmse += rmse

        avg_mae = total_mae / num_batches
        avg_mse = total_mse / num_batches
        avg_rmse = total_rmse / num_batches

        results = {
            'mae': float(avg_mae),
            'mse': float(avg_mse),
            'rmse': float(avg_rmse),
            'num_batches': num_batches,
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        logger.info(f"模型评估完成: MAE={avg_mae:.4f}, MSE={avg_mse:.4f}, RMSE={avg_rmse:.4f}")
        return results

    def get_model_info(self):
        """获取模型信息"""
        if not self.args:
            self.load_config()

        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / 1024 / 1024
        else:
            total_params = 0
            trainable_params = 0
            model_size_mb = 0

        info = {
            'model_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': round(model_size_mb, 2),
            'config': {
                'seq_len': self.args.seq_len,
                'label_len': self.args.label_len,
                'pred_len': self.args.pred_len,
                'd_model': self.args.d_model,
                'n_heads': self.args.n_heads,
                'e_layers': self.args.e_layers,
                'd_layers': self.args.d_layers,
                'batch_size': self.args.batch_size,
                'attn': self.args.attn,
                'device': str(self.args.device)
            },
            'data_config': {
                'csv_path': self.args.data_csv_path,
                'speed_min': self.args.speed_min,
                'speed_max': self.args.speed_max,
                'num_workers': self.args.num_workers
            }
        }

        return info