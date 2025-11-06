# tools/train.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict as edict  # 简化配置访问
import argparse

# 假设Informmer模型在上一级目录的models中
from models.model import Informer
# 导入新的数据加载器
from data.aircraft.data_loader import AircraftTrajectoryDataset, custom_collate_fn


def load_config(config_path):
    """加载YAML配置文件并转换为易于访问的edict对象"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # 将配置字典转换为类似args的edict对象
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


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self._data_cache = {}  # 用于缓存数据集对象，防止重复清洗

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if self.args.gpu is not None else "0"
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = Informer(
            self.args.enc_in,  # 输入特征维度 (lat, lon, geo_alt, baro_alt)
            self.args.dec_in,  # 解码器输入维度
            self.args.c_out,  # 输出维度
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device
        ).float()
        return model

    def _get_data(self, flag):
        # 检查缓存，如果已存在，直接返回
        if flag in self._data_cache:
            return self._data_cache[flag]

        print(f"--- 正在初始化 {flag} 数据集并进行清洗和标准化  ---")

        data_kwargs = {
            'csv_path': self.args.data_csv_path,
            'seq_len': self.args.seq_len,
            'label_len': self.args.label_len,
            'pred_len': self.args.pred_len,
            'mode': flag,
            'scale': True,
            'speed_min': self.args.speed_min,
            'speed_max': self.args.speed_max
        }

        # 仅实例化所需的 Dataset 对象
        dataset = AircraftTrajectoryDataset(**data_kwargs)

        # 缓存数据集对象
        self._data_cache[flag] = dataset

        return dataset

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self):
        # 首次调用 _get_data('train') 和 _get_data('val') 时会进行清洗和初始化
        train_data = self._get_data('train')
        val_data = self._get_data('val')

        train_loader = DataLoader(
            train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=min(4, self.args.num_workers),
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=min(4, self.args.num_workers),
            collate_fn=custom_collate_fn
        )

        path = os.path.join(self.args.checkpoints, 'checkpoint.pth')
        if not os.path.exists(self.args.checkpoints):
            os.makedirs(self.args.checkpoints)

        time_now = time.time()

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')

                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.4f}s")
            train_loss = np.average(train_loss)
            val_loss = self.val(val_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Val Loss: {val_loss:.7f}")
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def val(self, val_loader, criterion):
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
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

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self):
        # 仅在需要时调用 _get_data('test')，如果已缓存则直接获取
        test_data = self._get_data('test')
        test_loader = DataLoader(
            test_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=custom_collate_fn
        )

        self.model.eval()

        all_preds = []
        all_trues = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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

        total_mae, total_mse, total_rmse = 0, 0, 0
        num_batches = len(all_preds)

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

        print(f'测试 MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}')

        return all_preds, all_trues


def main():
    # 使用 argparse 来指定配置文件的路径
    parser = argparse.ArgumentParser(description='Aircraft Trajectory Prediction with Informer')
    parser.add_argument('--config', type=str, default='./config/aircraft.yaml', help='Configuration file path')
    cfg_args = parser.parse_args()

    # 加载配置
    args = load_config(cfg_args.config)

    # 打印关键参数
    print("--- 训练配置 ---")
    print(f"原始数据路径: {args.data_csv_path}")
    print(f"输入/预测长度: {args.seq_len}/{args.pred_len}")
    print(f"使用GPU: {args.use_gpu}, ID: {args.gpu}")
    print("----------------")

    # 创建训练器并开始训练
    trainer = Trainer(args)

    print("开始训练模型...")
    trainer.train()

    print("开始测试模型...")
    preds, trues = trainer.test()

    # 保存预测结果
    torch.save(preds, 'predictions.pt')
    torch.save(trues, 'ground_truth.pt')

    print("预测结果已保存为 predictions.pt 和 ground_truth.pt")

    print("训练和测试完成！")


if __name__ == '__main__':
    # 确保可以找到 yaml 和 easydict
    try:
        import yaml
        from easydict import EasyDict as edict
    except ImportError:
        print("请安装依赖: pip install pyyaml easydict")
        exit()

    main()
