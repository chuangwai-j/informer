import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.model import Informer
from data.data_loader import AircraftTrajectoryDataset, custom_collate_fn


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

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if self.args.gpu else "0"
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
        data_dict = {
            'train': AircraftTrajectoryDataset(
                self.args.data_path,
                seq_len=self.args.seq_len,
                label_len=self.args.label_len,
                pred_len=self.args.pred_len,
                mode='train',
                scale=True
            ),
            'val': AircraftTrajectoryDataset(
                self.args.data_path,
                seq_len=self.args.seq_len,
                label_len=self.args.label_len,
                pred_len=self.args.pred_len,
                mode='val',
                scale=True
            ),
            'test': AircraftTrajectoryDataset(
                self.args.data_path,
                seq_len=self.args.seq_len,
                label_len=self.args.label_len,
                pred_len=self.args.pred_len,
                mode='test',
                scale=True
            )
        }
        return data_dict[flag]

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self):
        train_data = self._get_data('train')
        val_data = self._get_data('val')

        train_loader = DataLoader(
            train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=min(4, self.args.num_workers),  # 减少worker数量
            collate_fn=custom_collate_fn  # 添加自定义collate函数
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=min(4, self.args.num_workers),  # 减少worker数量
            collate_fn=custom_collate_fn  # 添加自定义collate函数
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
        test_data = self._get_data('test')
        test_loader = DataLoader(
            test_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=custom_collate_fn
        )

        self.model.eval()

        all_preds = []  # 保存每个batch的预测
        all_trues = []  # 保存每个batch的真实值

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

        # 分别计算每个batch的指标，然后求平均
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

        # 返回列表而不是数组
        return all_preds, all_trues

    def save_predictions(preds, trues, prefix=''):
        """安全地保存预测结果"""
        # 方法1: 保存为torch文件
        torch.save(preds, f'{prefix}predictions.pt')
        torch.save(trues, f'{prefix}ground_truth.pt')

        # 方法2: 如果preds和trues是numpy数组，尝试保存
        try:
            if isinstance(preds, np.ndarray):
                np.save(f'{prefix}predictions.npy', preds)
            if isinstance(trues, np.ndarray):
                np.save(f'{prefix}ground_truth.npy', trues)
        except Exception as e:
            print(f"无法保存为numpy格式: {e}")

        print(f"预测结果已保存为 {prefix}predictions.pt 和 {prefix}ground_truth.pt")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Aircraft Trajectory Prediction with Informer')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='result.npy', help='data file path')
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='start token length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')

    # 模型参数
    parser.add_argument('--enc_in', type=int, default=4, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=4, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=4, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--distil', action='store_true', help='whether to use distilling in encoder', default=True)
    parser.add_argument('--mix', action='store_true', help='use mix attention in generative decoder', default=True)

    # 训练参数
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    # 检查点
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    args = parser.parse_args()

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
    main()