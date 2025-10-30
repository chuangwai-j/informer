# predict.py
import torch
import numpy as np
import argparse
from models.model import Informer
from data.data_loader import AircraftTrajectoryDataset


def load_model(checkpoint_path, args):
    """加载训练好的模型"""
    model = Informer(
        args.enc_in, args.dec_in, args.c_out,
        args.seq_len, args.label_len, args.pred_len,
        factor=args.factor, d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=args.d_layers, d_ff=args.d_ff,
        dropout=args.dropout, attn=args.attn, embed=args.embed, freq=args.freq,
        activation=args.activation, output_attention=args.output_attention,
        distil=args.distil, mix=args.mix, device=args.device
    ).float()

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def predict_trajectory(model, input_sequence, dataset):
    """使用模型预测轨迹"""
    with torch.no_grad():
        # 标准化输入
        input_std = (input_sequence - dataset.data_mean) / dataset.data_std

        # 添加批次维度
        input_tensor = torch.FloatTensor(input_std).unsqueeze(0).to(args.device)

        # 创建时间标记（这里简化处理，实际应该根据你的时间特征生成）
        seq_len = input_tensor.shape[1]
        time_mark = torch.zeros(1, seq_len, 1).to(args.device)

        # 解码器输入
        dec_inp = torch.zeros_like(input_tensor[:, -args.pred_len:, :])
        dec_inp = torch.cat([input_tensor[:, :args.label_len, :], dec_inp], dim=1)
        dec_time_mark = torch.zeros(1, args.label_len + args.pred_len, 1).to(args.device)

        # 预测
        outputs = model(input_tensor, time_mark, dec_inp, dec_time_mark)
        outputs = outputs[:, -args.pred_len:, :]

        # 反标准化
        pred_denorm = outputs.cpu().numpy()[0] * dataset.data_std + dataset.data_mean

        return pred_denorm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint.pth', help='模型检查点路径')
    parser.add_argument('--data_path', type=str, default='result_filtered.npy', help='数据路径')
    parser.add_argument('--seq_len', type=int, default=48, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=12, help='预测序列长度')
    args = parser.parse_args()

    # 设置设备
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = load_model(args.checkpoint, args)
    print(f"模型加载成功，设备: {args.device}")

    # 加载数据集用于标准化参数
    dataset = AircraftTrajectoryDataset(args.data_path, mode='test')

    # 这里可以添加你的预测逻辑-
    # 例如从数据集中取一个样本进行预测
    print("预测功能准备就绪！")