# tools/test.py
import torch
import numpy as np
import argparse
import yaml
from easydict import EasyDict as edict

# 导入模型和数据加载器
from models.model import Informer
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

        # 将args中的gpu信息配置到device
        if args.use_gpu and torch.cuda.is_available():
            args.device = torch.device(f'cuda:{args.gpu}')
        else:
            args.device = torch.device('cpu')

        return args
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit()
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit()


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
    ).float().to(args.device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    model.eval()
    return model


def predict_trajectory(model, input_sequence, data_mean, data_std, args):
    """
    使用模型预测轨迹
    input_sequence: 原始（未标准化）的N个时间步的序列 numpy array, shape (N, 4)
    data_mean, data_std: 标准化参数 numpy array, shape (4,)
    """
    with torch.no_grad():
        # 1. 标准化输入
        input_std = (input_sequence - data_mean) / data_std

        # 2. 转换为张量并添加批次维度
        input_tensor = torch.FloatTensor(input_std).unsqueeze(0).to(args.device)

        # 检查输入长度
        if input_tensor.shape[1] < args.seq_len:
            raise ValueError(f"输入序列长度不足 {args.seq_len}，无法进行预测。")

        # 提取用于Encoder的序列
        seq_x = input_tensor[:, -args.seq_len:, :]

        # 3. 创建时间标记 (这里需要一个真实的、基于时间的标记)
        # 由于我们无法获取预测时间步的原始时间戳，这里进行简化，仅提供一个占位符。
        # 实际应用中，需要根据预测时间长度生成对应的相对时间特征。

        # 假设 seq_x 对应的原始时间戳 (这里为简化，直接从最后一个时间步向前推 seq_len 个时间步)
        # 生产环境需要传入原始时间戳
        # 为了演示，我们使用与训练集特征数匹配的零张量作为占位符
        time_feature_dim = 4  # 对应 data_loader.py 中的4个特征
        seq_x_mark = torch.zeros(1, args.seq_len, time_feature_dim).float().to(args.device)

        # 4. 解码器输入
        dec_inp = torch.zeros(1, args.label_len + args.pred_len, args.c_out).float().to(args.device)
        dec_inp[:, :args.label_len, :] = seq_x[:, -args.label_len:, :]

        # 解码器时间标记
        dec_time_mark = torch.zeros(1, args.label_len + args.pred_len, time_feature_dim).float().to(args.device)

        # 5. 预测
        outputs = model(seq_x, seq_x_mark, dec_inp, dec_time_mark)
        outputs = outputs[:, -args.pred_len:, :]

        # 6. 反标准化
        pred_denorm = outputs.cpu().numpy()[0] * data_std + data_mean

        return pred_denorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/aircraft.yaml', help='Configuration file path')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint.pth', help='模型检查点路径')
    # 可以添加一个参数来指定要预测的轨迹 index
    parser.add_argument('--predict_index', type=int, default=0, help='选择测试集中哪个轨迹进行预测')

    cfg_args = parser.parse_args()

    # 加载配置
    args = load_config(cfg_args.config)
    args.checkpoint = cfg_args.checkpoint  # 覆盖checkpoint路径

    print(f"模型加载设备: {args.device}")

    # 1. 加载模型
    model = load_model(args.checkpoint, args)
    print(f"模型加载成功，设备: {args.device}")

    # 2. 加载数据集用于获取标准化参数和测试样本
    # mode='test' 会加载测试集数据，并进行标准化
    dataset = AircraftTrajectoryDataset(
        csv_path=args.data_csv_path,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        mode='test',
        speed_min=args.speed_min,
        speed_max=args.speed_max
    )

    if len(dataset.used_data) == 0:
        print("Warning: 测试集无足够长轨迹，无法进行预测演示。")
        return

    # 3. 选取一个样本进行预测
    try:
        # 获取原始（未标准化）轨迹
        trajectory_full = dataset.used_data[cfg_args.predict_index]

        # 提取用于输入模型的序列 (取最后 seq_len 个时间步)
        input_data_full = trajectory_full[:, 1:]  # 排除时间列

        # 确保输入数据长度满足要求
        if input_data_full.shape[0] < args.seq_len:
            print(f"错误: 选定的轨迹 ({cfg_args.predict_index}) 长度不足 {args.seq_len}，无法预测。")
            return

        input_sequence = input_data_full[-args.seq_len:]

        print(f"正在使用测试集中的第 {cfg_args.predict_index} 条轨迹的最后 {args.seq_len} 个点进行预测...")

        # 4. 预测
        predicted_trajectory = predict_trajectory(
            model,
            input_sequence,
            dataset.data_mean,
            dataset.data_std,
            args
        )

        print(f"预测完成。预测序列长度: {args.pred_len}, 预测特征维度: {predicted_trajectory.shape[1]}")
        # print("预测结果 (前5行):")
        # print(predicted_trajectory[:5])

    except IndexError:
        print(f"错误: 预测索引 {cfg_args.predict_index} 超出测试集范围 (0 到 {len(dataset.used_data) - 1})")
    except Exception as e:
        print(f"预测过程中发生错误: {e}")


if __name__ == "__main__":
    # 确保可以找到 yaml 和 easydict
    try:
        import yaml
        from easydict import EasyDict as edict
    except ImportError:
        print("请安装依赖: pip install pyyaml easydict")
        exit()

    main()