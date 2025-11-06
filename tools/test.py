# tools/test.py
import torch
import numpy as np
import argparse
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

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

    # 注意：这里假设检查点路径是完整的模型文件路径
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    model.eval()
    return model


def get_test_dataloader(args):
    """加载测试数据集并返回 DataLoader"""
    # 这里会触发数据的清洗和标准化
    print("--- 正在初始化 test 数据集并进行清洗和标准化 (仅执行一次) ---")
    test_data = AircraftTrajectoryDataset(
        csv_path=args.data_csv_path,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        mode='test',
        scale=True,
        speed_min=args.speed_min,
        speed_max=args.speed_max
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    return test_loader


def evaluate_model(model, test_loader, args):
    """对整个测试集进行评估，计算并打印指标"""
    model.eval()

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)

            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(args.device)

            all_preds.append(outputs.detach().cpu())
            all_trues.append(batch_y.detach().cpu())

    # 计算指标
    total_mae, total_mse, total_rmse = 0, 0, 0
    num_batches = len(all_preds)

    if num_batches == 0:
        print("Warning: 测试集中没有足够长的序列可用于评估。")
        return None, None

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

    print(f'\n--- 完整测试集评估结果 ---')
    print(f'测试 MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}')
    print(f'----------------------------')

    return all_preds, all_trues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/aircraft.yaml', help='Configuration file path')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint.pth', help='模型检查点路径')
    # 移除 --predict_index 参数，专注于完整评估

    cfg_args = parser.parse_args()

    # 加载配置
    args = load_config(cfg_args.config)
    args.checkpoint = cfg_args.checkpoint  # 覆盖checkpoint路径

    print(f"模型加载设备: {args.device}")

    # 1. 加载模型
    model = load_model(args.checkpoint, args)
    print(f"模型加载成功，设备: {args.device}")

    # 2. 加载测试数据的 DataLoader (执行清洗和标准化)
    test_loader = get_test_dataloader(args)

    # 3. 对整个测试集进行评估
    print("\n开始对整个测试集进行评估...")
    preds, trues = evaluate_model(model, test_loader, args)

    if preds is not None:
        # 4. 保存预测结果
        torch.save(preds, 'predictions_test.pt')
        torch.save(trues, 'ground_truth_test.pt')
        print("\n完整测试结果已保存为 predictions_test.pt 和 ground_truth_test.pt")


if __name__ == "__main__":
    # 确保可以找到 yaml 和 easydict
    try:
        import yaml
        from easydict import EasyDict as edict
    except ImportError:
        print("请安装依赖: pip install pyyaml easydict")
        exit()

    main()