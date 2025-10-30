# preprocess_data.py
import numpy as np


def filter_short_sequences(input_path, output_path, min_length=50):
    """过滤掉太短的序列"""
    data = np.load(input_path, allow_pickle=True)
    lengths = np.load(input_path.replace('.npy', '_lengths.npy'), allow_pickle=True)

    filtered_data = []
    filtered_lengths = []

    for i, trajectory in enumerate(data):
        if len(trajectory) >= min_length:
            filtered_data.append(trajectory)
            filtered_lengths.append(len(trajectory))

    print(f"原始序列数量: {len(data)}")
    print(f"过滤后序列数量: {len(filtered_data)}")
    print(f"保留比例: {len(filtered_data) / len(data) * 100:.2f}%")

    # 保存过滤后的数据
    np.save(output_path, np.array(filtered_data, dtype=object))
    np.save(output_path.replace('.npy', '_lengths.npy'), np.array(filtered_lengths))

    return filtered_data, filtered_lengths


if __name__ == "__main__":
    # 过滤掉长度小于50的序列
    filter_short_sequences("result.npy", "result_filtered.npy", min_length=50)