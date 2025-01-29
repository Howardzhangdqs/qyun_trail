import sys
import os
from rich import print
import random
from tqdm import tqdm
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from config import config
    from utils.data_utils import (
        parse_filename,
        get_image_paths,
        preprocess_image,
        metadata_to_filename,
    )
except ImportError:
    raise ImportError("Please run this script from project root directory")


if __name__ == "__main__":
    data_dir = config.dataset_path
    output_dir = config.preprocessed_path
    propotion = config.dataset_propotion

    # 删除已有目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    image_paths = get_image_paths(data_dir)

    # 创建训练集和测试集目录
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 打乱数据集
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * propotion)
    train_paths = image_paths[:split_idx]
    test_paths = image_paths[split_idx:]

    # 处理训练集图片
    for image_path in tqdm(train_paths, desc="Processing train images"):
        filename = os.path.basename(image_path)
        metadata = parse_filename(filename)
        new_filename = metadata_to_filename(metadata)
        preprocess_image(image_path, os.path.join(train_dir, new_filename))

    # 处理测试集图片
    for image_path in tqdm(test_paths, desc="Processing test images"):
        filename = os.path.basename(image_path)
        metadata = parse_filename(filename)
        new_filename = metadata_to_filename(metadata)
        preprocess_image(image_path, os.path.join(test_dir, new_filename))
