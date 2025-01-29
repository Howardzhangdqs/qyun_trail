import sys
import os
import shutil
import cv2
import numpy as np
from random import randint
from uuid import uuid4
from tqdm import trange

from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from config import config
except ImportError:
    raise ImportError("Please run this script from project root directory")


# 生成随机噪声图像
def generate_noise_image(size: Tuple[int, int]) -> np.ndarray:
    noise = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return noise


# 生成随机文件名
def generate_random_filename() -> str:
    frame_id = uuid4().hex[:8]
    steering_angle = np.random.uniform(-1, 1)
    speed = np.random.uniform(0, 10)

    return f"{frame_id}_{int((steering_angle + 1) * 1000)}_{int(speed * 10)}.jpg"


if __name__ == "__main__":
    FILE_NUM = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    if os.path.exists(config.dataset_path):
        shutil.rmtree(config.dataset_path)

    os.makedirs(config.dataset_path, exist_ok=True)

    for i in trange(FILE_NUM):
        image_size = (randint(320, 640), randint(240, 480))
        filename = generate_random_filename()
        image = generate_noise_image(image_size)
        folder = config.dataset_path
        cv2.imwrite(os.path.join(folder, filename), image)
