import os
import re
from PIL import Image
import cv2


def parse_filename(filename):
    """解析文件名提取元数据"""
    pattern = r"^([a-zA-Z0-9]+)_(\d+)_(\d+)\.(png|jpg)$"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")

    return {
        "imgid": match.group(1),
        "angle": int(match.group(2)) / 1000.0 - 1,  # 还原到[-1, 1]范围
        "speed": int(match.group(3)) / 10.0         # 还原到[0, 10]范围
    }


def metadata_to_filename(metadata):
    """根据元数据生成文件名"""
    return f"{metadata['imgid']}_{int((metadata['angle'] + 1) * 1000)}_{int(metadata['speed'] * 10)}.png"


def img_preprocess(src: cv2.Mat) -> cv2.Mat:

    # --------------------------
    # 在这里添加的预处理逻辑
    # 例如：尺寸调整、颜色空间转换、归一化等
    # --------------------------

    return src


def preprocess_image(src_path, dst_path):
    src = cv2.imread(src_path)
    dst = img_preprocess(src)
    cv2.imwrite(dst_path, dst)


def get_image_paths(data_dir):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return image_paths


def format_metadata(metadata):
    return f"imgid: {metadata['imgid']}, angle: {metadata['angle']}, speed: {metadata['speed']}"


if __name__ == "__main__":
    from config import config
    data_dir = config.dataset_path
    output_dir = config.preprocessed_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = get_image_paths(data_dir)
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        metadata = parse_filename(filename)

        print(f"Preprocessing {filename} with metadata: {format_metadata(metadata)}")

        preprocess_image(image_path, os.path.join(output_dir, metadata_to_filename(metadata)))
