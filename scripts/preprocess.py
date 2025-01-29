import sys
import os
from rich import print

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from config import config
    from utils.data_utils import parse_filename, get_image_paths, preprocess_image, format_metadata
except ImportError:
    raise ImportError("Please run this script from project root directory")


if __name__ == "__main__":
    data_dir = config.dataset_path
    output_dir = config.preprocessed_path

    os.makedirs(output_dir, exist_ok=True)

    image_paths = get_image_paths(data_dir)
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        metadata = parse_filename(filename)
        preprocess_image(image_path, os.path.join(output_dir, filename))
        print(f"Processed [yellow underline]{filename}[/yellow underline] -> {format_metadata(metadata)}")
