import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

try:
    from config import config
    from utils.data_utils import parse_filename
except ImportError:
    raise ImportError("Please run this script from project root directory")


class LaneDetectionDataset(Dataset):
    def __init__(self, mode: str = "train", data_dir=config.preprocessed_path, transform=None, size=tuple(config.img_size)):

        if mode not in ["train", "val", "test"]:
            raise ValueError("Invalid mode, must be one of ['train', 'val', 'test']")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Data directory is not a directory: {data_dir}")

        if not os.listdir(data_dir):
            raise FileNotFoundError(f"Data directory is empty: {data_dir}")

        data_dir = os.path.join(data_dir, mode)

        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.pretransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
        ])

        # 预加载数据信息
        for filename in os.listdir(data_dir):
            if filename.endswith(".png"):
                try:
                    info = parse_filename(filename)
                    self.samples.append({
                        "path": os.path.join(data_dir, filename),
                        "angle": torch.tensor(info["angle"], dtype=torch.float32)
                    })
                except ValueError:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["path"])

        img = self.pretransform(img)

        if self.transform:
            img = self.transform(img)

        return img, sample["angle"]


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = LaneDetectionDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Total samples: {len(dataset)}")

    for i, (imgs, angles) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  imgs.shape: {imgs.shape}")
        print(f"  angles: {angles}")
        break

    # 输出angles的均值和方差
    angles = torch.cat([sample["angle"].unsqueeze(0) for sample in dataset.samples])
    print(f"Mean angle: {angles.mean()}")
