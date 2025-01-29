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
    def __init__(self, data_dir=config.preprocessed_path, transform=None, size=(128, 128)):
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

    for i, (imgs, angles) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  imgs.shape: {imgs.shape}")
        print(f"  angles: {angles}")
        break
