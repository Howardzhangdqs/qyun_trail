import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.data_utils import parse_filename


class LaneDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

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

        if self.transform:
            img = self.transform(img)

        return img, sample["angle"]
