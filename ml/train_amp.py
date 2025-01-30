import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import sys
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

try:
    from .model import LaneDetectionModel
    from .data import LaneDetectionDataset
    from config import config as model_config
except ImportError:
    try:
        from model import LaneDetectionModel
        from data import LaneDetectionDataset
        from config import config as model_config
    except ImportError:
        raise ImportError("Please run this script from project root directory")


def abs_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


# 训练配置
class TrainingConfig:
    # 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = False  # 混合精度训练
    grad_accum_steps = 2  # 梯度累积步数

    # 优化参数
    lr = 3e-4
    weight_decay = 0.05
    max_grad_norm = 1.0  # 梯度裁剪

    # 训练参数
    epochs = 100
    batch_size = 16

    # 学习率预热
    warmup_epochs = 5

    # 正则化
    label_smoothing = 0.1

    # 路径配置
    log_dir = abs_path("../logs")
    checkpoint_dir = abs_path("../checkpoints")

    def print(self):
        # 输出一整行=
        # 获取行宽
        line_width = os.get_terminal_size().columns
        print("=" * line_width)
        print("训练配置:")
        print("\n".join(
            map(lambda x: "  " + x, [
                f"设备: {self.device}",
                f"混合精度 AMP: {self.amp_enabled}",
                f"梯度累积步数: {self.grad_accum_steps}",
                f"学习率: {self.lr}",
                f"权重衰减: {self.weight_decay}",
                f"最大梯度裁剪: {self.max_grad_norm}",
                f"训练轮数: {self.epochs}",
                f"Batch Size: {self.batch_size}",
                f"预热轮数: {self.warmup_epochs}",
                f"日志目录: {self.log_dir}",
                f"Check Point 目录: {self.checkpoint_dir}"
            ])
        ))
        print("=" * line_width)


class AlbumentationsWrapper:
    def __init__(self):
        self.transform = Compose([
            RandomBrightnessContrast(p=0.5),
            HorizontalFlip(p=0.5),
            ToTensorV2()
        ])

    def __call__(self, x):
        _ = self.transform(image=np.array(x))["image"].permute(1, 2, 0)
        return _


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        AlbumentationsWrapper(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class Trainer:
    def __init__(self, config):
        self.config = config
        self.scaler = GradScaler(device="cuda", enabled=config.amp_enabled)

        # 初始化模型
        self.model = LaneDetectionModel().to(config.device)

        # 数据加载
        transform = get_transforms()
        train_dataset = LaneDetectionDataset(
            mode="train",
            transform=transform
        )
        test_dataset = LaneDetectionDataset(
            mode="test",
            transform=transform
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=2,
            pin_memory=True
        )

        # 优化器和损失函数
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.loss_fn = nn.SmoothL1Loss(beta=0.1)

        # 学习率调度
        total_steps = len(self.train_loader) * config.epochs
        warmup_steps = len(self.train_loader) * config.warmup_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos'
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        accum_steps = 0

        process_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        process_bar.set_description(f"Epoch {epoch+1}/{self.config.epochs}")
        current_lr = 0  # Initialize current_lr

        for batch_idx, (images, angles) in process_bar:
            images = images.to(self.config.device)
            angles = angles.to(self.config.device)

            # print(images.shape, angles.shape)

            with autocast(device_type="cuda", enabled=self.config.amp_enabled):
                outputs = self.model(images)
                loss = self.loss_fn(outputs.squeeze(), angles)
                loss = loss / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()

            # 梯度累积
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # 参数更新
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

                accum_steps += 1
                total_loss += loss.item()

                # 记录学习率
                current_lr = self.optimizer.param_groups[0]["lr"]

            process_bar.set_postfix({
                "loss": total_loss / (accum_steps if accum_steps > 0 else 1),
                "lr": current_lr
            })

        return total_loss / accum_steps

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0

        process_bar = tqdm(self.val_loader, total=len(self.val_loader))
        process_bar.set_description(f"Val Epoch {epoch+1}/{self.config.epochs}")

        for images, angles in process_bar:
            images = images.to(self.config.device)
            angles = angles.to(self.config.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs.squeeze(), angles)
            total_loss += loss.item()

            process_bar.set_postfix({
                "val_loss": total_loss / (process_bar.n + 1)
            })

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_loss": self.best_loss
        }

        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"

        torch.save(
            state,
            os.path.join(self.config.checkpoint_dir, filename)
        )

    def train(self):
        self.best_loss = float("inf")

        for epoch in range(self.config.epochs):
            print()

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 保存检查点
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.save_checkpoint(epoch)


if __name__ == "__main__":
    os.makedirs(TrainingConfig.checkpoint_dir, exist_ok=True)
    os.makedirs(TrainingConfig.log_dir, exist_ok=True)

    trainer = Trainer(TrainingConfig())
    trainer.train()
