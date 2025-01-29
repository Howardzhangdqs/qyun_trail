import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

try:
    from model.model import LaneDetectionModel
    from data.dataset import LaneDetectionDataset
    from config import config as model_config
except ImportError:
    raise ImportError("Please run this script from project root directory")


# 训练配置
class TrainingConfig:
    # 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = True  # 混合精度训练
    grad_accum_steps = 2  # 梯度累积步数

    # 优化参数
    lr = 3e-4
    weight_decay = 0.05
    max_grad_norm = 1.0  # 梯度裁剪

    # 训练参数
    epochs = 100
    batch_size = 64
    warmup_epochs = 5  # 学习率预热

    # 正则化
    label_smoothing = 0.1

    # 路径配置
    log_dir = "./logs"
    checkpoint_dir = "./checkpoints"


# 数据增强配置
def get_transforms():
    from torchvision import transforms
    from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip
    from albumentations.pytorch import ToTensorV2
    import numpy as np

    # Albumentations转换（处理numpy数组）
    albumentations_transform = Compose([
        RandomBrightnessContrast(p=0.5),
        HorizontalFlip(p=0.5),
        ToTensorV2()
    ])

    # 标准转换
    standard_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: np.array(x)),  # 转换为numpy数组供Albumentations处理
        transforms.Lambda(lambda x: albumentations_transform(image=x)["image"]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return standard_transform


class Trainer:
    def __init__(self, config):
        self.config = config
        self.scaler = GradScaler(device="cuda", enabled=config.amp_enabled)

        # 初始化模型
        self.model = LaneDetectionModel().to(config.device)

        # 数据加载
        transform = get_transforms()
        train_dataset = LaneDetectionDataset(
            "./data/processed/train",
            transform=transform
        )
        val_dataset = LaneDetectionDataset(
            "./data/processed/val",
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
            val_dataset,
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

        # 日志记录
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                config.log_dir,
                datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        accum_steps = 0

        for batch_idx, (images, angles) in enumerate(self.train_loader):
            images = images.to(self.config.device)
            angles = angles.to(self.config.device)

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
                self.writer.add_scalar(
                    "train/lr",
                    current_lr,
                    epoch * len(self.train_loader) + batch_idx
                )

            # 日志记录
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

        return total_loss / accum_steps

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0

        for images, angles in self.val_loader:
            images = images.to(self.config.device)
            angles = angles.to(self.config.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs.squeeze(), angles)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("val/loss", avg_loss, epoch)
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
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 保存检查点
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.save_checkpoint(epoch)

            self.writer.add_scalars(
                "loss",
                {"train": train_loss, "val": val_loss},
                epoch
            )


if __name__ == "__main__":
    os.makedirs(TrainingConfig.checkpoint_dir, exist_ok=True)
    os.makedirs(TrainingConfig.log_dir, exist_ok=True)

    trainer = Trainer(TrainingConfig())
    trainer.train()
