import sys
import os
import argparse
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from ml import Trainer, TrainingConfig
except ImportError:
    raise ImportError("Please run this script from project root directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练模型")

    parser.add_argument("-d", "--device", type=str,
                        default=("cuda" if torch.cuda.is_available() else "cpu"),
                        help="用于训练模型的设备")
    parser.add_argument("-c", "--checkpoint", type=str, default=TrainingConfig.checkpoint_dir,
                        help="继续训练的模型检查点路径")
    parser.add_argument("-l", "--log", type=str, default=TrainingConfig.log_dir,
                        help="日志目录路径")
    parser.add_argument("-e", "--epochs", type=int, default=TrainingConfig.epochs,
                        help="训练的轮数")
    parser.add_argument("-b", "--batch_size", type=int, default=TrainingConfig.batch_size,
                        help="批处理大小")
    parser.add_argument("-lr", "--learning_rate", type=float, default=TrainingConfig.lr,
                        help="学习率")
    parser.add_argument("-w", "--warmup", type=int, default=TrainingConfig.warmup_epochs,
                        help="预热轮数")
    parser.add_argument("-wd", "--weight_decay", type=float, default=TrainingConfig.weight_decay,
                        help="权重衰减")
    parser.add_argument("-g", "--grad_accum", type=int, default=TrainingConfig.grad_accum_steps,
                        help="梯度累积步数")
    parser.add_argument("-a", "--amp", type=bool, default=TrainingConfig.amp_enabled,
                        help="启用自动混合精度训练")

    args = parser.parse_args()

    training_config = TrainingConfig()

    training_config.device = torch.device(args.device)
    training_config.checkpoint_dir = args.checkpoint
    training_config.log_dir = args.log
    training_config.epochs = args.epochs
    training_config.batch_size = args.batch_size
    training_config.lr = args.learning_rate
    training_config.warmup_epochs = args.warmup
    training_config.weight_decay = args.weight_decay
    training_config.grad_accum_steps = args.grad_accum
    training_config.amp_enabled = args.amp

    training_config.print()

    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)

    trainer = Trainer(training_config)
    trainer.train()
