import torch
import torch.nn as nn
import torch.nn.functional as F


# 骨干网络配置
backbone_cfg = {
    '0.25x': {'channels': [16, 24, 48, 96], 'layers': [2, 3, 3], 'se_stages': [2, 3]},
    '0.5x':  {'channels': [24, 32, 64, 128], 'layers': [3, 4, 4], 'se_stages': [1, 2, 3]},
    '1x':    {'channels': [32, 48, 96, 192], 'layers': [3, 6, 6], 'se_stages': [0, 1, 2, 3]},
    '2x':    {'channels': [48, 64, 128, 256], 'layers': [4, 8, 8], 'se_stages': [0, 1, 2, 3]}
}


# 动态蛇形卷积模块
class DynamicSnakeConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.offset = nn.Conv2d(in_ch, 2*kernel_size*kernel_size, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size//2)

    def forward(self, x):
        offset = self.offset(x)
        N, C, H, W = offset.size()
        offset = offset.view(N, 2, -1, H, W)  # 分解为x,y偏移量
        grid = self._create_grid(H, W, x.device)
        warped = F.grid_sample(x, (grid + offset).permute(0, 2, 3, 1), align_corners=True)
        return self.conv(warped)

    def _create_grid(self, H, W, device):
        return torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device)), dim=0).unsqueeze(0)


# 轻量级SE模块
class SlimSE(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Ghost模块
class GhostBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=False):
        super().__init__()
        self.stride = stride

        # Primary分支
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.SiLU(inplace=True)
        )

        # Cheap分支（深度可分离卷积）
        self.cheap = nn.Sequential(
            nn.Conv2d(out_ch // 2, out_ch // 2, 3, stride=stride, padding=1,
                      groups=out_ch // 2, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.SiLU(inplace=True)
        )

        # 下采样分支（如果需要）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        # SE注意力模块
        self.se = SlimSE(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)

        # 如果stride=2，需要对x1进行下采样以匹配x2的尺寸
        if self.stride != 1:
            x1 = F.avg_pool2d(x1, kernel_size=2, stride=2)

        out = torch.cat([x1, x2], dim=1)
        out = self.se(out)

        shortcut = self.shortcut(x)
        return out + shortcut


# 轻量级MobileNetV3骨干网络
class MobileNetV3Lite(nn.Module):
    def __init__(self, scale: str = '1x', in_channels: int = 3):
        super().__init__()
        cfg = backbone_cfg[scale]
        self.out_channels_list = []  # 新增输出通道记录

        # 初始stem层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, cfg['channels'][0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg['channels'][0]),
            nn.Hardswish()
        )
        current_channels = cfg['channels'][0]

        # 构建各阶段
        self.stages = nn.ModuleList()
        for i in range(3):
            in_ch = current_channels
            out_ch = cfg['channels'][i+1]
            stage = self._make_stage(in_ch, out_ch, cfg['layers'][i], use_se=(i in cfg['se_stages']))
            self.stages.append(stage)
            current_channels = out_ch
            self.out_channels_list.append(current_channels)  # 记录每阶段输出通道

    def _make_stage(self, in_ch, out_ch, num_blocks, use_se):
        blocks = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            blocks.append(GhostBottleneck(
                in_ch if i == 0 else out_ch,
                out_ch,
                stride=stride,
                use_se=use_se
            ))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# 轻量级BiFPN
class BiFPNLite(nn.Module):
    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        self.out_channels = out_channels

        # 上采样层
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 通道调整卷积
        self.p3_conv = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.p4_conv = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.p5_conv = nn.Conv2d(in_channels_list[2], out_channels, 1)

        # 修正p6_conv
        self.p6_conv = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, 1)  # 输入通道数改为out_channels
        )

        # 4个特征层的权重
        self.weights = nn.Parameter(torch.ones(4))

    def forward(self, features):
        p3, p4, p5 = features

        # 调整通道数
        p3 = self.p3_conv(p3)  # [B, C, H, W]
        p4 = self.p4_conv(p4)  # [B, C, H, W]
        p5 = self.p5_conv(p5)  # [B, C, H, W]

        # 生成p6
        p6 = self.p6_conv(p5)  # [B, C, H/2, W/2]

        # 上采样并融合
        p5_up = self.up_sample(p5)  # p5上采样到p4的尺寸
        p4_fused = p4 + p5_up       # 通道数一致，可以相加

        p4_up = self.up_sample(p4_fused)  # p4上采样到p3的尺寸
        p3_fused = p3 + p4_up             # 通道数一致，可以相加

        # 将所有特征图上采样到p3的尺寸
        p4_fused_up = self.up_sample(p4_fused)
        p5_up = self.up_sample(self.up_sample(p5))
        p6_up = self.up_sample(self.up_sample(self.up_sample(p6)))

        # 多尺度特征加权融合
        weights = F.softmax(self.weights, dim=0)

        fused = torch.cat([
            weights[0] * p3_fused,
            weights[1] * p4_fused_up,
            weights[2] * p5_up,
            weights[3] * p6_up
        ], dim=1)

        out = nn.functional.conv2d(
            fused,
            weight=torch.ones(self.out_channels, 4, 1, 1).to(fused.device),
            bias=None,
            groups=self.out_channels
        )
        return out


# 在模型定义部分添加转向预测头
class SteeringHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # 输出范围[-1,1]对应左/右转向
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# 修改完整模型类
class LaneDetectionModel(nn.Module):
    def __init__(self, scale='1x', in_channels=3):
        super().__init__()
        self.backbone = MobileNetV3Lite(scale, in_channels=in_channels)
        self.neck = BiFPNLite(self.backbone.out_channels_list)
        self.steering_head = SteeringHead(128)  # 新增转向头

    def forward(self, x):
        features = self.backbone(x)
        fused = self.neck(features)
        steering = self.steering_head(fused)  # 使用融合特征预测转向
        return steering.squeeze(1)


# 使用示例
if __name__ == '__main__':
    model = LaneDetectionModel('0.25x', in_channels=1)
    dummy = torch.randn(2, 1, 224, 224)
    outputs = model(dummy)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Steering output shape: {outputs.shape}")
    print(f"Steering sample: {outputs[0].item()}")
