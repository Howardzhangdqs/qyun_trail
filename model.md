## 动态蛇形卷积 (Dynamic Snake Convolution)

对于输入特征图 $X\in\mathbb{R}^{C\times H\times W}$，偏移量生成：

$$\Theta = \text{Conv}_{3\times3}(X) \in \mathbb{R}^{2K^2\times H\times W}$$

其中 $K$ 为卷积核尺寸。变形后的特征图：

$$X' = \text{GridSample}(X, \mathcal{G} + \Theta)$$

最终输出：

$$Y = \text{Conv}_{K\times K}(X') \in \mathbb{R}^{C'\times H'\times W'}$$


## Ghost 瓶颈模块 (Ghost Bottleneck)

输入 $X\in\mathbb{R}^{C\times H\times W}$，主要分支：

$$X_1 = \text{SiLU}(\text{BN}(\text{Conv}_{1\times1}(X))) \in\mathbb{R}^{C'/2\times H\times W}$$

Cheap分支：

$$X_2 = \text{SiLU}(\text{BN}(\text{DWConv}_{3\times3}(X_1))) \in\mathbb{R}^{C'/2\times H/s\times W/s}$$

当stride=2时：

$$X_1' = \text{AvgPool}_{2\times2}(X_1)$$

融合后：

$$Y = \text{SE}(\text{Concat}(X_1', X_2)) + \text{Shortcut}(X)$$


## 轻量级SE模块 (SlimSE)

全局上下文建模：

$$z_c = \frac{1}{HW}\sum_{i=1}^H\sum_{j=1}^W x_{c,i,j}$$

通道权重：

$$s = \sigma(\mathbf{W}_2 \cdot \text{SiLU}(\mathbf{W}_1 \cdot z))$$

最终输出：

$$y_{c,i,j} = s_c \cdot x_{c,i,j}$$


## BiFPN融合过程

设输入特征为 $\{P_3, P_4, P_5\}$，通道调整：

$$\hat{P}_l = \text{Conv}_{1\times1}(P_l),\ l\in\{3,4,5\}$$

特征融合：

$$
\begin{aligned}
& P_6 &&= \text{MaxPool}(\hat{P}_5) \\
& P_5^{up} &&= \mathcal{U}(\hat{P}_5) \\
& P_4^{fuse} &&= \hat{P}_4 + P_5^{up} \\
& P_3^{fuse} &&= \hat{P}_3 + \mathcal{U}(P_4^{fuse})
\end{aligned}
$$

加权融合：

$$
Y = \sum_{k=1}^4 \frac{e^{w_k}}{\sum_{i=1}^4 e^{w_i}} \cdot \mathcal{U}^n(\hat{P}_k)
$$

其中 $\mathcal{U}^n$ 表示 $n$ 次上采样操作


## 完整前向过程

$$
\begin{aligned}
X_0 &= \text{Stem}(I) \\
\{X_1, X_2, X_3\} &= \text{MobileNetV3Lite}(X_0) \\
F &= \text{BiFPNLite}(\{X_1, X_2, X_3\}) \\
\hat{\theta} &= \text{MLP}(\text{GAP}(\text{Conv}_{3\times3}(F)))
\end{aligned}
$$

其中各符号定义：
- $\text{Conv}_{k\times k}$: $k×k$ 卷积
- $\text{DWConv}$: 深度可分离卷积
- $\mathcal{U}$: 双线性上采样
- $\text{GAP}$: 全局平均池化
- $\sigma$: $\text{Sigmoid}$ 激活函数
- $\text{SiLU}$: $\text{Swish}$ 激活函数

模型最终输出转向角预测值 $\hat{\theta}\in[-1,1]$，其中 $-1$ 表示最大左转，$1$ 表示最大右转。