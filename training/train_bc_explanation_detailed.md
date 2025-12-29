# train_bc.py 逐行详细解释

## 文件整体功能

这个文件实现了一个**管理式行为克隆（Managed Behavior Cloning）训练脚本**，使用 ResNet18 作为视觉编码器，MLP 作为动作预测头。它包含完整的训练循环、验证、早停、学习率调度、指标记录和可视化功能。

---

## 第一部分：导入库（第14-36行）

```python
import os
import csv
import json
import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import MarkerDataset
```

**逐行解释：**

- **`os`**: 用于文件路径操作（`os.path.join`, `os.makedirs`等）
- **`csv`**: 用于写入 CSV 格式的指标文件
- **`json`**: 用于保存配置和 JSONL 格式的指标
- **`time`**: 用于记录训练时间
- **`math`**: 数学函数（虽然代码中可能未直接使用）
- **`random`**: 用于设置随机种子
- **`argparse`**: 用于解析命令行参数
- **`dataclass`**: 用于定义数据类（`EvalStats`）
- **`typing`**: 类型提示（`Dict`, `Any`, `Tuple`）

- **`numpy as np`**: 数值计算库
- **`torch`**: PyTorch 深度学习框架
- **`torch.nn as nn`**: 神经网络模块
- **`torch.optim as optim`**: 优化器
- **`DataLoader`**: PyTorch 数据加载器
- **`torchvision.models`**: 预训练模型（ResNet18）
- **`tqdm`**: 进度条库

- **`matplotlib`**: 绘图库
  - **`matplotlib.use("Agg")`**: 设置后端为 "Agg"（无 GUI，适合服务器环境）
- **`matplotlib.pyplot as plt`**: 绘图接口
- **`from dataset import MarkerDataset`**: 导入自定义数据集类

---

## 第二部分：工具函数（第39-86行）

### `set_seed(seed: int = 42)`（第41-45行）

```python
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

**功能**：设置所有随机数生成器的种子，确保实验可复现。

**逐行解释：**
- `random.seed(seed)`: 设置 Python 内置随机数生成器
- `np.random.seed(seed)`: 设置 NumPy 随机数生成器
- `torch.manual_seed(seed)`: 设置 PyTorch CPU 随机数生成器
- `torch.cuda.manual_seed_all(seed)`: 设置所有 CUDA 设备的随机数生成器

**为什么需要？**
- **可复现性**：相同的种子会产生相同的随机数序列
- **调试**：便于重现 bug 和对比不同实验

---

### `ensure_dir(p: str)`（第47-48行）

```python
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
```

**功能**：确保目录存在，如果不存在则创建。

**参数**：
- `p`: 目录路径

**`exist_ok=True`**：如果目录已存在，不抛出异常

---

### `to_float(x)`（第50-54行）

```python
def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")
```

**功能**：安全地将值转换为浮点数，如果转换失败返回 NaN。

**使用场景**：处理可能无效的数值（例如从 CSV 读取的数据）

---

### `safe_mean(xs)`（第56-58行）

```python
def safe_mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if len(xs) else float("nan")
```

**功能**：计算列表的平均值，自动过滤掉 NaN 和 Inf。

**逐行解释：**
- `[x for x in xs if np.isfinite(x)]`: 列表推导式，只保留有限数（不是 NaN 或 Inf）
- `np.mean(xs)`: 计算平均值
- `if len(xs) else float("nan")`: 如果列表为空，返回 NaN

**为什么需要？**
- 某些指标可能包含无效值，需要安全处理

---

### `cosine_similarity(a, b, eps)`（第60-64行）

```python
def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # a,b: [B, D]
    an = torch.norm(a, dim=1).clamp_min(eps)
    bn = torch.norm(b, dim=1).clamp_min(eps)
    return (a * b).sum(dim=1) / (an * bn)
```

**功能**：计算两个张量的余弦相似度（逐样本）。

**逐行解释：**
- **输入**：`a` 和 `b` 都是形状 `[B, D]` 的张量（B 是 batch size，D 是维度）
- **`torch.norm(a, dim=1)`**: 计算每个样本的 L2 范数，结果形状 `[B]`
- **`.clamp_min(eps)`**: 将范数限制在最小值 `eps` 以上，避免除以零
- **`(a * b).sum(dim=1)`**: 逐元素相乘后求和，得到内积，形状 `[B]`
- **`/ (an * bn)`**: 除以两个范数的乘积，得到余弦相似度

**公式**：`cos(θ) = (a · b) / (||a|| * ||b||)`

**为什么需要？**
- 评估预测动作的方向是否与真实动作一致（不关心幅度）

---

### `plot_curve(x, ys, title, xlabel, ylabel, out_path)`（第66-76行）

```python
def plot_curve(x, ys: Dict[str, list], title: str, xlabel: str, ylabel: str, out_path: str):
    plt.figure()
    for k, v in ys.items():
        plt.plot(x, v, label=k)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
```

**功能**：绘制一条或多条曲线。

**逐行解释：**
- **`plt.figure()`**: 创建新图形
- **`for k, v in ys.items()`**: 遍历字典，`k` 是曲线名称，`v` 是 y 值列表
- **`plt.plot(x, v, label=k)`**: 绘制曲线，`x` 是 x 轴值，`v` 是 y 轴值
- **`plt.title(title)`**: 设置标题
- **`plt.xlabel(xlabel)`**: 设置 x 轴标签
- **`plt.ylabel(ylabel)`**: 设置 y 轴标签
- **`plt.legend()`**: 显示图例
- **`plt.tight_layout()`**: 自动调整布局，避免标签重叠
- **`plt.savefig(out_path)`**: 保存图像到文件
- **`plt.close()`**: 关闭图形，释放内存

**使用场景**：绘制训练/验证 loss 曲线、学习率曲线等

---

### `plot_hist(data, title, xlabel, out_path, bins)`（第78-86行）

```python
def plot_hist(data: np.ndarray, title: str, xlabel: str, out_path: str, bins: int = 50):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
```

**功能**：绘制直方图。

**参数**：
- `data`: 数据数组
- `bins`: 直方图的 bin 数量（默认 50）

**使用场景**：可视化数据分布（例如动作范数分布）

---

## 第三部分：数据加载（第89-111行）

### `make_loaders(dataset_root, batch_size, num_workers, image_size_hw, only_success)`（第91-111行）

```python
def make_loaders(dataset_root, batch_size=64, num_workers=4, image_size_hw=(240, 320), only_success=False):
    train_set = MarkerDataset(dataset_root, split="train", image_size_hw=image_size_hw, only_success=only_success)
    val_set = MarkerDataset(dataset_root, split="val", image_size_hw=image_size_hw, only_success=only_success)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_set, val_set, train_loader, val_loader
```

**功能**：创建训练集、验证集和对应的 DataLoader。

**逐行解释：**

**第92-93行：创建数据集**
- `MarkerDataset(..., split="train")`: 创建训练集（80% 的数据）
- `MarkerDataset(..., split="val")`: 创建验证集（20% 的数据）
- `only_success=only_success`: 如果为 True，只加载成功的 episode

**第95-102行：训练集 DataLoader**
- `batch_size=batch_size`: 每批样本数量
- `shuffle=True`: **打乱数据顺序**（训练时需要）
- `num_workers=num_workers`: **并行加载数据的进程数**（加速数据加载）
- `pin_memory=True`: **固定内存**，加速 GPU 传输
- `drop_last=True`: **丢弃最后一个不完整的 batch**（保证 batch 大小一致）

**第103-110行：验证集 DataLoader**
- `shuffle=False`: **不打乱数据**（验证时不需要）
- `drop_last=False`: **保留最后一个不完整的 batch**（不浪费数据）

**为什么训练集 `drop_last=True`？**
- 某些操作（如 BatchNorm）需要固定 batch size
- 最后一个 batch 可能很小，导致统计不稳定

**为什么验证集 `drop_last=False`？**
- 验证时不需要固定 batch size
- 保留所有数据，评估更准确

---

## 第四部分：模型定义（第114-146行）

### `ResNetMLPPolicy` 类（第116-131行）

```python
class ResNetMLPPolicy(nn.Module):
    def __init__(self, out_dim=7):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # [B,512,1,1]
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        feat = self.backbone(x).flatten(1)  # [B,512]
        return self.head(feat)              # [B,7]
```

**功能**：定义 ResNet18 + MLP 策略网络。

**逐行解释：**

**`__init__` 方法：**

**第119行：加载预训练 ResNet18**
```python
backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
```
- 加载在 ImageNet 上预训练的 ResNet18
- **预训练的好处**：可以利用在自然图像上学到的特征

**第120行：移除最后一层（分类头）**
```python
self.backbone = nn.Sequential(*list(backbone.children())[:-1])
```
- `backbone.children()`: 获取 ResNet18 的所有子模块
- `list(...)`: 转换为列表
- `[:-1]`: 切片，去掉最后一个元素（分类头 `fc`）
- `nn.Sequential(*...)`: 重新组合为 Sequential
- **输出形状**：`[B, 512, 1, 1]`（B 是 batch size，512 是特征维度）

**第121-127行：定义 MLP 动作头**
```python
self.head = nn.Sequential(
    nn.Linear(512, 256),    # 512 -> 256
    nn.ReLU(inplace=True),  # ReLU 激活
    nn.Linear(256, 128),    # 256 -> 128
    nn.ReLU(inplace=True),  # ReLU 激活
    nn.Linear(128, out_dim), # 128 -> 7（7个关节）
)
```
- **`nn.Linear(512, 256)`**: 全连接层，输入 512 维，输出 256 维
- **`nn.ReLU(inplace=True)`**: ReLU 激活函数，`inplace=True` 表示原地操作（节省内存）
- **`out_dim=7`**: 输出 7 维（Panda 机械臂有 7 个关节）

**`forward` 方法：**

**第130行：提取视觉特征**
```python
feat = self.backbone(x).flatten(1)  # [B,512]
```
- `self.backbone(x)`: 输入图像 `[B, 3, H, W]`，输出特征 `[B, 512, 1, 1]`
- `.flatten(1)`: 将 `[B, 512, 1, 1]` 展平为 `[B, 512]`（保留 batch 维度）

**第131行：预测动作**
```python
return self.head(feat)  # [B,7]
```
- 输入 `[B, 512]`，输出 `[B, 7]`（7 个关节的增量命令）

---

### `freeze_backbone(model, freeze)`（第133-135行）

```python
def freeze_backbone(model: ResNetMLPPolicy, freeze: bool = True):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze
```

**功能**：冻结或解冻 backbone 的参数。

**逐行解释：**
- `model.backbone.parameters()`: 获取 backbone 的所有参数
- `p.requires_grad = not freeze`: 
  - 如果 `freeze=True`，则 `requires_grad=False`（参数不更新）
  - 如果 `freeze=False`，则 `requires_grad=True`（参数可更新）

**为什么需要？**
- **两阶段训练**：先冻结 backbone，只训练 head（更快、更稳定）
- 然后解冻 backbone，进行端到端微调

---

### `unfreeze_layer4_only(model)`（第137-146行）

```python
def unfreeze_layer4_only(model: ResNetMLPPolicy):
    # backbone is a Sequential of resnet children excluding fc:
    # [conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool]
    # We want layer4 trainable, others frozen.
    for p in model.backbone.parameters():
        p.requires_grad = False
    # layer4 is index 7 in that sequence
    layer4 = model.backbone[7]
    for p in layer4.parameters():
        p.requires_grad = True
```

**功能**：只解冻 ResNet18 的 layer4，其他层保持冻结。

**逐行解释：**

**第141-142行：冻结所有 backbone 参数**
```python
for p in model.backbone.parameters():
    p.requires_grad = False
```

**第144行：获取 layer4**
```python
layer4 = model.backbone[7]
```
- ResNet18 的 backbone Sequential 结构：
  - `[0]`: conv1
  - `[1]`: bn1
  - `[2]`: relu
  - `[3]`: maxpool
  - `[4]`: layer1
  - `[5]`: layer2
  - `[6]`: layer3
  - `[7]`: layer4 ← 我们要解冻的层
  - `[8]`: avgpool

**第145-146行：解冻 layer4**
```python
for p in layer4.parameters():
    p.requires_grad = True
```

**为什么只解冻 layer4？**
- **更精细的微调**：layer4 包含高级特征，对任务最相关
- **减少过拟合**：只训练少量参数，降低过拟合风险
- **更快训练**：需要更新的参数更少

---

## 第五部分：评估函数（第149-220行）

### `EvalStats` 数据类（第151-157行）

```python
@dataclass
class EvalStats:
    mse: float
    rmse_per_joint: np.ndarray
    action_norm_gt_mean: float
    action_norm_pred_mean: float
    cos_mean: float
```

**功能**：存储评估统计信息的数据类。

**字段说明：**
- `mse`: 平均平方误差（所有维度的平均）
- `rmse_per_joint`: 每个关节的 RMSE（7 维数组）
- `action_norm_gt_mean`: 真实动作的平均范数
- `action_norm_pred_mean`: 预测动作的平均范数
- `cos_mean`: 平均余弦相似度

---

### `evaluate(model, loader, device)`（第159-220行）

```python
@torch.no_grad()
def evaluate(model, loader, device) -> EvalStats:
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")

    total_mse_sum = 0.0
    total_n = 0
    # per joint
    se_sum = None  # [D]
    # norms and cosine
    gt_norms = []
    pred_norms = []
    cos_vals = []
```

**功能**：在验证集上评估模型性能。

**逐行解释：**

**第159行：装饰器**
```python
@torch.no_grad()
```
- **禁用梯度计算**：评估时不需要反向传播，节省内存和计算

**第160行：设置模型为评估模式**
```python
model.eval()
```
- **关闭 Dropout、BatchNorm 的更新**：评估时使用固定的统计量

**第161行：定义损失函数**
```python
loss_fn = nn.MSELoss(reduction="sum")
```
- `reduction="sum"`: 返回所有样本的损失之和（不是平均值）
- **为什么用 sum？** 需要手动计算总样本数，然后除以总样本数

**第163-171行：初始化统计变量**
- `total_mse_sum`: 累计 MSE 总和
- `total_n`: 总样本数
- `se_sum`: 每个关节的平方误差总和（`[7]` 数组）
- `gt_norms`: 真实动作范数列表
- `pred_norms`: 预测动作范数列表
- `cos_vals`: 余弦相似度列表

---

**第173-198行：遍历数据加载器**

```python
pbar = tqdm(loader, desc="Val", leave=False)
for batch in pbar:
    images = batch["image"].to(device, non_blocking=True)
    target = batch["delta_q"].to(device, non_blocking=True)  # [B,7]
    pred = model(images)

    mse_sum = loss_fn(pred, target).item()
    total_mse_sum += mse_sum
    bs = images.size(0)
    total_n += bs

    # per joint squared error
    se = (pred - target) ** 2  # [B,7]
    se_batch_sum = se.sum(dim=0).detach().cpu().numpy()  # [7]
    if se_sum is None:
        se_sum = se_batch_sum
    else:
        se_sum += se_batch_sum

    # norms
    gt_norms.append(torch.norm(target, dim=1).detach().cpu().numpy())
    pred_norms.append(torch.norm(pred, dim=1).detach().cpu().numpy())
    cos_vals.append(cosine_similarity(pred, target).detach().cpu().numpy())

    # show batch mse (mean) in bar
    pbar.set_postfix(mse=f"{(mse_sum / max(bs,1)):.4f}")
```

**逐行解释：**

**第173行：创建进度条**
```python
pbar = tqdm(loader, desc="Val", leave=False)
```
- `desc="Val"`: 进度条描述
- `leave=False`: 完成后不保留进度条

**第175-177行：加载数据并预测**
```python
images = batch["image"].to(device, non_blocking=True)
target = batch["delta_q"].to(device, non_blocking=True)  # [B,7]
pred = model(images)
```
- `.to(device)`: 将数据移动到 GPU（如果可用）
- `non_blocking=True`: **异步传输**，不阻塞 CPU

**第179-182行：计算并累计 MSE**
```python
mse_sum = loss_fn(pred, target).item()
total_mse_sum += mse_sum
bs = images.size(0)
total_n += bs
```
- `loss_fn(pred, target)`: 计算 MSE（返回标量张量）
- `.item()`: 将张量转换为 Python 浮点数
- `bs = images.size(0)`: 获取 batch size
- `total_n += bs`: 累计总样本数

**第184-190行：计算每个关节的平方误差**
```python
se = (pred - target) ** 2  # [B,7]
se_batch_sum = se.sum(dim=0).detach().cpu().numpy()  # [7]
if se_sum is None:
    se_sum = se_batch_sum
else:
    se_sum += se_batch_sum
```
- `(pred - target) ** 2`: 逐元素平方误差，形状 `[B, 7]`
- `.sum(dim=0)`: 沿 batch 维度求和，得到每个关节的平方误差总和，形状 `[7]`
- `.detach().cpu().numpy()`: 断开梯度，移到 CPU，转为 NumPy 数组
- 累计到 `se_sum`

**第192-195行：计算范数和余弦相似度**
```python
gt_norms.append(torch.norm(target, dim=1).detach().cpu().numpy())
pred_norms.append(torch.norm(pred, dim=1).detach().cpu().numpy())
cos_vals.append(cosine_similarity(pred, target).detach().cpu().numpy())
```
- `torch.norm(..., dim=1)`: 计算每个样本的 L2 范数，形状 `[B]`
- 将结果添加到列表中

**第197-198行：更新进度条**
```python
pbar.set_postfix(mse=f"{(mse_sum / max(bs,1)):.4f}")
```
- 显示当前 batch 的平均 MSE

---

**第200-220行：计算最终统计量**

```python
if total_n == 0:
    return EvalStats(mse=float("nan"),
                     rmse_per_joint=np.full((7,), np.nan),
                     action_norm_gt_mean=float("nan"),
                     action_norm_pred_mean=float("nan"),
                     cos_mean=float("nan"))

mse = total_mse_sum / (total_n * 7)  # average per-dim MSE
rmse_per_joint = np.sqrt(se_sum / total_n)  # [7]

gt_norms = np.concatenate(gt_norms) if len(gt_norms) else np.array([])
pred_norms = np.concatenate(pred_norms) if len(pred_norms) else np.array([])
cos_vals = np.concatenate(cos_vals) if len(cos_vals) else np.array([])

return EvalStats(
    mse=float(mse),
    rmse_per_joint=rmse_per_joint.astype(float),
    action_norm_gt_mean=float(gt_norms.mean()) if gt_norms.size else float("nan"),
    action_norm_pred_mean=float(pred_norms.mean()) if pred_norms.size else float("nan"),
    cos_mean=float(cos_vals.mean()) if cos_vals.size else float("nan"),
)
```

**逐行解释：**

**第200-206行：处理空数据集**
- 如果没有样本，返回 NaN 值

**第207行：计算平均 MSE**
```python
mse = total_mse_sum / (total_n * 7)
```
- `total_mse_sum`: 所有样本的 MSE 总和
- `total_n * 7`: 总元素数（样本数 × 7 个关节）
- **结果**：每个维度的平均 MSE

**第208行：计算每个关节的 RMSE**
```python
rmse_per_joint = np.sqrt(se_sum / total_n)  # [7]
```
- `se_sum / total_n`: 每个关节的平均平方误差
- `np.sqrt(...)`: 开平方，得到 RMSE
- **结果**：`[7]` 数组，每个关节的 RMSE

**第210-212行：合并列表**
```python
gt_norms = np.concatenate(gt_norms) if len(gt_norms) else np.array([])
pred_norms = np.concatenate(pred_norms) if len(pred_norms) else np.array([])
cos_vals = np.concatenate(cos_vals) if len(cos_vals) else np.array([])
```
- 将多个 batch 的结果合并为一个数组

**第214-220行：返回统计结果**
- 计算平均值并返回 `EvalStats` 对象

---

## 第六部分：训练函数（第222-247行）

### `train_one_epoch(model, loader, optimizer, device, grad_clip)`（第222-247行）

```python
def train_one_epoch(model, loader, optimizer, device, grad_clip: float = 0.0) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["delta_q"].to(device, non_blocking=True)

        pred = model(images)
        loss = loss_fn(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)
```

**功能**：训练一个 epoch。

**逐行解释：**

**第223行：设置模型为训练模式**
```python
model.train()
```
- **启用 Dropout、BatchNorm 的更新**

**第224行：定义损失函数**
```python
loss_fn = nn.MSELoss()
```
- 默认 `reduction="mean"`（返回平均值）

**第225-226行：初始化统计变量**
```python
total_loss = 0.0
n = 0
```

**第228-245行：训练循环**

**第230-231行：加载数据**
```python
images = batch["image"].to(device, non_blocking=True)
target = batch["delta_q"].to(device, non_blocking=True)
```

**第233-234行：前向传播**
```python
pred = model(images)
loss = loss_fn(pred, target)
```

**第236-240行：反向传播和优化**
```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
if grad_clip and grad_clip > 0:
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
optimizer.step()
```
- **`optimizer.zero_grad(set_to_none=True)`**: 清零梯度
  - `set_to_none=True`: 将梯度设为 `None` 而不是 0（更高效）
- **`loss.backward()`**: 反向传播，计算梯度
- **`nn.utils.clip_grad_norm_(..., grad_clip)`**: **梯度裁剪**，防止梯度爆炸
  - 如果梯度范数超过 `grad_clip`，将其缩放
- **`optimizer.step()`**: 更新参数

**第242-245行：累计损失**
```python
bs = images.size(0)
total_loss += loss.item() * bs
n += bs
pbar.set_postfix(loss=f"{loss.item():.4f}")
```
- `loss.item() * bs`: 将平均损失转换为总损失（因为 `loss` 是平均值）
- 累计总损失和样本数

**第247行：返回平均损失**
```python
return total_loss / max(n, 1)
```

---

## 第七部分：早停（Early Stopping）（第250-267行）

### `EarlyStopper` 类（第252-266行）

```python
class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, val: float) -> bool:
        # returns True if should stop
        if val + self.min_delta < self.best:
            self.best = val
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
```

**功能**：实现早停机制，防止过拟合。

**逐行解释：**

**`__init__` 方法：**
- `patience`: **容忍度**，连续多少个 epoch 没有改善就停止
- `min_delta`: **最小改善量**，只有改善超过这个值才认为是真正的改善
- `best`: 最佳验证值（初始为无穷大）
- `bad_epochs`: 连续没有改善的 epoch 数

**`step` 方法：**
- **输入**：当前验证值 `val`
- **返回**：`True` 表示应该停止，`False` 表示继续

**逻辑：**
1. 如果 `val + min_delta < self.best`（有显著改善）：
   - 更新 `best`
   - 重置 `bad_epochs = 0`
   - 返回 `False`（继续训练）
2. 否则（没有改善）：
   - `bad_epochs += 1`
   - 如果 `bad_epochs >= patience`，返回 `True`（停止训练）

**为什么需要 `min_delta`？**
- 避免因为微小的随机波动而重置计数器

---

## 第八部分：主函数（第269-489行）

### 参数解析（第272-288行）

```python
def main():
    parser = argparse.ArgumentParser(description="Managed BC Training (ResNet18 + MLP)")
    parser.add_argument("--dataset_root", type=str, default="/home/alphatok/ME5400/expert_data")
    parser.add_argument("--out_dir", type=str, default="./checkpoints_bc_managed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_head", type=float, default=1e-3, help="LR for head-only phase")
    parser.add_argument("--lr_finetune", type=float, default=1e-4, help="LR for finetune phase")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze_epochs", type=int, default=8, help="epochs to train head only")
    parser.add_argument("--unfreeze_layer4", action="store_true", help="finetune only layer4 instead of full backbone")
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()
```

**参数说明：**
- `--dataset_root`: 数据集根目录
- `--out_dir`: 输出目录（保存 checkpoints 和指标）
- `--batch_size`: batch 大小（默认 32）
- `--num_workers`: 数据加载的并行进程数
- `--lr_head`: head-only 阶段的学习率（默认 1e-3，较大）
- `--lr_finetune`: finetune 阶段的学习率（默认 1e-4，较小）
- `--epochs`: 最大训练 epoch 数
- `--freeze_epochs`: 只训练 head 的 epoch 数（默认 8）
- `--unfreeze_layer4`: 如果设置，只解冻 layer4（否则解冻整个 backbone）
- `--image_height/width`: 图像尺寸
- `--only_success`: 如果设置，只使用成功的 episode
- `--seed`: 随机种子
- `--patience`: 早停的容忍度
- `--grad_clip`: 梯度裁剪阈值

---

### 初始化（第290-308行）

```python
set_seed(args.seed)
ensure_dir(args.out_dir)
plots_dir = os.path.join(args.out_dir, "plots")
ensure_dir(plots_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

image_size_hw = (args.image_height, args.image_width)

# Data
train_set, val_set, train_loader, val_loader = make_loaders(
    args.dataset_root,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    image_size_hw=image_size_hw,
    only_success=args.only_success,
)
print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
```

**逐行解释：**
- 设置随机种子
- 创建输出目录和 plots 目录
- 选择设备（GPU 或 CPU）
- 创建数据加载器
- 打印数据集大小

---

### 模型和优化器初始化（第310-318行）

```python
# Model
model = ResNetMLPPolicy(out_dim=7).to(device)

# Phase 1: freeze backbone, train head
freeze_backbone(model, freeze=True)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

early = EarlyStopper(patience=args.patience, min_delta=1e-6)
```

**逐行解释：**
- 创建模型并移到设备
- **冻结 backbone**，只训练 head
- **创建优化器**：只优化需要梯度的参数（`filter(lambda p: p.requires_grad, ...)`）
- **创建学习率调度器**：`ReduceLROnPlateau`
  - `mode="min"`: 监控验证 loss，当 loss 不再下降时降低学习率
  - `factor=0.5`: 每次降低为原来的 0.5 倍
  - `patience=3`: 连续 3 个 epoch 没有改善就降低学习率
- 创建早停器

---

### 日志初始化（第320-332行）

```python
# Logging
metrics_csv = os.path.join(args.out_dir, "metrics.csv")
metrics_jsonl = os.path.join(args.out_dir, "metrics.jsonl")
with open(metrics_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "phase", "train_loss",
        "val_mse", "val_cos", "val_norm_gt", "val_norm_pred",
        "lr"
    ] + [f"val_rmse_j{i+1}" for i in range(7)])
# also dump config
with open(os.path.join(args.out_dir, "run_args.json"), "w") as f:
    json.dump(vars(args), f, indent=2)
```

**功能**：初始化 CSV 和 JSONL 日志文件，保存运行配置。

---

### 训练循环（第352-481行）

```python
for epoch in range(args.epochs):
    # switch phase
    if epoch < args.freeze_epochs:
        phase = "head_only"
    else:
        phase = "finetune"
    
    # 在切换到finetune阶段时，解冻backbone并重新初始化optimizer
    if epoch == args.freeze_epochs:
        # unfreeze
        if args.unfreeze_layer4:
            unfreeze_layer4_only(model)
            print("✅ finetune: 仅解冻 layer4")
        else:
            freeze_backbone(model, freeze=False)
            print("✅ finetune: 解冻整个 backbone")

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_finetune)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        early = EarlyStopper(patience=args.patience, min_delta=1e-6)

    # train
    train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
    # val stats
    val_stats = evaluate(model, val_loader, device)

    lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch:03d} [{phase}] "
        f"train_loss={train_loss:.6f} | val_mse={val_stats.mse:.6f} "
        f"| cos={val_stats.cos_mean:.3f} | lr={lr:.2e}"
    )

    # scheduler uses val mse
    scheduler.step(val_stats.mse)

    # save best
    if val_stats.mse < best_val:
        best_val = val_stats.mse
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_mse": val_stats.mse,
                "args": vars(args),
            },
            best_path,
        )
        print(f"  [saved] {best_path}")
```

**逐行解释：**

**第354-357行：确定当前阶段**
- 前 `freeze_epochs` 个 epoch 是 "head_only" 阶段
- 之后是 "finetune" 阶段

**第359-371行：切换到 finetune 阶段**
- 当 `epoch == args.freeze_epochs` 时：
  - 解冻 backbone（或只解冻 layer4）
  - **重新创建优化器**（因为可训练参数变了）
  - **重新创建调度器和早停器**（重置状态）

**第373-376行：训练和验证**
- 训练一个 epoch
- 在验证集上评估

**第378-383行：打印信息**
- 显示当前 epoch、阶段、loss、余弦相似度、学习率

**第385行：更新学习率**
```python
scheduler.step(val_stats.mse)
```
- 根据验证 MSE 调整学习率

**第387-400行：保存最佳模型**
- 如果当前验证 MSE 更好，保存 checkpoint

**第402-424行：记录指标**
- 写入 CSV 和 JSONL 文件

**第426-476行：更新历史并绘制图表**
- 更新历史记录
- 绘制 loss 曲线、每个关节的 RMSE、余弦相似度、动作范数

**第478-481行：早停检查**
```python
if early.step(val_stats.mse):
    print(f"🛑 Early stopping triggered at epoch {epoch} (best val_mse={early.best:.6f})")
    break
```
- 如果早停器返回 `True`，停止训练

---

## 总结

这个训练脚本实现了：

1. **两阶段训练**：先训练 head，再微调 backbone
2. **完整的评估指标**：MSE、RMSE、余弦相似度、动作范数
3. **自动学习率调整**：`ReduceLROnPlateau`
4. **早停机制**：防止过拟合
5. **详细的日志记录**：CSV、JSONL、可视化图表
6. **梯度裁剪**：防止梯度爆炸

这是一个**生产级别的训练脚本**，适合实际项目使用。

