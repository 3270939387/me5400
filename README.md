# 图像驱动机械臂控制 

## 🎯 任务目标

**训练一个纯视觉驱动的机械臂策略**：从随机初始关节位置出发，仅通过观察相机图像，自主运动到能够清晰捕捉 marker 的位置。

- **输入**: D405 相机 RGB 图像
- **输出**: 7 维关节位置增量 `Δq`
- **成功标准**: Marker 出现在图像中心附近（像素距离 < 50px）

---

## 📋 实现步骤

### Step 1: 数据收集
- 保存每步数据到 `meta.json`：
  - 图像路径
  - 当前关节角 `q_t`
  - RMPflow 动作 `Δq`（模仿目标）
- 按 episode 组织：`episodes/episode_0001/`

### Step 2: 模型实现
-  **架构**: ResNet18 (ImageNet 预训练) → MLP (512→256→128→7)
-  **输入**: RGB 图像 (640×480 或 320×240)
-  **输出**: 关节位置增量 `Δq` (7-dim)

### Step 3: 训练（Behavior Cloning）
-  **损失**: L2 损失 `|| Δq_pred - Δq_expert ||²`
-  **优化器**: Adam (lr=1e-4)
-  **批次**: 32-64
-  **数据分割**: 80% 训练 / 20% 验证

### Step 4: 评估（Isaac Sim 闭环测试）
- 随机初始位置测试
- 记录成功率、平均步数、动作平滑度

### Step 5: 迭代优化
- 分析失败案例
- 调整超参数/架构
- 如需要，考虑 RL 微调（PPO/SAC）

---

## 📦 数据格式

```
dataset_root/
├── episodes/
│   ├── episode_0001/
│   │   ├── rgb_0000.png
│   │   ├── rgb_0001.png
│   │   └── meta.json
│   └── ...
```

**`meta.json` 示例**:
```json
{
  "episode_idx": 0,
  "success": true,
  "end_reason": "success",
  "end_step": 199,
  "num_saved_frames": 40,
  "steps": [
    {
      "step": 0,
      "image_path": "frame_0000.png",
      "state": {
        "q": [
        ],
        "dq": [
        ],
        "ee_target_pos": [
        ],
        "ee_actual_pos": [
        ],
        "marker_pos_world": [
        ]
      },
      "action": {
        "command_positions": [
        ],
        "command_velocities": [
        ],
        "delta_q": [
        ]
      }
    },
```

---

## 🏗️ 模型架构

```
RGB Image (H×W×3)
    ↓
ResNet18 (ImageNet 预训练)
    ↓
特征向量 (512-dim)
    ↓
MLP (512 → 256 → 128 → 7)
    ↓
Δq (7-dim 关节位置增量)
```

---

## 📊 评估指标

- **成功率**: marker 到达中心的比例
- **平均步数**: 到达目标所需步数
- **动作平滑度**: 动作变化的标准差
- **离线误差**: `MAE(Δq_pred, Δq_expert)`

---

## ⚠️ 关键风险

1. **过拟合**: 增加数据量、数据增强
2. **动作抖动**: 添加动作平滑（移动平均）
3. **数据分布偏移**: 确保随机初始位置覆盖广

---

## 🎯 里程碑


