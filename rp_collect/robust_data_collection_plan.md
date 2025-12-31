# 稳健的数据收集计划（按专家建议实施）

## 核心改进

基于对当前失败模式的分析（"初始位置稍微偏离就会乱走、甚至朝别的方向走"），实施三种桶（Bucket）混合采样策略，解决**分布外行为（covariate shift）+ 多解/不确定性**的组合问题。

---

## 已实施的改进

### 1. ✅ 采样频率优化

**修改：** `CAPTURE_EVERY_N = 5 → 3`

**理由：**
- 样本数提升 ~1.67 倍（从 3,640 → 约 6,000+）
- 不会像 every=1 那样把大量重复帧塞进来
- 对当前 3.6k steps 的规模来说很划算

**保留：** 终止步强制保存最后一帧（关键纠偏瞬间）

---

### 2. ✅ Episode 数量增加

**修改：** `NUM_EPISODES = 100 → 200`

**预期数据量：**
- 200 集 × 平均 40 step/集 = 8,000 steps
- 再乘上 every=3 的增量 → 约 **13,000 steps**（如果每步记录）
- 实际：200 集 × 200 steps/集 ÷ 3 = 约 **13,333 个样本**

**注意：** 200 集仍然偏少，但作为下一轮迭代目标很合理。最终可能需要 3 万 step 才能达到"可用"的 success rate。

---

### 3. ✅ 三种桶混合采样策略（最重要）

#### Bucket A: 近处微调（40% = 80集）
- **目的**：学会最后稳定对齐、减少抖动
- **起始位置**：就在 marker 附近（10-20 cm）
- **解决的问题**：
  - ✅ 到了附近怎么稳稳地完成
  - ✅ 不要在成功区乱抖

#### Bucket B: 中等偏离纠偏（40% = 80集）
- **目的**：学会"看到偏了就朝正确方向拉回来"，避免乱走
- **起始位置**：marker 在图像边缘/偏离中心明显（20-35 cm）
- **解决的问题**：
  - ✅ 初始偏一点就跑偏
  - ✅ 往别处走的问题
  - ✅ **这是你现在最缺的数据！**

#### Bucket C: Hard cases（20% = 40集）
- **目的**：学会在"相似但不一样"的场景里不跑飞
- **起始位置**：EE 路径容易经过桌面/phantom 边缘、容易遮挡 marker
- **解决的问题**：
  - ✅ 路过桌面/phantom 却不去 marker
  - ✅ 视觉输入里"桌面/phantom 的纹理"比 marker 更强的问题

#### 正常分布（作为回退）
- 如果特定桶采样失败，回退到正常随机采样

---

## 实施细节

### 参数配置

```python
# 采集参数
NUM_EPISODES = 200        # 从100增加到200
CAPTURE_EVERY_N = 3       # 从5改为3

# 三种桶的比例
BUCKET_A_RATIO = 0.40  # 近处微调（40% = 80集）
BUCKET_B_RATIO = 0.40  # 中等偏离纠偏（40% = 80集）
BUCKET_C_RATIO = 0.20  # Hard cases（20% = 40集）

# Bucket A 参数
NEAR_TARGET_DISTANCE_MIN = 0.10  # 米
NEAR_TARGET_DISTANCE_MAX = 0.20  # 米

# Bucket B 参数（中等偏离）
MEDIUM_OFFSET_DISTANCE_MIN = 0.20  # 米
MEDIUM_OFFSET_DISTANCE_MAX = 0.35  # 米

# Bucket C 参数（Hard cases）
HARD_CASE_OFFSET_MIN = 0.15  # 米
HARD_CASE_OFFSET_MAX = 0.30  # 米
```

### Episode 分配逻辑

```python
def determine_bucket_type(episode_idx, num_episodes):
    # 按顺序分配（确保比例准确）
    if episode_idx < 80:        # 0-79: Bucket B
        return "bucket_b"
    elif episode_idx < 160:     # 80-159: 正常分布
        return "random"
    elif episode_idx < 200:     # 160-199: Bucket A + Bucket C
        if episode_idx < 180:   # 160-179: Bucket A
            return "bucket_a"
        else:                    # 180-199: Bucket C
            return "bucket_c"
    else:
        return "random"
```

---

## 预期效果

### 数据分布改进

**当前问题：**
- 86.9% 的样本只有 0–2 个关节显著移动（坐标下降）
- 平均 |Δq| ≈ 0.005 rad（动作太小）
- 缺少"偏离后纠偏"的数据

**改进后预期：**
- `non_zero_joint_count` 往右移（更多关节同时移动）
- `max_joint_ratio` 往 1/7 靠一点（更均匀的动作分布）
- `Δq` 幅值分布变宽（包含更多大动作的纠偏样本）

### 模型性能提升

**当前失败模式：**
- 初始位置稍微偏离就会乱走
- 路过桌面/phantom 却不去 marker
- 朝别的方向走

**改进后预期：**
- ✅ 学会从偏离位置纠偏（Bucket B）
- ✅ 学会在复杂场景中不跑飞（Bucket C）
- ✅ 学会精确到达目标（Bucket A）

---

## 验证步骤

### 1. 数据收集后立即验证

运行分析脚本：
```bash
python training/analyze_delta_q.py
```

**检查指标：**
- ✅ `non_zero_joint_count` 分布往右移（更多关节同时移动）
- ✅ `max_joint_ratio` 分布往 1/7 靠一点（更均匀）
- ✅ `Δq` 幅值分布变宽（包含更多大动作）

### 2. 训练模型

```bash
cd training
python train_bc.py --dataset_root ../expert_data
```

### 3. 评估改进效果

```bash
bash run_evaluate.sh <checkpoint_path> 20-30
```

**关键指标：**
- Success rate 是否提高
- "偏一点就跑飞"的问题是否改善
- 是否仍然"路过桌面/phantom 却不去 marker"

---

## 如果仍然失败

如果 success rate 仍然因为"偏一点就跑飞"而低，建议：

1. **增加数据量**：从 200 集 → 300-400 集
2. **调整桶比例**：增加 Bucket B 的比例（更多纠偏数据）
3. **考虑 DAgger**：比 RL 更对症，可以主动收集"模型犯错"的数据

---

## 关键洞察

### 为什么只加"near-marker"不够？

- **near-marker 数据**教的是**局部策略**，不教**全局纠偏**
- 它主要解决"到了附近怎么稳稳地完成"，但不解决"初始偏一点就跑偏"

### 为什么需要三种桶？

- **Bucket A**：解决末端微调
- **Bucket B**：解决偏离后纠偏（**最关键**）
- **Bucket C**：解决复杂场景鲁棒性

### 为什么 Bucket B 最重要？

- 当前失败模式的核心是"初始偏一点就跑飞"
- Bucket B 专门针对这个问题，让模型学习"看到偏了就朝正确方向拉回来"

---

## 总结

这个改进计划的核心是：

1. ✅ **加量**：200 集，CAPTURE_EVERY_N=3
2. ✅ **提高采样密度**：从 5 → 3
3. ✅ **三种桶混合**：既补近处，也补偏离后的纠偏，还补"诱导它犯错的区域"

这应该能显著改善模型的在线表现，特别是"偏一点就跑飞"的问题。


