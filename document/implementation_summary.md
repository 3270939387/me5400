# 数据收集改进实施总结

## ✅ 已完成的改进

### 1. 采样频率优化
- **修改**：`CAPTURE_EVERY_N = 5 → 3`
- **效果**：样本数提升 ~1.67 倍
- **保留**：终止步强制保存最后一帧（关键纠偏瞬间）

### 2. Episode 数量增加
- **修改**：`NUM_EPISODES = 100 → 200`
- **预期数据量**：约 13,333 个样本（200集 × 200步/集 ÷ 3）

### 3. 三种桶混合采样策略（核心改进）

#### Bucket A: 近处微调（40% = 80集）
- **函数**：`sample_near_target_config()`
- **距离**：10-20 cm
- **目的**：学会最后稳定对齐、减少抖动

#### Bucket B: 中等偏离纠偏（40% = 80集）⭐ **最重要**
- **函数**：`sample_medium_offset_config()`
- **距离**：20-35 cm
- **目的**：学会"看到偏了就朝正确方向拉回来"，避免乱走
- **解决**：当前最缺的数据！解决"初始偏一点就跑飞"的问题

#### Bucket C: Hard cases（20% = 40集）
- **函数**：`sample_hard_case_config()`
- **距离**：15-30 cm，偏向某个方向
- **目的**：学会在复杂场景中不跑飞
- **解决**：路过桌面/phantom 却不去 marker 的问题

#### 正常分布（作为回退）
- 如果特定桶采样失败，回退到正常随机采样

### 4. Episode 分配逻辑
- **函数**：`determine_bucket_type(episode_idx, num_episodes)`
- **分配**：
  - Episode 0-79: Bucket B（中等偏离纠偏）
  - Episode 80-159: 正常分布
  - Episode 160-179: Bucket A（近处微调）
  - Episode 180-199: Bucket C（Hard cases）

### 5. 终止步强制保存
- ✅ 碰撞终止时强制保存最后一帧
- ✅ 成功终止时强制保存最后一帧（关键纠偏瞬间）
- ✅ 记录 `bucket_type` 到 metadata（便于分析）

---

## 关键参数

```python
# 采集参数
NUM_EPISODES = 200
CAPTURE_EVERY_N = 3

# 三种桶的比例
BUCKET_A_RATIO = 0.40  # 近处微调
BUCKET_B_RATIO = 0.40  # 中等偏离纠偏（最重要）
BUCKET_C_RATIO = 0.20  # Hard cases

# 距离参数
NEAR_TARGET_DISTANCE_MIN = 0.10  # Bucket A
NEAR_TARGET_DISTANCE_MAX = 0.20
MEDIUM_OFFSET_DISTANCE_MIN = 0.20  # Bucket B
MEDIUM_OFFSET_DISTANCE_MAX = 0.35
HARD_CASE_OFFSET_MIN = 0.15  # Bucket C
HARD_CASE_OFFSET_MAX = 0.30
```

---

## 验证步骤

### 1. 数据收集后立即验证

```bash
python training/analyze_delta_q.py
```

**检查指标：**
- ✅ `non_zero_joint_count` 分布往右移
- ✅ `max_joint_ratio` 分布往 1/7 靠一点
- ✅ `Δq` 幅值分布变宽

### 2. 训练模型

```bash
cd training
python train_bc.py --dataset_root ../expert_data
```

### 3. 评估改进效果

```bash
bash run_evaluate.sh <checkpoint_path> 20-30
```

---

## 预期改进

### 数据分布
- 更多关节同时移动（non_zero_joint_count 增加）
- 动作分布更均匀（max_joint_ratio 接近 1/7）
- 包含更多大动作的纠偏样本（Δq 分布变宽）

### 模型性能
- ✅ 解决"初始偏一点就跑飞"的问题（Bucket B）
- ✅ 解决"路过桌面/phantom 却不去 marker"的问题（Bucket C）
- ✅ 提高精确到达目标的能力（Bucket A）

---

## 如果仍然失败

如果 success rate 仍然因为"偏一点就跑飞"而低：

1. **增加数据量**：200 集 → 300-400 集
2. **调整桶比例**：增加 Bucket B 的比例
3. **考虑 DAgger**：主动收集"模型犯错"的数据

