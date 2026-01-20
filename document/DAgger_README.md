# DAgger 集成说明（详细版）

## 总览
- 目标：在执行时用已训 BC 策略驱动机器人，同时用 RMPFlow 专家重标注动作，生成 DAgger 数据，提升分布偏移下的成功率。
- 兼容性：数据结构、字段完全沿用原始数据格式，训练脚本无需改动即可混训专家数据与 DAgger 数据。

## 目录与角色
- `rp_collect/dagger_collect.py`：DAgger 采集核心脚本（执行 + 专家标注 + 落盘）。
- `training/dataset.py`：数据加载器，支持多数据源合并，优先使用 DAgger 的 `expert_delta_q` 标签。
- 训练脚本：`training/train_bc.py` 直接复用，传入多个数据根目录即可。

## 代码分块讲解（dagger_collect.py）
1) **入口与依赖**  
   - `SimulationApp({"headless": False})`：启动 Isaac Sim，可改为 `True` 做无头采集。  
   - 添加 `ROOT_DIR` 到 `sys.path`，便于复用 `training/train_bc.py` 中的 `ResNetMLPPolicy`。

2) **默认配置常量**  
   - 路径：`ENV_USD_PATH`（场景）、`CAM_PATH`（相机）、`DEFAULT_DATASET_ROOT`（输出目录）。  
   - 控制周期 `DT`，成功阈值 `SUCCESS_DISTANCE`，工作空间约束（中心、半径、Z 上下界），关节限位。

3) **工具类与函数**  
   - `ViewportCamera`：绑定活跃视口，`capture()` 截图到文件。  
   - `NumpyEncoder`：JSON 序列化 numpy。  
   - `vec3_to_list()`：向量转 float list。  
   - `sample_random_joint_config()`：在关节限位内随机采样。  
   - `check_workspace()` / `world_to_base()`：确保 TCP 在工作空间内。  
   - `find_valid_start()`：拒绝采样找到有效初始姿态（先设关节，再物理预热，检查 TCP 位置）。

4) **BCPolicy 封装**  
   - 载入 checkpoint（支持包含 `model` 字段的 state_dict）。  
   - 预处理与前向 `predict(image_path)`，输出 `policy_delta_q`。

5) **参数解析 `parse_args()`**  
   - 关键参数：  
     - `--bc_checkpoint`：BC 模型路径  
     - `--behavior`：`policy` / `expert` / `mixture`  
     - `--mix_beta`：混合系数，`command = (1 - beta)*expert + beta*policy`  
     - 分辨率、episode 数、步数、输出目录等。

6) **目录创建 `ensure_dirs()`**  
   - 创建 `metadata` 与 `picture_data`，保持与训练数据结构一致。

7) **主流程 `main()`**  
   - 设备选择、场景加载、PhysicsScene 检查、桌子刚体化。  
   - 创建 `Articulation`，初始化 `SimulationContext`，预热物理引擎。  
   - 加载 RMPFlow 专家：`ArticulationMotionPolicy(robot, rmp)`。  
   - 初始化相机、BCPolicy，获取 marker 默认位姿。
   - Episode 循环：  
     a) `find_valid_start()`：采样有效初始姿态；失败则跳过。  
     b) 进入 step 循环（每步都存图）：  
        - **观测采集**：相机抓图存 `frame_xxxx.png`。  
        - **专家动作**：RMPFlow 设定目标，得到 `command_q_expert`，计算 `expert_delta_q`。  
        - **策略动作**：BC 前向得到 `policy_delta_q`，构造 `command_q_policy`。  
        - **混合/选择执行**：依据 `behavior` 和 `mix_beta` 生成最终 `command_q`，记录 `executed_delta_q`。  
        - **执行**：覆写专家动作的 `joint_positions` 为最终指令，`robot.apply_action()`，单步仿真。  
        - **记录**：存入 `step_data`（图像路径、状态、`expert_delta_q` / `policy_delta_q` / `executed_delta_q` / `command_positions`）。  
     c) 成功判定：最后一步计算 TCP 与 marker 的 xyz 误差，写入 `success` / `end_reason`。  
     d) 元数据落盘：`metadata/episode_xxxx.json`，图片已在 `picture_data/episode_xxxx/`。

8) **数据格式对齐**  
   - 与原数据完全同构：`metadata/episode_XXXX.json` + `picture_data/episode_XXXX/frame_YYYY.png`。  
   - 关键新增字段：`action.expert_delta_q`（训练时优先使用），`policy_delta_q`，`executed_delta_q`。

## 代码分块讲解（dataset.py）
1) **多数据源合并**  
   - `dataset_root` 支持 `"/path/a,/path/b"` 或 list，统一收集 `metadata` 列表后再随机打乱、按 80/20 划分 train/val。  
   - 这样 DAgger 数据与专家数据混合后再分割，避免分布偏差。

2) **标签优先级**  
   - 动作读取顺序：`expert_delta_q` → `delta_q` → `delta_q_cmd`。  
   - 兼容旧数据，同时优先利用 DAgger 的专家标注。

3) **样本校验**  
   - 检查 NaN/Inf、过大值、图片存在性；过滤无效样本，保证训练稳定。

4) **返回结构**  
   - `image`、`delta_q`（已按优先级选择）、`joint_positions`、`raw`。  
   - 图像预处理与 BC 模型保持一致（Resize + ToTensor + ImageNet Norm）。

## 运行示例
1) 采集 DAgger 数据：
```
python rp_collect/dagger_collect.py \
  --bc_checkpoint /home/wopubuntu/me5400/training/checkpoints_bc_managed/best.pt \
  --episodes 50 \
  --behavior mixture \
  --mix_beta 0.6 \
  --image_height 240 --image_width 320
```
2) 混合训练：
```
python training/train_bc.py \
  --dataset_root /home/wopubuntu/me5400/rp_collect/DATA/expert_data,/home/wopubuntu/me5400/rp_collect/DATA/dagger_data \
  --out_dir ./checkpoints_bc_dagger
```

## 为什么能工作（保障点）
- **数据同构**：目录与字段与旧数据一致，训练脚本零改动；仅新增 `expert_delta_q` 提升标签质量。  
- **多源统一划分**：先合并后打乱再切分，避免某一来源只落在 val/train。  
- **混合执行保障安全**：早期用专家或 mixture，避免策略失效导致坏数据；即便策略差，也有专家标签可用。  
- **动作优先级设计**：训练时始终优先用最新、最可靠的专家标签。  
- **工作空间约束**：起始位姿拒绝采样 + TCP 位置检查，减少无效或不可达数据。

## 小贴士
- 无头采集：改 `SimulationApp({"headless": False})` 为 `True`。  
- 采集后快速自检：检查 `metadata` 与 `picture_data` 数量一致。  
- 评估：可复用 `training/evaluate_bc.py`，指向新的 checkpoint。  
- 若混合权重想动态调整，可在采集脚本中按步数或成功率调节 `mix_beta`。 
