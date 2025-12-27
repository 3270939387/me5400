# 详细编码步骤

## Step 1: 完善数据收集 (`lula.py`)

### 1.1 添加必要的导入和工具函数

```python
import json
from pxr import UsdGeom, Gf
import omni.isaac.sensor as sensor
```

### 1.2 添加 marker 像素坐标计算函数

在 `lula.py` 中添加：

```python
def compute_marker_pixel_coords(stage, marker_prim_path, cam_prim_path, cam_resolution=(1920, 1080)):
    """
    计算 marker 在相机图像中的像素坐标
    返回: (u, v, visible) 或 None
    """
    from pxr import UsdGeom, Gf, UsdLux
    
    # 获取 marker 和相机 prim
    marker_prim = stage.GetPrimAtPath(marker_prim_path)
    cam_prim = stage.GetPrimAtPath(cam_prim_path)
    
    if not marker_prim.IsValid() or not cam_prim.IsValid():
        return None
    
    # 获取 marker 世界坐标
    marker_xform = UsdGeom.Xformable(marker_prim)
    marker_world_transform = marker_xform.ComputeLocalToWorldTransform(0)
    marker_world_pos = marker_world_transform.ExtractTranslation()
    
    # 获取相机参数（需要从 Camera prim 读取）
    # 这里简化处理，使用 Isaac Sim 的相机投影 API
    # 实际实现可能需要使用 omni.isaac.sensor 或手动投影
    
    # 临时方案：使用 Isaac Sim 的相机投影
    try:
        # 获取相机内参（需要从 Camera prim 属性读取）
        cam_xform = UsdGeom.Xformable(cam_prim)
        cam_world_transform = cam_xform.ComputeLocalToWorldTransform(0)
        cam_world_pos = cam_world_transform.ExtractTranslation()
        cam_world_rot = cam_world_transform.ExtractRotationMatrix()
        
        # 计算 marker 相对于相机的坐标
        marker_rel_pos = cam_world_rot.TransformDir(marker_world_pos - cam_world_pos)
        
        # 简化的投影（假设相机内参已知，需要根据实际相机调整）
        # 这里使用近似值，实际应该从 Camera prim 读取 focal length 等
        fx = fy = 1000  # 需要根据实际相机调整
        cx, cy = cam_resolution[0] / 2, cam_resolution[1] / 2
        
        if marker_rel_pos[2] > 0:  # 在相机前方
            u = fx * marker_rel_pos[0] / marker_rel_pos[2] + cx
            v = fy * marker_rel_pos[1] / marker_rel_pos[2] + cy
            
            # 检查是否在图像范围内
            visible = (0 <= u < cam_resolution[0] and 0 <= v < cam_resolution[1])
            if visible:
                return {"u": float(u), "v": float(v), "visible": True}
        
        return {"u": None, "v": None, "visible": False}
    except Exception as e:
        print(f"[WARN] 计算 marker 像素坐标失败: {e}")
        return None
```

### 1.3 修改主循环，收集并保存元数据

在 `lula.py` 的主循环前添加：

```python
# 创建 episode 目录
import datetime
episode_id = f"episode_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
episode_dir = os.path.join(OUTPUT_DIR, episode_id)
os.makedirs(episode_dir, exist_ok=True)

# 存储每步的数据
episode_data = {
    "episode_id": episode_id,
    "num_steps": 0,
    "env_path": ENV_PATH,
    "cam_path": CAM_PATH,
    "marker_path": MARKER_PATH,
    "initial_joint_positions": random_joint_positions.tolist(),
    "steps": []
}
```

修改主循环（在保存图片的地方）：

```python
for step_idx in range(NUM_STEPS):
    # 1) RMPflow 更新关节控制
    controller.update(dt)
    
    # 获取当前关节状态和动作（在 apply_action 之前）
    current_joint_pos = controller._articulation.get_joint_positions()
    current_joint_vel = controller._articulation.get_joint_velocities()
    
    # 获取 RMPflow 计算的动作（用于保存）
    # 注意：需要在 apply_action 之前获取
    action = controller._articulation_rmpflow.get_next_articulation_action(dt)
    action_delta_q = action.joint_positions - current_joint_pos  # 计算增量
    
    # 2) 推进一步仿真（含渲染）
    sim.step(render=True)
    
    # 3) 推进一步 Replicator
    rep.orchestrator.step()
    
    # 4) 按频率采集一帧
    if step_idx % CAPTURE_INTERVAL == 0:
        data = rgb_annotator.get_data()
        if data is not None:
            try:
                img = Image.fromarray(data)
                img_filename = f"rgb_{frame_id:04d}.png"
                img_path = os.path.join(episode_dir, img_filename)
                img.save(img_path)
                
                # 计算 marker 像素坐标
                marker_pixel = compute_marker_pixel_coords(
                    stage, MARKER_PATH, CAM_PATH, 
                    cam_resolution=(1920, 1080)
                )
                
                # 保存这一步的元数据
                step_data = {
                    "step": step_idx,
                    "image": img_filename,
                    "joint_positions": current_joint_pos.tolist(),
                    "joint_velocities": current_joint_vel.tolist(),
                    "action": {
                        "type": "joint_position_delta",
                        "delta_q": action_delta_q.tolist()
                    },
                    "marker_pixel": marker_pixel if marker_pixel else {
                        "u": None, "v": None, "visible": False
                    },
                    "done": False  # 可以后续添加终止条件判断
                }
                episode_data["steps"].append(step_data)
                
                print(f"[INFO] 已保存图片 {img_filename} (step={step_idx})")
                frame_id += 1
            except Exception as e:
                print(f"[WARN] 保存图片失败: {e}")
    
    # 5) 更新 UI
    app.update()
    time.sleep(0.001)

# 保存 episode 元数据
episode_data["num_steps"] = len(episode_data["steps"])
meta_path = os.path.join(episode_dir, "meta.json")
with open(meta_path, 'w') as f:
    json.dump(episode_data, f, indent=2)
print(f"[INFO] 已保存元数据到 {meta_path}")
```

---

## Step 2: 创建数据集加载器 (`dataset.py`)

创建文件 `/home/alphatok/ME5400/training/dataset.py`:

```python
import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MarkerDataset(Dataset):
    def __init__(self, dataset_root, split='train', image_size=(320, 240)):
        """
        dataset_root: 数据集根目录（包含 episodes/ 文件夹）
        split: 'train' 或 'val'
        image_size: 图像resize尺寸
        """
        self.dataset_root = dataset_root
        self.episodes_dir = os.path.join(dataset_root, "episodes")
        self.image_size = image_size
        
        # 加载所有 episodes
        self.episodes = []
        if os.path.exists(self.episodes_dir):
            for episode_name in sorted(os.listdir(self.episodes_dir)):
                episode_path = os.path.join(self.episodes_dir, episode_name)
                meta_path = os.path.join(episode_path, "meta.json")
                if os.path.exists(meta_path):
                    self.episodes.append(episode_path)
        
        # 划分 train/val (80/20)
        num_episodes = len(self.episodes)
        if split == 'train':
            self.episodes = self.episodes[:int(0.8 * num_episodes)]
        else:
            self.episodes = self.episodes[int(0.8 * num_episodes):]
        
        # 加载所有样本
        self.samples = []
        for episode_path in self.episodes:
            meta_path = os.path.join(episode_path, "meta.json")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            for step_data in meta["steps"]:
                self.samples.append({
                    "episode_path": episode_path,
                    "step_data": step_data
                })
        
        print(f"[INFO] {split} 数据集: {len(self.samples)} 个样本")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet 标准
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        episode_path = sample["episode_path"]
        step_data = sample["step_data"]
        
        # 加载图像
        img_path = os.path.join(episode_path, step_data["image"])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 获取动作目标
        delta_q = np.array(step_data["action"]["delta_q"], dtype=np.float32)
        
        return {
            "image": image,
            "delta_q": torch.from_numpy(delta_q),
            "joint_positions": torch.from_numpy(
                np.array(step_data["joint_positions"], dtype=np.float32)
            ),
            "marker_pixel": step_data["marker_pixel"]  # 用于评估
        }
```

---

## Step 3: 创建模型 (`model.py`)

创建文件 `/home/alphatok/ME5400/training/model.py`:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class VisionPolicy(nn.Module):
    def __init__(self, num_joints=7, feature_dim=512):
        super(VisionPolicy, self).__init__()
        
        # ResNet18 作为视觉编码器（ImageNet 预训练）
        resnet = models.resnet18(pretrained=True)
        # 移除最后的分类层
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 策略头：MLP
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_joints)  # 输出 delta_q
        )
    
    def forward(self, image):
        # 提取视觉特征
        features = self.visual_encoder(image)
        features = features.view(features.size(0), -1)  # Flatten
        
        # 预测动作
        delta_q = self.policy_head(features)
        
        return delta_q
```

---

## Step 4: 创建训练脚本 (`train.py`)

创建文件 `/home/alphatok/ME5400/training/train.py`:

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MarkerDataset
from model import VisionPolicy

# 配置
DATASET_ROOT = "/home/alphatok/ME5400/image data/output_lula_d405"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
IMAGE_SIZE = (320, 240)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建输出目录
OUTPUT_DIR = "/home/alphatok/ME5400/training/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)

# 数据集
train_dataset = MarkerDataset(DATASET_ROOT, split='train', image_size=IMAGE_SIZE)
val_dataset = MarkerDataset(DATASET_ROOT, split='val', image_size=IMAGE_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 模型
model = VisionPolicy(num_joints=7).to(DEVICE)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# TensorBoard
writer = SummaryWriter(os.path.join(OUTPUT_DIR, "logs"))

# 训练循环
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    
    for batch in train_bar:
        images = batch["image"].to(DEVICE)
        delta_q_target = batch["delta_q"].to(DEVICE)
        
        # 前向传播
        optimizer.zero_grad()
        delta_q_pred = model(images)
        
        # 计算损失
        loss = criterion(delta_q_pred, delta_q_target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_bar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    avg_train_loss = train_loss / len(train_loader)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
    
    with torch.no_grad():
        for batch in val_bar:
            images = batch["image"].to(DEVICE)
            delta_q_target = batch["delta_q"].to(DEVICE)
            
            delta_q_pred = model(images)
            loss = criterion(delta_q_pred, delta_q_target)
            
            val_loss += loss.item()
            val_bar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    avg_val_loss = val_loss / len(val_loader)
    
    # 学习率调度
    scheduler.step(avg_val_loss)
    
    # 记录到 TensorBoard
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Loss/Val", avg_val_loss, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"[INFO] 保存最佳模型 (val_loss={avg_val_loss:.6f})")
    
    # 定期保存
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", f"epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, checkpoint_path)

print("[INFO] 训练完成！")
writer.close()
```

---

## Step 5: 创建评估脚本 (`lula_eval.py`)

创建文件 `/home/alphatok/ME5400/lula/lula_eval.py`:

```python
# 配置区
ENV_PATH = "/home/alphatok/ME5400/env.setup/env.usda"
CAM_PATH = "/World/Panda/D405_rigid/D405/Camera_OmniVision_OV9782_Color"
MARKER_PATH = "/World/Phantom/marker"
MODEL_PATH = "/home/alphatok/ME5400/training/outputs/checkpoints/best_model.pth"
NUM_EPISODES = 10
NUM_STEPS_PER_EPISODE = 400

import os
import time
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

import omni.usd
import omni.kit.app
import omni.replicator.core as rep

from isaacsim.core.prims import SingleArticulation as Articulation
from omni.isaac.core import SimulationContext

# 导入模型
import sys
sys.path.append("/home/alphatok/ME5400/training")
from model import VisionPolicy

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionPolicy(num_joints=7).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"[INFO] 已加载模型: {MODEL_PATH}")

# 图像预处理（与训练时一致）
transform = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# 加载环境
usd_context = omni.usd.get_context()
stage = usd_context.get_stage()

if not stage or not stage.GetPrimAtPath("/World").IsValid():
    print(f"[INFO] 加载环境: {ENV_PATH}")
    usd_context.open_stage(ENV_PATH)
    for _ in range(60):
        omni.kit.app.get_app().update()
        time.sleep(0.01)
    stage = usd_context.get_stage()

# 初始化仿真
sim = SimulationContext()
dt = 1.0 / 60.0
sim.play()

# 初始化机器人
robot = Articulation("/World/Panda")
robot.initialize()

# 随机初始关节角（与训练时一致）
panda_joint_limits = [
    (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
    (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)
]

# 初始化相机
rp = rep.create.render_product(CAM_PATH, resolution=(1920, 1080))
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([rp])

app = omni.kit.app.get_app()

# 评估循环
success_count = 0
for episode in range(NUM_EPISODES):
    print(f"\n[Episode {episode+1}/{NUM_EPISODES}]")
    
    # 设置随机初始位置
    random_joint_positions = []
    for i in range(robot.num_dof):
        if i < len(panda_joint_limits):
            lower, upper = panda_joint_limits[i]
        else:
            lower, upper = -np.pi, np.pi
        random_joint_positions.append(np.random.uniform(lower, upper))
    
    robot.set_joint_positions(np.array(random_joint_positions))
    
    # 预热
    for _ in range(20):
        sim.step()
    
    # Episode 循环
    episode_success = False
    for step in range(NUM_STEPS_PER_EPISODE):
        # 获取当前图像
        sim.step(render=True)
        rep.orchestrator.step()
        
        data = rgb_annotator.get_data()
        if data is not None:
            # 预处理图像
            img = Image.fromarray(data).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # 模型推理
            with torch.no_grad():
                delta_q_pred = model(img_tensor)
                delta_q_pred = delta_q_pred.cpu().numpy()[0]
            
            # 应用动作（限制最大变化量）
            current_q = robot.get_joint_positions()
            max_delta = 0.1  # 限制单步最大变化
            delta_q_clamped = np.clip(delta_q_pred, -max_delta, max_delta)
            next_q = current_q + delta_q_clamped
            
            # 应用关节限制
            for i in range(robot.num_dof):
                if i < len(panda_joint_limits):
                    lower, upper = panda_joint_limits[i]
                    next_q[i] = np.clip(next_q[i], lower, upper)
            
            robot.set_joint_positions(next_q)
        
        app.update()
        time.sleep(0.001)
        
        # 检查成功条件（可以添加 marker 位置检查）
        # 这里简化处理，你可以添加实际的成功判断逻辑
    
    if episode_success:
        success_count += 1

print(f"\n[评估结果] 成功率: {success_count}/{NUM_EPISODES} ({100*success_count/NUM_EPISODES:.1f}%)")
```

---

## Step 6: 运行步骤总结

### 6.1 数据收集
```bash
cd /home/alphatok/ME5400
~/isaacsim/python.sh lula/lula.py
# 运行多次，生成多个 episodes
```

### 6.2 训练
```bash
cd /home/alphatok/ME5400/training
python train.py
```

### 6.3 评估
```bash
cd /home/alphatok/ME5400
~/isaacsim/python.sh lula/lula_eval.py
```

---

## 注意事项

1. **Marker 像素坐标计算**：上面的 `compute_marker_pixel_coords` 是简化版本，可能需要根据实际相机参数调整
2. **动作获取**：需要确保在 `apply_action` 之前获取 RMPflow 的动作
3. **数据量**：建议至少收集 50-100 个 episodes 再开始训练
4. **GPU 内存**：如果内存不足，可以减小 `BATCH_SIZE` 或 `IMAGE_SIZE`

