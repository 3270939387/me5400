#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavior Cloning 模型评估脚本
在Isaac Sim中加载训练好的模型并评估性能

运行方式（必须使用Isaac Sim的Python）:
    ~/isaacsim/python.sh /home/alphatok/ME5400/training/evaluate_bc.py --checkpoint <path> --num_episodes 20

或者使用提供的启动脚本:
    bash run_evaluate.sh <checkpoint_path> [num_episodes]
"""

from isaacsim import SimulationApp

# 启动 Isaac Sim
simulation_app = SimulationApp({"headless": False})

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

import omni.timeline
import omni.usd
from pxr import UsdPhysics, Gf, Usd

# 核心模块
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

# ===================== 模型定义 =====================

class ResNetMLPPolicy(nn.Module):
    """
    ResNet18 视觉编码 + MLP 动作头
    
    输入：
      - x: [B, 3, H, W] - RGB图像 → ResNet18提取512维特征
      - q: [B, 7] - 关节位置（proprioceptive state，解决非Markovian）
      - marker_geom: [B, 3] - marker的(u, v, s)信息
      - marker_visible: [B, 1] - marker可见性标志
    
    拼接顺序：image_features [512] + q [7] + marker_geom [3] + visible [1] = 523维
    
    设计理由：
      - q：解决非Markovian问题（同一图像可能对应不同关节配置）
      - marker_geom：显式提供视觉伺服的几何约束
      - marker_visible：显式告诉网络几何信息是否可靠
    """
    def __init__(self, out_dim=7, marker_geom_dim=3, q_dim=7, use_visible=True):
        super().__init__()
        # ResNet18 backbone (去掉最后的全连接层)
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # [B,512,1,1]
        
        self.use_visible = use_visible
        # MLP head 拼接所有信息
        # 输入维度 = image_features (512) + q (7) + marker_geom (3) + visible (0 or 1)
        extra_dim = marker_geom_dim + q_dim + (1 if use_visible else 0)
        mlp_input_dim = 512 + extra_dim
        
        self.head = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(self, x, q=None, marker_geom=None, marker_visible=None):
        """
        前向传播
        
        Args:
            x: [B, 3, H, W] - RGB图像
            q: [B, 7] - 当前关节位置，或None
            marker_geom: [B, 3] - marker的(u, v, s)信息，或None
                - 当marker不可见时，uvs已被清零为[0, 0, 0]
            marker_visible: [B, 1] - marker可见性标志，或None
                - 1.0: marker可见，uvs有效
                - 0.0: marker不可见，uvs已清零
        
        Returns:
            [B, 7] - 预测的delta_q
        """
        # 提取图像特征
        feat = self.backbone(x).flatten(1)  # [B, 512]
        
        # 拼接q (proprioceptive state)
        if q is not None:
            feat = torch.cat([feat, q], dim=1)  # [B, 512+7]
        
        # 拼接marker_geom (uvs信息，当visible=0时已清零)
        if marker_geom is not None:
            feat = torch.cat([feat, marker_geom], dim=1)  # [B, +3]
        
        # 拼接marker_visible (显式告诉网络几何是否可靠)
        if self.use_visible:
            # ⚠️ 关键：宁可crash也别默默补1
            # 如果marker_visible是None，说明数据加载有问题
            # 立即报错，而不是无声地创建ones张量
            assert marker_visible is not None, (
                "❌ marker_visible is required when use_visible=True!\n"
                "Check your dataset or dataloader - did you forget to return 'marker_visible'?"
            )
            feat = torch.cat([feat, marker_visible], dim=1)  # [B, +1]
        
        # 通过MLP head输出动作
        return self.head(feat)  # [B, 7]

# ===================== 配置 =====================

ENV_USD_PATH = "/home/alphatok/ME5400/env.setup/env.usda"
MARKER_PATH = "/World/Phantom/marker"
ROBOT_PATH = "/World/Panda"
TABLE_PATH = "/World/Table"
CAM_PATH = "/World/Panda/D405_rigid/D405/Camera_OmniVision_OV9782_Color"
TCP_PATH = "/World/Panda/TCP"

DT = 1.0 / 60.0

# Panda 关节限制
PANDA_JOINT_LIMITS = [
    (-2.8973, 2.8973),   # joint1
    (-1.7628, 1.7628),   # joint2
    (-2.8973, 2.8973),   # joint3
    (-3.0718, -0.0698),  # joint4
    (-2.8973, 2.8973),   # joint5
    (-0.0175, 3.7525),   # joint6
    (-2.8973, 2.8973),   # joint7
]

# 工作空间定义（与数据收集时一致）
WORKSPACE_CENTER = np.array([0.0, 0.50, 0.50])  # 米
WORKSPACE_RADIUS = 0.25  # 米（25cm）
WORKSPACE_Z_MIN = 0.20  # 米
WORKSPACE_Z_MAX = 0.75  # 米

# 成功条件（与数据收集时一致）
SUCCESS_DISTANCE_X_MAX = 0.1   # 米
SUCCESS_DISTANCE_Y_MAX = 0.1   # 米
SUCCESS_DISTANCE_Z_MAX = 0.3   # 米

# 碰撞检测阈值
COLLISION_VELOCITY_THRESHOLD = 10.0  # rad/s
COLLISION_ACCELERATION_THRESHOLD = 50.0  # rad/s²

# 预定义的中立姿态（与数据收集一致）
NEUTRAL_POSES = [
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],   # 01. 标准 Home
    [0.0, -0.500, 0.0, -2.000, 0.0, 1.500, 0.785],   # 02. 略高俯视
    [0.0, -1.000, 0.0, -2.500, 0.0, 1.700, 0.785],   # 03. 略低平视
    [0.4, -0.785, 0.2, -2.356, 0.0, 1.571, 0.785],   # 04. 右偏 23度
    [0.8, -0.600, 0.3, -2.100, 0.1, 1.600, 0.800],   # 05. 右深处
    [0.6, -0.850, 0.1, -2.400, 0.0, 1.550, 0.750],   # 06. 右中距离
    [-0.4, -0.785, -0.2, -2.356, 0.0, 1.571, 0.785], # 07. 左偏 23度
    [-0.8, -0.600, -0.3, -2.100, -0.1, 1.600, 0.800],# 08. 左深处
    [-0.6, -0.850, -0.1, -2.400, 0.0, 1.550, 0.750], # 09. 左中距离
    [0.0, -0.200, 0.0, -1.500, 0.0, 1.300, 0.785],   # 10. 高位中心
    [0.5, -0.200, 0.1, -1.600, 0.0, 1.400, 0.785],   # 11. 高位右视
    [-0.5, -0.200, -0.1, -1.600, 0.0, 1.400, 0.785], # 12. 高位左视
    [0.0, -1.200, 0.0, -2.800, 0.0, 1.800, 0.785],   # 13. 中心压低
    [0.3, -1.150, 0.1, -2.700, 0.0, 1.750, 0.785],   # 14. 右侧压低
    [-0.3, -1.150, -0.1, -2.700, 0.0, 1.750, 0.785], # 15. 左侧压低
    [0.0, -0.785, 0.0, -2.356, 0.5, 1.571, 1.200],   # 16. 手腕右旋
    [0.0, -0.785, 0.0, -2.356, -0.5, 1.571, 0.400],  # 17. 手腕左旋
    [0.0, -0.500, 0.0, -1.800, 0.0, 1.900, 0.785],   # 18. 中心前伸
    [0.2, -0.400, 0.0, -1.700, 0.0, 2.000, 0.785],   # 19. 右前斜伸
    [-0.2, -0.400, 0.0, -1.700, 0.0, 2.000, 0.785],  # 20. 左前斜伸
]

# 扰动参数（与数据收集一致）
PERTURBATION_SCALE = 0.15  # 扰动幅度（弧度），约8.6度

# ===================== 辅助函数 =====================

class ViewportCamera:
    """视口相机封装"""
    def __init__(self, camera_path, resolution=(1280, 720)):
        self.viewport_api = get_active_viewport()
        if not self.viewport_api:
            raise RuntimeError("❌ 无法找到活跃视口！")
        self.viewport_api.camera_path = camera_path
        self.viewport_api.set_texture_resolution(resolution)

    def capture(self, filename):
        try:
            capture_viewport_to_file(self.viewport_api, filename)
            return True
        except Exception as e:
            print(f"❌ 截图异常: {e}")
            return False

def sample_random_joint_config(num_joints):
    """在关节限位内随机采样关节配置"""
    random_joint_positions = []
    for i in range(num_joints):
        if i < len(PANDA_JOINT_LIMITS):
            lower, upper = PANDA_JOINT_LIMITS[i]
            random_joint_positions.append(np.random.uniform(lower, upper))
        else:
            random_joint_positions.append(np.random.uniform(-np.pi, np.pi))
    return np.array(random_joint_positions, dtype=np.float32)

def check_workspace_constraint(ee_pos_base):
    """
    检查末端执行器位置是否在工作空间内
    ee_pos_base: 末端执行器在 Panda base 坐标系下的位置 (x, y, z)
    返回: (is_valid, reason)
    """
    # 1. 检查球约束：||p_ee - center|| <= radius
    offset = ee_pos_base - WORKSPACE_CENTER
    distance = np.linalg.norm(offset)
    if distance > WORKSPACE_RADIUS:
        return False, f"超出球半径: {distance:.3f}m > {WORKSPACE_RADIUS}m"
    
    # 2. 检查Z范围约束：z_min <= z <= z_max
    z = ee_pos_base[2]
    if z < WORKSPACE_Z_MIN:
        return False, f"Z过低: {z:.3f}m < {WORKSPACE_Z_MIN}m"
    if z > WORKSPACE_Z_MAX:
        return False, f"Z过高: {z:.3f}m > {WORKSPACE_Z_MAX}m"
    
    return True, "OK"

def sample_valid_initial_config(robot, sim, max_attempts=100):
    """
    ✅ 简化版本：使用 Neutral Poses + 小扰动（与数据收集一致）
    
    由于 neutral poses 都是预定义的合理姿态，不需要复杂的工作空间检查
    只需要确保：
    1. 添加小扰动
    2. 在关节限位内截断
    3. 让物理引擎处理
    
    最多尝试 max_attempts 次（处理偶发的物理失败）
    
    返回: (joint_positions, ee_pos_base) 或 (None, None) 如果失败
    """
    num_joints = robot.num_dof
    
    for attempt in range(max_attempts):
        # 1. 随机选择一个 neutral pose
        base_pose = np.array(NEUTRAL_POSES[np.random.randint(0, len(NEUTRAL_POSES))], dtype=np.float32)
        
        # 2. 添加小扰动（每个关节独立扰动）
        perturbation = np.random.uniform(-PERTURBATION_SCALE, PERTURBATION_SCALE, size=num_joints)
        perturbed_pose = base_pose + perturbation
        
        # 3. 在关节限位内截断
        joint_positions = np.zeros(num_joints, dtype=np.float32)
        for i in range(num_joints):
            if i < len(PANDA_JOINT_LIMITS):
                lower, upper = PANDA_JOINT_LIMITS[i]
                joint_positions[i] = np.clip(perturbed_pose[i], lower, upper)
            else:
                joint_positions[i] = perturbed_pose[i]
        
        # 4. 应用到机器人并让物理稳定
        robot.set_joint_velocities(np.zeros(num_joints))
        robot.set_joint_positions(joint_positions)
        
        # 推进几帧让物理引擎处理
        for _ in range(10):
            sim.step(render=False)
        
        # 5. 获取末端执行器的世界坐标（仅用于验证，不做约束检查）
        try:
            tcp_prim = XFormPrim(TCP_PATH)
            tcp_world_pos, _ = tcp_prim.get_world_pose()
            tcp_base_pos = np.array([float(tcp_world_pos[0]), float(tcp_world_pos[1]), float(tcp_world_pos[2])])
            
            # ✅ 直接返回（不需要工作空间检查，neutral pose已经验证过）
            if attempt > 0:
                pass  # 评估时不打印太多信息
            return joint_positions, tcp_base_pos
            
        except Exception as e:
            continue
    
    return None, None

def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    model = ResNetMLPPolicy(out_dim=7, marker_geom_dim=3, q_dim=7, use_visible=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"✅ 已加载模型: {checkpoint_path}")
    print(f"   训练epoch: {checkpoint.get('epoch', 'N/A')}")
    val_loss = checkpoint.get('val_loss', None)
    if val_loss is not None:
        print(f"   验证loss: {val_loss:.6f}")
    else:
        val_mse = checkpoint.get('val_mse', None)
        if val_mse is not None:
            print(f"   验证MSE: {val_mse:.6f}")
        val_weighted_mse = checkpoint.get('val_weighted_mse', None)
        if val_weighted_mse is not None:
            print(f"   加权验证MSE: {val_weighted_mse:.6f}")
        if val_mse is None and val_weighted_mse is None:
            print(f"   验证loss: N/A")
    return model

def get_image_transform():
    """获取图像预处理（与训练时一致）"""
    return transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def check_success(ee_pos, marker_pos):
    """检查是否满足成功条件"""
    diff_x = abs(ee_pos[0] - marker_pos[0])
    diff_y = abs(ee_pos[1] - marker_pos[1])
    diff_z = abs(ee_pos[2] - marker_pos[2])
    
    return (diff_x < SUCCESS_DISTANCE_X_MAX) and \
           (diff_y < SUCCESS_DISTANCE_Y_MAX) and \
           (diff_z < SUCCESS_DISTANCE_Z_MAX)

# ===================== 主函数 =====================

def main():
    # 获取项目根目录（evaluate_bc.py 在 training/ 目录下）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # 项目根目录
    default_output_dir = os.path.join(project_root, "evaluation")
    
    parser = argparse.ArgumentParser(description="BC模型评估脚本")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--num_episodes", type=int, default=20, help="评估episode数量")
    parser.add_argument("--steps_per_episode", type=int, default=200, help="每个episode的最大步数")
    parser.add_argument("--save_images", action="store_true", help="是否保存评估过程中的图像")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="输出目录（默认: <project_root>/evaluation）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model(args.checkpoint, device)
    transform = get_image_transform()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_images:
        image_dir = os.path.join(args.output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

    # --- 1. 加载环境 ---
    print(f"\n正在加载场景: {ENV_USD_PATH}")
    omni.usd.get_context().open_stage(ENV_USD_PATH)
    for _ in range(100):
        simulation_app.update()

    # --- 2. 初始化仿真 ---
    timeline = omni.timeline.get_timeline_interface()
    
    stage = omni.usd.get_context().get_stage()
    has_physics = False
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            has_physics = True
            break
    if not has_physics:
        print("⚠️ 创建默认 PhysicsScene...")
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    # 稳定桌子
    table_prim = stage.GetPrimAtPath(TABLE_PATH)
    if table_prim.IsValid():
        if not table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(table_prim)
        UsdPhysics.RigidBodyAPI(table_prim).CreateKinematicEnabledAttr(True)

    # 创建机器人对象
    print("创建机器人对象...")
    robot = Articulation(ROBOT_PATH)

    # 初始化仿真
    print("初始化 SimulationContext...")
    sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    
    print("启动 Timeline...")
    timeline.play()
    
    print("强制初始化物理引擎...")
    sim.initialize_physics()
    
    if not sim.is_playing():
        sim.play()

    # 预热
    print("正在预热物理引擎 (60帧)...")
    for _ in range(60):
        sim.step(render=False)

    # 初始化机器人
    print("初始化机器人...")
    try:
        robot.initialize()
    except Exception as e:
        print(f"⚠️ 第一次初始化失败 ({e})，尝试重试...")
        for _ in range(10):
            sim.step(render=False)
        robot.initialize()

    # 初始化相机
    cam = ViewportCamera(CAM_PATH)

    # 获取marker位置
    marker_prim = XFormPrim(MARKER_PATH)
    marker_pos, marker_orn = marker_prim.get_world_pose()
    marker_pos = np.array([float(marker_pos[0]), float(marker_pos[1]), float(marker_pos[2])])

    # --- 3. 评估循环 ---
    print(f"\n🚀 开始评估: {args.num_episodes} 个episode，每episode最多 {args.steps_per_episode} 步")
    
    results = {
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "episode_details": []
    }

    for episode_idx in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode_idx + 1}/{args.num_episodes}")
        print(f"{'='*60}")

        # 为每个 episode 创建独立的子文件夹
        episode_output_dir = os.path.join(args.output_dir, f"episode_{episode_idx:02d}")
        os.makedirs(episode_output_dir, exist_ok=True)

        # 使用拒绝采样找到工作空间内的有效初始配置（与数据收集时一致）
        random_joint_positions, ee_pos_base = sample_valid_initial_config(robot, sim, max_attempts=100)
        
        if random_joint_positions is None:
            print(f"   ⚠️ 无法找到有效配置，使用随机配置（可能不在工作空间内）")
            random_joint_positions = sample_random_joint_config(robot.num_dof)
            robot.set_joint_velocities(np.zeros(robot.num_dof))
            robot.set_joint_positions(random_joint_positions)
        else:
            # 配置已设置，只需确保位置正确
            robot.set_joint_velocities(np.zeros(robot.num_dof))
            robot.set_joint_positions(random_joint_positions)
        
        # 物理稳定
        for _ in range(30):
            sim.step(render=True)

        episode_success = False
        has_collision = False
        end_reason = "timeout"
        prev_dq = None

        # Episode循环
        for step in range(args.steps_per_episode):
            if not simulation_app.is_running():
                break

            # 显示进度（每10步显示一次，或最后一步）
            if step % 10 == 0 or step == args.steps_per_episode - 1:
                progress_pct = (step + 1) / args.steps_per_episode * 100
                print(f"   步数: {step + 1}/{args.steps_per_episode} ({progress_pct:.1f}%)", end="\r", flush=True)

            # 1. 捕获图像（保存到该 episode 的子文件夹中）
            temp_img_path = os.path.join(episode_output_dir, f"frame_{step:04d}.png")
            image_tensor = None
            max_capture_retries = 2  # 减少重试次数，因为文件名已不同
            
            for capture_retry in range(max_capture_retries):
                # 强制渲染更新（确保 viewport 已渲染）
                simulation_app.update()
                
                # 捕获图像
                if not cam.capture(temp_img_path):
                    if capture_retry < max_capture_retries - 1:
                        simulation_app.update()
                        time.sleep(0.05)
                        continue
                    else:
                        print(f"   ⚠️ 第 {step} 步截图失败（已重试 {max_capture_retries} 次）")
                        break

                # 强制刷新（确保文件写入开始）
                simulation_app.update()
                
                # 等待文件写入完成（简化逻辑：只要文件大小 > 最小阈值即可）
                min_bytes = 10_000  # 最小文件大小阈值（1280x720 PNG 一般远大于这个）
                max_wait_attempts = 20
                wait_attempt = 0
                file_ready = False
                
                while wait_attempt < max_wait_attempts:
                    if os.path.exists(temp_img_path):
                        file_size = os.path.getsize(temp_img_path)
                        if file_size >= min_bytes:
                            file_ready = True
                            break
                    simulation_app.update()  # 每次检查时也更新
                    time.sleep(0.05)
                    wait_attempt += 1
                
                if not file_ready:
                    if capture_retry < max_capture_retries - 1:
                        simulation_app.update()
                        time.sleep(0.05)
                        continue
                    else:
                        print(f"   ⚠️ 第 {step} 步图像文件未就绪（已重试 {max_capture_retries} 次）")
                        break

                # 2. 预处理图像
                try:
                    # 尝试打开图像
                    image = Image.open(temp_img_path).convert('RGB')
                    # 验证图像完整性
                    image.verify()  # 验证但不加载数据
                    image = Image.open(temp_img_path).convert('RGB')  # 重新打开以加载数据
                    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]
                    break  # 成功加载，退出重试循环
                except Exception as e:
                    if capture_retry < max_capture_retries - 1:
                        simulation_app.update()
                        time.sleep(0.05)
                        continue
                    else:
                        print(f"   ⚠️ 图像加载失败: {e}（已重试 {max_capture_retries} 次）")
                        break
            
            if image_tensor is None:
                continue  # 跳过这一步，继续下一步

            # 3. 准备模型输入：q, marker_geom, marker_visible
            # 获取当前关节位置 [7]
            q_current = robot.get_joint_positions()
            q_tensor = torch.from_numpy(q_current).float().unsqueeze(0).to(device)  # [1, 7]
            
            # 对于evaluate，marker_geom和marker_visible在实际部署中需要从真实传感器获取
            # 这里简化处理：设置为有效值（因为我们有marker）
            marker_geom_tensor = torch.zeros(1, 3, dtype=torch.float32, device=device)  # [1, 3]
            marker_visible_tensor = torch.ones(1, 1, dtype=torch.float32, device=device)  # [1, 1] - 假设marker可见
            
            # 3. 模型预测
            with torch.no_grad():
                delta_q_pred = model(
                    image_tensor,
                    q=q_tensor,
                    marker_geom=marker_geom_tensor,
                    marker_visible=marker_visible_tensor
                ).cpu().numpy()[0]  # [7]

            # 4. 应用动作（delta_q -> 目标关节位置）
            # 将 delta_q 转换为目标关节位置（更符合 BC 训练的语义）
            q_current = robot.get_joint_positions()
            
            # 计算目标关节位置：q_target = q_current + delta_q
            q_target = q_current + delta_q_pred
            
            # 限制在关节限位内（避免超出物理限制）
            for i in range(len(q_target)):
                if i < len(PANDA_JOINT_LIMITS):
                    lower, upper = PANDA_JOINT_LIMITS[i]
                    q_target[i] = np.clip(q_target[i], lower, upper)
            
            # 使用 apply_action 应用目标位置（与数据收集时一致，更可靠）
            action = ArticulationAction(joint_positions=q_target)
            robot.apply_action(action)

            # 5. 推进仿真
            sim.step(render=True)
            simulation_app.update()  # 确保渲染更新

            # 6. 检查碰撞
            dq_after_step = robot.get_joint_velocities()
            max_velocity = np.max(np.abs(dq_after_step))
            
            if max_velocity > COLLISION_VELOCITY_THRESHOLD:
                has_collision = True
                end_reason = "collision"
                print(f"\n   ⚠️ 第 {step + 1} 步检测到碰撞（速度异常: {max_velocity:.2f} rad/s）")
                break

            if prev_dq is not None:
                acceleration = (dq_after_step - prev_dq) / DT
                max_acceleration = np.max(np.abs(acceleration))
                if max_acceleration > COLLISION_ACCELERATION_THRESHOLD:
                    has_collision = True
                    end_reason = "collision"
                    print(f"\n   ⚠️ 第 {step + 1} 步检测到碰撞（加速度异常: {max_acceleration:.2f} rad/s²）")
                    break

            prev_dq = dq_after_step.copy()

            # 7. 检查成功条件
            try:
                tcp_prim = XFormPrim(TCP_PATH)
                tcp_pos, _ = tcp_prim.get_world_pose()
                tcp_pos = np.array([float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])])
                
                if check_success(tcp_pos, marker_pos):
                    episode_success = True
                    end_reason = "success"
                    diff_x = abs(tcp_pos[0] - marker_pos[0])
                    diff_y = abs(tcp_pos[1] - marker_pos[1])
                    diff_z = abs(tcp_pos[2] - marker_pos[2])
                    print(f"\n   ✅ 第 {step + 1} 步成功到达目标！(X={diff_x:.3f}m, Y={diff_y:.3f}m, Z={diff_z:.3f}m)")
                    break
            except Exception as e:
                print(f"   ⚠️ 无法获取TCP位置: {e}")

        # 记录结果
        if episode_success:
            results["success"] += 1
        elif has_collision:
            results["collision"] += 1
        else:
            results["timeout"] += 1

        episode_result = {
            "episode": episode_idx,
            "success": episode_success,
            "end_reason": end_reason,
            "end_step": step
        }
        results["episode_details"].append(episode_result)

        # 保存该 episode 的单独结果文件
        episode_result_file = os.path.join(episode_output_dir, "episode_result.json")
        with open(episode_result_file, "w") as f:
            json.dump(episode_result, f, indent=2)

        # 清除进度显示行
        print(" " * 50, end="\r")  # 清除进度行
        
        status_emoji = "✅" if episode_success else "❌"
        print(f"{status_emoji} Episode {episode_idx} 完成: {end_reason} (共 {step + 1} 步)")

    # --- 4. 打印统计结果 ---
    print(f"\n{'='*60}")
    print("评估结果统计")
    print(f"{'='*60}")
    print(f"总episode数: {args.num_episodes}")
    print(f"成功: {results['success']} ({results['success']/args.num_episodes*100:.1f}%)")
    print(f"碰撞: {results['collision']} ({results['collision']/args.num_episodes*100:.1f}%)")
    print(f"超时: {results['timeout']} ({results['timeout']/args.num_episodes*100:.1f}%)")
    print(f"{'='*60}")

    # 保存总体结果
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {results_file}")

    print("\n评估完成！")
    simulation_app.close()

if __name__ == "__main__":
    main()

