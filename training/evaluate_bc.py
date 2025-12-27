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
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

# ===================== 模型定义 =====================

class ResNetMLPPolicy(nn.Module):
    """ResNet18 视觉编码 + MLP 动作头（与训练时一致）"""
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

# 成功条件（与数据收集时一致）
SUCCESS_DISTANCE_X_MAX = 0.1   # 米
SUCCESS_DISTANCE_Y_MAX = 0.1   # 米
SUCCESS_DISTANCE_Z_MAX = 0.3   # 米

# 碰撞检测阈值
COLLISION_VELOCITY_THRESHOLD = 10.0  # rad/s
COLLISION_ACCELERATION_THRESHOLD = 50.0  # rad/s²

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
    """随机采样关节配置"""
    random_joint_positions = []
    for i in range(num_joints):
        if i < len(PANDA_JOINT_LIMITS):
            lower, upper = PANDA_JOINT_LIMITS[i]
            random_joint_positions.append(np.random.uniform(lower, upper))
        else:
            random_joint_positions.append(np.random.uniform(-np.pi, np.pi))
    return np.array(random_joint_positions, dtype=np.float32)

def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    model = ResNetMLPPolicy(out_dim=7).to(device)
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
        else:
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
    parser = argparse.ArgumentParser(description="BC模型评估脚本")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--num_episodes", type=int, default=20, help="评估episode数量")
    parser.add_argument("--steps_per_episode", type=int, default=200, help="每个episode的最大步数")
    parser.add_argument("--save_images", action="store_true", help="是否保存评估过程中的图像")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="输出目录")
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

        # 随机初始配置
        random_joint_positions = sample_random_joint_config(robot.num_dof)
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

            # 1. 捕获图像
            temp_img_path = os.path.join(args.output_dir, "temp_frame.png")
            if not cam.capture(temp_img_path):
                print(f"   ⚠️ 第 {step} 步截图失败")
                continue

            # 等待文件写入完成（capture_viewport_to_file可能是异步的）
            max_wait_attempts = 10
            wait_attempt = 0
            while wait_attempt < max_wait_attempts:
                if os.path.exists(temp_img_path):
                    # 检查文件大小，确保不是空文件
                    file_size = os.path.getsize(temp_img_path)
                    if file_size > 0:
                        # 再等待一小段时间，确保文件完全写入
                        time.sleep(0.05)
                        break
                time.sleep(0.05)
                wait_attempt += 1
            
            if wait_attempt >= max_wait_attempts or not os.path.exists(temp_img_path):
                print(f"   ⚠️ 第 {step} 步图像文件未生成或为空")
                continue

            # 2. 预处理图像
            try:
                # 尝试打开图像，如果失败会抛出异常
                image = Image.open(temp_img_path).convert('RGB')
                # 尝试加载图像数据，确保文件完整
                image.load()  # 这会强制加载所有数据，如果文件损坏会抛出异常
                image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]
            except Exception as e:
                print(f"   ⚠️ 图像加载失败: {e}")
                continue

            # 3. 模型预测
            with torch.no_grad():
                delta_q_pred = model(image_tensor).cpu().numpy()[0]  # [7]

            # 4. 应用动作（delta_q -> 关节速度控制）
            # 使用简单的速度控制：将delta_q转换为速度命令
            q_current = robot.get_joint_positions()
            
            # 将delta_q转换为速度（带缩放因子，避免过大速度）
            max_velocity = 2.0  # rad/s，最大关节速度
            velocity_scale = 10.0  # 将delta_q转换为速度的缩放因子
            target_velocity = np.clip(delta_q_pred * velocity_scale, -max_velocity, max_velocity)
            
            # 应用速度命令
            robot.set_joint_velocities(target_velocity)

            # 5. 推进仿真
            sim.step(render=True)

            # 6. 检查碰撞
            dq_after_step = robot.get_joint_velocities()
            max_velocity = np.max(np.abs(dq_after_step))
            
            if max_velocity > COLLISION_VELOCITY_THRESHOLD:
                has_collision = True
                end_reason = "collision"
                print(f"   ⚠️ 第 {step} 步检测到碰撞（速度异常: {max_velocity:.2f} rad/s）")
                break

            if prev_dq is not None:
                acceleration = (dq_after_step - prev_dq) / DT
                max_acceleration = np.max(np.abs(acceleration))
                if max_acceleration > COLLISION_ACCELERATION_THRESHOLD:
                    has_collision = True
                    end_reason = "collision"
                    print(f"   ⚠️ 第 {step} 步检测到碰撞（加速度异常: {max_acceleration:.2f} rad/s²）")
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
                    print(f"   ✅ 第 {step} 步成功到达目标！(X={diff_x:.3f}m, Y={diff_y:.3f}m, Z={diff_z:.3f}m)")
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

        results["episode_details"].append({
            "episode": episode_idx,
            "success": episode_success,
            "end_reason": end_reason,
            "end_step": step
        })

        status_emoji = "✅" if episode_success else "❌"
        print(f"{status_emoji} Episode {episode_idx} 完成: {end_reason}")

    # --- 4. 打印统计结果 ---
    print(f"\n{'='*60}")
    print("评估结果统计")
    print(f"{'='*60}")
    print(f"总episode数: {args.num_episodes}")
    print(f"成功: {results['success']} ({results['success']/args.num_episodes*100:.1f}%)")
    print(f"碰撞: {results['collision']} ({results['collision']/args.num_episodes*100:.1f}%)")
    print(f"超时: {results['timeout']} ({results['timeout']/args.num_episodes*100:.1f}%)")
    print(f"{'='*60}")

    # 保存结果
    import json
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {results_file}")

    print("\n评估完成！")
    simulation_app.close()

if __name__ == "__main__":
    main()

