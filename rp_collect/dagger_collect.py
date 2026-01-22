#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAgger 数据采集脚本

核心思路：
- 控制：由已训练的 BC 策略执行（可选与专家混合）；安全/标签仍使用 RMPFlow 专家
- 记录：每一步都会记录图像 + 专家动作标签（expert_delta_q），同时存储策略输出与实际执行的动作
- 数据结构：与 MarkerDataset 兼容
    DATASET_ROOT/
      ├─ metadata/episode_XXXX.json
      └─ picture_data/episode_XXXX/frame_YYYY.png
"""

from isaacsim import SimulationApp

# DAgger 建议 headless=False 便于观察，但可按需修改
simulation_app = SimulationApp({"headless": False})

import sys
import os
import json
import time
import shutil
import argparse
import numpy as np
import omni.timeline
import omni.usd
from pxr import UsdPhysics, Gf, Usd
import torch
from torchvision import transforms
from PIL import Image

ROOT_DIR = "/home/wopubuntu/me5400"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from omni.isaac.core import SimulationContext
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

# 复用训练时的模型定义
from training.train_bc import ResNetMLPPolicy


# ===================== ⚙️ 默认配置 =====================
ENV_USD_PATH = "/home/wopubuntu/me5400/env.setup/env_single_arm.usda"
MARKER_PATH = "/World/Phantom/marker"
ROBOT_PATH = "/World/Panda"
TABLE_PATH = "/World/Table"
CAM_PATH = "/World/Panda/D405_rigid/D405/Camera_OmniVision_OV9782_Color"

DEFAULT_DATASET_ROOT = "/home/wopubuntu/me5400/rp_collect/DATA/dagger_data"
DT = 1.0 / 60.0
SUCCESS_DISTANCE = (0.1, 0.1, 0.3)  # (x,y,z) 阈值

# 工作空间
PANDA_JOINT_LIMITS = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]
WORKSPACE_CENTER = np.array([0.0, 0.50, 0.50])
WORKSPACE_RADIUS = 0.30
WORKSPACE_Z_MIN = 0.15
WORKSPACE_Z_MAX = 0.80


# ===================== 工具函数 =====================
class ViewportCamera:
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool8)):  # 处理numpy布尔类型
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8, np.uint16,
                              np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def vec3_to_list(v):
    if v is None:
        return None
    try:
        return [float(v[0]), float(v[1]), float(v[2])]
    except Exception:
        arr = np.array(v).astype(float)
        return [float(arr[0]), float(arr[1]), float(arr[2])]


def sample_random_joint_config(num_joints):
    cfg = []
    for i in range(num_joints):
        if i < len(PANDA_JOINT_LIMITS):
            lower, upper = PANDA_JOINT_LIMITS[i]
            cfg.append(np.random.uniform(lower, upper))
        else:
            cfg.append(np.random.uniform(-np.pi, np.pi))
    return np.array(cfg, dtype=np.float32)


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
    使用拒绝采样找到工作空间内的有效初始配置
    返回: (joint_positions, ee_pos_base) 或 (None, None) 如果失败
    """
    # 获取 Panda base 的世界变换
    base_quat = None
    base_pos = None
    
    try:
        base_prim = XFormPrim("/World/Panda")
        base_world_pos, base_world_orn = base_prim.get_world_pose()
        
        # 确保转换为 Python float（Gf 需要 double 类型）
        base_pos = [float(base_world_pos[0]), float(base_world_pos[1]), float(base_world_pos[2])]
        base_orn = [float(base_world_orn[0]), float(base_world_orn[1]), float(base_world_orn[2]), float(base_world_orn[3])]
        
        # 使用 Gf 库处理四元数和旋转
        # base_orn 是 (w, x, y, z) 格式
        # base_quat: Gf.Quatd(w, Vec3d(x,y,z))
        base_quat = Gf.Quatd(float(base_orn[0]), Gf.Vec3d(float(base_orn[1]), float(base_orn[2]), float(base_orn[3])))
        
    except Exception as e:
        print(f"⚠️ 无法获取base变换: {e}，使用简化方法（仅平移）")
        # 简化方法：只考虑平移（转换为 float 避免类型错误）
        try:
            base_pos = [float(base_world_pos[0]), float(base_world_pos[1]), float(base_world_pos[2])]
        except:
            # 如果 base_world_pos 也不存在，使用默认值
            base_pos = [0.0, 0.0, 0.0]
        base_quat = None
    
    # 构建从world到base的变换
    def world_to_base(p_world):
        # p_rel = p_world - base_pos
        p_world = np.array([float(p_world[0]), float(p_world[1]), float(p_world[2])], dtype=float)
        p_rel = Gf.Vec3d(p_world[0] - float(base_pos[0]),
                         p_world[1] - float(base_pos[1]),
                         p_world[2] - float(base_pos[2]))
        
        # 如果拿不到旋转，就退化为仅平移
        if base_quat is None:
            return np.array([float(p_rel[0]), float(p_rel[1]), float(p_rel[2])], dtype=float)
        
        # world -> base: 乘以 base 的逆旋转
        q_inv = base_quat.GetInverse()
        
        # ✅ 关键：用四元数旋转向量（Gf 支持 Transform）
        p_base = q_inv.Transform(p_rel)
        
        return np.array([float(p_base[0]), float(p_base[1]), float(p_base[2])], dtype=float)
    
    num_joints = robot.num_dof
    
    for attempt in range(max_attempts):
        # 1. 随机采样关节配置
        joint_positions = sample_random_joint_config(num_joints)
        
        # 2. 设置关节位置（先重置速度，避免突然变化）
        robot.set_joint_velocities(np.zeros(num_joints))
        robot.set_joint_positions(joint_positions)
        
        # 3. 推进更多帧让物理稳定（减少 PhysX 警告）
        for _ in range(10):
            sim.step(render=False)
        
        # 4. 获取TCP的世界坐标（使用真实路径 /World/Panda/TCP）
        try:
            tcp_prim = XFormPrim("/World/Panda/TCP")
            tcp_world_pos, _ = tcp_prim.get_world_pose()
        except Exception as e:
            # 如果TCP获取失败，跳过这次尝试
            continue
        
        # 5. 转换到base坐标系
        tcp_base_pos = world_to_base(tcp_world_pos)
        
        # 6. 检查工作空间约束
        is_valid, reason = check_workspace_constraint(tcp_base_pos)
        
        if is_valid:
            print(f"   ✅ 找到有效配置 (尝试 {attempt+1} 次): TCP_base=({tcp_base_pos[0]:.3f}, {tcp_base_pos[1]:.3f}, {tcp_base_pos[2]:.3f})")
            return joint_positions, tcp_base_pos
        else:
            if attempt < 5 or attempt % 20 == 0:  # 只打印前几次和每20次
                print(f"   ⏳ 尝试 {attempt+1}/{max_attempts}: {reason}")
    
    print(f"   ❌ 在 {max_attempts} 次尝试后未找到有效配置")
    return None, None


class BCPolicy:
    """封装训练好的 BC 策略，输出 delta_q"""
    def __init__(self, ckpt_path, device, image_size):
        self.device = device
        self.image_size = image_size
        self.model = ResNetMLPPolicy(out_dim=7).to(device)
        state = torch.load(ckpt_path, map_location=device)
        if "model" in state:
            state = state["model"]
        self.model.load_state_dict(state)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, image_path):
        """
        从图像路径加载并预测动作
        注意：图像文件应该在调用此方法之前已经完整写入
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            img = Image.open(image_path).convert("RGB")
            # 验证图像完整性
            img.verify()  # 验证但不加载数据
            img = Image.open(image_path).convert("RGB")  # 重新打开以加载数据
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            delta_q = self.model(tensor).squeeze(0).cpu().numpy()
            return delta_q.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"加载或处理图像失败 {image_path}: {e}") from e


# ===================== 主流程 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="DAgger 数据采集（BC 执行，RMPFlow 打标签）")
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT,
                        help="输出数据集根目录")
    parser.add_argument("--bc_checkpoint", type=str,
                        default="/home/wopubuntu/me5400/training/checkpoints_bc/best.pt",
                        help="BC 模型 checkpoint 路径（含 state_dict 或包含 model 字段）")
    parser.add_argument("--episodes", type=int, default=50, help="采集多少个 episode")
    parser.add_argument("--steps", type=int, default=200, help="每个 episode 的步数")
    parser.add_argument("--mix_beta", type=float, default=0.6,
                        help="混合系数：command = (1-beta)*expert + beta*policy")
    parser.add_argument("--behavior", type=str, default="policy", choices=["mixture", "policy", "expert"],
                        help="执行动作来源：mixture/policy/expert")
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--save_fail", action="store_true", help="保存失败 episode（默认也保存）")
    return parser.parse_args()


def ensure_dirs(root):
    meta_dir = os.path.join(root, "metadata")
    pic_dir = os.path.join(root, "picture_data")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(pic_dir, exist_ok=True)
    return meta_dir, pic_dir


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    meta_dir, pic_dir = ensure_dirs(args.dataset_root)
    print(f"数据将保存到: {args.dataset_root}")

    # --- 环境加载 ---
    print(f"正在加载场景: {ENV_USD_PATH}")
    omni.usd.get_context().open_stage(ENV_USD_PATH)
    for _ in range(100):
        simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    has_physics = any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse())
    if not has_physics:
        print("⚠️ 创建默认 PhysicsScene...")
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    table_prim = stage.GetPrimAtPath(TABLE_PATH)
    if table_prim.IsValid():
        if not table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(table_prim)
        UsdPhysics.RigidBodyAPI(table_prim).CreateKinematicEnabledAttr(True)

    print("创建机器人对象...")
    robot = Articulation(ROBOT_PATH)
    sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    timeline.play()
    sim.initialize_physics()
    if not sim.is_playing():
        sim.play()
    for _ in range(60):
        sim.step(render=False)
    robot.initialize()

    print("加载 RMPFlow (专家)...")
    mg_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    cfg_dir = os.path.join(mg_path, "motion_policy_configs")
    rmp = RmpFlow(
        robot_description_path=os.path.join(cfg_dir, "franka/rmpflow/robot_descriptor.yaml"),
        urdf_path=os.path.join(cfg_dir, "franka/lula_franka_gen.urdf"),
        rmpflow_config_path=os.path.join(cfg_dir, "franka/rmpflow/franka_rmpflow_common.yaml"),
        end_effector_frame_name="right_gripper",
        maximum_substep_size=0.00334,
    )
    expert_policy = ArticulationMotionPolicy(robot, rmp)
    cam = ViewportCamera(CAM_PATH)

    bc_policy = BCPolicy(args.bc_checkpoint, device, (args.image_height, args.image_width))
    target_prim = XFormPrim(MARKER_PATH)
    default_marker_pos, default_marker_orn = target_prim.get_world_pose()

    for ep_idx in range(args.episodes):
        ep_dir = os.path.join(pic_dir, f"episode_{ep_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        print(f"\n🎬 Episode {ep_idx}/{args.episodes}")

        print(f"   🎲 使用拒绝采样寻找工作空间内的初始配置...")
        print(f"      工作空间: 中心={WORKSPACE_CENTER}, 半径={WORKSPACE_RADIUS}m, Z范围=[{WORKSPACE_Z_MIN}, {WORKSPACE_Z_MAX}]m")
        random_joint_positions, ee_pos_base = sample_valid_initial_config(robot, sim, max_attempts=100)
        if random_joint_positions is None:
            print("   ⚠️ 无法找到工作空间内初始配置，跳过该 episode")
            shutil.rmtree(ep_dir, ignore_errors=True)
            continue
        robot.set_joint_positions(random_joint_positions)
        robot.set_joint_velocities(np.zeros(robot.num_dof))
        for _ in range(30):
            sim.step(render=True)

        episode_metadata = []
        end_reason = "timeout"
        last_command_q = None

        for step in range(args.steps):
            if not simulation_app.is_running():
                break

            # 1) 采集观测（保存到正式路径，便于直接训练）
            img_filename = f"frame_{step:04d}.png"
            img_path = os.path.join(ep_dir, img_filename)
            
            # 图像捕获（参考 evaluate_bc.py 的实现，但使用更宽松的重试策略）
            max_capture_retries = 3  # 增加重试次数
            file_ready = False
            
            for capture_retry in range(max_capture_retries):
                # 强制渲染更新（确保 viewport 已渲染）
                for _ in range(3):  # 多次更新确保渲染完成
                    simulation_app.update()
                
                # 捕获图像
                if not cam.capture(img_path):
                    if capture_retry < max_capture_retries - 1:
                        # 捕获失败，等待后重试
                        for _ in range(3):
                            simulation_app.update()
                        time.sleep(0.1)  # 增加等待时间
                        continue
                    else:
                        print(f"   ⚠️ 第 {step} 步截图失败（已重试 {max_capture_retries} 次）")
                        break
                
                # 强制刷新（确保文件写入开始）
                for _ in range(3):
                    simulation_app.update()
                
                # 等待文件写入完成（简化逻辑：只要文件大小 > 最小阈值即可）
                min_bytes = 10_000  # 最小文件大小阈值（1280x720 PNG 一般远大于这个）
                max_wait_attempts = 30  # 增加等待尝试次数
                wait_attempt = 0
                
                while wait_attempt < max_wait_attempts:
                    if os.path.exists(img_path):
                        try:
                            file_size = os.path.getsize(img_path)
                            if file_size >= min_bytes:
                                file_ready = True
                                break
                        except OSError:
                            # 文件可能正在写入，继续等待
                            pass
                    # 每次检查时更新
                    simulation_app.update()
                    time.sleep(0.05)
                    wait_attempt += 1
                
                if not file_ready:
                    if capture_retry < max_capture_retries - 1:
                        # 文件未就绪，等待后重试
                        for _ in range(3):
                            simulation_app.update()
                        time.sleep(0.1)
                        continue
                    else:
                        print(f"   ⚠️ 第 {step} 步图像文件未就绪（已重试 {max_capture_retries} 次，等待 {max_wait_attempts} 次）")
                        break
                else:
                    break  # 文件就绪，退出重试循环
            
            if not file_ready:
                print(f"   ⚠️ 第 {step} 步图像捕获失败，跳过该步")
                # 继续执行下一步，但不记录数据
                target_pose_world = np.array(default_marker_pos) + np.array([0.52, -0.07, -0.65])
                rmp.set_end_effector_target(target_pose_world, default_marker_orn)
                robot.apply_action(expert_policy.get_next_articulation_action(DT))
                sim.step(render=True)
                continue

            # 2) RMPFlow 专家动作
            target_pose_world = np.array(default_marker_pos) + np.array([0.52, -0.07, -0.65])
            rmp.set_end_effector_target(target_pose_world, default_marker_orn)
            expert_action = expert_policy.get_next_articulation_action(DT)
            q_current = robot.get_joint_positions()
            command_q_expert = np.array(expert_action.joint_positions) if expert_action.joint_positions is not None else np.array(q_current)
            delta_q_expert = command_q_expert - q_current

            # 3) BC 策略动作
            policy_delta_q = bc_policy.predict(img_path)
            command_q_policy = q_current + policy_delta_q

            # 4) 混合执行
            if args.behavior == "policy":
                command_q = command_q_policy
            elif args.behavior == "expert":
                command_q = command_q_expert
            else:  # mixture
                command_q = (1 - args.mix_beta) * command_q_expert + args.mix_beta * command_q_policy
            executed_delta_q = command_q - q_current
            last_command_q = command_q.copy()

            # 覆写动作后应用
            expert_action.joint_positions = command_q.tolist()
            expert_action.joint_velocities = None
            robot.apply_action(expert_action)
            sim.step(render=True)

            # 5) 记录
            try:
                ee_prim = XFormPrim("/World/Panda/TCP")
                ee_actual_pos, _ = ee_prim.get_world_pose()
            except Exception:
                ee_actual_pos = None
            step_data = {
                "step": step,
                "image_path": img_filename,
                "state": {
                    "q": q_current,
                    "dq": robot.get_joint_velocities(),
                    "ee_target_pos": vec3_to_list(target_pose_world),
                    "ee_actual_pos": vec3_to_list(ee_actual_pos),
                    "marker_pos_world": vec3_to_list(default_marker_pos)
                },
                "action": {
                    "expert_delta_q": delta_q_expert,
                    "policy_delta_q": policy_delta_q,
                    "executed_delta_q": executed_delta_q,
                    "command_positions": command_q,
                }
            }
            episode_metadata.append(step_data)

        # 成功判定（基于最后一步）
        try:
            ee_prim = XFormPrim("/World/Panda/TCP")
            ee_final_pos, _ = ee_prim.get_world_pose()
            ee_final_pos = np.array([float(ee_final_pos[0]), float(ee_final_pos[1]), float(ee_final_pos[2])])
            marker_final_pos = np.array(default_marker_pos)
            diff = np.abs(ee_final_pos - marker_final_pos)
            success = bool((diff[0] < SUCCESS_DISTANCE[0]) and (diff[1] < SUCCESS_DISTANCE[1]) and (diff[2] < SUCCESS_DISTANCE[2]))
            if success:
                end_reason = "success"
            else:
                end_reason = "timeout"
        except Exception:
            success = False
            end_reason = "timeout"

        meta = {
            "episode_idx": ep_idx,
            "success": success,
            "end_reason": end_reason,
            "end_step": len(episode_metadata) - 1,
            "num_saved_frames": len(episode_metadata),
            "behavior_type": args.behavior,
            "steps": episode_metadata,
        }
        meta_path = os.path.join(meta_dir, f"episode_{ep_idx:04d}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, cls=NumpyEncoder)

        status = "✅" if success else "❌"
        print(f"{status} 保存 Episode {ep_idx} ({end_reason}, 帧数 {len(episode_metadata)})")

    print("\n🎉 采集完成")
    simulation_app.close()


if __name__ == "__main__":
    main()
