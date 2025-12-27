# ============================================================================
# 独立运行脚本：大规模 RMPFlow 数据集采集器 (修复版 V2)
# 功能：生成 Episodes，保存图片 + JSON (q, action, etc.)
# ============================================================================

from isaacsim import SimulationApp

# 1. 启动 Isaac Sim
simulation_app = SimulationApp({"headless": False})

import os
import time
import json
import shutil
import numpy as np
import omni.timeline
import omni.usd
from pxr import UsdPhysics, Gf, Usd

# 核心模块
from omni.isaac.core import SimulationContext
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

# ===================== ⚙️ 数据集配置 =====================
# ⚠️ 再次确认路径正确
ENV_USD_PATH = "/home/alphatok/ME5400/env.setup/env.usda"
MARKER_PATH = "/World/Phantom/marker"
ROBOT_PATH = "/World/Panda"
PHANTOM_PATH = "/World/Phantom"
TABLE_PATH = "/World/Table"
CAM_PATH = "/World/Panda/D405_rigid/D405/Camera_OmniVision_OV9782_Color"

# 输出根目录
DATASET_ROOT = "/home/alphatok/ME5400/DATA2"

# 采集参数
NUM_EPISODES = 100        # 总共采集多少集
STEPS_PER_EPISODE = 200   # 每一集跑多少步
DT = 1.0 / 60.0
CAPTURE_EVERY_N = 5       # 每5步保存一次
TARGET_OFFSET = [0.52, -0.07, -0.65]

# 随机化参数：Panda 关节限制（用于在工作空间内随机采样）
PANDA_JOINT_LIMITS = [
    (-2.8973, 2.8973),   # joint1
    (-1.7628, 1.7628),   # joint2
    (-2.8973, 2.8973),   # joint3
    (-3.0718, -0.0698),  # joint4
    (-2.8973, 2.8973),   # joint5
    (-0.0175, 3.7525),   # joint6
    (-2.8973, 2.8973),   # joint7
]

# 工作空间定义（相对于 Panda base 坐标系）
WORKSPACE_CENTER = np.array([0.0, 0.50, 0.50])  # 米
WORKSPACE_RADIUS = 0.25  # 米（25cm）
WORKSPACE_Z_MIN = 0.20  # 米
WORKSPACE_Z_MAX = 0.75  # 米

# ===================== 辅助类 =====================
class ViewportCamera:
    def __init__(self, camera_path, resolution=(1280, 720)):
        self.viewport_api = get_active_viewport()
        if not self.viewport_api:
            raise RuntimeError("❌ 无法找到活跃视口！")
        
        print(f"✅ 已绑定活跃视口")
        self.viewport_api.camera_path = camera_path
        # 传入元组，不要解包
        self.viewport_api.set_texture_resolution(resolution)

    def capture(self, filename):
        try:
            capture_viewport_to_file(self.viewport_api, filename)
            return True
        except Exception as e:
            print(f"❌ 截图异常: {e}")
            return False

class NumpyEncoder(json.JSONEncoder):
    """ 处理 NumPy 数组转 JSON 的辅助类 """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def vec3_to_list(v):
    """
    将 Vec3 类型（Gf.Vec3d, np.ndarray, list等）转换为 float list
    用于 JSON 序列化
    """
    if isinstance(v, (list, tuple)):
        return [float(v[0]), float(v[1]), float(v[2])]
    elif isinstance(v, np.ndarray):
        return [float(v[0]), float(v[1]), float(v[2])]
    else:
        # 处理 Gf.Vec3d 等类型
        try:
            return [float(v[0]), float(v[1]), float(v[2])]
        except (TypeError, IndexError):
            # 如果无法转换，尝试转换为 numpy 数组再转 list
            v_arr = np.array(v)
            return [float(v_arr[0]), float(v_arr[1]), float(v_arr[2])]

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
    # p_base = q_inv * (p_world - t) * q_inv_conj 这种思想
    # 使用 Gf.Quatd 的 Transform 方法
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
        for _ in range(10):  # 从10帧增加到20帧
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

# ===================== 主函数 =====================
def main():
    # --- 1. 环境加载 ---
    print(f"正在加载场景: {ENV_USD_PATH}")
    omni.usd.get_context().open_stage(ENV_USD_PATH)
    # 等待资源加载
    for _ in range(100): 
        simulation_app.update()

    # --- 2. 初始化核心对象 ---
    timeline = omni.timeline.get_timeline_interface()
    
    # 确保 PhysicsScene
    stage = omni.usd.get_context().get_stage()
    has_physics = False
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene): has_physics = True; break
    if not has_physics:
        print("⚠️ 创建默认 PhysicsScene...")
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    # 稳定桌子
    table_prim = stage.GetPrimAtPath(TABLE_PATH)
    if table_prim.IsValid():
        if not table_prim.HasAPI(UsdPhysics.RigidBodyAPI): UsdPhysics.RigidBodyAPI.Apply(table_prim)
        UsdPhysics.RigidBodyAPI(table_prim).CreateKinematicEnabledAttr(True)

    # 机器人对象创建 (先不要 initialize)
    print("创建机器人对象...")
    robot = Articulation(ROBOT_PATH)
    
    # --- 3. 启动仿真 & 物理初始化 (关键修复) ---
    print("初始化 SimulationContext...")
    sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    
    print("启动 Timeline...")
    timeline.play()
    
    # 强制让 Physics Engine 启动
    print("强制初始化物理引擎...")
    sim.initialize_physics() 
    
    # 再次确认处于播放状态
    if not sim.is_playing():
        sim.play()

    # 预热几帧
    print("正在预热物理引擎 (60帧)...")
    for _ in range(60):
        sim.step(render=False)

    # 现在可以安全初始化机器人了
    print("初始化机器人...")
    try:
        robot.initialize()
    except Exception as e:
        print(f"⚠️ 第一次初始化失败 ({e})，尝试重试...")
        for _ in range(10): sim.step(render=False)
        robot.initialize()
    
    # --- 3.5. 初始化碰撞检测参数 ---
    # 使用关节速度异常检测碰撞（更可靠的方法）
    # 碰撞时关节速度会突然变化，超过阈值则判定为碰撞
    COLLISION_VELOCITY_THRESHOLD = 10.0  # rad/s，正常运动时关节速度不会超过这个值
    COLLISION_ACCELERATION_THRESHOLD = 50.0  # rad/s²，加速度阈值
    print("✅ 碰撞检测已启用（基于关节速度和加速度异常）")
    
    # --- 4. RMPFlow & 策略 ---
    print("加载 RMPFlow...")
    mg_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    cfg_dir = os.path.join(mg_path, "motion_policy_configs")
    rmp = RmpFlow(
        robot_description_path=os.path.join(cfg_dir, "franka/rmpflow/robot_descriptor.yaml"),
        urdf_path=os.path.join(cfg_dir, "franka/lula_franka_gen.urdf"),
        rmpflow_config_path=os.path.join(cfg_dir, "franka/rmpflow/franka_rmpflow_common.yaml"),
        end_effector_frame_name="right_gripper",
        maximum_substep_size=0.00334,
    )
    policy = ArticulationMotionPolicy(robot, rmp)
    target_prim = XFormPrim(MARKER_PATH)
    
    # 相机
    cam = ViewportCamera(CAM_PATH)
    
    # 保存基础位置用于随机化（⚠️ 只读，不再移动 marker 本身）
    # marker 永远固定在 Phantom 上，我们只对"RMPFlow 的目标点"加噪声
    default_marker_pos, default_marker_orn = target_prim.get_world_pose()

    # --- 5. 开始 Episode 循环 ---
    print(f"🚀 开始采集任务: 目标 {NUM_EPISODES} 集，每集 {STEPS_PER_EPISODE} 步")
    
    # 创建成功和失败的目录结构
    success_metadata_dir = os.path.join(DATASET_ROOT, "success", "metadata")
    success_picture_dir = os.path.join(DATASET_ROOT, "success", "picture_data")
    fail_metadata_dir = os.path.join(DATASET_ROOT, "fail", "metadata")
    fail_picture_dir = os.path.join(DATASET_ROOT, "fail", "picture_data")
    
    os.makedirs(success_metadata_dir, exist_ok=True)
    os.makedirs(success_picture_dir, exist_ok=True)
    os.makedirs(fail_metadata_dir, exist_ok=True)
    os.makedirs(fail_picture_dir, exist_ok=True)
    
    # 临时目录：在episode运行期间先保存到这里，结束后根据成功/失败移动到对应目录
    temp_metadata_dir = os.path.join(DATASET_ROOT, "temp_metadata")
    temp_picture_dir = os.path.join(DATASET_ROOT, "temp_picture_data")
    os.makedirs(temp_metadata_dir, exist_ok=True)
    os.makedirs(temp_picture_dir, exist_ok=True)
    
    for episode_idx in range(NUM_EPISODES):
        # ----------------------------------------
        # (A) Episode 初始化与随机化
        # ----------------------------------------
        # 先保存到临时目录，episode结束后根据成功/失败移动到对应目录
        ep_dir = os.path.join(temp_picture_dir, f"episode_{episode_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        
        # 1. 使用拒绝采样找到工作空间内的有效初始配置
        print(f"   🎲 使用拒绝采样寻找工作空间内的初始配置...")
        print(f"      工作空间: 中心={WORKSPACE_CENTER}, 半径={WORKSPACE_RADIUS}m, Z范围=[{WORKSPACE_Z_MIN}, {WORKSPACE_Z_MAX}]m")
        
        random_joint_positions, ee_pos_base = sample_valid_initial_config(robot, sim, max_attempts=100)
        
        if random_joint_positions is None:
            print(f"   ⚠️ 无法找到有效配置，使用随机配置（可能不在工作空间内）")
            random_joint_positions = sample_random_joint_config(robot.num_dof)
            robot.set_joint_positions(random_joint_positions)
        else:
            # 配置已设置，只需确保位置正确
            robot.set_joint_positions(random_joint_positions)
        
        print(f"   📍 初始关节配置: {[f'{q:.3f}' for q in random_joint_positions]}")
        
        # 2. 物理稳态预热
        robot.set_joint_velocities(np.zeros(robot.num_dof))
        for _ in range(30):
            sim.step(render=True)
        

        current_marker_pos = default_marker_pos  # marker 真实位置维持不变
        
        episode_metadata = []
        prev_dq = None  # 上一时刻的关节速度，用于计算加速度
        episode_success = False  # 初始假设失败，在最后一步检查成功条件
        
        # 成功条件：在最后一步检查末端执行器是否接近marker
        SUCCESS_DISTANCE_X_MAX = 0.1   # 米，X方向最大距离
        SUCCESS_DISTANCE_Y_MAX = 0.1   # 米，Y方向最大距离
        SUCCESS_DISTANCE_Z_MAX = 0.3   # 米，Z方向最大距离
        
        # 初始化episode结束信息（默认超时）
        end_reason = "timeout"
        end_step = STEPS_PER_EPISODE - 1
        
        print(f"🎬 Episode {episode_idx}/{NUM_EPISODES} 开始...")
        print(f"   将运行固定 {STEPS_PER_EPISODE} 步，最后一步检查成功条件")
        print(f"   成功条件: diff_x < {SUCCESS_DISTANCE_X_MAX}m, diff_y < {SUCCESS_DISTANCE_Y_MAX}m, diff_z < {SUCCESS_DISTANCE_Z_MAX}m")

        # ----------------------------------------
        # (B) Step 循环
        # ----------------------------------------
        for step in range(STEPS_PER_EPISODE):
            if not simulation_app.is_running(): break

            # --- 1. 计算动作（每一步都需要，用于控制）---
            # 目标 = marker 当前位置 + 固定 offset（完全不加随机扰动）
            target_pose_world = current_marker_pos + np.array(TARGET_OFFSET)
            rmp.set_end_effector_target(target_pose_world, default_marker_orn)
            
            action = policy.get_next_articulation_action(DT)

            # 取出当前这一步的"关节命令位置"作为 q_cmd(t)
            if action.joint_positions is not None:
                command_q = np.array(action.joint_positions)
            else:
                # 退化方案：如果没有给出绝对关节位置，就把当前 q 当作命令
                q_current = robot.get_joint_positions()
                command_q = np.array(q_current)

            # --- 2. 只在每5步时：计算delta_q、截图、记录数据 ---
            if step % CAPTURE_EVERY_N == 0:
                # 获取当前状态（用于记录）
                q_current = robot.get_joint_positions()
                dq_current = robot.get_joint_velocities()
                
                # 计算 delta_q = 当前命令 - 当前状态
                # 这是经典的 BC 监督信号：image(t) -> delta_q(t)
                # delta_q 表示"在当前图像下，expert 希望关节朝命令方向移动多少"
                delta_q = command_q - q_current
                
                # 截图：捕获当前状态的图片
                img_filename = f"frame_{step:04d}.png"
                img_path = os.path.join(ep_dir, img_filename)
                cam.capture(img_path)
                
                # 获取实际末端执行器位置（TCP 的物理位置）
                try:
                    ee_prim = XFormPrim("/World/Panda/TCP")
                    ee_actual_pos, ee_actual_orn = ee_prim.get_world_pose()
                except:
                    ee_actual_pos = None
                    ee_actual_orn = None
                
                # 记录数据（使用 vec3_to_list 确保 JSON 序列化安全）
                step_data = {
                    "step": step,
                    "image_path": img_filename,
                    "state": {
                        "q": q_current,
                        "dq": dq_current,
                        "ee_target_pos": vec3_to_list(target_pose_world),
                        "ee_actual_pos": vec3_to_list(ee_actual_pos) if ee_actual_pos is not None else None,
                        "marker_pos_world": vec3_to_list(current_marker_pos)
                    },
                    "action": {
                        "command_positions": command_q,
                        "command_velocities": action.joint_velocities,
                        # delta_q: 当前命令 - 当前状态，表示"在当前图像下应该移动多少"
                        "delta_q": delta_q
                    }
                }
                episode_metadata.append(step_data)

            # --- 3. 应用动作并推进仿真（每一步都需要）---
            robot.apply_action(action)
            sim.step(render=True)
            
            # --- 4. 检查碰撞（检测到碰撞立即终止episode）---
            # 获取step后的关节速度（碰撞会导致速度突然变化）
            dq_after_step = robot.get_joint_velocities()
            
            has_collision = False
            collision_reason = ""
            
            # 方法1: 检查速度是否超过阈值（碰撞时速度会突然增大）
            max_velocity = np.max(np.abs(dq_after_step))
            if max_velocity > COLLISION_VELOCITY_THRESHOLD:
                has_collision = True
                collision_reason = f"速度异常: {max_velocity:.2f} rad/s > {COLLISION_VELOCITY_THRESHOLD} rad/s"
            
            # 方法2: 检查加速度是否超过阈值（碰撞时加速度会突然增大）
            if prev_dq is not None and not has_collision:
                acceleration = (dq_after_step - prev_dq) / DT
                max_acceleration = np.max(np.abs(acceleration))
                if max_acceleration > COLLISION_ACCELERATION_THRESHOLD:
                    has_collision = True
                    collision_reason = f"加速度异常: {max_acceleration:.2f} rad/s² > {COLLISION_ACCELERATION_THRESHOLD} rad/s²"
            
            if has_collision:
                episode_success = False
                end_reason = "collision"
                end_step = step
                print(f"   ⚠️ Episode {episode_idx} 在第 {step} 步发生碰撞 ({collision_reason})，立即结束该episode")
                
                # 如果终止步不是记录帧，强制保存最后一帧
                if step % CAPTURE_EVERY_N != 0:
                    q_current = robot.get_joint_positions()
                    dq_current = robot.get_joint_velocities()
                    delta_q = command_q - q_current
                    
                    img_filename = f"frame_{step:04d}.png"
                    img_path = os.path.join(ep_dir, img_filename)
                    cam.capture(img_path)
                    
                    try:
                        ee_prim = XFormPrim("/World/Panda/TCP")
                        ee_actual_pos, ee_actual_orn = ee_prim.get_world_pose()
                    except:
                        ee_actual_pos = None
                        ee_actual_orn = None
                    
                    step_data = {
                        "step": step,
                        "image_path": img_filename,
                        "state": {
                            "q": q_current,
                            "dq": dq_current,
                            "ee_target_pos": vec3_to_list(target_pose_world),
                            "ee_actual_pos": vec3_to_list(ee_actual_pos) if ee_actual_pos is not None else None,
                            "marker_pos_world": vec3_to_list(current_marker_pos)
                        },
                        "action": {
                            "command_positions": command_q,
                            "command_velocities": action.joint_velocities,
                            "delta_q": delta_q
                        }
                    }
                    episode_metadata.append(step_data)
                    print(f"   💾 已强制保存终止帧 {step}")
                
                break  # 立即结束当前episode
            
            # 保存当前速度供下一step使用
            prev_dq = dq_after_step.copy()

        # ----------------------------------------
        # (C) 检查成功条件（在最后一步检查）---
        # ----------------------------------------
        # 如果episode没有因为碰撞提前终止，检查最后一步是否满足成功条件
        if end_reason != "collision":
            try:
                # 获取最终末端执行器位置
                ee_prim = XFormPrim("/World/Panda/TCP")
                ee_final_pos, _ = ee_prim.get_world_pose()
                ee_final_pos = np.array([float(ee_final_pos[0]), float(ee_final_pos[1]), float(ee_final_pos[2])])
                
                # 获取marker位置
                marker_final_pos = np.array(current_marker_pos)
                
                # 计算x、y、z方向的绝对差值
                diff_x = abs(ee_final_pos[0] - marker_final_pos[0])
                diff_y = abs(ee_final_pos[1] - marker_final_pos[1])
                diff_z = abs(ee_final_pos[2] - marker_final_pos[2])
                
                # 检查是否满足距离阈值（简单上限：diff_x < 0.1m, diff_y < 0.1m, diff_z < 0.3m）
                if (diff_x < SUCCESS_DISTANCE_X_MAX) and \
                   (diff_y < SUCCESS_DISTANCE_Y_MAX) and \
                   (diff_z < SUCCESS_DISTANCE_Z_MAX):
                    episode_success = True
                    end_reason = "success"
                    print(f"   ✅ Episode {episode_idx} 在最后一步满足成功条件 "
                          f"(X={diff_x:.3f}m, Y={diff_y:.3f}m, Z={diff_z:.3f}m)")
                else:
                    episode_success = False
                    end_reason = "timeout"
                    print(f"   ⚠️ Episode {episode_idx} 在最后一步不满足成功条件 "
                          f"(X={diff_x:.3f}m, Y={diff_y:.3f}m, Z={diff_z:.3f}m)")
            except Exception as e:
                print(f"   ⚠️ Episode {episode_idx} 无法获取末端执行器位置进行成功检查: {e}")
                episode_success = False
                end_reason = "timeout"
        
        # ----------------------------------------
        # (D) 保存 Episode 元数据并移动到对应目录
        # ----------------------------------------
        # 添加episode级别的元数据
        episode_info = {
            "episode_idx": episode_idx,
            "success": episode_success,
            "end_reason": end_reason,  # "success", "collision", "timeout"
            "end_step": end_step,  # 实际结束的仿真步数（0-based）
            "num_saved_frames": len(episode_metadata),  # 实际保存的截图帧数
            "steps": episode_metadata
        }
        
        # 先保存metadata到临时目录
        json_filename = f"episode_{episode_idx:04d}.json"
        temp_json_path = os.path.join(temp_metadata_dir, json_filename)
        with open(temp_json_path, "w") as f:
            json.dump(episode_info, f, indent=2, cls=NumpyEncoder)
        
        # 根据成功/失败，将数据移动到对应目录
        if episode_success:
            # 移动到success目录
            final_metadata_dir = success_metadata_dir
            final_picture_dir = success_picture_dir
            final_ep_dir = os.path.join(final_picture_dir, f"episode_{episode_idx:04d}")
            final_json_path = os.path.join(final_metadata_dir, json_filename)
        else:
            # 移动到fail目录
            final_metadata_dir = fail_metadata_dir
            final_picture_dir = fail_picture_dir
            final_ep_dir = os.path.join(final_picture_dir, f"episode_{episode_idx:04d}")
            final_json_path = os.path.join(final_metadata_dir, json_filename)
        
        # 移动metadata文件
        shutil.move(temp_json_path, final_json_path)
        
        # 移动picture_data目录
        if os.path.exists(ep_dir):
            if os.path.exists(final_ep_dir):
                # 如果目标目录已存在，先删除
                shutil.rmtree(final_ep_dir)
            shutil.move(ep_dir, final_ep_dir)
        
        # 如果episode成功，记录到seed.txt
        if episode_success:
            seed_file_path = os.path.join(DATASET_ROOT, "seed.txt")
            with open(seed_file_path, "a") as f:
                f.write(f"{episode_idx}\n")
        
        status_emoji = "✅" if episode_success else "❌"
        if episode_success:
            status_text = "成功（最后一步满足距离条件）"
        else:
            # 根据end_reason生成状态文本
            if end_reason == "collision":
                status_text = "失败（碰撞）"
            elif end_reason == "timeout":
                status_text = "失败（最后一步不满足距离条件）"
            else:
                status_text = f"失败（{end_reason}）"
        print(f"{status_emoji} Saved Episode {episode_idx} ({status_text}, 运行 {STEPS_PER_EPISODE} 步, 保存 {len(episode_metadata)} 帧) to {final_ep_dir}")

    print("🎉 所有任务完成！")
    simulation_app.close()

if __name__ == "__main__":
    main()