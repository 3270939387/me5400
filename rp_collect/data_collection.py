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
ENV_USD_PATH = "/home/wopubuntu/me5400/env.setup/env.usda"
MARKER_PATH = "/World/Phantom/marker"
ROBOT_PATH = "/World/Panda"
PHANTOM_PATH = "/World/Phantom"
TABLE_PATH = "/World/Table"
CAM_PATH = "/World/Panda/D405_rigid/D405/Camera_OmniVision_OV9782_Color"

# 输出根目录
DATASET_ROOT = "/home/wopubuntu/me5400/rp_collect/DATA"

# 采集参数
NUM_EPISODES = 500        # 总共采集多少集（从100增加到200）
STEPS_PER_EPISODE = 200   # 每一集跑多少步
DT = 1.0 / 60.0
CAPTURE_EVERY_N_DEFAULT = 5  # 默认每3步保存一次
CAPTURE_EVERY_N_CLOSE = 3    # 距离小于0.3m时每2步保存一次
threshold_distance_close = 0.3  # 米
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
WORKSPACE_RADIUS = 0.30  # 米（30cm，从25cm扩大）
WORKSPACE_Z_MIN = 0.15  # 米（从0.20降低）
WORKSPACE_Z_MAX = 0.80  # 米（从0.75增加）

# ===================== 预定义的中立姿态（Neutral Poses）=====================
# 这些姿态是从人工演示中精选的，覆盖工作空间的各个角度
# 在每个episode中，我们会随机选择一个neutral pose，然后添加小扰动
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

# 扰动参数
PERTURBATION_SCALE = 0.15  # 扰动幅度（弧度），约8.6度

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

def sample_random_joint_config(num_joints, base_config=None, variation_scale=1.0):
    """
    在关节限位内随机采样关节配置
    
    Args:
        num_joints: 关节数量
        base_config: 基础配置（如果提供，会在其附近采样，用于增加姿态多样性）
        variation_scale: 变化幅度（如果提供base_config，控制变化范围）
    """
    if base_config is not None and len(base_config) == num_joints:
        # 在基础配置附近采样（增加姿态多样性）
        random_joint_positions = []
        for i in range(num_joints):
            if i < len(PANDA_JOINT_LIMITS):
                lower, upper = PANDA_JOINT_LIMITS[i]
                # 在基础配置附近采样，但限制在关节限位内
                center = base_config[i]
                variation = (upper - lower) * 0.3 * variation_scale  # 30%的范围
                new_value = center + np.random.uniform(-variation, variation)
                random_joint_positions.append(np.clip(new_value, lower, upper))
            else:
                random_joint_positions.append(np.random.uniform(-np.pi, np.pi))
        return np.array(random_joint_positions, dtype=np.float32)
    else:
        # 完全随机采样
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

def compute_marker_geometry(marker_world_pos, camera_world_pos, camera_quat, camera_intrinsics, img_resolution):
    """
    计算marker在相机图像坐标系中的几何信息 (u, v, s)
    
    Args:
        marker_world_pos: marker的世界坐标 [x, y, z]
        camera_world_pos: 相机的世界坐标 [x, y, z]
        camera_quat: 相机的世界四元数 (w, x, y, z)
        camera_intrinsics: 相机内参 {'fx': float, 'fy': float, 'cx': float, 'cy': float}
        img_resolution: 图像分辨率 (width, height)
    
    Returns:
        dict: {
            "visible": bool,  # marker是否在相机视野内
            "u": float,       # marker中心的像素X坐标（归一化到[0, W)）
            "v": float,       # marker中心的像素Y坐标（归一化到[0, H)）
            "s": float,       # marker的尺度（1/depth，用于尺度信息）
            "Zc": float,      # marker在相机坐标系下的深度
            "marker_cam": [float, float, float]  # marker在相机坐标系下的坐标
        }
    """
    try:
        marker_world_pos = np.array([float(marker_world_pos[0]), float(marker_world_pos[1]), float(marker_world_pos[2])], dtype=np.float32)
        camera_world_pos = np.array([float(camera_world_pos[0]), float(camera_world_pos[1]), float(camera_world_pos[2])], dtype=np.float32)
        
        # 构建相机的旋转矩阵（从world到camera）
        # 相机四元数格式：(w, x, y, z) - 表示相机相对于世界坐标系的方向
        w, qx, qy, qz = float(camera_quat[0]), float(camera_quat[1]), float(camera_quat[2]), float(camera_quat[3])
        
        # 四元数转旋转矩阵 (标准格式 q = w + xi + yj + zk)
        # R_cam_to_world 表示从相机坐标系到世界坐标系的旋转
        # 我们需要 R_world_to_cam = R_cam_to_world.T
        R_cam_to_world = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*w), 2*(qx*qz + qy*w)],
            [2*(qx*qy + qz*w), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*w)],
            [2*(qx*qz - qy*w), 2*(qy*qz + qx*w), 1 - 2*(qx**2 + qy**2)]
        ], dtype=np.float32)
        
        # 反向旋转：从世界坐标系到相机坐标系
        R_world_to_cam = R_cam_to_world.T
        
        # marker在相机坐标系下的坐标
        # p_cam = R_world_to_cam * (p_world - t_camera)
        p_relative = marker_world_pos - camera_world_pos
        p_cam = R_world_to_cam @ p_relative
        
        Xc, Yc, Zc = p_cam[0], p_cam[1], p_cam[2]
        
        # 注意：Isaac Sim 使用 OpenGL 风格的相机坐标系
        # 标准针孔相机坐标系 vs Isaac Sim OpenGL坐标系的对应关系：
        # 
        # 标准针孔相机（计算机视觉）：
        #   +X → 右，+Y → 下，+Z → 前（物体方向）
        #
        # Isaac Sim (OpenGL):
        #   +X → 右，+Y → 上（反向），-Z → 前（物体方向，反向）
        #
        # 因此需要进行坐标系转换：
        Yc = -Yc   # Y轴反向：Isaac Sim的+Y对应标准的-Y
        Zc = -Zc   # Z轴反向：Isaac Sim的-Z对应标准的+Z
        
        # 如果marker在相机后面，标记为不可见
        # 现在 Zc > 0 表示物体在相机前方（可见）
        if Zc <= 0.01:  # 0.01m的最小深度阈值
            return {
                "visible": False,
                "u": -1.0,
                "v": -1.0,
                "s": 0.0,
                "u_raw": -1.0,
                "v_raw": -1.0,
                "Zc": float(Zc),
                "marker_cam": [float(Xc), float(Yc), float(Zc)]
            }
        
        # 针孔相机模型投影
        # u = cx + fx * X / Z
        # v = cy + fy * Y / Z
        fx = camera_intrinsics.get('fx', 640.0)
        fy = camera_intrinsics.get('fy', 640.0)
        cx = camera_intrinsics.get('cx', 640.0)
        cy = camera_intrinsics.get('cy', 360.0)
        
        u = cx + fx * (Xc / Zc)
        v = cy + fy * (Yc / Zc)
        
        img_width, img_height = img_resolution
        
        # ========== 处理超出范围的投影 ==========
        # 当marker距离相机很近时，投影坐标可能会非常大（u, v > 10000）
        # 这会导致网络输入不稳定。需要进行截断和归一化处理。
        
        # 1. 首先检查投影是否在合理范围内
        #    合理范围：[-2倍图像宽度, 3倍图像宽度]（允许marker部分超出图像范围）
        reasonable_u_min = -2.0 * img_width
        reasonable_u_max = 3.0 * img_width
        reasonable_v_min = -2.0 * img_height
        reasonable_v_max = 3.0 * img_height
        
        # 如果投影完全超出合理范围，标记为不可见
        if u < reasonable_u_min or u > reasonable_u_max or v < reasonable_v_min or v > reasonable_v_max:
            return {
                "visible": False,
                "u": -1.0,
                "v": -1.0,
                "s": 0.0,
                "u_raw": float(u),
                "v_raw": float(v),
                "Zc": float(Zc),
                "marker_cam": [float(Xc), float(Yc), float(Zc)]
            }
        
        # 2. 归一化 u, v 到 [0, 1]（基于原始图像分辨率）
        #    即使投影超出范围，也会映射到合理的范围
        u_norm = u / img_width
        v_norm = v / img_height
        
        # 3. 截断到 [-1.5, 2.5] 范围
        #    这允许marker在图像边界附近也有有效的坐标表示
        #    -1.5 ~ 2.5 映射到归一化空间表示：
        #      -1.5: marker在图像左边1.5倍宽度处
        #       0.0: marker在图像左边界处
        #       0.5: marker在图像中心
        #       1.0: marker在图像右边界处
        #       2.5: marker在图像右边2.5倍宽度处
        u_norm_clipped = np.clip(u_norm, -1.5, 2.5)
        v_norm_clipped = np.clip(v_norm, -1.5, 2.5)
        
        # 4. 处理深度和尺度信息
        #    当Z很小时，s = 1/Z 会变得非常大，导致数值不稳定
        #    方案：使用深度截断 + 归一化
        
        # 深度截断：确保最小深度为 0.05m（5cm）
        Z_min = 0.05  # 最小深度5cm
        Z_clipped = max(Zc, Z_min)
        
        # s = 1/Z，但要截断
        # 通常深度范围是 0.05m ~ 2.0m，所以 s 范围大约是 0.5 ~ 20
        s = 1.0 / Z_clipped
        s_max = 20.0  # 深度倒数的上限
        s_norm = np.clip(s, 0.0, s_max) / s_max  # 归一化到 [0, 1]
        
        # 检查投影是否在图像范围内（定义可见性）
        is_in_image = (u >= 0 and u < img_width and v >= 0 and v < img_height)
        
        return {
            "visible": bool(is_in_image),
            "u": float(u_norm_clipped),          # 归一化后的u
            "v": float(v_norm_clipped),          # 归一化后的v
            "s": float(s_norm),                  # 归一化后的尺度
            "u_raw": float(u),                   # 原始像素坐标u
            "v_raw": float(v),                   # 原始像素坐标v
            "Zc": float(Zc),
            "marker_cam": [float(Xc), float(Yc), float(Zc)]
        }
    
    except Exception as e:
        print(f"[WARN] 计算marker几何信息失败: {e}")
        return {
            "visible": False,
            "u": -1.0,
            "v": -1.0,
            "s": 0.0,
            "u_raw": -1.0,
            "v_raw": -1.0,
            "Zc": -1.0,
            "marker_cam": [-1.0, -1.0, -1.0]
        }

def sample_valid_initial_config(robot, sim, max_attempts=100):
    """
    ✅ 改进版本：使用 Neutral Poses + 小扰动
    
    替代原来的拒绝采样方法。优点：
    - 避免奇怪的、不自然的姿态
    - 确保数据在合理的工作空间范围内
    - 更快收敛（不需要反复拒绝采样）
    
    流程：
    1. 随机选择一个 neutral pose
    2. 添加小扰动（每个关节 ±0.15rad，约 ±8.6度）
    3. 在关节限位内截断扰动后的配置
    4. 验证末端执行器是否在工作空间内
    5. 如果不在，重试（最多100次）
    
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
        
        # 5. 获取末端执行器的世界坐标
        try:
            tcp_prim = XFormPrim("/World/Panda/TCP")
            tcp_world_pos, _ = tcp_prim.get_world_pose()
            
            # 转换到 base 坐标系（简化：仅考虑平移）
            # 因为 base 通常在原点，所以 tcp_base ≈ tcp_world
            tcp_base_pos = np.array([float(tcp_world_pos[0]), float(tcp_world_pos[1]), float(tcp_world_pos[2])])
        except Exception as e:
            # 如果获取失败，跳过这次尝试
            if attempt < 5:
                print(f"   ⏳ 尝试 {attempt+1}/{max_attempts}: 无法获取TCP位置，重试...")
            continue
        
        # 6. 检查工作空间约束
        is_valid, reason = check_workspace_constraint(tcp_base_pos)
        
        if is_valid:
            if attempt > 0:
                print(f"   ✅ 找到有效配置 (尝试 {attempt+1} 次): TCP_base=({tcp_base_pos[0]:.3f}, {tcp_base_pos[1]:.3f}, {tcp_base_pos[2]:.3f})")
            return joint_positions, tcp_base_pos
        else:
            if attempt < 5 or attempt % 20 == 0:
                print(f"   ⏳ 尝试 {attempt+1}/{max_attempts}: {reason}")
    
    print(f"   ❌ 在 {max_attempts} 次尝试后未找到有效配置（所有neutral pose都不在工作空间内）")
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
    
    # ========== 设置相机内参（针孔相机模型）==========
    # D405 RealSense相机的物理参数（从USD配置中提取）
    # 根据物理参数计算焦距（像素单位）
    img_resolution = (1280, 720)  # 图像分辨率 (width, height)
    img_width, img_height = img_resolution
    
    # 相机物理参数（来自USD配置）
    focal_length_mm = 1.9299999475479126  # 焦距（毫米）
    horiz_aperture_mm = 3.8959999084472656  # 水平光圈（毫米）
    vert_aperture_mm = 2.453000068664551  # 垂直光圈（毫米）
    
    # 计算焦距（像素单位）
    # fx = focal_length * (img_width / horiz_aperture)
    # fy = focal_length * (img_height / vert_aperture)
    fx = focal_length_mm * (img_width / horiz_aperture_mm)
    fy = focal_length_mm * (img_height / vert_aperture_mm)
    
    # 主点通常位于图像中心
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    camera_intrinsics = {
        'fx': fx,   # 焦距X（像素单位）
        'fy': fy,   # 焦距Y（像素单位）
        'cx': cx,   # 主点X（像素坐标）
        'cy': cy    # 主点Y（像素坐标）
    }
    print(f"✅ 相机内参已计算（D405实际物理参数）:")
    print(f"   分辨率: {img_resolution}")
    print(f"   焦距: {focal_length_mm} mm, 光圈: {horiz_aperture_mm}×{vert_aperture_mm} mm")
    print(f"   计算内参: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    
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
        ep_dir = os.path.join(temp_picture_dir, f"episode_{episode_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        # 1. 获取marker位置
        try:
            marker_prim = XFormPrim(MARKER_PATH)
            marker_world_pos, _ = marker_prim.get_world_pose()
            marker_world_pos = np.array([float(marker_world_pos[0]), float(marker_world_pos[1]), float(marker_world_pos[2])])
        except Exception as e:
            print(f"   ⚠️ 无法获取marker位置: {e}，使用默认位置")
            marker_world_pos = None

        # ✅ 改进：使用 Neutral Poses + 小扰动，而不是完全随机拒绝采样
        print(f"   🎲 从 {len(NEUTRAL_POSES)} 个预定义姿态中随机选择，添加小扰动({PERTURBATION_SCALE:.2f}rad)...")
        print(f"      工作空间: 中心={WORKSPACE_CENTER}, 半径={WORKSPACE_RADIUS}m, Z范围=[{WORKSPACE_Z_MIN}, {WORKSPACE_Z_MAX}]m")

        # 尝试找到有效配置（最多100次尝试）
        random_joint_positions, ee_pos_base = sample_valid_initial_config(robot, sim, max_attempts=100)
        
        if random_joint_positions is None:
            # 如果找不到有效配置，使用第一个 neutral pose（保证至少有一个合理的起始点）
            print(f"   ⚠️ 无法找到在工作空间内的配置，使用第一个 neutral pose 作为后备")
            random_joint_positions = np.array(NEUTRAL_POSES[0], dtype=np.float32)
            robot.set_joint_positions(random_joint_positions)
            config_type = "neutral_pose_default"
        else:
            config_type = "neutral_pose_perturbed"
            robot.set_joint_positions(random_joint_positions)

        print(f"   📍 初始关节配置类型: {config_type}")
        print(f"   📍 初始关节配置: {[f'{q:.3f}' for q in random_joint_positions]}")

        # 2. 物理稳态预热
        robot.set_joint_velocities(np.zeros(robot.num_dof))
        for _ in range(30):
            sim.step(render=True)

        current_marker_pos = default_marker_pos  # marker 真实位置维持不变

        episode_metadata = []
        prev_dq = None
        episode_success = False
        last_command_q = None

        # 成功条件
        SUCCESS_DISTANCE_X_MAX = 0.1
        SUCCESS_DISTANCE_Y_MAX = 0.1
        SUCCESS_DISTANCE_Z_MAX = 0.3

        end_reason = "timeout"
        end_step = STEPS_PER_EPISODE - 1

        print(f"🎬 Episode {episode_idx}/{NUM_EPISODES} 开始...")
        print(f"   将运行固定 {STEPS_PER_EPISODE} 步，最后一步检查成功条件")
        print(f"   成功条件: diff_x < {SUCCESS_DISTANCE_X_MAX}m, diff_y < {SUCCESS_DISTANCE_Y_MAX}m, diff_z < {SUCCESS_DISTANCE_Z_MAX}m")

        # ----------------------------------------
        # (B) Step 循环
        # ----------------------------------------
        capture_every_n = CAPTURE_EVERY_N_DEFAULT
        for step in range(STEPS_PER_EPISODE):
            if not simulation_app.is_running(): break

            # --- 1. 计算动作（每一步都需要，用于控制）---
            target_pose_world = current_marker_pos + np.array(TARGET_OFFSET)
            rmp.set_end_effector_target(target_pose_world, default_marker_orn)

            action = policy.get_next_articulation_action(DT)

            if action.joint_positions is not None:
                command_q = np.array(action.joint_positions)
            else:
                q_current = robot.get_joint_positions()
                command_q = np.array(q_current)

            last_command_q = command_q.copy()

            # --- 动态调整采集频率 ---
            try:
                ee_prim = XFormPrim("/World/Panda/TCP")
                ee_actual_pos, _ = ee_prim.get_world_pose()
                marker_prim = XFormPrim(MARKER_PATH)
                marker_pos, _ = marker_prim.get_world_pose()
                ee_actual_pos = np.array([float(ee_actual_pos[0]), float(ee_actual_pos[1]), float(ee_actual_pos[2])])
                marker_pos = np.array([float(marker_pos[0]), float(marker_pos[1]), float(marker_pos[2])])
                tcp_marker_dist = np.linalg.norm(ee_actual_pos - marker_pos)
                if tcp_marker_dist < threshold_distance_close:
                    capture_every_n = CAPTURE_EVERY_N_CLOSE
                else:
                    capture_every_n = CAPTURE_EVERY_N_DEFAULT
            except Exception as e:
                # 如果获取失败，保持默认
                capture_every_n = CAPTURE_EVERY_N_DEFAULT

            # --- 2. 只在每capture_every_n步时采集 ---
            if step % capture_every_n == 0:
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
                
                # ========== 计算marker的几何信息 (u, v, s) ==========
                # 获取相机的世界位姿
                try:
                    cam_prim = XFormPrim(CAM_PATH)
                    cam_world_pos, cam_world_orn = cam_prim.get_world_pose()
                    cam_world_pos = np.array([float(cam_world_pos[0]), float(cam_world_pos[1]), float(cam_world_pos[2])])
                    cam_world_orn = [float(cam_world_orn[0]), float(cam_world_orn[1]), float(cam_world_orn[2]), float(cam_world_orn[3])]
                    marker_geometry = compute_marker_geometry(current_marker_pos, cam_world_pos, cam_world_orn, camera_intrinsics, img_resolution)
                except Exception as e:
                    print(f"   [WARN] 无法计算marker几何信息: {e}")
                    marker_geometry = {
                        "visible": False,
                        "u": -1.0,
                        "v": -1.0,
                        "s": 0.0,
                        "Zc": -1.0,
                        "marker_cam": [-1.0, -1.0, -1.0]
                    }
                
                step_data = {
                    "image_path": img_filename,        # str: 图像文件名 (e.g., "frame_0000.png")
                    "q": q_current.tolist(),           # list: 关节位置 [7] - proprioception信息
                    "delta_q": delta_q.tolist(),       # list: 关节增量命令 [7] - BC的训练标签
                    "marker": {
                        "uvs_normalized": {            # 归一化后的marker几何
                            "u": marker_geometry.get("u", -1.0),
                            "v": marker_geometry.get("v", -1.0),
                            "s": marker_geometry.get("s", 0.0)
                        },
                        "uvs_raw": {                   # 原始像素坐标（方便debug）
                            "u_raw": marker_geometry.get("u_raw", -1.0),
                            "v_raw": marker_geometry.get("v_raw", -1.0)
                        },
                        "visible": marker_geometry.get("visible", False),  # bool: marker是否可见
                        "Zc": marker_geometry.get("Zc", -1.0),            # float: 相机坐标系中的深度
                        "marker_cam": marker_geometry.get("marker_cam", [-1.0, -1.0, -1.0])  # [Xc, Yc, Zc]
                    }
                }
                episode_metadata.append(step_data)

            # --- 3. 应用动作并推进仿真（每一步都需要）---
            robot.apply_action(action)
            sim.step(render=True)

            # --- 4. 检查碰撞（检测到碰撞立即终止episode）---
            dq_after_step = robot.get_joint_velocities()
            has_collision = False
            collision_reason = ""
            max_velocity = np.max(np.abs(dq_after_step))
            if max_velocity > COLLISION_VELOCITY_THRESHOLD:
                has_collision = True
                collision_reason = f"速度异常: {max_velocity:.2f} rad/s > {COLLISION_VELOCITY_THRESHOLD} rad/s"
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
                if step % capture_every_n != 0:
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
                    
                    # ========== 计算marker的几何信息 (u, v, s) ==========
                    try:
                        cam_prim = XFormPrim(CAM_PATH)
                        cam_world_pos, cam_world_orn = cam_prim.get_world_pose()
                        cam_world_pos = np.array([float(cam_world_pos[0]), float(cam_world_pos[1]), float(cam_world_pos[2])])
                        cam_world_orn = [float(cam_world_orn[0]), float(cam_world_orn[1]), float(cam_world_orn[2]), float(cam_world_orn[3])]
                        marker_geometry = compute_marker_geometry(current_marker_pos, cam_world_pos, cam_world_orn, camera_intrinsics, img_resolution)
                    except Exception as e:
                        marker_geometry = {
                            "visible": False,
                            "u": -1.0,
                            "v": -1.0,
                            "s": 0.0,
                            "Zc": -1.0,
                            "marker_cam": [-1.0, -1.0, -1.0]
                        }
                    
                    step_data = {
                        "image_path": img_filename,        # str: 图像文件名
                        "q": q_current.tolist(),           # list: 关节位置 [7] - proprioception信息
                        "delta_q": delta_q.tolist(),       # list: 关节增量命令 [7] - BC的训练标签
                        "marker": {
                            "uvs_normalized": {            # 归一化后的marker几何
                                "u": marker_geometry.get("u", -1.0),
                                "v": marker_geometry.get("v", -1.0),
                                "s": marker_geometry.get("s", 0.0)
                            },
                            "uvs_raw": {                   # 原始像素坐标（方便debug）
                                "u_raw": marker_geometry.get("u_raw", -1.0),
                                "v_raw": marker_geometry.get("v_raw", -1.0)
                            },
                            "visible": marker_geometry.get("visible", False),  # bool: marker是否可见
                            "Zc": marker_geometry.get("Zc", -1.0),            # float: 相机坐标系中的深度
                            "marker_cam": marker_geometry.get("marker_cam", [-1.0, -1.0, -1.0])  # [Xc, Yc, Zc]
                        }
                    }
                    episode_metadata.append(step_data)
                    print(f"   💾 已强制保存终止帧 {step}")
                break
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
                    end_step = STEPS_PER_EPISODE - 1  # 更新end_step
                    print(f"   ✅ Episode {episode_idx} 在最后一步满足成功条件 "
                          f"(X={diff_x:.3f}m, Y={diff_y:.3f}m, Z={diff_z:.3f}m)")
                    
                    # 如果成功终止步不是记录帧，强制保存最后一帧（关键纠偏瞬间）
                    if end_step % capture_every_n != 0:
                        q_current = robot.get_joint_positions()
                        dq_current = robot.get_joint_velocities()
                        if last_command_q is not None:
                            command_q = last_command_q
                        else:
                            command_q = q_current
                        delta_q = command_q - q_current
                        img_filename = f"frame_{end_step:04d}.png"
                        img_path = os.path.join(ep_dir, img_filename)
                        cam.capture(img_path)
                        try:
                            ee_prim = XFormPrim("/World/Panda/TCP")
                            ee_actual_pos, ee_actual_orn = ee_prim.get_world_pose()
                        except:
                            ee_actual_pos = None
                            ee_actual_orn = None
                        
                        # ========== 计算marker的几何信息 (u, v, s) ==========
                        try:
                            cam_prim = XFormPrim(CAM_PATH)
                            cam_world_pos, cam_world_orn = cam_prim.get_world_pose()
                            cam_world_pos = np.array([float(cam_world_pos[0]), float(cam_world_pos[1]), float(cam_world_pos[2])])
                            cam_world_orn = [float(cam_world_orn[0]), float(cam_world_orn[1]), float(cam_world_orn[2]), float(cam_world_orn[3])]
                            marker_geometry = compute_marker_geometry(current_marker_pos, cam_world_pos, cam_world_orn, camera_intrinsics, img_resolution)
                        except Exception as e:
                            marker_geometry = {
                                "visible": False,
                                "u": -1.0,
                                "v": -1.0,
                                "s": 0.0,
                                "Zc": -1.0,
                                "marker_cam": [-1.0, -1.0, -1.0]
                            }
                        
                        step_data = {
                            "image_path": img_filename,        # str: 图像文件名
                            "q": q_current.tolist(),           # list: 关节位置 [7] - proprioception信息
                            "delta_q": delta_q.tolist(),       # list: 关节增量命令 [7] - BC的训练标签
                            "marker": {
                                "uvs_normalized": {            # 归一化后的marker几何
                                    "u": marker_geometry.get("u", -1.0),
                                    "v": marker_geometry.get("v", -1.0),
                                    "s": marker_geometry.get("s", 0.0)
                                },
                                "uvs_raw": {                   # 原始像素坐标（方便debug）
                                    "u_raw": marker_geometry.get("u_raw", -1.0),
                                    "v_raw": marker_geometry.get("v_raw", -1.0)
                                },
                                "visible": marker_geometry.get("visible", False),  # bool: marker是否可见
                                "Zc": marker_geometry.get("Zc", -1.0),            # float: 相机坐标系中的深度
                                "marker_cam": marker_geometry.get("marker_cam", [-1.0, -1.0, -1.0])  # [Xc, Yc, Zc]
                            }
                        }
                        episode_metadata.append(step_data)
                        print(f"   💾 已强制保存成功终止帧 {end_step}")
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
            "bucket_type": config_type,  # 记录使用的桶类型（便于分析）
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