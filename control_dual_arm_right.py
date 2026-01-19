#!/usr/bin/env python3
"""
Control Dual Arm Right Arm to Marker using RMPflow - Multiple Episodes
"""

# MUST be first import
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
import omni.timeline
import omni.usd
from pxr import UsdPhysics, Gf, Usd

from omni.isaac.core import SimulationContext
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy

# ===================== Configuration =====================
DUAL_ARM_USD_PATH = "/home/wopubuntu/me5400/dual arm.usda"
ROBOT_RIGHT_PATH = "/World/panda_right"
MARKER_PATH = "/World/Phantom/marker"
TARGET_OFFSET = [0.605, -0.05, -0.55]  # Offset from marker to actual target
DT = 1.0 / 60.0
NUM_EPISODES = 5  # Run 5 episodes
STEPS_PER_EPISODE = 300  # Steps per episode
SUCCESS_DISTANCE = 0.05  # 5cm

# Franka joint limits
PANDA_JOINT_LIMITS = [
    (-2.8973, 2.8973),   # joint1
    (-1.7628, 1.7628),   # joint2
    (-2.8973, 2.8973),   # joint3
    (-3.0718, -0.0698),  # joint4
    (-2.8973, 2.8973),   # joint5
    (-0.0175, 3.7525),   # joint6
    (-2.8973, 2.8973),   # joint7
]

def sample_random_joint_config(num_joints):
    """Sample random joint configuration within limits"""
    random_joint_positions = []
    for i in range(num_joints):
        if i < len(PANDA_JOINT_LIMITS):
            lower, upper = PANDA_JOINT_LIMITS[i]
            random_joint_positions.append(np.random.uniform(lower, upper))
        else:
            random_joint_positions.append(0.0)
    return np.array(random_joint_positions, dtype=np.float32)

def main():
    print("=" * 70)
    print("🤖 Dual Arm Control - Right Arm Multi-Episode with Random Start")
    print("=" * 70)
    
    # Load dual arm environment
    print(f"\n📂 Loading environment: {DUAL_ARM_USD_PATH}")
    omni.usd.get_context().open_stage(DUAL_ARM_USD_PATH)
    
    # Wait for resources
    for _ in range(100):
        simulation_app.update()
    
    # Get stage
    stage = omni.usd.get_context().get_stage()
    print("✅ Stage loaded")
    
    # Check for right arm
    robot_prim = stage.GetPrimAtPath(ROBOT_RIGHT_PATH)
    if not robot_prim.IsValid():
        print(f"❌ Right arm not found at {ROBOT_RIGHT_PATH}")
        return
    print(f"✅ Found right arm at {ROBOT_RIGHT_PATH}")
    
    # Check for marker
    marker_prim = stage.GetPrimAtPath(MARKER_PATH)
    if not marker_prim.IsValid():
        print(f"❌ Marker not found at {MARKER_PATH}")
        return
    print(f"✅ Found marker at {MARKER_PATH}")
    
    # Check for left arm (to set it to safe position)
    robot_left_prim = stage.GetPrimAtPath("/World/panda_left")
    if robot_left_prim.IsValid():
        print(f"✅ Found left arm at /World/panda_left")
    else:
        print(f"⚠️ Left arm not found at /World/panda_left")
    
    # Initialize timeline
    timeline = omni.timeline.get_timeline_interface()
    
    # Ensure PhysicsScene exists
    has_physics = False
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            has_physics = True
            break
    if not has_physics:
        print("   Creating PhysicsScene...")
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    
    # Create right arm articulation
    print("\n📍 Creating right arm articulation...")
    robot_right = Articulation(ROBOT_RIGHT_PATH)
    
    # Initialize SimulationContext
    print("⚙️ Initializing SimulationContext...")
    sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    
    # Play timeline
    print("▶️ Playing timeline...")
    timeline.play()
    
    # Initialize physics
    print("⚡ Initializing physics...")
    sim.initialize_physics()
    
    # Warm up physics
    print("🔥 Warming up physics (60 frames)...")
    for _ in range(60):
        sim.step(render=False)
    
    # Initialize robot
    print("🔧 Initializing right arm robot...")
    try:
        robot_right.initialize()
    except Exception as e:
        print(f"⚠️ First initialization failed ({e}), retrying...")
        for _ in range(10):
            sim.step(render=False)
        robot_right.initialize()
    
    print(f"✅ Right arm initialized with {robot_right.num_dof} DOFs")
    
    # Initialize left arm and set to safe position
    print("\n🔧 Initializing left arm robot...")
    robot_left = Articulation("/World/panda_left")
    try:
        robot_left.initialize()
    except Exception as e:
        print(f"⚠️ Left arm initialization failed ({e}), retrying...")
        for _ in range(10):
            sim.step(render=False)
        robot_left.initialize()
    
    print(f"✅ Left arm initialized with {robot_left.num_dof} DOFs")
    
    # Set left arm to safe upright position
    # For Franka, upright position is typically: [0, -π/4, 0, -3π/4, 0, π/2, π/4]
    print("🛡️ Setting left arm to safe upright position...")
    upright_config = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    
    # Pad with zeros if the arm has more DOFs (for fingers)
    if robot_left.num_dof > len(upright_config):
        upright_config = np.concatenate([upright_config, np.zeros(robot_left.num_dof - len(upright_config))])
    
    robot_left.set_joint_positions(upright_config)
    robot_left.set_joint_velocities(np.zeros(robot_left.num_dof))
    
    # Simulate a few steps to let the left arm settle
    for _ in range(30):
        sim.step(render=False)
    
    print("✅ Left arm set to safe upright position")
    
    # Initialize RMPflow
    print("\n🎯 Loading RMPflow...")
    mg_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    cfg_dir = os.path.join(mg_path, "motion_policy_configs")
    
    rmp_right = RmpFlow(
        robot_description_path=os.path.join(cfg_dir, "franka/rmpflow/robot_descriptor.yaml"),
        urdf_path=os.path.join(cfg_dir, "franka/lula_franka_gen.urdf"),
        rmpflow_config_path=os.path.join(cfg_dir, "franka/rmpflow/franka_rmpflow_common.yaml"),
        end_effector_frame_name="panda_hand",
        maximum_substep_size=0.00334,
    )
    print("✅ RMPflow loaded")
    
    # Create motion policy
    policy_right = ArticulationMotionPolicy(robot_right, rmp_right)
    print("✅ Motion policy created")
    
    # Get marker
    target_prim = XFormPrim(MARKER_PATH)
    marker_pos, marker_orn = target_prim.get_world_pose()
    marker_pos = np.array([float(marker_pos[0]), float(marker_pos[1]), float(marker_pos[2])])
    target_pos_world = marker_pos + np.array(TARGET_OFFSET)
    
    print(f"\n📍 Marker position: {marker_pos}")
    print(f"📍 Target position (marker + offset): {target_pos_world}")
    
    # ===================== Main Episode Loop =====================
    print(f"\n{'='*70}")
    print(f"🚀 Running {NUM_EPISODES} episodes")
    print(f"{'='*70}\n")
    
    for episode in range(NUM_EPISODES):
        print(f"\n{'='*70}")
        print(f"📌 Episode {episode + 1}/{NUM_EPISODES}")
        print(f"{'='*70}")
        
        # Sample random initial configuration for right arm
        random_joint_config = sample_random_joint_config(robot_right.num_dof)
        print(f"\n🎲 Setting random initial joint configuration...")
        print(f"   Joint config: {[f'{q:.3f}' for q in random_joint_config[:7]]}")
        
        robot_right.set_joint_positions(random_joint_config)
        robot_right.set_joint_velocities(np.zeros(robot_right.num_dof))
        
        # Warm up physics for this episode
        for _ in range(30):
            sim.step(render=False)
        
        # Get initial EE position
        try:
            ee_prim = XFormPrim(f"{ROBOT_RIGHT_PATH}/panda_hand")
            ee_init_pos, _ = ee_prim.get_world_pose()
            ee_init_pos = np.array([float(ee_init_pos[0]), float(ee_init_pos[1]), float(ee_init_pos[2])])
            init_distance = np.linalg.norm(ee_init_pos - target_pos_world)
            print(f"   Initial EE position: {ee_init_pos}")
            print(f"   Initial distance to target: {init_distance:.4f}m")
        except:
            pass
        
        print(f"\n⏱️  Starting control loop ({STEPS_PER_EPISODE} steps)...")
        
        # Control loop for this episode
        episode_success = False
        for step in range(STEPS_PER_EPISODE):
            if not simulation_app.is_running():
                break
            
            if step % 100 == 0 and step > 0:
                print(f"   Step {step}/{STEPS_PER_EPISODE}")
            
            # Set RMPflow target to marker position + offset
            target_pos_with_offset = marker_pos + np.array(TARGET_OFFSET)
            rmp_right.set_end_effector_target(target_pos_with_offset, marker_orn)
            
            # Get next action
            action_right = policy_right.get_next_articulation_action(DT)
            
            # Apply action
            robot_right.apply_action(action_right)
            
            # Step simulation
            sim.step(render=True)
            
            # Check progress every 50 steps
            if step % 50 == 0:
                try:
                    ee_prim = XFormPrim(f"{ROBOT_RIGHT_PATH}/panda_hand")
                    ee_pos, _ = ee_prim.get_world_pose()
                    ee_pos = np.array([float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])])
                    distance = np.linalg.norm(ee_pos - target_pos_with_offset)
                    
                    if step % 100 == 0:
                        print(f"   Distance: {distance:.4f}m")
                    
                    # Check if reached target
                    if distance < SUCCESS_DISTANCE:
                        print(f"\n✅ 🎉 Reached target at step {step}!")
                        print(f"   Final distance: {distance:.4f}m")
                        episode_success = True
                        break
                except:
                    pass
        
        # Episode summary
        if episode_success:
            print(f"\n✅ Episode {episode + 1} SUCCESS")
        else:
            print(f"\n⚠️ Episode {episode + 1} TIMEOUT (did not reach target)")
    
    print(f"\n{'='*70}")
    print("✅ All episodes completed!")
    print(f"{'='*70}")
    print("\n⏸️  Keep Isaac Sim window open. Close manually to exit.")

if __name__ == "__main__":
    main()

def main():
    print("=" * 70)
    print("🤖 Dual Arm Control - Right Arm to Marker using RMPflow")
    print("=" * 70)
    
    # Load dual arm environment
    print(f"\n📂 Loading environment: {DUAL_ARM_USD_PATH}")
    omni.usd.get_context().open_stage(DUAL_ARM_USD_PATH)
    
    # Wait for resources
    for _ in range(100):
        simulation_app.update()
    
    # Get stage
    stage = omni.usd.get_context().get_stage()
    print("✅ Stage loaded")
    
    # Check for right arm
    robot_prim = stage.GetPrimAtPath(ROBOT_RIGHT_PATH)
    if not robot_prim.IsValid():
        print(f"❌ Right arm not found at {ROBOT_RIGHT_PATH}")
        return
    print(f"✅ Found right arm at {ROBOT_RIGHT_PATH}")
    
    # Check for marker
    marker_prim = stage.GetPrimAtPath(MARKER_PATH)
    if not marker_prim.IsValid():
        print(f"❌ Marker not found at {MARKER_PATH}")
        return
    print(f"✅ Found marker at {MARKER_PATH}")
    
    # Check for left arm (to set it to safe position)
    robot_left_prim = stage.GetPrimAtPath("/World/panda_left")
    if robot_left_prim.IsValid():
        print(f"✅ Found left arm at /World/panda_left")
    else:
        print(f"⚠️ Left arm not found at /World/panda_left")
    
    # Initialize timeline
    timeline = omni.timeline.get_timeline_interface()
    
    # Ensure PhysicsScene exists
    has_physics = False
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            has_physics = True
            break
    if not has_physics:
        print("   Creating PhysicsScene...")
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    
    # Create right arm articulation
    print("\n📍 Creating right arm articulation...")
    robot_right = Articulation(ROBOT_RIGHT_PATH)
    
    # Initialize SimulationContext
    print("⚙️ Initializing SimulationContext...")
    sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    
    # Play timeline
    print("▶️ Playing timeline...")
    timeline.play()
    
    # Initialize physics
    print("⚡ Initializing physics...")
    sim.initialize_physics()
    
    # Warm up physics
    print("🔥 Warming up physics (60 frames)...")
    for _ in range(60):
        sim.step(render=False)
    
    # Initialize robot
    print("🔧 Initializing right arm robot...")
    try:
        robot_right.initialize()
    except Exception as e:
        print(f"⚠️ First initialization failed ({e}), retrying...")
        for _ in range(10):
            sim.step(render=False)
        robot_right.initialize()
    
    print(f"✅ Right arm initialized with {robot_right.num_dof} DOFs")
    
    # Initialize left arm and set to safe position
    print("\n🔧 Initializing left arm robot...")
    robot_left = Articulation("/World/panda_left")
    try:
        robot_left.initialize()
    except Exception as e:
        print(f"⚠️ Left arm initialization failed ({e}), retrying...")
        for _ in range(10):
            sim.step(render=False)
        robot_left.initialize()
    
    print(f"✅ Left arm initialized with {robot_left.num_dof} DOFs")
    
    # Set left arm to safe upright position
    # For Franka, upright position is typically: [0, -π/4, 0, -3π/4, 0, π/2, π/4]
    print("🛡️ Setting left arm to safe upright position...")
    upright_config = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    
    # Pad with zeros if the arm has more DOFs (for fingers)
    if robot_left.num_dof > len(upright_config):
        upright_config = np.concatenate([upright_config, np.zeros(robot_left.num_dof - len(upright_config))])
    
    robot_left.set_joint_positions(upright_config)
    robot_left.set_joint_velocities(np.zeros(robot_left.num_dof))
    
    # Simulate a few steps to let the left arm settle
    for _ in range(30):
        sim.step(render=False)
    
    print("✅ Left arm set to safe upright position")
    
    print("\n🎯 Loading RMPflow...")
    mg_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    cfg_dir = os.path.join(mg_path, "motion_policy_configs")
    
    rmp_right = RmpFlow(
        robot_description_path=os.path.join(cfg_dir, "franka/rmpflow/robot_descriptor.yaml"),
        urdf_path=os.path.join(cfg_dir, "franka/lula_franka_gen.urdf"),
        rmpflow_config_path=os.path.join(cfg_dir, "franka/rmpflow/franka_rmpflow_common.yaml"),
        end_effector_frame_name="panda_hand",
        maximum_substep_size=0.00334,
    )
    print("✅ RMPflow loaded")
    
    # Create motion policy
    policy_right = ArticulationMotionPolicy(robot_right, rmp_right)
    print("✅ Motion policy created")
    
    # Get marker
    target_prim = XFormPrim(MARKER_PATH)
    
    # Debug: get initial positions
    print("\n🔍 Initial Positions:")
    try:
        marker_pos, marker_orn = target_prim.get_world_pose()
        marker_pos = np.array([float(marker_pos[0]), float(marker_pos[1]), float(marker_pos[2])])
        print(f"   Marker position: {marker_pos}")
        print(f"   TARGET_OFFSET: {TARGET_OFFSET}")
        target_with_offset = marker_pos + np.array(TARGET_OFFSET)
        print(f"   Target position (marker + offset): {target_with_offset}")
        
        ee_prim = XFormPrim(f"{ROBOT_RIGHT_PATH}/panda_hand")
        ee_pos, _ = ee_prim.get_world_pose()
        ee_pos = np.array([float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])])
        print(f"   Right arm EE position: {ee_pos}")
        
        init_distance = np.linalg.norm(ee_pos - target_with_offset)
        print(f"   Initial distance to target: {init_distance:.4f}m")
        
        if init_distance < 0.1:
            print(f"\n⚠️ EE already very close to target! Generating new marker position...")
            # Move marker to a new position
            new_marker_pos = marker_pos + np.array([0.3, 0.0, 0.1])
            new_marker_pos[2] = np.clip(new_marker_pos[2], 0.2, 1.5)
            print(f"   New marker position: {new_marker_pos}")
            target_prim.set_world_pose(new_marker_pos, marker_orn)
            marker_pos = new_marker_pos
    except Exception as e:
        print(f"   ⚠️ Error getting positions: {e}")
    
    # Main control loop
    print(f"\n{'='*70}")
    print(f"🚀 Starting control loop ({NUM_STEPS} steps)")
    print(f"{'='*70}\n")
    
    for step in range(NUM_STEPS):
        if not simulation_app.is_running():
            break
        
        if step % 50 == 0 and step > 0:
            print(f"⏱️  Step {step}/{NUM_STEPS}")
        
        # Get current marker position (fixed throughout episode)
        if step == 0:
            marker_pos_target, marker_orn_target = target_prim.get_world_pose()
            marker_pos_target = np.array([float(marker_pos_target[0]), float(marker_pos_target[1]), float(marker_pos_target[2])])
        
        # Set RMPflow target to marker position + offset
        target_pos_with_offset = marker_pos_target + np.array(TARGET_OFFSET)
        rmp_right.set_end_effector_target(target_pos_with_offset, marker_orn_target)
        
        # Get next action
        action_right = policy_right.get_next_articulation_action(DT)
        
        # Apply action
        robot_right.apply_action(action_right)
        
        # Step simulation
        sim.step(render=True)
        
        # Print progress every 100 steps
        if step % 100 == 0:
            try:
                ee_prim = XFormPrim(f"{ROBOT_RIGHT_PATH}/panda_hand")
                ee_pos, _ = ee_prim.get_world_pose()
                ee_pos = np.array([float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])])
                target_pos_with_offset = marker_pos_target + np.array(TARGET_OFFSET)
                distance = np.linalg.norm(ee_pos - target_pos_with_offset)
                
                print(f"   EE: {ee_pos}")
                print(f"   Target: {target_pos_with_offset}")
                print(f"   Distance: {distance:.4f}m")
                
                if distance < SUCCESS_DISTANCE:
                    print(f"\n✅ 🎉 Reached target at step {step}!")
                    print(f"   Final distance: {distance:.4f}m")
                    print(f"   Stopping...\n")
                    break
            except Exception as e:
                pass
    
    print(f"\n{'='*70}")
    print("✅ Control complete!")
    print(f"{'='*70}")
    print("\n⏸️  Keep Isaac Sim window open. Close manually to exit.")

if __name__ == "__main__":
    main()