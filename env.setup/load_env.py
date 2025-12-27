#!/usr/bin/env python3
"""
加载并显示 env.usda 环境
运行方式: ~/isaacsim/python.sh /home/alphatok/ME5400/env.setup/load_env.py
"""

from omni.isaac.kit import SimulationApp

# GUI 模式
simulation_app = SimulationApp({"headless": False})

import omni.usd

def load_environment():
    """
    加载 env.usda 环境
    """
    stage = omni.usd.get_context().get_stage()
    
    # 环境文件路径
    env_path = "/home/alphatok/ME5400/env.setup/env.usda"
    
    # 检查文件是否存在
    import os
    if not os.path.exists(env_path):
        print(f"错误: 找不到环境文件 {env_path}")
        return False
    
    # 尝试使用新 API，如果失败则使用旧 API
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
        print("✅ 使用 isaacsim.core API")
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage
        print("⚠️ 使用 omni.isaac.core API（已弃用）")
    
    # 直接打开 env.usda 文件（而不是引用）
    # env.usda 已经通过 references 引用了：
    # - ./Props/VentionTableWithBlackCover/table_with_cover.usd
    # - ./Props/ABDPhantom/phantom.usda  
    # - ./Robots/Franka/Collected_panda_assembly/panda_assembly.usda
    try:
        # 使用 omni.usd 打开文件（这会自动解析所有 references）
        # open_stage 可能需要一些时间，所以等待一下
        import time
        omni.usd.get_context().open_stage(env_path)
        print(f"✅ 正在打开环境: {env_path}")
        
        # 等待场景加载（给 USD 时间解析 references）
        for i in range(10):
            time.sleep(0.1)
            # 检查是否已加载
            current_stage = omni.usd.get_context().get_stage()
            if current_stage and current_stage.GetPrimAtPath("/World").IsValid():
                break
        
        print("   注意: env.usda 通过 references 引用了以下文件:")
        print("   - ./Props/VentionTableWithBlackCover/table_with_cover.usd")
        print("   - ./Props/ABDPhantom/phantom.usda")
        print("   - ./Robots/Franka/Collected_panda_assembly/panda_assembly.usda")
        
        # 重新获取 stage（因为打开了新文件）
        stage = omni.usd.get_context().get_stage()
        
        # 验证主要组件是否存在（检查大小写变体）
        paths_to_check = [
            ("Phantom", ["/World/Phantom", "/World/phantom"]),
            ("Panda", ["/World/Panda", "/World/panda"]),
            ("Table", ["/World/Table", "/World/table"])
        ]
        
        for name, path_list in paths_to_check:
            found = False
            for path in path_list:
                if stage.GetPrimAtPath(path).IsValid():
                    print(f"✅ {name} 已加载: {path}")
                    found = True
                    break
            if not found:
                print(f"⚠️ 未找到 {name}，尝试的路径: {path_list}")
        
        return True
    except Exception as e:
        print(f"❌ 加载环境时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("正在加载环境...")
    if load_environment():
        print("\n✅ 环境加载完成！")
        print("场景包含: Panda 机械臂、Vention 桌子、ABD Phantom")
        print("\n进入主循环，按 Ctrl+C 退出...")
    else:
        print("\n❌ 环境加载失败")

if __name__ == "__main__":
    try:
        main()
        # 保持应用运行，但要让 GUI 保持响应
        import omni.kit.app
        app = omni.kit.app.get_app()
        
        # 使用更短的时间间隔，并确保应用可以处理事件
        import time
        print("\n✅ Isaac Sim 窗口已打开，场景已加载")
        print("你可以在窗口中查看和操作场景")
        print("关闭窗口或按 Ctrl+C 退出...")
        
        while app.is_running():
            # 非常短的休眠，让 GUI 事件循环处理事件
            app.update()
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n正在关闭应用...")
        simulation_app.close()

