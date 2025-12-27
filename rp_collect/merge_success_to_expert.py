#!/usr/bin/env python3
"""
合并两个success目录到expert_data，并重新排序episode编号
"""

import os
import json
import shutil
from pathlib import Path

# 源目录
DATA_SUCCESS_DIR = "/home/alphatok/ME5400/DATA/success"
DATA2_SUCCESS_DIR = "/home/alphatok/ME5400/DATA2/success"

# 目标目录
EXPERT_DATA_DIR = "/home/alphatok/ME5400/expert_data"
EXPERT_METADATA_DIR = os.path.join(EXPERT_DATA_DIR, "metadata")
EXPERT_PICTURE_DIR = os.path.join(EXPERT_DATA_DIR, "picture_data")

def extract_episode_number(filename):
    """从文件名中提取episode编号"""
    # episode_0001.json -> 1
    # episode_0001/ -> 1
    if filename.startswith("episode_"):
        num_str = filename.split("_")[1].split(".")[0]
        return int(num_str)
    return None

def collect_all_episodes():
    """收集所有episode文件"""
    episodes = []
    
    # 从DATA/success收集
    data_metadata_dir = os.path.join(DATA_SUCCESS_DIR, "metadata")
    data_picture_dir = os.path.join(DATA_SUCCESS_DIR, "picture_data")
    
    if os.path.exists(data_metadata_dir):
        for meta_file in os.listdir(data_metadata_dir):
            if meta_file.endswith(".json"):
                ep_num = extract_episode_number(meta_file)
                if ep_num is not None:
                    meta_path = os.path.join(data_metadata_dir, meta_file)
                    pic_dir = os.path.join(data_picture_dir, f"episode_{ep_num:04d}")
                    episodes.append({
                        "episode_num": ep_num,
                        "metadata_path": meta_path,
                        "picture_dir": pic_dir if os.path.exists(pic_dir) else None,
                        "source": "DATA"
                    })
    
    # 从DATA2/success收集
    data2_metadata_dir = os.path.join(DATA2_SUCCESS_DIR, "metadata")
    data2_picture_dir = os.path.join(DATA2_SUCCESS_DIR, "picture_data")
    
    if os.path.exists(data2_metadata_dir):
        for meta_file in os.listdir(data2_metadata_dir):
            if meta_file.endswith(".json"):
                ep_num = extract_episode_number(meta_file)
                if ep_num is not None:
                    meta_path = os.path.join(data2_metadata_dir, meta_file)
                    pic_dir = os.path.join(data2_picture_dir, f"episode_{ep_num:04d}")
                    episodes.append({
                        "episode_num": ep_num,
                        "metadata_path": meta_path,
                        "picture_dir": pic_dir if os.path.exists(pic_dir) else None,
                        "source": "DATA2"
                    })
    
    # 按原始episode编号和来源排序（保留所有episode，包括重复编号）
    # 先按episode_num排序，然后按source排序（DATA在前，DATA2在后）
    episodes.sort(key=lambda x: (x["episode_num"], 0 if x["source"] == "DATA" else 1))
    
    # 不去重，保留所有episode（包括重复编号的）
    return episodes

def update_episode_metadata(metadata_path, new_episode_idx):
    """更新metadata中的episode_idx"""
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    
    meta["episode_idx"] = new_episode_idx
    return meta

def main():
    print("🔍 收集所有episode...")
    episodes = collect_all_episodes()
    print(f"   找到 {len(episodes)} 个episode")
    
    # 创建目标目录
    os.makedirs(EXPERT_METADATA_DIR, exist_ok=True)
    os.makedirs(EXPERT_PICTURE_DIR, exist_ok=True)
    
    print(f"\n📦 开始合并到 {EXPERT_DATA_DIR}...")
    
    for new_idx, ep in enumerate(episodes):
        old_ep_num = ep["episode_num"]
        new_ep_num = new_idx
        
        # 1. 复制并更新metadata
        old_meta_path = ep["metadata_path"]
        new_meta_filename = f"episode_{new_ep_num:04d}.json"
        new_meta_path = os.path.join(EXPERT_METADATA_DIR, new_meta_filename)
        
        # 读取并更新metadata
        updated_meta = update_episode_metadata(old_meta_path, new_ep_num)
        
        # 保存更新后的metadata
        with open(new_meta_path, 'w') as f:
            json.dump(updated_meta, f, indent=2)
        
        # 2. 复制picture_data目录
        if ep["picture_dir"] and os.path.exists(ep["picture_dir"]):
            new_pic_dir = os.path.join(EXPERT_PICTURE_DIR, f"episode_{new_ep_num:04d}")
            if os.path.exists(new_pic_dir):
                shutil.rmtree(new_pic_dir)
            shutil.copytree(ep["picture_dir"], new_pic_dir)
        
        if (new_idx + 1) % 10 == 0:
            print(f"   已处理 {new_idx + 1}/{len(episodes)} 个episode")
    
    print(f"\n✅ 合并完成！")
    print(f"   总共合并了 {len(episodes)} 个episode")
    print(f"   输出目录: {EXPERT_DATA_DIR}")
    print(f"   - metadata: {EXPERT_METADATA_DIR}")
    print(f"   - picture_data: {EXPERT_PICTURE_DIR}")

if __name__ == "__main__":
    main()

