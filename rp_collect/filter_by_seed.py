#!/usr/bin/env python
"""
根据 seed.txt 过滤数据集，只保留成功的 episodes
删除所有不在 seed.txt 中的 episodes
"""

import os
import json
import shutil
from pathlib import Path

def filter_dataset_by_seed(dataset_path):
    """根据 seed.txt 过滤单个数据集"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"⚠️  数据集不存在: {dataset_path}")
        return
    
    seed_file = dataset_path / "seed.txt"
    metadata_dir = dataset_path / "metadata"
    picture_data_dir = dataset_path / "picture_data"
    
    if not seed_file.exists():
        print(f"⚠️  {dataset_path.name} 没有 seed.txt 文件，跳过")
        return
    
    if not metadata_dir.exists() or not picture_data_dir.exists():
        print(f"⚠️  {dataset_path.name} 数据不完整，跳过")
        return
    
    # 读取成功的 episode 索引
    successful_episodes = set()
    with open(seed_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                successful_episodes.add(int(line))
    
    print(f"\n处理数据集: {dataset_path.name}")
    print(f"  成功的 episodes: {len(successful_episodes)} 个")
    
    # 获取所有 episode 文件
    all_episode_files = sorted(
        [f for f in metadata_dir.iterdir() if f.name.startswith("episode_") and f.name.endswith(".json")],
        key=lambda x: int(x.stem.split("_")[1])
    )
    
    total_episodes = len(all_episode_files)
    deleted_count = 0
    kept_count = 0
    
    # 遍历所有 episodes
    for episode_file in all_episode_files:
        episode_idx = int(episode_file.stem.split("_")[1])
        
        if episode_idx in successful_episodes:
            # 保留这个 episode
            kept_count += 1
        else:
            # 删除这个 episode
            # 删除 metadata 文件
            episode_file.unlink()
            
            # 删除 picture_data 文件夹
            picture_dir = picture_data_dir / f"episode_{episode_idx:04d}"
            if picture_dir.exists():
                shutil.rmtree(picture_dir)
            
            deleted_count += 1
    
    print(f"  总 episodes: {total_episodes}")
    print(f"  保留: {kept_count}")
    print(f"  删除: {deleted_count}")
    
    # 验证：确保所有成功的 episodes 都存在
    missing_episodes = []
    for episode_idx in successful_episodes:
        metadata_file = metadata_dir / f"episode_{episode_idx:04d}.json"
        picture_dir = picture_data_dir / f"episode_{episode_idx:04d}"
        if not metadata_file.exists() or not picture_dir.exists():
            missing_episodes.append(episode_idx)
    
    if missing_episodes:
        print(f"  ⚠️  警告：以下成功的 episodes 缺失: {missing_episodes}")
    else:
        print(f"  ✅ 所有成功的 episodes 都存在")


def main():
    """主函数：处理所有数据集"""
    base_dir = Path("/home/alphatok/ME5400/rp_collect")
    
    # 要处理的数据集列表
    datasets = [
        "dataset_v1",
        "dataset_v2",
        "dataset_v3",
        "dataset_v4",
        "dataset_v5"
    ]
    
    print("="*60)
    print("开始根据 seed.txt 过滤数据集...")
    print("="*60)
    
    for dataset_name in datasets:
        dataset_path = base_dir / dataset_name
        filter_dataset_by_seed(dataset_path)
    
    print("\n" + "="*60)
    print("过滤完成！")
    print("="*60)
    print("\n注意：")
    print("  - 只保留了 seed.txt 中列出的成功 episodes")
    print("  - 所有失败的 episodes 已被删除")
    print("  - 请检查数据完整性")


if __name__ == "__main__":
    main()

