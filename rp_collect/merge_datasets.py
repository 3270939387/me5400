#!/usr/bin/env python
"""
合并 dataset_v1 到 dataset_v5 的数据到新的 dataset 文件夹
重新编号所有 episodes，使其连续
"""

import os
import json
import shutil
from pathlib import Path

def merge_datasets():
    """合并所有数据集"""
    base_dir = Path("/home/alphatok/ME5400/rp_collect")
    
    # 源数据集列表（按顺序）
    source_datasets = [
        "dataset_v1",
        "dataset_v2", 
        "dataset_v3",
        "dataset_v4",
        "dataset_v5"
    ]
    
    # 目标数据集
    target_dataset = base_dir / "dataset"
    target_metadata = target_dataset / "metadata"
    target_picture_data = target_dataset / "picture_data"
    
    # 如果目标文件夹已存在，询问是否删除
    if target_dataset.exists():
        print(f"⚠️  目标文件夹 {target_dataset} 已存在")
        response = input("是否删除并重新创建？(y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(target_dataset)
            print("已删除旧文件夹")
        else:
            print("取消操作")
            return
    
    # 创建目标文件夹
    target_metadata.mkdir(parents=True, exist_ok=True)
    target_picture_data.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_episodes = 0
    successful_episodes = []
    dataset_stats = {}
    all_seeds = []
    
    print("="*60)
    print("开始合并数据集...")
    print("="*60)
    
    # 遍历每个源数据集
    for dataset_name in source_datasets:
        source_path = base_dir / dataset_name
        if not source_path.exists():
            print(f"⚠️  跳过不存在的数据集: {dataset_name}")
            continue
        
        source_metadata = source_path / "metadata"
        source_picture_data = source_path / "picture_data"
        source_seed = source_path / "seed.txt"
        
        if not source_metadata.exists() or not source_picture_data.exists():
            print(f"⚠️  跳过不完整的数据集: {dataset_name}")
            continue
        
        print(f"\n处理数据集: {dataset_name}")
        dataset_start_idx = total_episodes
        dataset_episode_count = 0
        
        # 读取该数据集的 seed.txt（成功的 episodes）
        dataset_successful_episodes = set()
        if source_seed.exists():
            with open(source_seed, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line.isdigit():
                        dataset_successful_episodes.add(int(line))
        
        # 获取所有episode文件（按数字排序）
        episode_files = sorted(
            [f for f in source_metadata.iterdir() if f.name.startswith("episode_") and f.name.endswith(".json")],
            key=lambda x: int(x.stem.split("_")[1])
        )
        
        for episode_file in episode_files:
            # 读取metadata
            with open(episode_file, 'r') as f:
                episode_data = json.load(f)
            
            old_episode_idx = episode_data.get("episode_idx", 
                                              int(episode_file.stem.split("_")[1]))
            
            # 新的episode索引（连续编号）
            new_episode_idx = total_episodes
            
            # 更新metadata中的episode_idx
            episode_data["episode_idx"] = new_episode_idx
            episode_data["source_dataset"] = dataset_name
            episode_data["original_episode_idx"] = old_episode_idx
            
            # 保存新的metadata文件
            new_metadata_file = target_metadata / f"episode_{new_episode_idx:04d}.json"
            with open(new_metadata_file, 'w') as f:
                json.dump(episode_data, f, indent=2)
            
            # 复制picture_data文件夹
            old_picture_dir = source_picture_data / f"episode_{old_episode_idx:04d}"
            new_picture_dir = target_picture_data / f"episode_{new_episode_idx:04d}"
            
            if old_picture_dir.exists():
                shutil.copytree(old_picture_dir, new_picture_dir)
            else:
                print(f"  ⚠️  图片文件夹不存在: {old_picture_dir}")
            
            # 如果是成功的episode，记录到列表
            if episode_data.get("success", False) or old_episode_idx in dataset_successful_episodes:
                successful_episodes.append(new_episode_idx)
                all_seeds.append(new_episode_idx)
            
            total_episodes += 1
            dataset_episode_count += 1
        
        dataset_stats[dataset_name] = {
            "start_idx": dataset_start_idx,
            "end_idx": total_episodes - 1,
            "count": dataset_episode_count,
            "successful_count": len([idx for idx in successful_episodes if dataset_start_idx <= idx < total_episodes])
        }
        print(f"  ✅ 处理了 {dataset_episode_count} 个episodes (索引 {dataset_start_idx} - {total_episodes - 1})")
    
    # 保存合并后的seed.txt
    seed_file = target_dataset / "seed.txt"
    with open(seed_file, 'w') as f:
        for episode_idx in sorted(successful_episodes):
            f.write(f"{episode_idx}\n")
    
    # 保存合并统计信息
    stats_file = target_dataset / "merge_stats.json"
    merge_stats = {
        "total_episodes": total_episodes,
        "successful_episodes": len(successful_episodes),
        "dataset_stats": dataset_stats,
        "source_datasets": source_datasets
    }
    with open(stats_file, 'w') as f:
        json.dump(merge_stats, f, indent=2)
    
    print("\n" + "="*60)
    print("合并完成！")
    print("="*60)
    print(f"总episodes数: {total_episodes}")
    print(f"成功episodes数: {len(successful_episodes)}")
    print(f"\n数据集统计:")
    for dataset_name, stats in dataset_stats.items():
        print(f"  {dataset_name}: {stats['count']} episodes (索引 {stats['start_idx']}-{stats['end_idx']}), "
              f"成功: {stats['successful_count']}")
    print(f"\n输出目录: {target_dataset}")
    print(f"  - metadata: {target_metadata}")
    print(f"  - picture_data: {target_picture_data}")
    print(f"  - seed.txt: {seed_file}")
    print(f"  - merge_stats.json: {stats_file}")


if __name__ == "__main__":
    merge_datasets()
