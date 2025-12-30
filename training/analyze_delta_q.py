#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 expert_data 中 delta_q 的统计特性
检查模型是否学到了"保守的、近似坐标下降式的策略"
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_delta_q_statistics(dataset_root):
    """
    分析 delta_q 的统计特性
    
    检查项：
    1. 每个关节的 |Δq| 均值/方差
    2. 每个样本里"最大关节幅值占比"的分布
    3. 平均同时显著非零（>|ε|）的关节数量
    """
    metadata_dir = os.path.join(dataset_root, "metadata")
    
    if not os.path.exists(metadata_dir):
        raise ValueError(f"metadata目录不存在: {metadata_dir}")
    
    # 收集所有 delta_q
    all_delta_q = []  # 存储所有有效的 delta_q 数组
    joint_abs_delta_q = defaultdict(list)  # 每个关节的 |Δq| 值
    max_joint_ratios = []  # 每个样本的最大关节幅值占比
    non_zero_counts = []  # 每个样本中显著非零的关节数量
    
    # 阈值：判断关节是否"显著非零"
    epsilon = 0.01  # rad，如果 |delta_q| > 0.01 认为显著非零
    
    print("正在加载数据...")
    meta_files = sorted(
        [f for f in os.listdir(metadata_dir) if f.startswith("episode_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    
    total_samples = 0
    skipped_samples = 0
    
    for meta_name in meta_files:
        meta_path = os.path.join(metadata_dir, meta_name)
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[WARN] 加载 {meta_path} 失败: {e}")
            continue
        
        steps = meta.get("steps", [])
        
        for step_data in steps:
            action = step_data.get("action", {})
            delta_q_raw = action.get("delta_q", action.get("delta_q_cmd", []))
            delta_q_arr = np.array(delta_q_raw, dtype=np.float32)
            
            # 跳过无效数据
            if delta_q_arr.size == 0 or not np.isfinite(delta_q_arr).all():
                skipped_samples += 1
                continue
            
            # 确保是7维
            if delta_q_arr.shape[0] != 7:
                if delta_q_arr.shape[0] < 7:
                    delta_q_arr = np.pad(delta_q_arr, (0, 7 - delta_q_arr.shape[0]), 'constant')[:7]
                else:
                    delta_q_arr = delta_q_arr[:7]
            
            all_delta_q.append(delta_q_arr)
            total_samples += 1
            
            # 1. 收集每个关节的 |Δq|
            abs_delta_q = np.abs(delta_q_arr)
            for j in range(7):
                joint_abs_delta_q[j].append(abs_delta_q[j])
            
            # 2. 计算"最大关节幅值占比"
            # 即：max(|Δq_i|) / sum(|Δq_i|)
            abs_sum = np.sum(abs_delta_q)
            if abs_sum > 1e-6:  # 避免除以零
                max_ratio = np.max(abs_delta_q) / abs_sum
                max_joint_ratios.append(max_ratio)
            else:
                max_joint_ratios.append(0.0)
            
            # 3. 统计显著非零的关节数量
            non_zero_mask = abs_delta_q > epsilon
            non_zero_count = np.sum(non_zero_mask)
            non_zero_counts.append(non_zero_count)
    
    print(f"\n数据加载完成:")
    print(f"  总样本数: {total_samples}")
    print(f"  跳过样本数: {skipped_samples}")
    
    if total_samples == 0:
        print("❌ 没有有效数据！")
        return
    
    # ========== 分析1：每个关节的 |Δq| 均值/方差 ==========
    print(f"\n{'='*60}")
    print("分析1: 每个关节的 |Δq| 统计")
    print(f"{'='*60}")
    
    joint_stats = {}
    for j in range(7):
        abs_values = np.array(joint_abs_delta_q[j])
        mean_val = np.mean(abs_values)
        std_val = np.std(abs_values)
        median_val = np.median(abs_values)
        max_val = np.max(abs_values)
        min_val = np.min(abs_values)
        
        joint_stats[j] = {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'max': max_val,
            'min': min_val
        }
        
        print(f"关节 {j+1}:")
        print(f"  均值: {mean_val:.6f} rad")
        print(f"  标准差: {std_val:.6f} rad")
        print(f"  中位数: {median_val:.6f} rad")
        print(f"  最大值: {max_val:.6f} rad")
        print(f"  最小值: {min_val:.6f} rad")
        print()
    
    # ========== 分析2：最大关节幅值占比的分布 ==========
    print(f"{'='*60}")
    print("分析2: 最大关节幅值占比分布")
    print(f"{'='*60}")
    print("定义: max(|Δq_i|) / sum(|Δq_i|)")
    print("如果接近1.0，说明每次只有一个关节大幅移动（坐标下降）")
    print("如果接近1/7≈0.14，说明所有关节均匀移动")
    print()
    
    max_ratios_arr = np.array(max_joint_ratios)
    print(f"均值: {np.mean(max_ratios_arr):.4f}")
    print(f"中位数: {np.median(max_ratios_arr):.4f}")
    print(f"标准差: {np.std(max_ratios_arr):.4f}")
    print(f"最小值: {np.min(max_ratios_arr):.4f}")
    print(f"最大值: {np.max(max_ratios_arr):.4f}")
    print()
    
    # 统计不同区间的分布
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(max_ratios_arr, bins=bins)
    print("分布区间:")
    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / len(max_ratios_arr) * 100
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} ({pct:.1f}%)")
    
    # 统计"高度集中"的样本（最大关节占比 > 0.7）
    high_concentration = np.sum(max_ratios_arr > 0.7)
    print(f"\n最大关节占比 > 0.7 的样本数: {high_concentration} ({high_concentration/len(max_ratios_arr)*100:.1f}%)")
    print("（这些样本表现出明显的'坐标下降'特征）")
    
    # ========== 分析3：平均同时显著非零的关节数量 ==========
    print(f"\n{'='*60}")
    print(f"分析3: 平均同时显著非零（>|{epsilon}| rad）的关节数量")
    print(f"{'='*60}")
    
    non_zero_counts_arr = np.array(non_zero_counts)
    print(f"均值: {np.mean(non_zero_counts_arr):.2f} 个关节")
    print(f"中位数: {np.median(non_zero_counts_arr):.2f} 个关节")
    print(f"标准差: {np.std(non_zero_counts_arr):.2f} 个关节")
    print(f"最小值: {np.min(non_zero_counts_arr)} 个关节")
    print(f"最大值: {np.max(non_zero_counts_arr)} 个关节")
    print()
    
    # 统计不同非零关节数量的分布
    unique_counts, unique_freqs = np.unique(non_zero_counts_arr, return_counts=True)
    print("非零关节数量分布:")
    for count, freq in zip(unique_counts, unique_freqs):
        pct = freq / len(non_zero_counts_arr) * 100
        print(f"  {int(count)} 个关节: {freq} ({pct:.1f}%)")
    
    # 统计"单关节主导"的样本（只有1-2个关节显著非零）
    single_joint_dominant = np.sum(non_zero_counts_arr <= 2)
    print(f"\n只有1-2个关节显著非零的样本数: {single_joint_dominant} ({single_joint_dominant/len(non_zero_counts_arr)*100:.1f}%)")
    print("（这些样本表现出明显的'坐标下降'特征）")
    
    # ========== 可视化 ==========
    output_dir = os.path.join(dataset_root, "delta_q_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1: 每个关节的 |Δq| 分布（箱线图）
    plt.figure(figsize=(12, 6))
    data_to_plot = [np.array(joint_abs_delta_q[j]) for j in range(7)]
    plt.boxplot(data_to_plot, labels=[f"Joint {j+1}" for j in range(7)])
    plt.ylabel("|Δq| (rad)")
    plt.title("每个关节的 |Δq| 分布")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "joint_abs_delta_q_boxplot.png"), dpi=150)
    plt.close()
    
    # 图2: 最大关节幅值占比的直方图
    plt.figure(figsize=(10, 6))
    plt.hist(max_ratios_arr, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(max_ratios_arr), color='r', linestyle='--', label=f'均值: {np.mean(max_ratios_arr):.3f}')
    plt.axvline(1.0/7, color='g', linestyle='--', label='均匀分布: 1/7≈0.143')
    plt.xlabel("最大关节幅值占比")
    plt.ylabel("样本数")
    plt.title("最大关节幅值占比分布\n(接近1.0 = 坐标下降, 接近0.14 = 均匀移动)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "max_joint_ratio_hist.png"), dpi=150)
    plt.close()
    
    # 图3: 非零关节数量的分布
    plt.figure(figsize=(10, 6))
    unique_counts_sorted = sorted(unique_counts)
    unique_freqs_sorted = [unique_freqs[list(unique_counts).index(c)] for c in unique_counts_sorted]
    plt.bar(unique_counts_sorted, unique_freqs_sorted, edgecolor='black', alpha=0.7)
    plt.xlabel("显著非零的关节数量")
    plt.ylabel("样本数")
    plt.title(f"每个样本中显著非零（>|{epsilon}| rad）的关节数量分布")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "non_zero_joint_count_dist.png"), dpi=150)
    plt.close()
    
    # 图4: 每个关节的 |Δq| 均值对比
    plt.figure(figsize=(10, 6))
    joint_means = [joint_stats[j]['mean'] for j in range(7)]
    joint_stds = [joint_stats[j]['std'] for j in range(7)]
    x_pos = np.arange(7)
    plt.bar(x_pos, joint_means, yerr=joint_stds, capsize=5, edgecolor='black', alpha=0.7)
    plt.xlabel("关节编号")
    plt.ylabel("|Δq| 均值 (rad)")
    plt.title("每个关节的 |Δq| 均值（带标准差）")
    plt.xticks(x_pos, [f"Joint {j+1}" for j in range(7)])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "joint_means_with_std.png"), dpi=150)
    plt.close()
    
    print(f"\n✅ 可视化图表已保存到: {output_dir}/")
    
    # ========== 总结 ==========
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    
    mean_max_ratio = np.mean(max_ratios_arr)
    mean_non_zero = np.mean(non_zero_counts_arr)
    
    print(f"1. 最大关节幅值占比均值: {mean_max_ratio:.3f}")
    if mean_max_ratio > 0.6:
        print("   → 数据表现出明显的'坐标下降'特征（每次主要移动一个关节）")
    elif mean_max_ratio > 0.4:
        print("   → 数据表现出中等程度的'坐标下降'特征")
    else:
        print("   → 数据表现出'多关节协调'特征（多个关节同时移动）")
    
    print(f"\n2. 平均显著非零关节数: {mean_non_zero:.2f}")
    if mean_non_zero < 2.5:
        print("   → 数据表现出明显的'坐标下降'特征（每次只有1-2个关节显著移动）")
    elif mean_non_zero < 4.0:
        print("   → 数据表现出中等程度的'坐标下降'特征")
    else:
        print("   → 数据表现出'多关节协调'特征（多个关节同时显著移动）")
    
    print(f"\n3. 如果模型学到了这种模式，它可能会：")
    print("   - 预测较小的 delta_q（保守）")
    print("   - 每次主要移动一个关节（坐标下降）")
    print("   - 导致在线评估时移动很慢")

if __name__ == "__main__":
    dataset_root = "/home/alphatok/ME5400/expert_data"
    analyze_delta_q_statistics(dataset_root)

