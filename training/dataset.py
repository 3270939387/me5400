#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _list_episode_json(meta_dir: str):
    """返回排序后的 episode_XXXX.json 文件名列表"""
    if not os.path.exists(meta_dir):
        raise ValueError(f"metadata 目录不存在: {meta_dir}")

    files = [f for f in os.listdir(meta_dir) if f.startswith("episode_") and f.endswith(".json")]
    files = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    if len(files) == 0:
        raise ValueError(f"在 {meta_dir} 中找不到任何 episode_*.json")
    return files


def _split_episodes(episodes, split: str, seed: int = 42, train_ratio: float = 0.8):
    """episode-level 80/20 split，可复现"""
    rng = random.Random(seed)
    episodes = list(episodes)
    rng.shuffle(episodes)
    split_idx = int(train_ratio * len(episodes))
    if split == "train":
        return episodes[:split_idx]
    else:
        return episodes[split_idx:]


def _safe_pad_7(x: np.ndarray) -> np.ndarray:
    """确保向量长度为 7，不足补 0，超出截断"""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] != 7:
        x = np.pad(x, (0, max(0, 7 - x.shape[0])), "constant")[:7]
    return x


class MixedDataset(Dataset):
    """
    混合 BC(expert_dir) + DAgger(dagger_dir) 数据。
    - BC 动作从 step["action"]["delta_q"] 或 delta_q_cmd 读取
    - DAgger 动作从 step["expert_delta_q"] 读取
    """
    def __init__(
        self,
        expert_dir: str,
        dagger_dir: str,
        split: str = "train",
        image_size_hw=(240, 320),
        only_success: bool = False,
        seed: int = 42,
        train_ratio: float = 0.8,
        max_abs_action: float = 1e3,
        verbose: bool = True,
    ):
        super().__init__()
        assert split in ["train", "val"], "split 必须是 'train' 或 'val'"

        self.expert_dir = expert_dir
        self.dagger_dir = dagger_dir
        self.image_size = image_size_hw
        self.only_success = only_success
        self.split = split

        # 路径
        expert_meta_dir = os.path.join(expert_dir, "metadata")
        expert_pic_dir = os.path.join(expert_dir, "picture_data")
        dagger_meta_dir = os.path.join(dagger_dir, "metadata")
        dagger_pic_dir = os.path.join(dagger_dir, "picture_data")

        # 列 episode
        expert_episodes_all = _list_episode_json(expert_meta_dir)
        dagger_episodes_all = _list_episode_json(dagger_meta_dir)

        # train/val split（episode-level，可复现）
        expert_episodes = _split_episodes(expert_episodes_all, split=split, seed=seed, train_ratio=train_ratio)
        dagger_episodes = _split_episodes(dagger_episodes_all, split=split, seed=seed, train_ratio=train_ratio)

        # 图像 transform（与你训练 ResNet 的 ImageNet 归一化一致）
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),     # 注意：torchvision 的 Resize 接受 (H,W)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # 展开 samples（关键：避免 __getitem__ 读 JSON）
        self.samples = []
        skipped_img = 0
        skipped_action = 0
        skipped_success = 0
        skipped_json = 0

        def add_episode_samples(source: str, meta_dir: str, pic_dir: str, epi_filename: str):
            nonlocal skipped_img, skipped_action, skipped_success, skipped_json

            meta_path = os.path.join(meta_dir, epi_filename)
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                skipped_json += 1
                return

            # only_success 过滤（episode 级）
            if self.only_success and not meta.get("success", False):
                skipped_success += 1
                return

            steps = meta.get("steps", [])
            ep_idx = int(epi_filename.split("_")[1].split(".")[0])
            ep_picture_dir = os.path.join(pic_dir, f"episode_{ep_idx:04d}")

            for step in steps:
                # 图像路径
                img_filename = step.get("image_path", f"frame_{step.get('step', 0):04d}.png")
                img_path = os.path.join(ep_picture_dir, img_filename)
                if not os.path.exists(img_path):
                    skipped_img += 1
                    continue

                # 动作读取
                if source == "bc":
                    action = step.get("action", {})
                    dq = action.get("delta_q", action.get("delta_q_cmd", []))
                else:
                    dq = step.get("expert_delta_q", [])

                dq = np.asarray(dq, dtype=np.float32)

                # 动作质量过滤
                if dq.size == 0 or (not np.isfinite(dq).all()):
                    skipped_action += 1
                    continue
                if np.abs(dq).max() > max_abs_action:
                    skipped_action += 1
                    continue

                dq = _safe_pad_7(dq)

                # 关节位置
                q = step.get("state", {}).get("q", [0.0] * 7)
                q = _safe_pad_7(np.asarray(q, dtype=np.float32))

                self.samples.append({
                    "image_path": img_path,
                    "delta_q": dq,
                    "joint_positions": q,
                    "raw": step,       # 原始 step_data，方便 debug
                    "source": source,  # "bc" or "dagger"
                })

        # DAgger
        for epi in dagger_episodes:
            add_episode_samples("dagger", dagger_meta_dir, dagger_pic_dir, epi)

        # BC
        for epi in expert_episodes:
            add_episode_samples("bc", expert_meta_dir, expert_pic_dir, epi)

        if len(self.samples) == 0:
            raise ValueError("MixedDataset 未加载到任何样本，请检查目录结构/only_success/数据内容")

        # 统计打印
        if verbose:
            num_bc = sum(1 for s in self.samples if s["source"] == "bc")
            num_dag = len(self.samples) - num_bc
            dag_over_bc = num_dag / max(1, num_bc)

            # 取一小部分估计 dagger 动作幅值，防止 dagger 标签被读成 0
            dag_abs = [float(np.mean(np.abs(s["delta_q"]))) for s in self.samples if s["source"] == "dagger"]
            mean_abs_dag = float(np.mean(dag_abs[:min(1000, len(dag_abs))])) if len(dag_abs) > 0 else 0.0

            print(f"[MixedDataset] split={split} samples={len(self.samples)} "
                  f"(bc={num_bc}, dagger={num_dag}, dagger/bc={dag_over_bc:.3f}), "
                  f"mean|dq|_dagger(first<=1000)={mean_abs_dag:.6f}")
            print(f"[MixedDataset] skipped: json={skipped_json}, only_success={skipped_success}, "
                  f"bad_img={skipped_img}, bad_action={skipped_action}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # 读图像 + transform
        img_path = sample["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"加载图像失败 {img_path}: {e}") from e
        image = self.transform(image)

        # 注意：这里必须用 sample["delta_q"]，不要再从 raw/action 取
        delta_q = sample["delta_q"].astype(np.float32)
        joint_positions = sample["joint_positions"].astype(np.float32)

        return {
            "image": image,
            "delta_q": torch.from_numpy(delta_q),
            "joint_positions": torch.from_numpy(joint_positions),
            "raw": sample["raw"],
        }


# （可选）保留 MarkerDataset：如果你训练脚本仍需要它也能用
class MarkerDataset(Dataset):
    """你原来的单一数据集加载器（只读一个 dataset_root），保留以兼容旧训练。"""
    def __init__(self, dataset_root, split="train", image_size_hw=(240, 320), only_success=False):
        self.dataset_root = dataset_root
        self.metadata_dir = os.path.join(dataset_root, "metadata")
        self.picture_dir = os.path.join(dataset_root, "picture_data")
        self.image_size = image_size_hw
        self.only_success = only_success
        self.split = split

        if not os.path.exists(self.metadata_dir) or not os.path.exists(self.picture_dir):
            raise ValueError(f"数据结构不符合预期，缺少 metadata/ 或 picture_data/ 目录: {dataset_root}")

        meta_files = _list_episode_json(self.metadata_dir)
        episodes = _split_episodes(meta_files, split=split, seed=42, train_ratio=0.8)

        self.samples = []
        invalid_actions = 0

        for epi in episodes:
            meta_path = os.path.join(self.metadata_dir, epi)
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                continue

            if only_success and not meta.get("success", False):
                continue

            steps = meta.get("steps", [])
            ep_idx = int(epi.split("_")[1].split(".")[0])
            ep_picture_dir = os.path.join(self.picture_dir, f"episode_{ep_idx:04d}")
            if not os.path.exists(ep_picture_dir):
                continue

            for step in steps:
                action = step.get("action", {})
                dq = np.asarray(action.get("delta_q", action.get("delta_q_cmd", [])), dtype=np.float32)
                if dq.size == 0 or (not np.isfinite(dq).all()) or (np.abs(dq).max() > 1e3):
                    invalid_actions += 1
                    continue
                dq = _safe_pad_7(dq)

                img_filename = step.get("image_path", f"frame_{step.get('step', 0):04d}.png")
                img_path = os.path.join(ep_picture_dir, img_filename)
                if not os.path.exists(img_path):
                    continue

                q = _safe_pad_7(np.asarray(step.get("state", {}).get("q", [0.0]*7), dtype=np.float32))

                self.samples.append({
                    "image_path": img_path,
                    "delta_q": dq,
                    "joint_positions": q,
                    "raw": step,
                })

        if len(self.samples) == 0:
            raise ValueError("MarkerDataset 未加载到任何样本")

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"[MarkerDataset] split={split}: samples={len(self.samples)}, invalid_actions_skipped={invalid_actions}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_path = s["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "delta_q": torch.from_numpy(s["delta_q"]),
            "joint_positions": torch.from_numpy(s["joint_positions"]),
            "raw": s["raw"],
        }
