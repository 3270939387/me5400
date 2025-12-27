#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Managed Behavior Cloning training (ResNet18 + MLP)
Features:
- metrics.csv / metrics.jsonl logging
- early stopping
- ReduceLROnPlateau scheduler
- freeze backbone then unfreeze layer4
- extra validation metrics: per-joint RMSE, action norm stats, cosine similarity
- periodic visualization: plots saved to out_dir/plots/
"""

import os
import csv
import json
import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import MarkerDataset


# ----------------------------- Utilities -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def safe_mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if len(xs) else float("nan")

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # a,b: [B, D]
    an = torch.norm(a, dim=1).clamp_min(eps)
    bn = torch.norm(b, dim=1).clamp_min(eps)
    return (a * b).sum(dim=1) / (an * bn)

def plot_curve(x, ys: Dict[str, list], title: str, xlabel: str, ylabel: str, out_path: str):
    plt.figure()
    for k, v in ys.items():
        plt.plot(x, v, label=k)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_hist(data: np.ndarray, title: str, xlabel: str, out_path: str, bins: int = 50):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ----------------------------- Data -----------------------------

def make_loaders(dataset_root, batch_size=64, num_workers=4, image_size_hw=(240, 320), only_success=False):
    train_set = MarkerDataset(dataset_root, split="train", image_size_hw=image_size_hw, only_success=only_success)
    val_set = MarkerDataset(dataset_root, split="val", image_size_hw=image_size_hw, only_success=only_success)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_set, val_set, train_loader, val_loader


# ----------------------------- Model -----------------------------

class ResNetMLPPolicy(nn.Module):
    def __init__(self, out_dim=7):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # [B,512,1,1]
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        feat = self.backbone(x).flatten(1)  # [B,512]
        return self.head(feat)              # [B,7]

def freeze_backbone(model: ResNetMLPPolicy, freeze: bool = True):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze

def unfreeze_layer4_only(model: ResNetMLPPolicy):
    # backbone is a Sequential of resnet children excluding fc:
    # [conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool]
    # We want layer4 trainable, others frozen.
    for p in model.backbone.parameters():
        p.requires_grad = False
    # layer4 is index 7 in that sequence
    layer4 = model.backbone[7]
    for p in layer4.parameters():
        p.requires_grad = True


# ----------------------------- Train/Eval -----------------------------

@dataclass
class EvalStats:
    mse: float
    rmse_per_joint: np.ndarray
    action_norm_gt_mean: float
    action_norm_pred_mean: float
    cos_mean: float

@torch.no_grad()
def evaluate(model, loader, device) -> EvalStats:
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")

    total_mse_sum = 0.0
    total_n = 0
    # per joint
    se_sum = None  # [D]
    # norms and cosine
    gt_norms = []
    pred_norms = []
    cos_vals = []

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["delta_q"].to(device, non_blocking=True)  # [B,7]
        pred = model(images)

        mse_sum = loss_fn(pred, target).item()
        total_mse_sum += mse_sum
        bs = images.size(0)
        total_n += bs

        # per joint squared error
        se = (pred - target) ** 2  # [B,7]
        se_batch_sum = se.sum(dim=0).detach().cpu().numpy()  # [7]
        if se_sum is None:
            se_sum = se_batch_sum
        else:
            se_sum += se_batch_sum

        # norms
        gt_norms.append(torch.norm(target, dim=1).detach().cpu().numpy())
        pred_norms.append(torch.norm(pred, dim=1).detach().cpu().numpy())
        cos_vals.append(cosine_similarity(pred, target).detach().cpu().numpy())

        # show batch mse (mean) in bar
        pbar.set_postfix(mse=f"{(mse_sum / max(bs,1)):.4f}")

    if total_n == 0:
        return EvalStats(mse=float("nan"),
                         rmse_per_joint=np.full((7,), np.nan),
                         action_norm_gt_mean=float("nan"),
                         action_norm_pred_mean=float("nan"),
                         cos_mean=float("nan"))

    mse = total_mse_sum / (total_n * 7)  # average per-dim MSE
    rmse_per_joint = np.sqrt(se_sum / total_n)  # [7]

    gt_norms = np.concatenate(gt_norms) if len(gt_norms) else np.array([])
    pred_norms = np.concatenate(pred_norms) if len(pred_norms) else np.array([])
    cos_vals = np.concatenate(cos_vals) if len(cos_vals) else np.array([])

    return EvalStats(
        mse=float(mse),
        rmse_per_joint=rmse_per_joint.astype(float),
        action_norm_gt_mean=float(gt_norms.mean()) if gt_norms.size else float("nan"),
        action_norm_pred_mean=float(pred_norms.mean()) if pred_norms.size else float("nan"),
        cos_mean=float(cos_vals.mean()) if cos_vals.size else float("nan"),
    )

def train_one_epoch(model, loader, optimizer, device, grad_clip: float = 0.0) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["delta_q"].to(device, non_blocking=True)

        pred = model(images)
        loss = loss_fn(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


# ----------------------------- Early Stopping -----------------------------

class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, val: float) -> bool:
        # returns True if should stop
        if val + self.min_delta < self.best:
            self.best = val
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Managed BC Training (ResNet18 + MLP)")
    parser.add_argument("--dataset_root", type=str, default="/home/alphatok/ME5400/expert_data")
    parser.add_argument("--out_dir", type=str, default="./checkpoints_bc_managed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_head", type=float, default=1e-3, help="LR for head-only phase")
    parser.add_argument("--lr_finetune", type=float, default=1e-4, help="LR for finetune phase")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze_epochs", type=int, default=8, help="epochs to train head only")
    parser.add_argument("--unfreeze_layer4", action="store_true", help="finetune only layer4 instead of full backbone")
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    plots_dir = os.path.join(args.out_dir, "plots")
    ensure_dir(plots_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    image_size_hw = (args.image_height, args.image_width)

    # Data
    train_set, val_set, train_loader, val_loader = make_loaders(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size_hw=image_size_hw,
        only_success=args.only_success,
    )
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    # Model
    model = ResNetMLPPolicy(out_dim=7).to(device)

    # Phase 1: freeze backbone, train head
    freeze_backbone(model, freeze=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    early = EarlyStopper(patience=args.patience, min_delta=1e-6)

    # Logging
    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    metrics_jsonl = os.path.join(args.out_dir, "metrics.jsonl")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "phase", "train_loss",
            "val_mse", "val_cos", "val_norm_gt", "val_norm_pred",
            "lr"
        ] + [f"val_rmse_j{i+1}" for i in range(7)])
    # also dump config
    with open(os.path.join(args.out_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")

    # store history for plotting
    hist = {
        "epoch": [],
        "train_loss": [],
        "val_mse": [],
        "val_cos": [],
        "val_norm_gt": [],
        "val_norm_pred": [],
        "lr": [],
        "rmse_per_joint": [],
        "phase": [],
    }

    start_time = time.time()

    for epoch in range(args.epochs):
        # switch phase
        if epoch < args.freeze_epochs:
            phase = "head_only"
        else:
            phase = "finetune"
        
        # 在切换到finetune阶段时，解冻backbone并重新初始化optimizer
        if epoch == args.freeze_epochs:
            # unfreeze
            if args.unfreeze_layer4:
                unfreeze_layer4_only(model)
                print("✅ finetune: 仅解冻 layer4")
            else:
                freeze_backbone(model, freeze=False)
                print("✅ finetune: 解冻整个 backbone")

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_finetune)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
            early = EarlyStopper(patience=args.patience, min_delta=1e-6)

        # train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        # val stats
        val_stats = evaluate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} [{phase}] "
            f"train_loss={train_loss:.6f} | val_mse={val_stats.mse:.6f} "
            f"| cos={val_stats.cos_mean:.3f} | lr={lr:.2e}"
        )

        # scheduler uses val mse
        scheduler.step(val_stats.mse)

        # save best
        if val_stats.mse < best_val:
            best_val = val_stats.mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_stats.mse,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  [saved] {best_path}")

        # log
        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "val_mse": val_stats.mse,
            "val_cos": val_stats.cos_mean,
            "val_norm_gt": val_stats.action_norm_gt_mean,
            "val_norm_pred": val_stats.action_norm_pred_mean,
            "lr": lr,
            "val_rmse_per_joint": val_stats.rmse_per_joint.tolist(),
        }

        with open(metrics_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, phase, train_loss, val_stats.mse, val_stats.cos_mean,
                 val_stats.action_norm_gt_mean, val_stats.action_norm_pred_mean, lr]
                + list(val_stats.rmse_per_joint)
            )

        # update history
        hist["epoch"].append(epoch)
        hist["phase"].append(phase)
        hist["train_loss"].append(train_loss)
        hist["val_mse"].append(val_stats.mse)
        hist["val_cos"].append(val_stats.cos_mean)
        hist["val_norm_gt"].append(val_stats.action_norm_gt_mean)
        hist["val_norm_pred"].append(val_stats.action_norm_pred_mean)
        hist["lr"].append(lr)
        hist["rmse_per_joint"].append(val_stats.rmse_per_joint.copy())

        # plots every epoch (小数据集建议每次都画)
        plot_curve(
            hist["epoch"],
            {"train_loss": hist["train_loss"], "val_mse": hist["val_mse"]},
            title="Loss curves",
            xlabel="epoch",
            ylabel="loss",
            out_path=os.path.join(plots_dir, "loss_curves.png"),
        )
        # per joint RMSE
        rmse_arr = np.stack(hist["rmse_per_joint"], axis=0) if len(hist["rmse_per_joint"]) else None
        if rmse_arr is not None:
            plt.figure()
            for j in range(rmse_arr.shape[1]):
                plt.plot(hist["epoch"], rmse_arr[:, j], label=f"j{j+1}")
            plt.title("Val RMSE per joint")
            plt.xlabel("epoch")
            plt.ylabel("RMSE")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "val_rmse_per_joint.png"))
            plt.close()

        plot_curve(
            hist["epoch"],
            {"val_cos": hist["val_cos"]},
            title="Val cosine similarity (direction)",
            xlabel="epoch",
            ylabel="cosine",
            out_path=os.path.join(plots_dir, "val_cosine.png"),
        )

        plot_curve(
            hist["epoch"],
            {"gt_norm": hist["val_norm_gt"], "pred_norm": hist["val_norm_pred"]},
            title="Val action norm (||Δq||)",
            xlabel="epoch",
            ylabel="norm",
            out_path=os.path.join(plots_dir, "val_action_norm.png"),
        )

        # early stopping on val mse
        if early.step(val_stats.mse):
            print(f"🛑 Early stopping triggered at epoch {epoch} (best val_mse={early.best:.6f})")
            break

    elapsed = time.time() - start_time
    print(f"训练完成，用时 {elapsed/60:.1f} 分钟。best: {best_path} (val_mse={best_val:.6f})")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
