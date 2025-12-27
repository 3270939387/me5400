"""
MarkerDataset - 机器人视觉模仿学习数据集加载器

这个类负责从磁盘加载机器人任务执行数据，包括：
- 图像：RGB相机拍摄的场景图像
- 动作：机器人关节增量命令（delta_q_cmd）
- 状态：当前关节位置等信息

数据结构：
    DATA/
      ├─ metadata/episode_XXXX.json  (每个episode的元数据，包含所有时间步的状态和动作)
      └─ picture_data/episode_XXXX/  (每个episode的图像文件夹)
          └─ frame_YYYY.png          (每个时间步对应的图像)
"""

import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MarkerDataset(Dataset):
    """
    适配新的数据结构：
    DATA/
      ├─ metadata/episode_XXXX.json
      └─ picture_data/episode_XXXX/frame_YYYY.png
    
    工作流程：
    1. 初始化时扫描所有episodes，过滤无效数据，建立样本索引
    2. __getitem__时根据索引加载图像和动作，返回PyTorch张量
    """

    def __init__(self, dataset_root, split='train', image_size_hw=(240, 320), only_success=False):
        """
        初始化数据集，扫描所有episodes并建立样本索引
        
        Args:
            dataset_root: 数据集根目录路径，例如 "/home/alphatok/ME5400/DATA"
            split: 'train' 或 'val'，用于80/20划分训练集和验证集
            image_size_hw: 图像目标尺寸 (高度, 宽度)，例如 (240, 320) 表示240×320像素
            only_success: 如果为True，只加载success=True的episodes，过滤掉失败的任务
        
        工作流程：
            1. 检查目录结构
            2. 扫描并排序所有metadata文件
            3. 加载并过滤episodes（根据only_success）
            4. 固定seed打乱后按80/20划分train/val
            5. 遍历所有episodes，提取每个时间步的样本
            6. 对每个样本进行质量检查（动作有效性、图像存在性）
            7. 设置图像预处理管道
        """
        # ========== 第一步：设置路径 ==========
        self.dataset_root = dataset_root
        self.metadata_dir = os.path.join(dataset_root, "metadata")  # metadata文件夹路径
        self.picture_dir = os.path.join(dataset_root, "picture_data")  # 图片文件夹路径
        self.image_size = image_size_hw  # (H, W) 图像目标尺寸

        # ========== 第二步：验证目录结构 ==========
        # 在开始加载数据前，先检查必要的目录是否存在
        # 如果不存在，立即抛出异常，避免后续错误
        if not os.path.exists(self.metadata_dir) or not os.path.exists(self.picture_dir):
            raise ValueError(f"数据结构不符合预期，缺少 metadata/ 或 picture_data/ 目录: {dataset_root}")

        # ========== 第三步：收集所有metadata文件 ==========
        # 列出metadata目录下所有以"episode_"开头、".json"结尾的文件
        # 然后按episode编号排序（例如：episode_0000.json, episode_0001.json, ...）
        # 
        # 排序逻辑：
        #   "episode_0048.json" -> split("_") -> ["episode", "0048.json"]
        #   -> [1] -> "0048.json" -> split(".") -> ["0048", "json"]
        #   -> [0] -> "0048" -> int("0048") -> 48
        #   最终按数字0, 1, 2, ...排序
        meta_files = sorted(
            [f for f in os.listdir(self.metadata_dir) if f.startswith("episode_") and f.endswith(".json")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        if len(meta_files) == 0:
            raise ValueError(f"在 {self.metadata_dir} 中找不到任何 episode_*.json")

        # ========== 第四步：加载并过滤episodes ==========
        # 遍历每个metadata文件，尝试加载JSON内容
        # 如果only_success=True，只保留success=True的episode
        episodes = []
        for meta_name in meta_files:
            meta_path = os.path.join(self.metadata_dir, meta_name)
            try:
                # 尝试加载JSON文件
                # 使用try-except捕获可能的JSON解析错误（文件损坏、格式错误等）
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception as e:
                # 如果加载失败，打印警告但继续处理其他文件
                # 这样即使某些文件损坏，也不会中断整个数据集加载
                print(f"[WARN] 加载 {meta_path} 失败: {e}")
                continue

            # 如果only_success=True，检查episode是否成功
            # meta.get("success", False) 表示：如果metadata中没有"success"字段，默认返回False
            # 这样可以避免因字段缺失导致程序崩溃
            if only_success and not meta.get("success", False):
                continue  # 跳过失败的episode

            # 将成功加载的episode信息存入列表
            episodes.append({
                "meta_path": meta_path,  # 文件路径，用于后续提取episode编号
                "meta": meta,            # JSON内容，包含所有时间步的数据
            })

        if len(episodes) == 0:
            raise ValueError("没有可用的 episodes（可能全部被过滤掉）")

        # ========== 第五步：数据集划分（Train/Val Split） ==========
        # 使用固定随机种子（42）打乱episodes，然后按80/20划分
        # 
        # 为什么固定seed？
        #   - 保证每次运行程序，划分结果完全一致（可复现性）
        #   - 便于对比不同模型在相同数据上的表现
        # 
        # 为什么80/20？
        #   - 训练集需要更多数据来学习模式
        #   - 验证集用于评估模型泛化能力，不需要太多数据
        # 
        # 示例：假设有100个episodes
        #   split_idx = int(0.8 * 100) = 80
        #   train: episodes[0:80] (80个)
        #   val: episodes[80:100] (20个)
        rng = random.Random(42)  # 固定随机种子
        rng.shuffle(episodes)     # 打乱顺序

        num_episodes = len(episodes)
        split_idx = int(0.8 * num_episodes)  # 80%的位置
        if split == 'train':
            episodes = episodes[:split_idx]  # 前80%作为训练集
        else:
            episodes = episodes[split_idx:]  # 后20%作为验证集

        # ========== 第六步：加载所有样本（Samples） ==========
        # 遍历每个episode的所有时间步，提取图像路径和动作数据
        # 对每个样本进行质量检查，只保留有效的样本
        self.samples = []  # 存储所有有效样本的列表
        invalid_actions = 0  # 统计无效动作的数量
        
        for ep in episodes:
            meta = ep["meta"]
            
            # 从文件名中提取episode编号（而不是从metadata中读取）
            # 为什么？因为文件名是数据结构的"真相来源"（source of truth）
            # metadata中的episode_idx可能在数据合并/重命名后过时
            # 例如："episode_0048.json" -> 提取出 48
            ep_idx = int(os.path.basename(ep["meta_path"]).split("_")[1].split(".")[0])

            # 获取该episode的所有时间步数据
            # 每个step_data包含：状态、动作、图像路径等信息
            steps = meta.get("steps", [])
            
            # 构建对应的图片文件夹路径
            # 例如：episode_0048 -> "/DATA/picture_data/episode_0048"
            ep_picture_dir = os.path.join(self.picture_dir, f"episode_{ep_idx:04d}")
            if not os.path.exists(ep_picture_dir):
                print(f"[WARN] 图片文件夹不存在: {ep_picture_dir}")
                continue  # 如果图片文件夹不存在，跳过整个episode

            # 遍历该episode的每个时间步
            for step_data in steps:
                # ========== 动作有效性检查 ==========
                # 这是数据质量保证的关键步骤！
                # 
                # 1. 提取动作向量
                #    优先使用delta_q（当前命令 - 当前状态，经典的BC监督信号）
                #    如果没有，则回退到delta_q_cmd（兼容旧数据）
                action = step_data.get("action", {})
                delta_q_raw = action.get("delta_q", action.get("delta_q_cmd", []))
                delta_q_arr = np.array(delta_q_raw, dtype=np.float32)
                
                # 2. 检查动作是否为空或包含NaN/Inf
                #    np.isfinite() 检查每个元素是否是有限数（不是NaN或Inf）
                #    .all() 确保所有元素都是有限数
                #    如果动作包含NaN/Inf，训练时loss会变成NaN，导致训练失败
                if delta_q_arr.size == 0 or not np.isfinite(delta_q_arr).all():
                    invalid_actions += 1
                    continue  # 跳过这个无效样本
                
                # 3. 检查动作值是否过大（避免梯度爆炸）
                #    如果动作值过大（>1000），可能导致梯度爆炸，模型无法收敛
                #    这是防御式编程的体现：在数据加载阶段就过滤掉坏数据
                if np.abs(delta_q_arr).max() > 1e3:
                    invalid_actions += 1
                    continue  # 跳过这个极端值样本

                # ========== 图像路径验证 ==========
                # 从step_data中获取图像文件名
                # 如果没有image_path字段，使用默认格式：frame_XXXX.png
                img_filename = step_data.get("image_path", f"frame_{step_data.get('step', 0):04d}.png")
                img_path = os.path.join(ep_picture_dir, img_filename)
                
                # 检查图像文件是否存在
                # 某些图像可能被删除或损坏，提前跳过避免后续加载失败
                if not os.path.exists(img_path):
                    print(f"[WARN] 图像不存在，跳过: {img_path}")
                    continue
                
                # ========== 添加有效样本 ==========
                # 只有通过所有检查的样本才会被添加到self.samples
                # 每个样本包含：
                #   - image_path: 图像文件的完整路径
                #   - step_data: 该时间步的所有原始数据（状态、动作等）
                self.samples.append({
                    "image_path": img_path,
                    "step_data": step_data
                })

        # ========== 验证样本数量 ==========
        if len(self.samples) == 0:
            raise ValueError("未能从选定的 episodes 中加载到任何样本")

        # 打印统计信息，便于了解数据集情况
        print(f"[INFO] {split} 数据集: {len(episodes)} 个 episodes, {len(self.samples)} 个样本")
        if invalid_actions > 0:
            print(f"[INFO] 无效动作样本已跳过: {invalid_actions}")

        # ========== 第七步：设置图像预处理管道 ==========
        # 这个管道会在__getitem__时对每张图像应用以下变换：
        # 
        # 1. Resize: 将图像调整到指定尺寸（例如240×320）
        #    - 统一图像尺寸，便于批处理（batch processing）
        #    - 不同图像可能有不同尺寸，必须统一
        # 
        # 2. ToTensor: 将PIL Image转换为PyTorch张量
        #    - PIL Image是Python对象，无法直接用于神经网络
        #    - 自动将像素值从[0, 255]缩放到[0.0, 1.0]
        #    - 自动将H×W×C转换为C×H×W（通道优先格式）
        # 
        # 3. Normalize: 使用ImageNet的均值和标准差进行归一化
        #    - 如果使用预训练的ResNet等模型，它们是在ImageNet上训练的
        #    - 使用相同的归一化参数，可以更好地利用预训练权重
        #    - 归一化公式：normalized = (pixel - mean) / std
        #    - 归一化后像素值大约在[-2.118, 2.249]范围内
        # 
        # 示例变换过程：
        #   原始图像: PIL Image, 尺寸可能不同, 像素值[0, 255]
        #   -> Resize: PIL Image, 240×320, 像素值[0, 255]
        #   -> ToTensor: torch.Tensor, [3, 240, 320], 像素值[0.0, 1.0]
        #   -> Normalize: torch.Tensor, [3, 240, 320], 像素值约[-2.118, 2.249]
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),  # 调整尺寸
            transforms.ToTensor(),               # 转换为张量并缩放
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值（RGB三通道）
                                 std=[0.229, 0.224, 0.225])  # ImageNet标准差（RGB三通道）
        ])

    def __len__(self):
        """
        返回数据集中样本的总数
        
        PyTorch的DataLoader需要这个方法来确定数据集大小
        例如：1000个样本，batch_size=32 -> 需要32个batch（最后一个可能不足32）
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本，返回PyTorch张量格式的数据
        
        这是PyTorch Dataset的核心方法，DataLoader会调用它来获取训练数据
        
        Args:
            idx: 样本索引（0, 1, 2, ..., len(self.samples)-1）
        
        Returns:
            dict: 包含以下键值对：
                - "image": torch.Tensor, 形状 [3, H, W]，预处理后的图像
                - "delta_q": torch.Tensor, 形状 [7]，动作向量（关节增量命令）
                - "joint_positions": torch.Tensor, 形状 [7]，当前关节位置
                - "raw": dict，原始step_data（便于调试和扩展）
        
        工作流程：
            1. 根据索引获取样本信息
            2. 加载并预处理图像
            3. 提取并处理动作向量
            4. 提取关节位置（可选）
            5. 转换为PyTorch张量并返回
        """
        # ========== 第一步：获取样本信息 ==========
        sample = self.samples[idx]  # 从预加载的样本列表中获取
        step_data = sample["step_data"]  # 提取该时间步的所有原始数据

        # ========== 第二步：加载和预处理图像 ==========
        img_path = sample["image_path"]
        try:
            # 使用PIL打开图像文件
            # .convert('RGB')确保图像是RGB格式（即使原图是灰度图或RGBA）
            # 模型期望输入是3通道（RGB），统一格式避免维度不匹配错误
            image = Image.open(img_path).convert('RGB')
            
            # 应用预处理管道（resize、归一化等）
            # 结果：torch.Tensor, 形状 [3, H, W], 像素值已归一化
            image = self.transform(image)
        except Exception as e:
            # 如果加载失败，抛出异常（而不是返回黑图）
            # 为什么？因为返回黑图（全零）会污染训练数据
            # 抛出异常可以让DataLoader跳过这个样本或终止训练，便于发现数据问题
            raise RuntimeError(f"加载图像失败 {img_path}: {e}") from e

        # ========== 第三步：提取动作向量 ==========
        # 动作目标：优先使用delta_q（当前命令 - 当前状态，经典的BC监督信号）
        # 如果没有，则回退到delta_q_cmd（兼容旧数据）
        action = step_data.get("action", {})
        delta_q = action.get("delta_q", action.get("delta_q_cmd", []))
        delta_q = np.array(delta_q, dtype=np.float32)
        
        # 确保动作向量长度为7（7个关节）
        # 如果长度不足7，用0填充；如果超过7，截断到7
        # 为什么？机器人有7个关节（Panda机械臂），模型输出固定为7维
        # 
        # np.pad示例：
        #   delta_q = [1.0, 2.0] (只有2维)
        #   np.pad(delta_q, (0, 5), 'constant') -> [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #   [:7] 确保不超过7维
        if delta_q.shape[0] != 7:
            delta_q = np.pad(delta_q, (0, max(0, 7 - delta_q.shape[0])), 'constant')[:7]

        # ========== 第四步：提取关节位置（可选信息） ==========
        # 某些模型可能需要当前状态作为输入（例如：状态-动作联合预测）
        # 这里提取当前关节位置q（7个关节的角度）
        state = step_data.get("state", {})
        q = np.array(state.get("q", [0.0] * 7), dtype=np.float32)  # 如果没有，默认全零
        if q.shape[0] != 7:
            q = np.pad(q, (0, max(0, 7 - q.shape[0])), 'constant')[:7]  # 同样pad/cut到7维

        # ========== 第五步：返回数据字典 ==========
        # 将处理好的数据打包成字典返回
        # 使用字典的好处：
        #   - 清晰的键值对，便于访问
        #   - 可以返回多个数据（图像、动作、状态等）
        #   - 模型可以按需使用不同的字段
        # 
        # torch.from_numpy() 将numpy数组转换为PyTorch张量
        # 注意：它会共享内存，提高效率（不会复制数据）
        return {
            "image": image,  # torch.Tensor, [3, H, W]，预处理后的图像
            "delta_q": torch.from_numpy(delta_q),  # torch.Tensor, [7]，动作向量
            "joint_positions": torch.from_numpy(q),  # torch.Tensor, [7]，关节位置
            "raw": step_data  # dict，原始step_data（便于调试和扩展）
        }