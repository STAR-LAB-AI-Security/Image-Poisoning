#!/usr/bin/env python
"""防御代码 —— Baseline：数据增强（Data Augmentation）。

原理：对可能被投毒的训练集做鲁棒训练类防御。为每个样本生成 1 份随机平移
      副本（水平/垂直方向 ±2 像素，np.roll 实现），将训练集扩充为 2 倍
      规模后统一训练。通过引入平移不变性先验、扩大有效样本量，提升模型
      泛化能力、降低对个别投毒样本的敏感度，从而恢复模型在干净测试集上
      的性能。实现简单、模型无关，对 MLP/CNN 均有效。

接口：
    defend(images, labels, seed=42, max_shift=2, copies=1)
        -> {"images": [2N,1,28,28], "labels": [2N]}

用法：
    python defense.py                  # 读取投毒数据，输出 data/defended.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from dataset import load_npz, save_npz


import numpy as np

def defend(
    images: np.ndarray, 
    labels: np.ndarray, 
    seed: int = 42, 
    max_shift: int = 2, 
    copies: int = 1
) -> dict:
    """
    防御策略：数据增强 + 样本重加权
    1. 随机平移图像，打乱攻击者可能设置的固定像素触发器
    2. 计算样本与类内均值的距离，给异常样本降权，减轻投毒影响
    """
    
    # 固定随机种子，保证评测结果可复现
    rng = np.random.default_rng(seed)
    
    N, C, H, W = images.shape
    
    # 保存原始数据和增强后的数据
    augmented_images = [images]
    augmented_labels = [labels]
    
    # 数据增强（随机平移）
    if copies > 0:
        for _ in range(copies):
            shifted_images = np.zeros_like(images)
            for i in range(N):
                # 随机生成上下左右的偏移量
                dx = rng.integers(-max_shift, max_shift + 1)
                dy = rng.integers(-max_shift, max_shift + 1)
                
                # 对单张图做平移
                img = images[i, 0]  # 取出[H, W]大小的单通道图
                shifted = np.roll(img, shift=(dx, dy), axis=(0, 1))
                
                # np.roll 会把移出边界的内容循环补到另一边，这里把补过来的边缘置0
                if dx > 0:
                    shifted[:dx, :] = 0
                elif dx < 0:
                    shifted[dx:, :] = 0
                if dy > 0:
                    shifted[:, :dy] = 0
                elif dy < 0:
                    shifted[:, dy:] = 0
                    
                shifted_images[i, 0] = shifted
                
            augmented_images.append(shifted_images)
            augmented_labels.append(labels)
            
    # 拼接原始图和增强图
    final_images = np.concatenate(augmented_images, axis=0)
    final_labels = np.concatenate(augmented_labels, axis=0)
    
    # 计算样本权重以防投毒
    sample_weights = np.ones(len(final_labels), dtype=np.float32)
    
    # 按类别求均值图，然后算每个样本跟均值图的欧氏距离
    for cls in np.unique(final_labels):
        cls_mask = (final_labels == cls)
        cls_images = final_images[cls_mask]
        if len(cls_images) > 0:
            cls_mean = np.mean(cls_images, axis=0)  # 这个类别的平均长相
            
            # 计算欧氏距离，距离越远说明长得越不合群
            distances = np.linalg.norm(cls_images.reshape(len(cls_images), -1) - cls_mean.reshape(1, -1), axis=1)
            
            # 距离越远权重越低，用倒数做个映射，然后归一化
            weights = 1.0 / (1.0 + distances)
            weights = weights / np.mean(weights)
            sample_weights[cls_mask] = weights

    # 把权重截断在0.1到2.0之间，防止个别极端数据把模型带偏
    sample_weights = np.clip(sample_weights, 0.1, 2.0)
    
    # 确保数据类型符合模型接口要求
    final_images = final_images.astype(np.float32)
    final_labels = final_labels.astype(np.int64)
    
    # 返回清洗后的数据和权重
    return {
        "images": final_images, 
        "labels": final_labels,
        "sample_weights": sample_weights,
        "n_augmented": int(copies * N)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据增强防御（Baseline）")
    parser.add_argument("--input", default="data/poisoned.npz", help="输入投毒数据")
    parser.add_argument("--output", default="data/defended.npz", help="输出路径")
    parser.add_argument("--max-shift", type=int, default=2, help="随机平移最大像素数")
    parser.add_argument("--copies", type=int, default=1, help="每样本副本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    poisoned = load_npz(args.input)
    out = defend(poisoned["images"], poisoned["labels"],
                 seed=args.seed, max_shift=args.max_shift, copies=args.copies)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_npz(args.output, out["images"], out["labels"])
    print(f"[defense] 数据增强完成：{len(poisoned['labels'])} -> {len(out['labels'])} 个样本")
