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


def defend(images, labels, seed=42, max_shift=2, copies=1):
    """通过随机平移数据增强降低少量投毒样本对模型训练的影响。"""
    # 放在函数内部导入，使本函数可以直接粘贴到受管评测模板中。
    import numpy as np

    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if images.ndim != 4:
        raise ValueError("images 必须是形状为 [N, C, H, W] 的四维数组")
    if labels.ndim != 1:
        raise ValueError("labels 必须是一维数组")
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images 和 labels 的样本数量必须相同")
    if max_shift < 0:
        raise ValueError("max_shift 不能小于 0")
    if copies < 0:
        raise ValueError("copies 不能小于 0")

    # 复制输入，避免修改评测程序传入的原始训练数据。
    images = images.copy()
    labels = labels.copy()

    rng = np.random.default_rng(seed)
    n_samples = images.shape[0]
    image_parts = [images]
    label_parts = [labels]

    for _ in range(copies):
        augmented = images.copy()
        shifts = rng.integers(
            -max_shift,
            max_shift + 1,
            size=(n_samples, 2),
        )

        for index in range(n_samples):
            dy = int(shifts[index, 0])
            dx = int(shifts[index, 1])
            augmented[index] = np.roll(
                augmented[index],
                shift=(dy, dx),
                axis=(1, 2),
            )

        image_parts.append(augmented)
        label_parts.append(labels.copy())

    defended_images = np.concatenate(image_parts, axis=0).astype(np.float32)
    defended_labels = np.concatenate(label_parts, axis=0).astype(np.int64)

    return {
        "images": defended_images,
        "labels": defended_labels,
        "n_augmented": n_samples * copies,
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
