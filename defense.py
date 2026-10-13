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


#!/usr/bin/env python
"""MNIST 数据投毒防御：随机平移数据增强。

该实现遵循课程项目给定的防御接口。它保留原始训练样本，并为每个样本
生成 ``copies`` 份随机平移副本，以降低模型对少量投毒样本的敏感度。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _validate_inputs(
    images: np.ndarray,
    labels: np.ndarray,
    max_shift: int,
    copies: int,
) -> tuple[np.ndarray, np.ndarray]:
    """检查并规范化防御函数的输入。"""
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if images.ndim != 4:
        raise ValueError(
            f"images 应为 [N, C, H, W] 四维数组，实际形状为 {images.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"labels 应为 [N] 一维数组，实际形状为 {labels.shape}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            "图像和标签数量不一致："
            f"{images.shape[0]} 张图像，{labels.shape[0]} 个标签"
        )
    if isinstance(max_shift, (bool, np.bool_)) or not isinstance(
        max_shift, (int, np.integer)
    ):
        raise TypeError("max_shift 必须是整数")
    if isinstance(copies, (bool, np.bool_)) or not isinstance(
        copies, (int, np.integer)
    ):
        raise TypeError("copies 必须是整数")
    if max_shift < 0:
        raise ValueError("max_shift 不能小于 0")
    if copies < 0:
        raise ValueError("copies 不能小于 0")

    # copy 可确保本函数不会修改调用者持有的原数组。
    return images.copy(), labels.copy()


def defend(
    images: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    max_shift: int = 2,
    copies: int = 1,
) -> dict:
    """使用随机平移数据增强防御训练集投毒。

    Parameters
    ----------
    images:
        可能被投毒的图像，形状为 ``[N, 1, 28, 28]``。
    labels:
        与图像对应的标签，形状为 ``[N]``。
    seed:
        随机种子；相同输入与种子会产生相同结果。
    max_shift:
        水平和垂直方向的最大平移像素数。每个方向均从闭区间
        ``[-max_shift, max_shift]`` 独立均匀采样。
    copies:
        为每个原始样本生成的增强副本数。默认值 1 会把数据集扩充为两倍。

    Returns
    -------
    dict
        ``images`` 和 ``labels`` 为增强后的训练集；``n_augmented`` 为新增
        样本数。输出图像类型为 float32，标签类型为 int64。
    """
    images, labels = _validate_inputs(images, labels, max_shift, copies)
    n_samples = images.shape[0]
    rng = np.random.default_rng(seed)

    image_parts = [images]
    label_parts = [labels]

    for _ in range(copies):
        augmented = images.copy()
        shifts = rng.integers(
            -max_shift,
            max_shift + 1,
            size=(n_samples, 2),
        )

        for index, (dy, dx) in enumerate(shifts):
            # 单个样本形状为 [C, H, W]，所以空间轴为 1 和 2。
            augmented[index] = np.roll(
                augmented[index],
                shift=(int(dy), int(dx)),
                axis=(1, 2),
            )

        image_parts.append(augmented)
        label_parts.append(labels.copy())

    defended_images = np.concatenate(image_parts, axis=0).astype(
        np.float32, copy=False
    )
    defended_labels = np.concatenate(label_parts, axis=0).astype(
        np.int64, copy=False
    )

    return {
        "images": defended_images,
        "labels": defended_labels,
        "n_augmented": n_samples * copies,
    }


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """从项目约定的 NPZ 文件读取图像与标签。"""
    with np.load(path, allow_pickle=False) as data:
        if "images" not in data or "labels" not in data:
            raise ValueError(f"{path} 必须包含 images 和 labels 两个数组")
        return data["images"], data["labels"]


def main() -> None:
    """命令行入口：读取投毒数据并保存增强后的数据。"""
    parser = argparse.ArgumentParser(description="随机平移数据增强防御")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("code/data/poisoned.npz"),
        help="输入的投毒数据（默认：code/data/poisoned.npz）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("code/data/defended.npz"),
        help="防御后数据的保存位置（默认：code/data/defended.npz）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-shift", type=int, default=2, help="最大平移像素数")
    parser.add_argument("--copies", type=int, default=1, help="每个样本的副本数")
    args = parser.parse_args()

    images, labels = _load_npz(args.input)
    result = defend(
        images,
        labels,
        seed=args.seed,
        max_shift=args.max_shift,
        copies=args.copies,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        images=result["images"],
        labels=result["labels"],
    )
    print(
        "[defence] 数据增强完成："
        f"{len(labels)} -> {len(result['labels'])} 个样本；已保存到 {args.output}"
    )


if __name__ == "__main__":
    main()


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
