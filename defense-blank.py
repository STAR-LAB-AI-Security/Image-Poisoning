import numpy as np


def defend(
    images: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    max_shift: int = 2,
    copies: int = 1,
) -> dict:
    """数据增强防御：每个样本附加 copies 份随机平移副本。"""
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = images.shape[0]
    new_imgs = [images]
    new_lbls = [labels]
    for _ in range(copies):
        aug = images.copy()
        sh = rng.integers(-max_shift, max_shift + 1, size=(n, 2))
        for i in range(n):
            dy, dx = int(sh[i, 0]), int(sh[i, 1])
            aug[i] = np.roll(aug[i], shift=(dy, dx), axis=(1, 2))
        new_imgs.append(aug)
        new_lbls.append(labels.copy())
    return {
        "images": np.concatenate(new_imgs, axis=0).astype(np.float32),
        "labels": np.concatenate(new_lbls, axis=0).astype(np.int64),
    }
