import numpy as np


def defend(
    images: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    max_shift: int = 2,
    copies: int = 1,
) -> dict:
    """数据增强防御（Baseline）：每样本附加 copies 份 ±max_shift 像素随机平移副本。"""
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = images.shape[0]

    imgs = [images]
    lbls = [labels]
    for _ in range(copies):
        aug = images.copy()
        sh = rng.integers(-max_shift, max_shift + 1, size=(n, 2))
        for i in range(n):
            dy, dx = int(sh[i, 0]), int(sh[i, 1])
            aug[i] = np.roll(aug[i], shift=(dy, dx), axis=(1, 2))
        imgs.append(aug)
        lbls.append(labels.copy())

    return {"images": np.concatenate(imgs, axis=0).astype(np.float32),
            "labels": np.concatenate(lbls, axis=0).astype(np.int64)}