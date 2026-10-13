import numpy as np

NUM_CLASSES = 10


def attack(
    images: np.ndarray,
    labels: np.ndarray,
    poison_rate: float = 0.05,
    seed: int = 42,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """集中式标签翻转 + 原型引导投毒。

    5% 预算下随机对称标签噪声对 MNIST 是弱攻击（ASR≈1.2%）。本方法：
      1) 用类内均值原型找每个类的「最易混淆异类」作为翻转目标；
      2) 把预算集中投向少数几个「类内异质性最高」的类，使错误信号形成
         高密度簇，显著拉偏决策边界。
    保证：输出规模一致、修改数 <= 预算、结果可复现、纯 CPU 秒级完成。
    """
    images = np.asarray(images, dtype=np.float32).copy()
    labels = np.asarray(labels, dtype=np.int64).copy()
    n = len(labels)
    if n == 0:
        return {"images": images, "labels": labels}

    budget = max(1, min(int(np.floor(poison_rate * n)), n))
    flat = images.reshape(n, -1).astype(np.float64)
    cls_idx = [np.flatnonzero(labels == c) for c in range(num_classes)]

    cent = np.zeros((num_classes, flat.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        if len(cls_idx[c]) > 0:
            cent[c] = flat[cls_idx[c]].mean(axis=0)
    dist = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    target_of = dist.argmin(axis=1).astype(np.int64)

    hetero = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        ci = cls_idx[c]
        if len(ci) == 0:
            continue
        sub = flat[ci]
        if len(sub) > 400:
            sub = sub[::max(1, len(sub) // 400)][:400]
        hetero[c] = float(np.linalg.norm(sub - cent[c], axis=1).mean())
    src_order = np.argsort(-hetero)

    out_labels = labels.copy()
    remaining = budget
    for c in src_order:
        if remaining <= 0:
            break
        c = int(c)
        ci = cls_idx[c]
        if len(ci) == 0:
            continue
        take = min(remaining, len(ci))
        out_labels[ci[:take]] = int(target_of[c])
        remaining -= take

    return {"images": images, "labels": out_labels}