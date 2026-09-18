import numpy as np

NUM_CLASSES = 10


def attack(
    images: np.ndarray,
    labels: np.ndarray,
    poison_rate: float = 0.05,
    seed: int = 42,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """随机标签翻转攻击。"""
    images = np.asarray(images, dtype=np.float32).copy()
    labels = np.asarray(labels, dtype=np.int64).copy()
    n_train = len(labels)
    budget = max(1, int(poison_rate * n_train))
    budget = min(budget, n_train)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_train, size=budget, replace=False)
    for i in idx:
        old = int(labels[i])
        candidates = [c for c in range(num_classes) if c != old]
        labels[i] = rng.choice(candidates)
    return {"images": images, "labels": labels, "n_modified": budget}
