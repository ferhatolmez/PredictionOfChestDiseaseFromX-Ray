"""Train the from-scratch MLP on the chest X-rays.

Uses the same patient-grouped split (seed 0) as the CNN, so the numbers can
be compared. Each image is reduced to 32x32 gray values (1,024 inputs).

Usage:
    python -m from_scratch.train_on_xrays
"""

import copy
import json
from pathlib import Path

import numpy as np

from chestxray.config import Config
from chestxray.data import load_dataset, split_by_patient
from chestxray.duplicates import duplicate_sets, merge_groups
from chestxray.evaluate import compute_metrics
from from_scratch.neural_net import MLP, SGDMomentum, train

SMALL = 32  # 224 / 7


def downsample(images: np.ndarray, size: int = SMALL) -> np.ndarray:
    """Average 224x224x3 images into size x size gray blocks, one row per image."""
    n, h, w, _ = images.shape
    block = h // size
    out = np.empty((n, size * size), dtype=np.float32)
    for start in range(0, n, 512):  # in chunks, to keep memory low
        chunk = images[start : start + 512].astype(np.float32)
        chunk = chunk.reshape(-1, size, block, size, block, 3).mean(axis=(2, 4, 5))
        out[start : start + 512] = chunk.reshape(len(chunk), -1)
    return out


def balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Return one weight per example so that every class counts equally.

    w_c = N / (C * n_c): rare classes get larger weights.
    """
    counts = np.bincount(y, minlength=n_classes)
    return (len(y) / (n_classes * counts))[y]


def main(epochs: int = 40) -> None:
    """Train, keep the epoch with the best validation macro F1, test once."""
    config = Config()
    data = load_dataset(config)
    class_index = {name: i for i, name in enumerate(config.classes)}
    y = np.array([class_index[label] for label in data.labels])
    groups = merge_groups(data.groups, duplicate_sets(data.images))
    split = split_by_patient(data.labels, groups, 0.15, 0.15, seed=0)

    x = downsample(data.images)
    mean, std = x[split.train].mean(axis=0), x[split.train].std(axis=0) + 1e-6
    x = (x - mean) / std  # standardise with training statistics only

    model = MLP([SMALL * SMALL, 256, 64, len(config.classes)], seed=0)
    optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
    weights = balanced_weights(y[split.train], len(config.classes))

    history, best_f1, best_model = [], -1.0, None
    for epoch in range(epochs):
        loss = train(
            model,
            x[split.train],
            y[split.train],
            epochs=1,
            optimizer=optimizer,
            weights=weights,
            seed=epoch,
        )[0]
        val = compute_metrics(
            y[split.val], model.predict_proba(x[split.val]), config.classes
        )
        history.append(
            {"epoch": epoch + 1, "loss": loss, "val_macro_f1": val["macro_f1"]}
        )
        print(
            f"epoch {epoch + 1:2d}  loss {loss:.4f}  val macro F1 {val['macro_f1']:.4f}"
        )
        if val["macro_f1"] > best_f1:
            best_f1, best_model = val["macro_f1"], copy.deepcopy(model)

    probs = best_model.predict_proba(x[split.test])
    metrics = compute_metrics(y[split.test], probs, config.classes)
    print(
        f"test: accuracy {metrics['accuracy']:.4f}, "
        f"macro F1 {metrics['macro_f1']:.4f}, "
        f"ROC-AUC {metrics['roc_auc_macro']:.4f}"
    )
    out = Path("results/mlp_from_scratch.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"history": history, "test": metrics}, indent=2))


if __name__ == "__main__":
    main()
