"""Metrics and plots for a trained model."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray, probs: np.ndarray, class_names: tuple[str, ...]
) -> dict:
    """Return accuracy, macro F1, macro ROC-AUC, per-class report and confusion matrix.

    Args:
        y_true: integer class indices, shape (N,).
        probs: predicted class probabilities, shape (N, C).
        class_names: names in index order.
    """
    y_pred = probs.argmax(axis=1)
    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        # one-vs-rest: each class against all others, then averaged
        "roc_auc_macro": roc_auc_score(y_true, probs, multi_class="ovr"),
        "report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(class_names))
        ).tolist(),
    }


def plot_history(history: dict, phase1_epochs: int, ax=None):
    """Plot training and validation accuracy and loss over both phases."""
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for axis, metric in zip(ax, ("accuracy", "loss"), strict=True):
        axis.plot(epochs, history[metric], label=f"training {metric}")
        axis.plot(epochs, history[f"val_{metric}"], label=f"validation {metric}")
        axis.axvline(
            phase1_epochs + 0.5, color="gray", ls="--", label="fine-tuning starts"
        )
        axis.set(xlabel="epoch", ylabel=metric, title=f"Model {metric}")
        axis.legend()
    return ax


def plot_confusion_matrix(matrix, class_names: tuple[str, ...], ax=None):
    """Draw the confusion matrix with counts in each cell."""
    matrix = np.asarray(matrix)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    for (row, col), count in np.ndenumerate(matrix):
        color = "white" if count > matrix.max() / 2 else "black"
        ax.text(col, row, count, ha="center", va="center", color=color)
    ticks = range(len(class_names))
    ax.set_xticks(ticks, class_names, rotation=30)
    ax.set_yticks(ticks, class_names)
    ax.set(xlabel="predicted class", ylabel="true class", title="Confusion matrix")
    return ax


def plot_roc_curves(y_true: np.ndarray, probs: np.ndarray, class_names, ax=None):
    """Draw one-vs-rest ROC curves, one per class, with their AUC."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for index, name in enumerate(class_names):
        is_class = y_true == index
        fpr, tpr, _ = roc_curve(is_class, probs[:, index])
        auc = roc_auc_score(is_class, probs[:, index])
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", label="random guess")
    ax.set(
        xlabel="false positive rate", ylabel="true positive rate", title="ROC curves"
    )
    ax.legend()
    return ax
