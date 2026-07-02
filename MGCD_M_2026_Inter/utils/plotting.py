"""Reusable plotting helpers for all course labs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_class_distribution(
    series,
    labels: list[str] | None = None,
    title: str = "Class distribution",
    ax=None,
):
    """Bar chart of value counts for a binary (or multi-class) target Series."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    counts = series.value_counts().sort_index()
    if labels is None:
        labels = [str(x) for x in counts.index]

    bars = ax.bar(labels, counts.values,
                  color=_PALETTE[: len(labels)], edgecolor="white")
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.005,
            f"{count:,}  ({count / counts.sum():.1%})",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return ax


def plot_confusion_matrix(
    cm,
    class_names: list[str] | None = None,
    title: str = "Confusion matrix",
    ax=None,
):
    """Annotated heatmap of a 2-D confusion matrix (from sklearn)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    ticks = np.arange(n)
    ax.set(
        xticks=ticks, xticklabels=class_names,
        yticks=ticks, yticklabels=class_names,
        xlabel="Predicted label", ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    return ax


def plot_training_curves(
    train_scores: list[float],
    val_scores: list[float],
    metric: str = "Score",
    title: str = "Training curves",
):
    """Line chart comparing train vs. validation performance across iterations."""
    _, ax = plt.subplots(figsize=(6, 4))
    x = range(1, len(train_scores) + 1)
    ax.plot(x, train_scores, label="Train", marker="o", markersize=4, color=_PALETTE[0])
    ax.plot(x, val_scores, label="Validation", marker="s", markersize=4,
            linestyle="--", color=_PALETTE[1])
    ax.set_xlabel("Training size / iteration")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return ax


def plot_feature_importance(
    importances,
    feature_names: list[str],
    top_n: int = 15,
    title: str = "Feature importances",
):
    """Horizontal bar chart of the top-N most important features."""
    imp = np.array(importances)
    idx = np.argsort(imp)[::-1][:top_n][::-1]

    _, ax = plt.subplots(figsize=(6, max(3, top_n * 0.35)))
    ax.barh(
        [feature_names[i] for i in idx],
        imp[idx],
        color=_PALETTE[0], edgecolor="white",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return ax
