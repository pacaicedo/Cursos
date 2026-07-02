"""Exercise validation helpers.

Each function prints a ✓ or ✗ line so students get immediate feedback
without crashing the notebook on a wrong answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ok(tag: str, extra: str = ""):
    msg = f"  ✓  [{tag}] Correct!"
    if extra:
        msg += f"  {extra}"
    print(msg)


def _fail(tag: str, msg: str):
    print(f"  ✗  [{tag}] {msg}")


# ── DataFrame checks ─────────────────────────────────────────────────────────

def check_dataframe(
    tag: str,
    df,
    min_rows: int | None = None,
    required_cols: list[str] | None = None,
    null_free_cols: list[str] | None = None,
):
    """Validate basic DataFrame properties."""
    if not isinstance(df, pd.DataFrame):
        _fail(tag, f"Expected a DataFrame, got {type(df).__name__}.")
        return
    if min_rows is not None and len(df) < min_rows:
        _fail(tag, f"Expected at least {min_rows:,} rows, got {len(df):,}.")
        return
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            _fail(tag, f"Missing column(s): {missing}")
            return
    if null_free_cols:
        for col in null_free_cols:
            if col in df.columns and df[col].isnull().any():
                n = df[col].isnull().sum()
                _fail(tag, f"Column '{col}' still has {n} null value(s).")
                return
    _ok(tag, f"shape={df.shape}")


# ── Split checks ─────────────────────────────────────────────────────────────

def check_split(
    tag: str,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    expected_fracs: tuple[float, ...] = (0.70, 0.15, 0.15),
    tol: float = 0.05,
):
    """Validate approximate proportions of a train / val / test split."""
    total = len(X_train) + len(X_val) + len(X_test)
    if total == 0:
        _fail(tag, "All splits are empty.")
        return

    fracs = [len(X_train) / total, len(X_val) / total, len(X_test) / total]
    for actual, expected, name in zip(fracs, expected_fracs, ["train", "val", "test"]):
        if abs(actual - expected) > tol:
            _fail(tag, f"{name} split is {actual:.1%}, expected ~{expected:.1%}.")
            return

    for X, y, name in [(X_train, y_train, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
        if len(X) != len(y):
            _fail(tag, f"X_{name} and y_{name} have different lengths ({len(X)} vs {len(y)}).")
            return

    sizes = f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    _ok(tag, sizes)


# ── Classic ML model checks ───────────────────────────────────────────────────

def check_model(
    tag: str,
    model,
    X_test,
    y_test,
    min_f1: float = 0.4,
):
    """Check that a fitted sklearn-compatible model achieves reasonable performance."""
    from sklearn.metrics import f1_score

    if not hasattr(model, "predict"):
        _fail(tag, "Object does not have a predict() method.")
        return
    try:
        y_pred = model.predict(X_test)
    except Exception as exc:
        _fail(tag, f"model.predict() raised an error: {exc}")
        return

    score = f1_score(y_test, y_pred, average="macro")
    if score < min_f1:
        _fail(tag, f"Macro F1 = {score:.3f} < {min_f1}. Check your training setup.")
        return
    _ok(tag, f"macro F1 = {score:.3f}")


# ── GNN model checks ─────────────────────────────────────────────────────────

def check_gnn_model(
    tag: str,
    model,
    data,
    split: str = "val",
    min_acc: float = 0.70,
):
    """Check a trained PyG GNN achieves reasonable accuracy on a given split.

    Parameters
    ----------
    tag:
        Exercise identifier shown in output.
    model:
        A fitted PyTorch model with a ``forward(x, edge_index)`` signature.
    data:
        A PyG ``Data`` object with ``x``, ``edge_index``, ``y``, and split masks.
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.
    min_acc:
        Minimum acceptable accuracy.
    """
    import torch

    if not hasattr(model, "parameters"):
        _fail(tag, "Expected a PyTorch nn.Module.")
        return

    mask = getattr(data, f"{split}_mask", None)
    if mask is None:
        _fail(tag, f"data.{split}_mask not found.")
        return

    model.eval()
    try:
        with torch.no_grad():
            out = model(data.x, data.edge_index)
    except Exception as exc:
        _fail(tag, f"model.forward() raised: {exc}")
        return

    pred = out.argmax(dim=1)
    acc = (pred[mask] == data.y[mask]).float().mean().item()

    if acc < min_acc:
        _fail(tag, f"{split} accuracy = {acc:.3f} < {min_acc}. Check model or training.")
        return
    _ok(tag, f"{split}_acc = {acc:.3f}")


def check_graph(
    tag: str,
    G,
    min_nodes: int | None = None,
    min_edges: int | None = None,
):
    """Validate a NetworkX graph."""
    import networkx as nx

    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        _fail(tag, f"Expected a NetworkX Graph, got {type(G).__name__}.")
        return
    if min_nodes is not None and G.number_of_nodes() < min_nodes:
        _fail(tag, f"Expected ≥{min_nodes} nodes, got {G.number_of_nodes()}.")
        return
    if min_edges is not None and G.number_of_edges() < min_edges:
        _fail(tag, f"Expected ≥{min_edges} edges, got {G.number_of_edges()}.")
        return
    _ok(tag, f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
