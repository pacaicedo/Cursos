"""Graph visualization and analysis helpers for Module III labs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

_CORA_CLASSES = [
    "Case-Based", "Genetic Algorithms", "Neural Networks",
    "Probabilistic Methods", "Reinforcement Learning",
    "Rule Learning", "Theory",
]


# ── Graph drawing ─────────────────────────────────────────────────────────────

def plot_graph(
    G,
    labels: dict | None = None,
    label_names: list[str] | None = None,
    title: str = "",
    node_size: int = 400,
    seed: int = 42,
    ax=None,
):
    """Draw a NetworkX graph, optionally coloured by integer node labels.

    Parameters
    ----------
    G:
        A ``networkx.Graph`` or ``DiGraph``.
    labels:
        Dict mapping node id → integer class index.
    label_names:
        Human-readable class names indexed by class integer.
    """
    import networkx as nx

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    pos = nx.spring_layout(G, seed=seed)

    if labels is not None:
        unique = sorted(set(labels.values()))
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 2)))
        node_colors = [cmap[labels.get(n, 0)] for n in G.nodes()]
    else:
        node_colors = "#4C72B0"

    nx.draw(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_size,
        edge_color="#cccccc", with_labels=True, font_size=8,
    )

    if labels is not None and label_names:
        unique = sorted(set(labels.values()))
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 2)))
        handles = [
            plt.scatter([], [], c=[cmap[i]], label=label_names[i], s=60)
            for i in unique if i < len(label_names)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7)

    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_degree_distribution(degrees: list[int], title: str = "Degree distribution", ax=None):
    """Bar chart of degree counts."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    unique, counts = np.unique(degrees, return_counts=True)
    ax.bar(unique, counts, color="#4C72B0", edgecolor="white", width=0.8)
    ax.set_xlabel("Degree (number of neighbours)")
    ax.set_ylabel("Number of nodes")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return ax


# ── Embedding visualisation ───────────────────────────────────────────────────

def plot_embeddings(
    embeddings,
    labels,
    label_names: list[str] | None = None,
    method: str = "tsne",
    title: str = "Node embeddings",
    seed: int = 42,
    ax=None,
):
    """2-D projection of node embeddings using t-SNE or PCA.

    Parameters
    ----------
    embeddings:
        NumPy array of shape ``(N, D)``.
    labels:
        Integer class labels of length N.
    method:
        ``"tsne"`` or ``"pca"``.
    """
    emb = np.array(embeddings)
    lab = np.array(labels)

    if method == "tsne":
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, random_state=seed, perplexity=30,
                    n_iter=1000).fit_transform(emb)
    else:
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=seed).fit_transform(emb)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(set(lab.tolist()))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 2)))

    for i, lbl in enumerate(unique_labels):
        mask = lab == lbl
        name = label_names[lbl] if (label_names and lbl < len(label_names)) else str(lbl)
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[cmap[i]],
                   label=name, s=10, alpha=0.75)

    ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_title(f"{title} ({method.upper()})")
    ax.axis("off")
    plt.tight_layout()
    return ax


# ── Attention visualisation ───────────────────────────────────────────────────

def plot_attention_subgraph(
    edge_index,
    alpha,
    center_node: int,
    node_labels=None,
    label_names: list[str] | None = None,
    head: int = 0,
    title: str = "Attention weights",
    seed: int = 0,
):
    """Visualise a node's 1-hop neighbourhood with edges weighted by GAT attention.

    Parameters
    ----------
    edge_index:
        PyG edge index tensor of shape ``(2, E)``.
    alpha:
        Attention weights of shape ``(E, H)`` where H = number of heads.
    center_node:
        The focal node whose neighbourhood is visualised.
    node_labels:
        1-D tensor / array of integer class labels for all nodes.
    head:
        Which attention head to visualise.
    """
    import networkx as nx

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    weights = alpha[:, head].detach().cpu().numpy()

    # Keep only edges incident on center_node
    mask = (src == center_node) | (dst == center_node)
    src_l, dst_l, w_l = src[mask], dst[mask], weights[mask]

    # Build subgraph
    G = nx.DiGraph()
    nodes = list(set(src_l.tolist() + dst_l.tolist()))
    G.add_nodes_from(nodes)
    for s, d, w in zip(src_l, dst_l, w_l):
        G.add_edge(int(s), int(d), weight=float(w))

    pos = nx.spring_layout(G, seed=seed)

    edge_widths = [G[u][v]["weight"] * 12 for u, v in G.edges()]
    edge_colors = [G[u][v]["weight"] for u, v in G.edges()]

    if node_labels is not None:
        import numpy as _np
        lab_arr = _np.array(node_labels)
        unique = sorted(set(lab_arr[nodes].tolist()))
        cmap = plt.cm.tab10(_np.linspace(0, 1, max(len(unique), 2)))
        nc = [cmap[int(lab_arr[n])] for n in G.nodes()]
    else:
        nc = ["#4C72B0" if n != center_node else "#DD8452" for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_nodes(G, pos, node_color=nc, node_size=400, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    edges_drawn = nx.draw_networkx_edges(
        G, pos, width=edge_widths, edge_color=edge_colors,
        edge_cmap=plt.cm.Blues, ax=ax, arrows=True,
        arrowsize=15, connectionstyle="arc3,rad=0.1",
    )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                                norm=plt.Normalize(vmin=min(w_l), vmax=max(w_l)))
    plt.colorbar(sm, ax=ax, label="Attention weight (head 0)")
    ax.set_title(f"{title} — node {center_node}")
    ax.axis("off")
    plt.tight_layout()
    return ax
