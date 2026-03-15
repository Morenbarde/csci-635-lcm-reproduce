from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from Democritus.graph_creation import Graph
from Democritus.sentence_embedding import EdgeFeatureStore, NodeEmbeddingStore


@dataclass
class SubjectDomainEmbeddingStore:
    """Stores one embedding vector per (subject node, domain) pair."""

    subject_node_ids: List[int]
    subject_texts: List[str]
    domains: List[str]
    vectors: np.ndarray  # shape: (N, dim)
    key_to_row: Dict[Tuple[int, str], int]  # (subject_node_id, domain) -> row index


@dataclass
class UMAP2DStore:
    """Stores reduced 2D coordinates for subject-domain embeddings."""

    subject_node_ids: List[int]
    subject_texts: List[str]
    domains: List[str]
    coords: np.ndarray  # shape: (N, 2)


def _weighted_mean(vectors: List[np.ndarray], weights: List[float], dim: int) -> np.ndarray:
    if not vectors:
        return np.zeros(dim, dtype=np.float32)

    arr = np.asarray(vectors, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)

    total = float(w.sum())
    if total <= 0.0:
        return arr.mean(axis=0)

    return (arr * w[:, None]).sum(axis=0) / total


def aggregate_subject_domain_embeddings(
    graph: Graph,
    node_store: NodeEmbeddingStore,
    edge_store: EdgeFeatureStore,
    node_vectors: np.ndarray | None = None,
    edge_weight_scale: float = 0.5,
) -> SubjectDomainEmbeddingStore:
    """
    Aggregate embeddings per subject, per domain.

    For each (subject, domain):
    1) Start from the subject node embedding.
    2) Aggregate outgoing edge context using relation/domain/object embeddings.
    3) Blend both signals into a single vector.

    Args:
        graph: Causal graph.
        node_store: Node embedding store.
        edge_store: Relation/domain embedding store.
        node_vectors: Optional updated node vectors (e.g., post message passing).
            If omitted, node_store.vectors are used.
        edge_weight_scale: Blend factor in [0, 1].
            0.0 uses only subject embedding; 1.0 uses only edge context.

    Returns:
        SubjectDomainEmbeddingStore with one row per (subject, domain).
    """
    if node_vectors is None:
        node_vectors = node_store.vectors

    if node_vectors.shape != node_store.vectors.shape:
        raise ValueError(
            "node_vectors shape must match node_store.vectors shape: "
            f"expected {node_store.vectors.shape}, got {node_vectors.shape}"
        )

    if not (0.0 <= edge_weight_scale <= 1.0):
        raise ValueError("edge_weight_scale must be between 0.0 and 1.0")

    dim = node_vectors.shape[1]

    outgoing_by_subject_domain: Dict[Tuple[int, str], List] = {}
    for edge in graph.edges.values():
        key = (edge.src_node_id, edge.domain)
        outgoing_by_subject_domain.setdefault(key, []).append(edge)

    subject_node_ids: List[int] = []
    subject_texts: List[str] = []
    domains: List[str] = []
    rows: List[np.ndarray] = []
    key_to_row: Dict[Tuple[int, str], int] = {}

    for key in sorted(outgoing_by_subject_domain.keys(), key=lambda x: (x[0], x[1])):
        subject_id, domain = key
        edges = outgoing_by_subject_domain[key]

        subj_row = node_store.id_to_row[subject_id]
        subject_vec = node_vectors[subj_row]

        edge_messages: List[np.ndarray] = []
        edge_weights: List[float] = []

        for edge in edges:
            rel_row = edge_store.relation_to_row[edge.relation]
            dom_row = edge_store.domain_to_row[edge.domain]
            dst_row = node_store.id_to_row[edge.dst_node_id]

            rel_vec = edge_store.relation_vectors[rel_row]
            dom_vec = edge_store.domain_vectors[dom_row]
            obj_vec = node_vectors[dst_row]

            edge_msg = (rel_vec + dom_vec + obj_vec) / 3.0
            edge_messages.append(edge_msg)
            edge_weights.append(float(max(edge.count, 1)))

        edge_context = _weighted_mean(edge_messages, edge_weights, dim)
        blended = (1.0 - edge_weight_scale) * subject_vec + edge_weight_scale * edge_context

        row_idx = len(rows)
        key_to_row[key] = row_idx
        rows.append(blended.astype(np.float32))
        subject_node_ids.append(subject_id)
        subject_texts.append(graph.nodes[subject_id].canonical_text)
        domains.append(domain)

    if rows:
        vectors = np.vstack(rows)
    else:
        vectors = np.zeros((0, dim), dtype=np.float32)

    return SubjectDomainEmbeddingStore(
        subject_node_ids=subject_node_ids,
        subject_texts=subject_texts,
        domains=domains,
        vectors=vectors,
        key_to_row=key_to_row,
    )


def reduce_subject_embeddings_umap(
    store: SubjectDomainEmbeddingStore,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> UMAP2DStore:
    """
    Reduce subject-domain embeddings to 2D coordinates using UMAP.

    Requires: `pip install umap-learn`
    """
    n = store.vectors.shape[0]

    if n == 0:
        coords = np.zeros((0, 2), dtype=np.float32)
    elif n == 1:
        coords = np.zeros((1, 2), dtype=np.float32)
    else:
        try:
            import umap.umap_ as umap
        except ImportError as exc:
            raise ImportError(
                "UMAP is not installed. Install with `pip install umap-learn`."
            ) from exc

        effective_neighbors = max(2, min(int(n_neighbors), n - 1))

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        coords = np.asarray(reducer.fit_transform(store.vectors), dtype=np.float32)

    return UMAP2DStore(
        subject_node_ids=store.subject_node_ids,
        subject_texts=store.subject_texts,
        domains=store.domains,
        coords=coords,
    )


def plot_subject_umap(
    umap_store: UMAP2DStore,
    title: str = "Subject Embeddings by Domain (UMAP)",
    annotate: bool = False,
    hover_labels: bool = True,
):
    """Create a quick matplotlib scatter plot for reduced subject embeddings."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is not installed. Install with `pip install matplotlib`."
        ) from exc

    coords = umap_store.coords
    if coords.shape[0] == 0:
        raise ValueError("No points to plot: UMAP2DStore is empty.")

    unique_domains = sorted(set(umap_store.domains))

    fig, ax = plt.subplots(figsize=(10, 8))
    all_scatter_points = []
    point_labels = []

    for domain in unique_domains:
        idx = [i for i, d in enumerate(umap_store.domains) if d == domain]
        points = coords[idx]
        scatter = ax.scatter(points[:, 0], points[:, 1], label=domain, alpha=0.85)
        all_scatter_points.append(scatter)
        point_labels.extend([f"{umap_store.subject_texts[i]} [{umap_store.domains[i]}]" for i in idx])

    if hover_labels and coords.shape[0] > 0:
        # Single tooltip annotation reused as the mouse moves across points.
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.95},
            fontsize=8,
        )
        annot.set_visible(False)

        point_offsets = np.vstack([s.get_offsets() for s in all_scatter_points])

        def _on_move(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                return

            mouse = np.array([event.xdata, event.ydata], dtype=np.float32)
            deltas = point_offsets - mouse
            dists = np.sqrt((deltas * deltas).sum(axis=1))
            idx = int(np.argmin(dists))

            # Threshold in data units; keeps tooltips from firing far from points.
            x_span = max(float(coords[:, 0].max() - coords[:, 0].min()), 1e-6)
            y_span = max(float(coords[:, 1].max() - coords[:, 1].min()), 1e-6)
            threshold = 0.03 * max(x_span, y_span)

            if float(dists[idx]) <= threshold:
                annot.xy = (float(point_offsets[idx, 0]), float(point_offsets[idx, 1]))
                annot.set_text(point_labels[idx])
                if not annot.get_visible():
                    annot.set_visible(True)
                fig.canvas.draw_idle()
            elif annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", _on_move)

    if annotate:
        for i, label in enumerate(umap_store.subject_texts):
            ax.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.75)

        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(loc="best")
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    from Democritus.graph_creation import build_graph_from_triples
    from Democritus.message_passing import fixed_message_passing
    from Democritus.sentence_embedding import build_edge_feature_embeddings, build_node_embeddings
    from utils.utils import load_jsonl

    triples = load_jsonl("Causal_Triples\\triples_econ.jsonl")
    domain = "econ"

    graph = build_graph_from_triples(triples, domain)
    node_store = build_node_embeddings(graph, "sentence-transformers/all-MiniLM-L6-v2")
    edge_store = build_edge_feature_embeddings(graph, "sentence-transformers/all-MiniLM-L6-v2")

    updated_node_vectors = fixed_message_passing(
        graph=graph,
        node_store=node_store,
        edge_store=edge_store,
        num_layers=2,
        alpha=0.6,
        beta=0.3,
        use_triangles=False,
    )

    subject_store = aggregate_subject_domain_embeddings(
        graph=graph,
        node_store=node_store,
        edge_store=edge_store,
        node_vectors=updated_node_vectors,
        edge_weight_scale=0.5,
    )

    umap_store = reduce_subject_embeddings_umap(subject_store)
    print("Subject-domain vectors:", subject_store.vectors.shape)
    print("UMAP coords:", umap_store.coords.shape)

    # Set annotate=True if you want text labels on each point.
    plot_subject_umap(umap_store, annotate=False)
