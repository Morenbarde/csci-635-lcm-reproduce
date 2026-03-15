from Democritus.sentence_embedding import NodeEmbeddingStore, EdgeFeatureStore
from Democritus.graph_creation import Graph
import numpy as np
from typing import List, Tuple, Set


# Note: this function was written by Claude
def detect_triangles(graph: Graph) -> List[Tuple[int, int, int]]:
    """
    Detect actual undirected triangles in the graph.

    Returns:
        List of triangles as sorted tuples (a, b, c), with a < b < c.
    """
    neighbors: dict[int, Set[int]] = {}

    # Build undirected adjacency
    for edge in graph.edges.values():
        u = edge.src_node_id
        v = edge.dst_node_id

        if u == v:
            continue

        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)

    triangles = set()

    # Find triples (a, b, c) such that all three pairwise edges exist
    for a in neighbors:
        for b in neighbors[a]:
            if b <= a:
                continue
            common = neighbors[a].intersection(neighbors[b])
            for c in common:
                if c <= b:
                    continue
                triangles.add((a, b, c))

    return sorted(triangles)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def fixed_message_passing(graph: Graph,
                          node_store: NodeEmbeddingStore,
                          edge_store: EdgeFeatureStore,
                          num_layers: int = 2,
                          alpha: float = 0.5,
                          beta: float = 0.1,
                          use_triangles: bool = False):
    """
    Perform fixed message passing on the graph.

    returns:
        node_embeddings: np.ndarray of shape (num_nodes, dim)
        edge_embeddings: np.ndarray of shape (num_edges, dim)
    """
    h = node_store.vectors.copy()
    num_nodes, dim = h.shape

    triangles = detect_triangles(graph) if use_triangles else []

    for layer in range(num_layers):
        edge_agg = np.zeros_like(h)
        edge_counts = np.zeros(num_nodes, dtype=np.float32)

        # Aggregate messages from edges
        for edge in graph.edges.values():
            src_row = node_store.id_to_row[edge.src_node_id]
            dst_row = node_store.id_to_row[edge.dst_node_id]

            rel_row = edge_store.relation_to_row[edge.relation]
            dom_row = edge_store.domain_to_row[edge.domain]

            h_src = h[src_row]
            r_vec = edge_store.relation_vectors[rel_row]
            d_vec = edge_store.domain_vectors[dom_row]

            # Message is the sum of source node embedding, relation embedding, and domain embedding
            message = h_src + r_vec + d_vec
            edge_agg[dst_row] += message
            edge_counts[dst_row] += 1

        # Average the messages for each node
        for i in range(num_nodes):
            if edge_counts[i] > 0:
                edge_agg[i] /= edge_counts[i]

        tri_agg = np.zeros_like(h)
        tri_counts = np.zeros(num_nodes, dtype=np.float32)

        # Triangle level aggregation
        if use_triangles:
            for u, v, w in triangles:
                u_row = node_store.id_to_row[u]
                v_row = node_store.id_to_row[v]
                w_row = node_store.id_to_row[w]

                dom_row = edge_store.domain_to_row[domain]

                d_vec = edge_store.domain_vectors[dom_row]

                tri_msg = (h[u_row] + h[v_row] + h[w_row]) / 3.0
                # Update the triangle message with domain information
                tri_msg = 0.8 * tri_msg + 0.2 * d_vec

                for row in (u_row, v_row, w_row):
                    tri_agg[row] += tri_msg
                    tri_counts[row] += 1.0

            # Average the triangle messages for each node
            for i in range(num_nodes):
                if tri_counts[i] > 0:
                    tri_agg[i] /= tri_counts[i]

        # This style was suggested by Claude.
        # It is a residual-style update that combines the original embedding,
        # edge aggregation, and triangle aggregation.
        #
        # Residual-style update
        # Do not want to replace the original embedding entirely,
        # so we use a weighted sum of the original embedding,
        # edge aggregation, and triangle aggregation.
        #
        # Without it, the embeddings can collapse to zero if a node has no edges or triangles.
        h_new = (1.0 - alpha - beta) * h + alpha * edge_agg + beta * tri_agg

        # Keep isolated nodes from collapsing to zero
        isolated = (edge_counts == 0) & (tri_counts == 0)
        h_new[isolated] = h[isolated]

        h = l2_normalize_rows(h_new)

    return h


if __name__ == "__main__":
    from Democritus.sentence_embedding import build_node_embeddings, build_edge_feature_embeddings
    from Democritus.graph_creation import build_graph_from_triples
    from utils.utils import load_jsonl

    triples = load_jsonl("Causal_Triples\\triangle_test.jsonl")
    domain = "econ"

    graph = build_graph_from_triples(triples, domain)

    node_store = build_node_embeddings(graph, "sentence-transformers/all-MiniLM-L6-v2")
    edge_store = build_edge_feature_embeddings(graph, "sentence-transformers/all-MiniLM-L6-v2")

    updated_vectors = fixed_message_passing(
        graph=graph,
        node_store=node_store,
        edge_store=edge_store,
        num_layers=2,
        alpha=0.6,
        beta=0.3,
        use_triangles=True,
    )

    print("Original shape:", node_store.vectors.shape)
    print("Updated shape:", updated_vectors.shape)

    diff = np.linalg.norm(updated_vectors - node_store.vectors, axis=1)
    print("Max node change:", diff.max())
    print("Mean node change:", diff.mean())
    print("Changed nodes:", np.sum(diff > 1e-6), "out of", len(diff))

    triangles = detect_triangles(graph)
    print("Num triangles:", len(triangles))
