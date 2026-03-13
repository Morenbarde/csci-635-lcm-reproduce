import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph


def to_graphviz(g, max_edges=None):
    dot = Digraph('CausalGraph')
    dot.attr(rankdir='LR')  # Set the graph direction from left to right

    # Nodes
    for nid, node in g.nodes.items():
        label = f"{nid}: {node.canonical_text}"
        dot.node(str(nid), label=label)

    # Edges
    items = list(g.edges.values())
    if max_edges is not None:
        items = items[:max_edges]

    for e in items:
        edge_label = f"{e.relation} ({e.domain}) x{e.count}"
        dot.edge(str(e.src_node_id), str(e.dst_node_id), label=edge_label)

    return dot


def to_networkx(g):
    G = nx.MultiDiGraph()

    for nid, node in g.nodes.items():
        G.add_node(nid, label=node.canonical_text, raw_texts=node.raw_texts)

    for e in g.edges.values():
        G.add_edge(
            e.src_node_id,
            e.dst_node_id,
            relation=e.relation,
            domain=e.domain,
            count=e.count,
            label=f"{e.relation} [{e.domain}] x{e.count}",
        )

    return G


def draw_networkx_graph(G, figsize=(14, 10)):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=figsize)

    node_labels = {n: f"{n}: {d['label']}" for n, d in G.nodes(data=True)}
    nx.draw_networkx_nodes(G, pos, node_size=900, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", alpha=0.5)

    # Multi-edge labels: include edge key
    edge_labels = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        edge_labels[(u, v, k)] = d["label"]

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from Democritus.graph_creation import build_graph_from_triples
    from utils.utils import load_jsonl

    triples = load_jsonl("Causal_Triples\\triples_econ.jsonl")
    domain = "econ"
    graph = build_graph_from_triples(triples, domain)
    print("raw triples:", len(triples))
    print("unique nodes:", len(graph.nodes))
    print("unique edges:", len(graph.edges))
    print("sum edge counts:", sum(e.count for e in graph.edges.values()))

    dot = to_graphviz(graph, max_edges=None)
    dot.render('causal_graph', format='png', cleanup=True)