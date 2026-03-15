from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GraphNode:
    node_id: int
    canonical_text: str
    raw_texts: List[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    src_node_id: int
    relation: str
    dst_node_id: int
    domain: str
    count: int = 0
    evidence: List[dict] = field(default_factory=list)


@dataclass
class Graph:
    node_index: Dict[str, int] = field(default_factory=dict)  # Maps canonical_text to node_id
    nodes: Dict[int, GraphNode] = field(default_factory=dict)  # Maps node_id to GraphNode
    edges: Dict[Tuple[int, str, int, str], GraphEdge] = field(default_factory=dict)  # Maps (src_node_id, relation, dst_node_id, domain) to GraphEdge
    next_node_id: int = 0  # Counter for generating unique node IDs


def normalize_node_text(text: str) -> str:
    text = text.lower().strip()
    text = text.lstrip('-*').strip()  # Remove leading '-' or '*' characters
    text = " ".join(text.split())  # Replace multiple spaces with a single space
    # remove articles anywhere, not just at the start
    # found that it removes a lot of noise in the node names,
    #  e.g. "the" in "the United States" or "a" in "a decrease"
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    text = " ".join(tokens)

    return text


def normalize_relation(text: str) -> str:
    rel = normalize_node_text(text)
    # This mapping is used to standardize relation names to a consistent format.
    # Will probably need to expand this mapping as more relations are encountered
    rel_map = {
        "leads": "leads_to",
        "leads to": "leads_to",
        "causes": "causes",
        "causes increase in": "causes_increase",
        "causes decrease in": "causes_decrease",
        "causes reduction in": "causes_decrease",
    }
    return rel_map.get(rel, rel.replace(" ", "_"))


def normalize_object_text(raw_object: str, normalized_relation: str) -> str:
    obj = normalize_node_text(raw_object)
    # May need to add more rules as testing continues
    if normalized_relation == "leads_to" and obj.startswith("to "):
        obj = obj[3:].strip()
    return obj


def get_or_create_node(graph: Graph, raw_text: str) -> int:
    canonical = normalize_node_text(raw_text)
    node_id = graph.node_index.get(canonical)

    if node_id is not None:
        if raw_text not in graph.nodes[node_id].raw_texts:
            graph.nodes[node_id].raw_texts.append(raw_text)
        return node_id

    node_id = graph.next_node_id
    graph.next_node_id += 1

    graph.node_index[canonical] = node_id
    graph.nodes[node_id] = GraphNode(
        node_id=node_id,
        canonical_text=canonical,
        raw_texts=[raw_text],
    )
    return node_id


def build_graph_from_triples(triples: List[dict], domain: str) -> Graph:
    graph = Graph()

    add_triples_to_graph(graph=graph, triples=triples, domain=domain)

    return graph


def add_triples_to_graph(graph: Graph, triples: List[dict], domain: str) -> None:
    """Add triples from one domain into an existing graph in-place."""

    for triple in triples:
        s_raw = triple["subject"]
        r_raw = triple["relation"]
        o_raw = triple["object"]

        rel = normalize_relation(r_raw)
        subj_id = get_or_create_node(graph, s_raw)
        obj_norm = normalize_object_text(o_raw, rel)
        obj_id = get_or_create_node(graph, obj_norm)

        edge_key = (subj_id, rel, obj_id, domain)
        if edge_key not in graph.edges:
            graph.edges[edge_key] = GraphEdge(
                src_node_id=subj_id,
                relation=rel,
                dst_node_id=obj_id,
                domain=domain,
                count=1,
            )
        else:
            graph.edges[edge_key].count += 1


def build_merged_graph_from_domain_triples(domain_to_triples: Dict[str, List[dict]]) -> Graph:
    """Build one graph from multiple domains while preserving per-edge domain labels."""
    graph = Graph()

    for domain in sorted(domain_to_triples.keys()):
        triples = domain_to_triples[domain]
        add_triples_to_graph(graph=graph, triples=triples, domain=domain)

    return graph


if __name__ == "__main__":
    # test build graph with triples
    from Democritus.sentence_embedding import build_node_embeddings, build_edge_feature_embeddings
    from utils.utils import load_jsonl
    triples = load_jsonl("Causal_Triples\\triples_econ.jsonl")

    # note: in this example domain is based on filename
    domain = "econ"
    graph = build_graph_from_triples(triples, domain)
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

    for edge in list(graph.edges.values())[:5]:
        s = graph.nodes[edge.src_node_id].canonical_text
        t = graph.nodes[edge.dst_node_id].canonical_text
        print(f"{edge.src_node_id}:{s} --[{edge.relation}]--> {edge.dst_node_id}:{t}")

    store = build_node_embeddings(graph, "sentence-transformers/all-MiniLM-L6-v2")
    edge_store = build_edge_feature_embeddings(graph, "sentence-transformers/all-MiniLM-L6-v2")
    print(store.vectors.shape)  # (num_nodes, dim)

    nid = 0
    vec = store.vectors[store.id_to_row[nid]]
    print(f"Node {nid} embedding: {vec[:5]}...")

    rid = 0
    rel_vec = edge_store.relation_vectors[edge_store.relation_to_row[graph.edges[list(graph.edges.keys())[rid]].relation]]
    print(f"Relation {graph.edges[list(graph.edges.keys())[rid]].relation} embedding: {rel_vec[:5]}...")