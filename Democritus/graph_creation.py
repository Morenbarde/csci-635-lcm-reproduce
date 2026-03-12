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
    for article in ("a ", "an ", "the "):
        if text.startswith(article):
            text = text[len(article):]
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

    for triple in triples:
        src_id = get_or_create_node(graph, triple["subject"])
        dst_id = get_or_create_node(graph, triple["object"])
        rel = normalize_relation(triple["relation"])

        edge_key = (src_id, rel, dst_id, domain)
        if edge_key not in graph.edges:
            graph.edges[edge_key] = GraphEdge(
                src_node_id=src_id,
                relation=rel,
                dst_node_id=dst_id,
                domain=domain,
                count=1,
            )
        else:
            graph.edges[edge_key].count += 1

    return graph


if __name__ == "__main__":
    # test build graph with triples
    from Democritus.sentence_embedding import build_node_embeddings
    from utils.utils import load_jsonl
    triples = load_jsonl("Causal_Triples\\triples_econ.jsonl")

    # note: in this example domain is based on filename
    domain = "econ"
    graph = build_graph_from_triples(triples, domain)

    for edge in list(graph.edges.values())[:5]:
        s = graph.nodes[edge.src_node_id].canonical_text
        t = graph.nodes[edge.dst_node_id].canonical_text
        print(f"{edge.src_node_id}:{s} --[{edge.relation}]--> {edge.dst_node_id}:{t}")

    store = build_node_embeddings(graph, "sentence-transformers/all-MiniLM-L6-v2")
    print(store.vectors.shape)  # (num_nodes, dim)

    nid = 0
    vec = store.vectors[store.id_to_row[nid]]
    print(f"Node {nid} embedding: {vec[:5]}...") 