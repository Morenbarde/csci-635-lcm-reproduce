# Democritus/sentence_embedding.py
# This file contains the dataclasses for storing the embeddings of causal triples.
# The embeddings are generated using a sentence transformer model,
# which is specified by the model_name attribute in the SentenceEmbedding dataclass.
# The SentenceNode dataclass holds the text and optional span information for each node in the causal triple,
# while the SentenceEmbedding dataclass holds the model name,
# embedding dimension, list of nodes,
# and the corresponding embedding vectors.
#
# The end goal is to have a Graph of shape (V, E, rel, dom)
# Where V is the head or tail string
# E is the embedding
# rel is the relation string
# domain is the domain string

from dataclasses import dataclass
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from Democritus.graph_creation import Graph


@dataclass
class NodeEmbeddingStore:
    model_name: str
    dim: int
    node_ids: List[int]
    vectors: np.ndarray  # shape (N, dim) where N is the number of nodes
    id_to_row: Dict[int, int]  # node_id -> row index in vectors


def build_node_embeddings(graph: Graph, model_name: str) -> NodeEmbeddingStore:
    model = SentenceTransformer(model_name)

    node_ids = sorted(graph.nodes.keys())
    node_texts = [graph.nodes[node_id].canonical_text for node_id in node_ids]

    vectors = np.asarray(model.encode(node_texts, convert_to_numpy=True))
    id_to_row = {node_id: i for i, node_id in enumerate(node_ids)}

    return NodeEmbeddingStore(
        model_name=model_name,
        dim=vectors.shape[1],
        node_ids=node_ids,
        vectors=vectors,
        id_to_row=id_to_row
    )


if __name__ == "__main__":
    from utils.utils import load_jsonl
    # example usage
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    path = "Causal_Triples\\triples_econ.jsonl"

    # load causal triples
    triples = load_jsonl(path)
    print(triples)
