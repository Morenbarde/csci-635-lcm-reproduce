# Built with the help of Claude
from pathlib import Path

from Democritus.domain_projection import (
    aggregate_subject_domain_embeddings,
    plot_subject_umap,
    reduce_subject_embeddings_umap,
)
from Democritus.graph_creation import build_merged_graph_from_domain_triples
from Democritus.message_passing import fixed_message_passing
from Democritus.sentence_embedding import build_edge_feature_embeddings, build_node_embeddings
from utils.utils import load_jsonl


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    domain_files = {
        "bio": repo_root / "Causal_Triples" / "triples_bio_depth0.jsonl",
        "econ": repo_root / "Causal_Triples" / "triples_econ_depth0.jsonl",
        "indus": repo_root / "Causal_Triples" / "triples_indus_depth0.jsonl",
    }

    domain_to_triples = {
        domain: load_jsonl(path)
        for domain, path in domain_files.items()
    }

    graph = build_merged_graph_from_domain_triples(domain_to_triples)
    print(f"Merged graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    node_store = build_node_embeddings(graph, model_name)
    edge_store = build_edge_feature_embeddings(graph, model_name)

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
    umap_store = reduce_subject_embeddings_umap(
        subject_store,
        n_neighbors=20,
        min_dist=0.15,
        metric="cosine",
    )

    print("Subject-domain vectors:", subject_store.vectors.shape)
    print("UMAP coords:", umap_store.coords.shape)

    plot_subject_umap(
        umap_store,
        title="Merged Multi-Domain Subject Embeddings (UMAP)",
        annotate=False,
        hover_labels=True,
    )


if __name__ == "__main__":
    main()
