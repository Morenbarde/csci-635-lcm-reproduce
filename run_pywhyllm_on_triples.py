import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pywhyllm import RelationshipStrategy
from pywhyllm.suggesters.identification_suggester import IdentificationSuggester
from pywhyllm.suggesters.model_suggester import ModelSuggester
from pywhyllm.suggesters.validation_suggester import ValidationSuggester


def load_triples(path: Path) -> List[dict]:
    triples: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            triples.append(json.loads(line))
    return triples


def normalize_text(text: str) -> str:
    text = text.strip().lower().lstrip("-* ")
    return " ".join(text.split())


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        converted: Dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, (str, int, float, bool)) or key is None:
                safe_key: Any = key
            elif isinstance(key, tuple):
                safe_key = " | ".join(str(part) for part in key)
            else:
                safe_key = str(key)
            converted[safe_key] = make_json_safe(item)
        return converted
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return value


def factor_ranking(triples: List[dict]) -> List[Tuple[str, int]]:
    counts: Counter = Counter()
    for t in triples:
        subj = normalize_text(t.get("subject", ""))
        obj = normalize_text(t.get("object", ""))
        if subj:
            counts[subj] += 1
        if obj:
            counts[obj] += 1
    return counts.most_common()


def choose_default_treatment_outcome(triples: List[dict]) -> Tuple[str, str]:
    edge_counts: Dict[Tuple[str, str], int] = {}
    for t in triples:
        src = normalize_text(t.get("subject", ""))
        dst = normalize_text(t.get("object", ""))
        if not src or not dst or src == dst:
            continue
        key = (src, dst)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    if not edge_counts:
        raise ValueError("No usable (subject, object) pairs were found in the triple file.")

    return max(edge_counts.items(), key=lambda x: x[1])[0]


def extracted_edge_counts(
    triples: List[dict],
    allowed_factors: List[str],
) -> Dict[Tuple[str, str], int]:
    allowed = set(allowed_factors)
    counts: Dict[Tuple[str, str], int] = {}
    for t in triples:
        src = normalize_text(t.get("subject", ""))
        dst = normalize_text(t.get("object", ""))
        if not src or not dst or src == dst:
            continue
        if src not in allowed or dst not in allowed:
            continue
        key = (src, dst)
        counts[key] = counts.get(key, 0) + 1
    return counts


def undirected_edge_set(directed_edges: set[Tuple[str, str]]) -> set[Tuple[str, str]]:
    return {tuple(sorted((a, b))) for a, b in directed_edges}


def compute_agreement_metrics(
    extracted_edges: Dict[Tuple[str, str], int],
    pywhyllm_edges: Dict[Tuple[str, str], int],
    min_votes: int,
) -> dict:
    extracted_set = set(extracted_edges.keys())
    pywhyllm_set = {edge for edge, votes in pywhyllm_edges.items() if votes >= min_votes}

    directed_overlap = extracted_set & pywhyllm_set

    pywhyllm_count = len(pywhyllm_set)
    extracted_count = len(extracted_set)
    overlap_count = len(directed_overlap)

    precision = overlap_count / pywhyllm_count if pywhyllm_count else 0.0
    recall = overlap_count / extracted_count if extracted_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    extracted_undirected = undirected_edge_set(extracted_set)
    pywhyllm_undirected = undirected_edge_set(pywhyllm_set)
    undirected_overlap = extracted_undirected & pywhyllm_undirected

    reverse_match_count = 0
    for a, b in undirected_overlap:
        forward = (a, b)
        backward = (b, a)
        extracted_has_forward = forward in extracted_set
        extracted_has_backward = backward in extracted_set
        pywhyllm_has_forward = forward in pywhyllm_set
        pywhyllm_has_backward = backward in pywhyllm_set
        if ((extracted_has_forward and pywhyllm_has_forward) or (extracted_has_backward and pywhyllm_has_backward)):
            reverse_match_count += 1

    direction_agreement = (
        reverse_match_count / len(undirected_overlap) if undirected_overlap else 0.0
    )

    return {
        "min_votes": min_votes,
        "directed": {
            "extracted_edges": extracted_count,
            "pywhyllm_edges": pywhyllm_count,
            "overlap_edges": overlap_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "undirected": {
            "extracted_pairs": len(extracted_undirected),
            "pywhyllm_pairs": len(pywhyllm_undirected),
            "overlap_pairs": len(undirected_overlap),
            "direction_agreement_on_overlap": direction_agreement,
        },
        "overlap_examples": {
            "directed": [
                {"cause": src, "effect": dst}
                for src, dst in sorted(directed_overlap)[:20]
            ],
            "undirected": [
                {"node_a": a, "node_b": b}
                for a, b in sorted(undirected_overlap)[:20]
            ],
        },
    }


def build_llm(args: argparse.Namespace):
    import guidance

    if args.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Set it, or use --backend openai_compatible for a local endpoint."
            )
        return guidance.models.OpenAI(args.model)

    if args.backend == "openai_compatible":
        # Guidance reads OpenAI-style configuration from environment variables.
        # For local servers, a placeholder API key is often accepted.
        if args.api_base:
            base_url = args.api_base.rstrip("/")
            if base_url.endswith("/api/generate"):
                base_url = base_url[: -len("/api/generate")] + "/v1"
            elif not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
            os.environ["OPENAI_BASE_URL"] = base_url
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "dummy"

        return guidance.models.OpenAI(args.model)

    raise ValueError(f"Unsupported backend: {args.backend}")


def build_modeler(args: argparse.Namespace) -> ModelSuggester:
    def make_suggester(model_name: str) -> ModelSuggester:
        import guidance

        suggester = ModelSuggester()
        suggester.llm = guidance.models.OpenAI(model_name)
        return suggester

    llm = build_llm(args)
    suggester = make_suggester(args.model)
    suggester.llm = llm
    return suggester


def build_identifier(args: argparse.Namespace) -> IdentificationSuggester:
    identifier = IdentificationSuggester()
    identifier.llm = build_llm(args)
    # Initialize the model_suggester dependency
    model_suggester = ModelSuggester()
    model_suggester.llm = build_llm(args)
    identifier.model_suggester = model_suggester
    return identifier


def build_validator(args: argparse.Namespace) -> ValidationSuggester:
    validator = ValidationSuggester()
    validator.llm = build_llm(args)
    # Initialize the model_suggester dependency
    model_suggester = ModelSuggester()
    model_suggester.llm = build_llm(args)
    validator.model_suggester = model_suggester
    return validator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pywhyllm suggesters on causal triples generated by this repository."
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "compare_filtered", "broader"],
        default="standard",
        help=(
            "standard: run modeler outputs only. "
            "compare_filtered: run on filtered triples and compute agreement metrics. "
            "broader: run modeler + identifier + validator outputs in one unified JSON."
        ),
    )
    parser.add_argument(
        "--triples",
        type=Path,
        default=Path("Causal_Triples/triples_swe_depth1.jsonl"),
        help="Path to a JSONL triples file.",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "openai_compatible"],
        default="openai",
        help="openai: official OpenAI API. openai_compatible: local/server endpoint using OpenAI-compatible API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model name for the selected backend (e.g., gpt-4 for OpenAI, llama3.1 for local endpoints).",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help=(
            "Optional API base URL for --backend openai_compatible. "
            "If you pass an Ollama /api/generate URL, it will be rewritten to the Ollama /v1 OpenAI-compatible base."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override. If omitted in openai_compatible mode, defaults to env OPENAI_API_KEY or 'dummy'.",
    )
    parser.add_argument(
        "--analysis-context",
        type=str,
        default="software engineering and computer science",
        help="Analysis context string used in prompts.",
    )
    parser.add_argument(
        "--max-factors",
        type=int,
        default=25,
        help="Top-N factors (by frequency) to include in pywhyllm reasoning.",
    )
    parser.add_argument(
        "--n-experts",
        type=int,
        default=3,
        help="Number of domain expertises to request.",
    )
    parser.add_argument(
        "--treatment",
        type=str,
        default=None,
        help="Optional treatment override; default uses the most frequent directed pair source.",
    )
    parser.add_argument(
        "--outcome",
        type=str,
        default=None,
        help="Optional outcome override; default uses the most frequent directed pair target.",
    )
    parser.add_argument(
        "--filtered-triples",
        type=Path,
        default=None,
        help="Path to filtered triples JSONL. If omitted in compare_filtered mode, inferred from --triples basename in Filtered_Causal_Triples.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help="Minimum pywhyllm vote count for an edge to be included in agreement metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write JSON output.",
    )
    args = parser.parse_args()

    if args.output is None:
        if args.mode == "compare_filtered":
            args.output = Path("Run_Info/pywhyllm_compare_results.json")
        elif args.mode == "broader":
            args.output = Path("Run_Info/pywhyllm_broader_results.json")
        else:
            args.output = Path("Run_Info/pywhyllm_swe_depth1_results.json")

    source_triples_path = args.triples
    if args.mode == "compare_filtered":
        if args.filtered_triples is not None:
            source_triples_path = args.filtered_triples
        else:
            source_triples_path = Path("Filtered_Causal_Triples") / args.triples.name

    if not source_triples_path.exists():
        raise FileNotFoundError(f"Triple file not found: {source_triples_path}")

    triples = load_triples(source_triples_path)
    ranked_factors = factor_ranking(triples)
    all_factors = [name for name, _ in ranked_factors[: args.max_factors]]

    if len(all_factors) < 2:
        raise ValueError("Need at least two factors from triples to run pywhyllm.")

    if args.treatment and args.outcome:
        treatment = normalize_text(args.treatment)
        outcome = normalize_text(args.outcome)
    else:
        treatment, outcome = choose_default_treatment_outcome(triples)

    # Ensure treatment/outcome are represented in the factor list used by pywhyllm.
    if treatment not in all_factors:
        all_factors.append(treatment)
    if outcome not in all_factors:
        all_factors.append(outcome)

    modeler = build_modeler(args)

    domain_expertises = modeler.suggest_domain_expertises(
        all_factors,
        n_experts=args.n_experts,
        analysis_context=args.analysis_context,
    )

    confounders_counter, confounders = modeler.suggest_confounders(
        treatment=treatment,
        outcome=outcome,
        all_factors=all_factors,
        expertise_list=domain_expertises,
        analysis_context=args.analysis_context,
    )

    pairwise_edges = modeler.suggest_relationships(
        treatment=treatment,
        outcome=outcome,
        all_factors=all_factors,
        expertise_list=domain_expertises,
        relationship_strategy=RelationshipStrategy.Pairwise,
        analysis_context=args.analysis_context,
    )

    output_data = {
        "mode": args.mode,
        "triples_file": str(source_triples_path),
        "analysis_context": args.analysis_context,
        "backend": args.backend,
        "model": args.model,
        "max_factors": args.max_factors,
        "treatment": treatment,
        "outcome": outcome,
        "domain_expertises": domain_expertises,
        "confounders_counter": confounders_counter,
        "confounders": confounders,
        "pairwise_edges": [
            {"cause": k[0], "effect": k[1], "votes": v}
            for k, v in sorted(pairwise_edges.items(), key=lambda x: x[1], reverse=True)
        ],
        "factor_frequency_top": [
            {"factor": name, "count": count}
            for name, count in ranked_factors[: args.max_factors]
        ],
    }

    if args.mode == "compare_filtered":
        extracted_counts = extracted_edge_counts(triples, all_factors)
        metrics = compute_agreement_metrics(
            extracted_edges=extracted_counts,
            pywhyllm_edges=pairwise_edges,
            min_votes=args.min_votes,
        )
        output_data["agreement_metrics"] = metrics
        output_data["extracted_edges"] = [
            {"cause": k[0], "effect": k[1], "count": v}
            for k, v in sorted(extracted_counts.items(), key=lambda x: x[1], reverse=True)
        ]

    if args.mode == "broader":
        identifier = build_identifier(args)
        validator = build_validator(args)

        backdoor_counter, backdoor = identifier.suggest_backdoor(
            treatment=treatment,
            outcome=outcome,
            all_factors=all_factors,
            expertise_list=domain_expertises,
            analysis_context=args.analysis_context,
        )
        mediators_edges, mediators = identifier.suggest_mediators(
            treatment=treatment,
            outcome=outcome,
            all_factors=all_factors,
            expertise_list=domain_expertises,
            analysis_context=args.analysis_context,
        )
        iv_edges, ivs = identifier.suggest_ivs(
            treatment=treatment,
            outcome=outcome,
            all_factors=all_factors,
            expertise_list=domain_expertises,
            analysis_context=args.analysis_context,
        )

        latent_confounders_counter, latent_confounders = validator.suggest_latent_confounders(
            treatment=treatment,
            outcome=outcome,
            expertise_list=domain_expertises,
            analysis_context=args.analysis_context,
        )
        negative_controls_counter, negative_controls = validator.suggest_negative_controls(
            treatment=treatment,
            outcome=outcome,
            all_factors=all_factors,
            expertise_list=domain_expertises,
            analysis_context=args.analysis_context,
        )
        original_edges, critiqued_edges = validator.critique_graph(
            all_factors=all_factors,
            edges=pairwise_edges,
            experts=domain_expertises,
            relationship_strategy=RelationshipStrategy.Pairwise,
            analysis_context=args.analysis_context,
        )

        output_data["identification"] = {
            "backdoor_counter": backdoor_counter,
            "backdoor": backdoor,
            "mediators_edges": mediators_edges,
            "mediators": mediators,
            "iv_edges": iv_edges,
            "ivs": ivs,
        }
        output_data["validation"] = {
            "latent_confounders_counter": latent_confounders_counter,
            "latent_confounders": latent_confounders,
            "negative_controls_counter": negative_controls_counter,
            "negative_controls": negative_controls,
            "original_edges": original_edges,
            "critiqued_edges": critiqued_edges,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(output_data), f, indent=2)

    print(f"Saved pywhyllm results to: {args.output}")
    print(f"Treatment: {treatment}")
    print(f"Outcome: {outcome}")
    print(f"Factors used: {len(all_factors)}")
    print(f"Domain expertises: {domain_expertises}")
    print(f"Pairwise edges suggested: {len(pairwise_edges)}")
    if args.mode == "compare_filtered":
        directed = output_data["agreement_metrics"]["directed"]
        print(
            "Agreement (directed): "
            f"P={directed['precision']:.3f}, R={directed['recall']:.3f}, F1={directed['f1']:.3f}, "
            f"overlap={directed['overlap_edges']}"
        )
    if args.mode == "broader":
        print(f"Backdoor candidates: {len(output_data['identification']['backdoor'])}")
        print(f"Mediators: {len(output_data['identification']['mediators'])}")
        print(f"IVs: {len(output_data['identification']['ivs'])}")
        print(f"Latent confounders: {len(output_data['validation']['latent_confounders'])}")
        print(f"Negative controls: {len(output_data['validation']['negative_controls'])}")
        print(f"Critiqued edges suggested: {len(output_data['validation']['critiqued_edges'])}")


if __name__ == "__main__":
    main()
