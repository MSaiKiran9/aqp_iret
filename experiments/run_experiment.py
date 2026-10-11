"""Reproducible comparison of all retrieval methods and models."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import demo_dataset, load_hotpotqa, load_json_dataset
from src.retrieval.bm25 import BM25Retriever
from src.planner.adaptive_planner import AdaptiveQueryPlanner, PlannerConfig
from src.stopping.stopping import StoppingConfig
from src.query_generation.providers import build_provider
from src.pipeline import run_adaptive
from src.evaluation.metrics import save_results, summarise
from src.evaluation.reporting import generate_candidate_traces, generate_report, plot_comparison, save_comparison
from baselines.one_shot_rag import run_one_shot
from baselines.react import run_react_style


def load_data(config: dict, dataset_path: str | None):
    data = config["dataset"]
    if dataset_path:
        return load_json_dataset(dataset_path, data.get("sample_size"), config["seed"])
    if data["name"] == "hotpotqa":
        return load_hotpotqa(data["split"], data.get("sample_size"), config["seed"])
    return demo_dataset()


def sanitize_tag(model_id: str) -> str:
    name = model_id.split("/")[-1].lower()
    return name.replace("-", "_").replace(".", "_")


def main():
    parser = argparse.ArgumentParser(description="Run AQP-IRET and baseline retrieval experiments.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--dataset", help="Optional repository-neutral JSON dataset path")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument("--models", help="Comma-separated model IDs to evaluate")
    parser.add_argument("--include-fallback", action="store_true", help="Also evaluate deterministic fallback alongside LLM")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    docs, examples = load_data(config, args.dataset)
    retriever = BM25Retriever(docs)

    retrieval = config["retrieval"]
    stop_values = config["stopping"]
    stopping = StoppingConfig(
        retrieval["max_iterations"],
        retrieval["budget"],
        stop_values["min_iterations"],
        stop_values["min_new_documents"],
        stop_values["min_missing_coverage"],
        stop_values["no_gain_limit"]
    )
    values = config["planner"]
    planner = AdaptiveQueryPlanner(
        retriever,
        PlannerConfig(
            values["relevance_weight"],
            values["missing_information_weight"],
            values["redundancy_weight"],
            values["cost_weight"],
            tuple(values["enabled"])
        ),
        retrieval["positive_scores_only"]
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output or config["output"]["directory"]) / stamp
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": stamp,
        "config_file": str(args.config),
        "dataset_name": config.get("dataset", {}).get("name"),
        "dataset_sample_size": config.get("dataset", {}).get("sample_size"),
        "question_count": len(examples),
        "corpus_document_count": len(docs),
        "seed": config.get("seed", 42),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    query_cfg = config["query_generation"]
    top_k = retrieval["top_k"]
    pos_only = retrieval["positive_scores_only"]
    cand_count = query_cfg["candidate_count"]
    verbose = config.get("verbose", False)

    summaries = {}
    all_adaptive_records = []

    # 1. One-Shot Baseline (shared across all methods/models)
    print("Evaluating One-Shot RAG baseline...")
    one_shot_records = [run_one_shot(example, retriever, top_k, pos_only) for example in examples]
    summaries["one_shot"] = summarise(one_shot_records)
    summaries["one_shot"]["model_id"] = "retrieval_only"
    save_results(one_shot_records, summaries["one_shot"], output, "one_shot", config, metadata)
    print("one_shot:", summaries["one_shot"])

    # 2. LLM Evaluation (multi-model or single model)
    llm_enabled = config.get("llm", {}).get("enabled", False)
    configured_models = config.get("llm", {}).get("models", [])
    if args.models:
        configured_models = [{"id": m.strip(), "trust_remote_code": False} for m in args.models.split(",") if m.strip()]

    run_multi = len(configured_models) > 1 or args.include_fallback

    if llm_enabled and configured_models:
        for model_spec in configured_models:
            model_id = model_spec["id"]
            model_tag = sanitize_tag(model_id)
            react_name = f"react_style_{model_tag}" if run_multi else "react_style"
            adaptive_name = f"adaptive_{model_tag}" if run_multi else "adaptive"

            print(f"\nEvaluating with model: {model_id}...")
            provider = build_provider(config["llm"], model_spec)

            # ReAct baseline reusing the loaded provider
            print(f"Running ReAct-style [{model_id}]...")
            react_records = [run_react_style(example, retriever, top_k, cand_count, stopping, provider, pos_only) for example in examples]
            summaries[react_name] = summarise(react_records)
            summaries[react_name]["model_id"] = model_id
            save_results(react_records, summaries[react_name], output, react_name, config, metadata)
            print(f"{react_name}:", summaries[react_name])

            # AQP-IRET reusing the loaded provider
            print(f"Running AQP-IRET [{model_id}]...")
            adaptive_records = [run_adaptive(example, retriever, planner, stopping, top_k, cand_count, provider, verbose) for example in examples]
            summaries[adaptive_name] = summarise(adaptive_records)
            summaries[adaptive_name]["model_id"] = model_id
            save_results(adaptive_records, summaries[adaptive_name], output, adaptive_name, config, metadata)
            print(f"{adaptive_name}:", summaries[adaptive_name])
            all_adaptive_records.extend(adaptive_records)

            if hasattr(provider, "release"):
                provider.release()

    # 3. Deterministic Fallback Run (if LLMs disabled or explicitly requested)
    if not llm_enabled or args.include_fallback:
        react_name = "react_style_fallback" if run_multi else "react_style"
        adaptive_name = "adaptive_fallback" if run_multi else "adaptive"

        print(f"\nEvaluating deterministic fallback ({react_name}, {adaptive_name})...")
        react_records = [run_react_style(example, retriever, top_k, cand_count, stopping, None, pos_only) for example in examples]
        summaries[react_name] = summarise(react_records)
        summaries[react_name]["model_id"] = "deterministic_fallback"
        save_results(react_records, summaries[react_name], output, react_name, config, metadata)
        print(f"{react_name}:", summaries[react_name])

        adaptive_records = [run_adaptive(example, retriever, planner, stopping, top_k, cand_count, None, verbose) for example in examples]
        summaries[adaptive_name] = summarise(adaptive_records)
        summaries[adaptive_name]["model_id"] = "deterministic_fallback"
        save_results(adaptive_records, summaries[adaptive_name], output, adaptive_name, config, metadata)
        print(f"{adaptive_name}:", summaries[adaptive_name])
        all_adaptive_records.extend(adaptive_records)

    # 4. Save Artifacts: Comparison, Plots, Traces, Report
    rows = save_comparison(summaries, output)
    plot_comparison(rows, output)
    generate_report(summaries, rows, output, config, metadata)
    generate_candidate_traces(all_adaptive_records, output)
    print(f"\nSaved all experimental evaluation artifacts to: {output}")


if __name__ == "__main__":
    main()
