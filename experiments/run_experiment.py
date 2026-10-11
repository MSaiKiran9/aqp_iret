"""Reproducible comparison of all retrieval methods."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
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
from src.evaluation.reporting import plot_comparison, save_comparison
from baselines.one_shot_rag import run_one_shot
from baselines.react import run_react_style


def load_data(config: dict, dataset_path: str | None):
    data = config["dataset"]
    if dataset_path:
        return load_json_dataset(dataset_path, data.get("sample_size"), config["seed"])
    if data["name"] == "hotpotqa":
        return load_hotpotqa(data["split"], data.get("sample_size"), config["seed"])
    return demo_dataset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", help="Optional repository-neutral JSON dataset")
    parser.add_argument("--output", help="Override output directory")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    docs, examples = load_data(config, args.dataset)
    retriever = BM25Retriever(docs)
    retrieval = config["retrieval"]; stop_values = config["stopping"]
    stopping = StoppingConfig(retrieval["max_iterations"], retrieval["budget"], stop_values["min_iterations"], stop_values["min_new_documents"], stop_values["min_missing_coverage"], stop_values["no_gain_limit"])
    values = config["planner"]
    planner = AdaptiveQueryPlanner(retriever, PlannerConfig(values["relevance_weight"], values["missing_information_weight"], values["redundancy_weight"], values["cost_weight"], tuple(values["enabled"])), retrieval["positive_scores_only"])
    model_spec = config["llm"]["models"][0] if config.get("llm", {}).get("models") else None
    provider = build_provider(config["llm"], model_spec)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output or config["output"]["directory"]) / stamp
    query = config["query_generation"]
    methods = {
        "one_shot": [run_one_shot(example, retriever, retrieval["top_k"], retrieval["positive_scores_only"]) for example in examples],
        "react_style": [run_react_style(example, retriever, retrieval["top_k"], query["candidate_count"], stopping, provider, retrieval["positive_scores_only"]) for example in examples],
        "adaptive": [run_adaptive(example, retriever, planner, stopping, retrieval["top_k"], query["candidate_count"], provider, config.get("verbose", False)) for example in examples],
    }
    summaries = {}
    for name, records in methods.items():
        summaries[name] = summarise(records); save_results(records, summaries[name], output, name, config)
        print(name, summaries[name])
    rows = save_comparison(summaries, output); plot_comparison(rows, output)
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
