"""Run Hugging Face models one at a time and persist resumable Drive-friendly results."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.one_shot_rag import run_one_shot
from baselines.react import run_react_style
from experiments.run_experiment import load_data
from src.evaluation.metrics import summarise
from src.evaluation.reporting import plot_comparison
from src.pipeline import run_adaptive
from src.planner.adaptive_planner import AdaptiveQueryPlanner, PlannerConfig
from src.query_generation.providers import build_provider
from src.retrieval.bm25 import BM25Retriever
from src.stopping.stopping import StoppingConfig


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def append_record(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def run_records(root: Path, run_id: str) -> list[dict]:
    """Read every method/model record already saved for this run, including resumed stages."""
    records = []
    for path in (root / "raw_results").glob(f"**/{run_id}.jsonl"):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return records


def csv_write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_hotpotqa_dev.yaml")
    parser.add_argument("--output", required=True, help="Drive results root or a local equivalent")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--models", help="Comma-separated Hugging Face model ids; defaults to config models")
    parser.add_argument("--methods", default="one_shot,react_style,adaptive")
    parser.add_argument("--smoke", action="store_true", help="Run only the first evaluation question")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["llm"]["enabled"] = True
    root = Path(args.output)
    directories = {name: root / name for name in ("configs", "logs", "model_metadata", "raw_results", "traces", "summaries", "comparisons", "plots", "failures")}
    for directory in directories.values(): directory.mkdir(parents=True, exist_ok=True)
    (directories["configs"] / f"{args.run_id}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    docs, examples = load_data(config, None)
    if args.smoke: examples = examples[:1]
    retriever = BM25Retriever(docs)
    retrieval, stop_data, query = config["retrieval"], config["stopping"], config["query_generation"]
    stopping = StoppingConfig(retrieval["max_iterations"], retrieval["budget"], stop_data["min_iterations"], stop_data["min_new_documents"], stop_data["min_missing_coverage"], stop_data["no_gain_limit"])
    plan = config["planner"]
    planner = AdaptiveQueryPlanner(retriever, PlannerConfig(plan["relevance_weight"], plan["missing_information_weight"], plan["redundancy_weight"], plan["cost_weight"], tuple(plan["enabled"])), retrieval["positive_scores_only"])
    methods = set(args.methods.split(","))
    all_records, failures = [], []
    if "one_shot" in methods:
        path = directories["raw_results"] / "retrieval_only" / "one_shot" / f"{args.run_id}.jsonl"
        done = existing_ids(path)
        for example in examples:
            if example.id not in done:
                record = run_one_shot(example, retriever, retrieval["top_k"], retrieval["positive_scores_only"])
                append_record(path, record)
        all_records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    selected = args.models.split(",") if args.models else [item["id"] for item in config["llm"]["models"]]
    model_specs = {item["id"]: item for item in config["llm"]["models"]}
    for model_id in selected if methods & {"react_style", "adaptive"} else []:
        spec = model_specs.get(model_id, {"id": model_id, "trust_remote_code": False})
        provider = None
        try:
            provider = build_provider(config["llm"], spec)
            provider._load()  # Fail before any method is reported as successful.
            (directories["model_metadata"] / f"{safe_name(model_id)}_{args.run_id}.json").write_text(json.dumps(provider.metadata, indent=2), encoding="utf-8")
            for method in ("react_style", "adaptive"):
                if method not in methods: continue
                path = directories["raw_results"] / safe_name(model_id) / method / f"{args.run_id}.jsonl"
                done = existing_ids(path)
                for example in examples:
                    if example.id in done: continue
                    record = run_react_style(example, retriever, retrieval["top_k"], query["candidate_count"], stopping, provider, retrieval["positive_scores_only"]) if method == "react_style" else run_adaptive(example, retriever, planner, stopping, retrieval["top_k"], query["candidate_count"], provider, False)
                    append_record(path, record)
                    (directories["traces"] / safe_name(model_id) / method).mkdir(parents=True, exist_ok=True)
                    (directories["traces"] / safe_name(model_id) / method / f"{example.id}_{args.run_id}.json").write_text(json.dumps(record.get("state", {}), indent=2), encoding="utf-8")
                all_records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        except Exception as error:
            failure = {"run_id": args.run_id, "model_id": model_id, "status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "provider_events": getattr(provider, "events", [])}
            failures.append(failure); append_record(directories["failures"] / f"{args.run_id}.jsonl", failure)
        finally:
            if provider:
                (directories["logs"] / f"{safe_name(model_id)}_{args.run_id}.json").write_text(json.dumps(provider.events, indent=2), encoding="utf-8")
                provider.release()
    valid = [record for record in run_records(root, args.run_id) if "best_step_recall_at_k" in record]
    grouped = {}
    for record in valid: grouped.setdefault((record["model_id"], record["method"]), []).append(record)
    summary_rows = [{"model_id": model, "method": method, **summarise(records)} for (model, method), records in grouped.items()]
    csv_write(directories["comparisons"] / "model_method_comparison.csv", summary_rows)
    csv_write(directories["comparisons"] / "per_question_results.csv", valid)
    csv_write(directories["comparisons"] / "model_comparison.csv", [row for row in summary_rows if row["method"] != "one_shot"])
    method_rows = []
    for method in sorted({row["method"] for row in summary_rows}):
        records = [record for record in valid if record["method"] == method]
        method_rows.append({"method": method, **summarise(records)})
    csv_write(directories["comparisons"] / "method_comparison.csv", method_rows)
    (directories["summaries"] / "comparison_summary.json").write_text(json.dumps({"run_id": args.run_id, "records": len(valid), "failures": failures, "model_method": summary_rows, "method": method_rows}, indent=2), encoding="utf-8")
    if method_rows:
        plot_comparison(method_rows, directories["plots"])
        source = directories["plots"] / "comparison.png"
        source.replace(directories["plots"] / "comparison_dashboard.png")
    print(json.dumps({"run_id": args.run_id, "records": len(valid), "failures": len(failures), "root": str(root)}, indent=2))


if __name__ == "__main__": main()
