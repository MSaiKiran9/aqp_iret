"""Evidence metrics shared by all methods."""
from pathlib import Path
import csv
import json


def evidence_recall(retrieved_ids: list[str], supporting_ids: list[str]) -> float:
    """Fraction of required evidence documents present in a retrieved document set."""
    return len(set(retrieved_ids) & set(supporting_ids)) / len(supporting_ids) if supporting_ids else 0.0


def evidence_metrics(history: list[dict], accumulated_ids: list[str], supporting_ids: list[str], top_k: int) -> dict:
    """Per-step Recall@k, any-step Recall@k, and accumulated evidence recall are distinct."""
    per_step = [evidence_recall(item["retrieved_ids"][:top_k], supporting_ids) for item in history]
    return {"first_step_recall_at_k": per_step[0] if per_step else 0.0, "best_step_recall_at_k": max(per_step, default=0.0), "accumulated_evidence_recall": evidence_recall(accumulated_ids, supporting_ids)}


def summarise(records: list[dict]) -> dict:
    count = len(records) or 1
    reasons = {reason: sum(record.get("stopping_reason") == reason for record in records) for reason in sorted({r.get("stopping_reason") for r in records if r.get("stopping_reason")})}
    return {
        "questions": len(records),
        "first_step_recall_at_k": round(sum(r.get("first_step_recall_at_k", 0.0) for r in records) / count, 4),
        "evidence_recall_at_k": round(sum(r.get("best_step_recall_at_k", 0.0) for r in records) / count, 4),
        "accumulated_evidence_recall": round(sum(r.get("accumulated_evidence_recall", 0.0) for r in records) / count, 4),
        "average_retrieval_steps": round(sum(r.get("steps", 0) for r in records) / count, 4),
        "average_candidate_evaluation_cost": round(sum(r.get("candidate_evaluation_cost", 0) for r in records) / count, 4),
        "average_selected_retrieval_cost": round(sum(r.get("selected_retrieval_cost", 0) for r in records) / count, 4),
        "average_retrieval_cost": round(sum(r.get("retrieval_cost", 0) for r in records) / count, 4),
        "average_generation_latency_seconds": round(sum(r.get("generation_latency_seconds", 0.0) for r in records) / count, 4),
        "average_retrieval_latency_seconds": round(sum(r.get("retrieval_latency_seconds", 0.0) for r in records) / count, 4),
        "average_end_to_end_latency_seconds": round(sum(r.get("end_to_end_latency_seconds", 0.0) for r in records) / count, 4),
        "average_generated_tokens": round(sum(r.get("generated_tokens", 0) for r in records) / count, 4),
        "total_generated_tokens": sum(r.get("generated_tokens", 0) for r in records),
        "total_model_calls": sum(r.get("model_calls", 0) for r in records),
        "total_retries": sum(r.get("retries", 0) for r in records),
        "total_structured_output_failures": sum(r.get("structured_output_failures", 0) for r in records),
        "successful_evidence_cases": sum(r.get("accumulated_evidence_recall", 0.0) == 1.0 for r in records),
        "failed_cases": sum(1 for r in records if r.get("error") or str(r.get("stopping_reason", "")).startswith("error")),
        "stopping_reasons": reasons
    }


def save_results(records: list[dict], summary: dict, output_dir: str | Path, name: str, config: dict | None = None, metadata: dict | None = None):
    path = Path(output_dir); path.mkdir(parents=True, exist_ok=True)
    (path / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (path / f"{name}_raw.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if config is not None:
        (path / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if metadata is not None:
        (path / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fields = [
        "id", "method", "model_id", "first_step_recall_at_k", "best_step_recall_at_k",
        "accumulated_evidence_recall", "steps", "candidate_evaluation_cost", "selected_retrieval_cost",
        "retrieval_cost", "generation_latency_seconds", "retrieval_latency_seconds",
        "end_to_end_latency_seconds", "generated_tokens", "model_calls", "retries",
        "structured_output_failures", "stopping_reason", "error", "retrieved_ids"
    ]
    with (path / f"{name}_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
