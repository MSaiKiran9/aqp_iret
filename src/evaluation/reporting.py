"""Comparison-table and plot helpers for experiment outputs."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_comparison(summaries: dict[str, dict], output_dir: str | Path) -> list[dict]:
    rows = [{
        "method": method,
        "evidence_recall_at_k": summary["evidence_recall_at_k"],
        "accumulated_evidence_recall": summary["accumulated_evidence_recall"],
        "average_retrieval_steps": summary["average_retrieval_steps"],
        "average_candidate_preview_cost": summary.get("average_candidate_evaluation_cost", 0.0),
        "average_selected_retrieval_cost": summary.get("average_selected_retrieval_cost", 0.0),
        "average_retrieval_cost": summary["average_retrieval_cost"]
    } for method, summary in summaries.items()]
    with (Path(output_dir) / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader(); writer.writerows(rows)
    return rows


def plot_comparison(rows: list[dict], output_dir: str | Path):
    labels = [row["method"] for row in rows]
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    axes[0].bar(labels, [row["evidence_recall_at_k"] for row in rows], color="#2b5c8f")
    axes[0].set_title("Evidence Recall@k (Best Step)")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(labels, [row["accumulated_evidence_recall"] for row in rows], color="#3f8f5c")
    axes[1].set_title("Accumulated Evidence Recall")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(labels, [row["average_retrieval_steps"] for row in rows], color="#d97724")
    axes[2].set_title("Average Steps")
    axes[2].tick_params(axis="x", rotation=15)

    preview_costs = [row.get("average_candidate_preview_cost", 0.0) for row in rows]
    selected_costs = [row.get("average_selected_retrieval_cost", 0.0) for row in rows]
    axes[3].bar(labels, selected_costs, label="Selected Retrieval", color="#5c6bc0")
    axes[3].bar(labels, preview_costs, bottom=selected_costs, label="Candidate Previews", color="#ab47bc")
    axes[3].set_title("Retrieval Cost Breakdown")
    axes[3].legend(loc="upper left")
    axes[3].tick_params(axis="x", rotation=15)

    figure.tight_layout()
    figure.savefig(Path(output_dir) / "comparison.png", dpi=160)
    plt.close(figure)
