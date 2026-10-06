"""Comparison-table and plot helpers for experiment outputs."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_comparison(summaries: dict[str, dict], output_dir: str | Path) -> list[dict]:
    rows = [{"method": method, "evidence_recall_at_k": summary["evidence_recall_at_k"], "accumulated_evidence_recall": summary["accumulated_evidence_recall"], "average_retrieval_steps": summary["average_retrieval_steps"], "average_retrieval_cost": summary["average_retrieval_cost"]} for method, summary in summaries.items()]
    with (Path(output_dir) / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader(); writer.writerows(rows)
    return rows


def plot_comparison(rows: list[dict], output_dir: str | Path):
    labels = [row["method"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, key, title in zip(axes, ["evidence_recall_at_k", "average_retrieval_steps", "average_retrieval_cost"], ["Evidence Recall@k", "Average Steps", "Retrieval Cost"]):
        axis.bar(labels, [row[key] for row in rows])
        axis.set_title(title); axis.tick_params(axis="x", rotation=20)
    figure.tight_layout(); figure.savefig(Path(output_dir) / "comparison.png", dpi=160); plt.close(figure)
