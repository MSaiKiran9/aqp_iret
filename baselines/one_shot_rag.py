"""One-shot BM25 evidence-retrieval baseline."""
from src.evaluation.metrics import evidence_metrics


def run_one_shot(example, retriever, top_k: int, positive_scores_only: bool = True) -> dict:
    results = retriever.search(example.question, top_k, positive_scores_only)
    ids = [doc.id for doc, _ in results]
    history = [{"retrieved_ids": ids}]
    return {"id": example.id, "method": "one_shot", **evidence_metrics(history, ids, example.supporting_ids, top_k), "steps": 1, "candidate_evaluation_cost": 0, "selected_retrieval_cost": len(results), "retrieval_cost": len(results), "stopping_reason": "one_shot", "retrieved_ids": " ".join(ids), "state": {"query": example.question, "retrieved_ids": ids}}
