"""Deterministic, inspectable adaptive query planner."""
from dataclasses import dataclass
import time
from src.retrieval.bm25 import tokenize


@dataclass
class PlannerConfig:
    relevance_weight: float = .45
    missing_information_weight: float = .35
    redundancy_weight: float = .15
    cost_weight: float = .05
    enabled: tuple[str, ...] = ("relevance", "missing", "redundancy", "cost")


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


class AdaptiveQueryPlanner:
    def __init__(self, retriever, config: PlannerConfig, positive_scores_only: bool = True):
        self.retriever, self.config, self.positive_scores_only = retriever, config, positive_scores_only

    def score_candidates(self, candidates: list[str], state, top_k: int) -> list[dict]:
        previews = []
        for query in candidates:
            started = time.perf_counter()
            previews.append((query, self.retriever.search(query, top_k, self.positive_scores_only), time.perf_counter() - started))
        max_bm25 = max((score for _, results, _ in previews for _, score in results), default=0.0) or 1.0
        evidence_tokens = set(tokenize(" ".join(doc.text for doc in state.accumulated_evidence)))
        need_tokens = set(tokenize(" ".join(state.missing_information.get("target_terms", []))))
        scores = []
        for query, results, elapsed in previews:
            relevance = max((score for _, score in results), default=0.0) / max_bm25
            missing = jaccard(set(tokenize(query)), need_tokens)
            redundancy = sum(jaccard(set(tokenize(doc.text)), evidence_tokens) for doc, _ in results) / max(1, len(results))
            cost = len(results) / max(1, top_k)
            c = self.config
            total = (c.relevance_weight * relevance if "relevance" in c.enabled else 0.0) + (c.missing_information_weight * missing if "missing" in c.enabled else 0.0) - (c.redundancy_weight * redundancy if "redundancy" in c.enabled else 0.0) - (c.cost_weight * cost if "cost" in c.enabled else 0.0)
            scores.append({"query": query, "relevance": round(relevance, 4), "missing": round(missing, 4), "redundancy": round(redundancy, 4), "cost": round(cost, 4), "total": round(total, 4), "preview_ids": [doc.id for doc, _ in results], "preview_cost": len(results), "preview_seconds": round(elapsed, 6)})
        return scores

    def select(self, candidates, state, top_k):
        scores = self.score_candidates(candidates, state, top_k)
        return max(scores, key=lambda row: row["total"]), scores
