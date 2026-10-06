"""Inference-only, explainable stopping policy."""
from dataclasses import dataclass


@dataclass
class StoppingConfig:
    max_iterations: int = 3
    retrieval_budget: int = 24
    min_iterations: int = 2
    min_new_documents: int = 1
    min_missing_coverage: float = .75
    no_gain_limit: int = 1


def can_afford_iteration(state, candidate_count: int, top_k: int, config: StoppingConfig) -> bool:
    """Reserve cost for every candidate preview plus one selected retrieval."""
    return state.retrieval_cost + (candidate_count + 1) * top_k <= config.retrieval_budget


def should_stop(state, config: StoppingConfig) -> tuple[bool, str]:
    if state.retrieval_cost >= config.retrieval_budget:
        return True, "budget_exhausted"
    if state.iteration >= config.max_iterations:
        return True, "max_iterations"
    if state.iteration < config.min_iterations:
        return False, "continue"
    coverage = float(state.missing_information.get("coverage", 0.0))
    recent = state.history[-config.no_gain_limit:]
    no_gain = bool(recent) and all(item["new_document_count"] < config.min_new_documents for item in recent)
    if coverage >= config.min_missing_coverage and not no_gain:
        return True, "evidence_sufficient"
    if no_gain:
        return True, "no_information_gain"
    return False, "continue"
