"""Fair simplified ReAct-style retrieval baseline (not a paper reproduction)."""
from src.evidence.state import EvidenceState
from src.query_generation.generator import generate_candidates, get_missing_information
from src.stopping.stopping import should_stop
from src.evaluation.metrics import evidence_metrics


def run_react_style(example, retriever, top_k: int, candidate_count: int, stopping_config, provider=None, positive_scores_only: bool = True) -> dict:
    """Uses only current state; it never reads answer or supporting annotations for control flow."""
    state = EvidenceState(example.question)
    reason = "max_iterations"
    for _ in range(stopping_config.max_iterations):
        state.missing_information = get_missing_information(example.question, state, provider)
        candidates = generate_candidates(example.question, state, state.missing_information, candidate_count, provider)
        if not candidates or state.retrieval_cost + top_k > stopping_config.retrieval_budget:
            reason = "budget_exhausted"; break
        query = candidates[0]
        results = retriever.search(query, top_k, positive_scores_only)
        state.add_retrieval(query, results, [], 0)
        stop, reason = should_stop(state, stopping_config)
        if stop: break
    metrics = evidence_metrics(state.history, state.evidence_ids, example.supporting_ids, top_k)
    return {"id": example.id, "method": "react_style", **metrics, "steps": state.iteration, "candidate_evaluation_cost": 0, "selected_retrieval_cost": state.selected_retrieval_cost, "retrieval_cost": state.retrieval_cost, "stopping_reason": reason, "retrieved_ids": " ".join(state.evidence_ids), "state": state.to_dict()}
