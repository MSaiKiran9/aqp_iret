"""Proposed adaptive query-planning retrieval loop."""
from src.evidence.state import EvidenceState
from src.query_generation.generator import generate_candidates, get_missing_information
from src.stopping.stopping import can_afford_iteration, should_stop
from src.evaluation.metrics import evidence_metrics
import time


def run_adaptive(example, retriever, planner, stopping_config, top_k: int, candidate_count: int, provider=None, verbose: bool = False) -> dict:
    started = time.perf_counter()
    event_start = len(getattr(provider, "events", []))
    retrieval_seconds = 0.0
    state = EvidenceState(example.question)
    reason = "continue"
    while True:
        state.missing_information = get_missing_information(example.question, state, provider)
        state.candidate_queries = generate_candidates(example.question, state, state.missing_information, candidate_count, provider)
        if not state.candidate_queries:
            reason = "no_candidates"; break
        if not can_afford_iteration(state, len(state.candidate_queries), top_k, stopping_config):
            reason = "budget_exhausted"; break
        selected, scores = planner.select(state.candidate_queries, state, top_k)
        retrieval_seconds += sum(row["preview_seconds"] for row in scores)
        evaluation_cost = sum(row["preview_cost"] for row in scores)
        retrieval_started = time.perf_counter()
        results = retriever.search(selected["query"], top_k, planner.positive_scores_only)
        retrieval_seconds += time.perf_counter() - retrieval_started
        state.add_retrieval(selected["query"], results, scores, evaluation_cost)
        if verbose:
            print(f"Iteration {state.iteration}: selected={selected['query']} retrieved={[d.id for d, _ in results]} cost={state.retrieval_cost}")
            for row in scores: print({key: value for key, value in row.items() if key != "preview_ids"})
        stop, reason = should_stop(state, stopping_config)
        if stop: break
    metrics = evidence_metrics(state.history, state.evidence_ids, example.supporting_ids, top_k)
    events = getattr(provider, "events", [])[event_start:]
    generation_seconds = sum(item.get("seconds", 0.0) for item in events if item["kind"] == "generation")
    return {"id": example.id, "method": "adaptive", **metrics, "steps": state.iteration, "candidate_evaluation_cost": state.candidate_evaluation_cost, "selected_retrieval_cost": state.selected_retrieval_cost, "retrieval_cost": state.retrieval_cost, "generation_latency_seconds": round(generation_seconds, 6), "retrieval_latency_seconds": round(retrieval_seconds, 6), "end_to_end_latency_seconds": round(time.perf_counter() - started, 6), "model_id": getattr(provider, "model_id", "deterministic_fallback"), "generation_events": events, "stopping_reason": reason, "retrieved_ids": " ".join(state.evidence_ids), "state": state.to_dict()}
