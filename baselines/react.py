"""Fair simplified ReAct-style retrieval baseline (not a paper reproduction)."""
from src.evidence.state import EvidenceState
from src.query_generation.generator import generate_candidates, get_missing_information
from src.stopping.stopping import should_stop
from src.evaluation.metrics import evidence_metrics
import time


def run_react_style(example, retriever, top_k: int, candidate_count: int, stopping_config, provider=None, positive_scores_only: bool = True) -> dict:
    """Uses only current state; it never reads answer or supporting annotations for control flow."""
    started = time.perf_counter()
    event_start = len(getattr(provider, "events", []))
    retrieval_seconds = 0.0
    state = EvidenceState(example.question)
    reason = "max_iterations"
    error = None
    try:
        for _ in range(stopping_config.max_iterations):
            state.missing_information = get_missing_information(example.question, state, provider)
            candidates = generate_candidates(example.question, state, state.missing_information, candidate_count, provider)
            if not candidates or state.retrieval_cost + top_k > stopping_config.retrieval_budget:
                reason = "budget_exhausted"; break
            query = candidates[0]
            retrieval_started = time.perf_counter()
            results = retriever.search(query, top_k, positive_scores_only)
            retrieval_seconds += time.perf_counter() - retrieval_started
            state.add_retrieval(query, results, [], 0)
            stop, reason = should_stop(state, stopping_config)
            if stop: break
    except Exception as exc:
        error = str(exc)
        reason = f"error: {type(exc).__name__}"
    metrics = evidence_metrics(state.history, state.evidence_ids, example.supporting_ids, top_k)
    events = getattr(provider, "events", [])[event_start:]
    generation_seconds = sum(item.get("seconds", 0.0) for item in events if item.get("kind") == "generation")
    generated_tokens = sum(e.get("output", {}).get("new_token_count", 0) for e in events if "output" in e)
    model_calls = sum(1 for e in events if e.get("kind") in ("generation_attempt", "generation"))
    retries = sum(1 for e in events if e.get("kind") == "generation_attempt" and not e.get("success", False))
    structured_output_failures = sum(1 for e in events if not e.get("success", True))
    return {
        "id": example.id,
        "method": "react_style",
        **metrics,
        "steps": state.iteration,
        "candidate_evaluation_cost": 0,
        "selected_retrieval_cost": state.selected_retrieval_cost,
        "retrieval_cost": state.retrieval_cost,
        "generation_latency_seconds": round(generation_seconds, 6),
        "retrieval_latency_seconds": round(retrieval_seconds, 6),
        "end_to_end_latency_seconds": round(time.perf_counter() - started, 6),
        "generated_tokens": generated_tokens,
        "model_calls": model_calls,
        "retries": retries,
        "structured_output_failures": structured_output_failures,
        "model_id": getattr(provider, "model_id", "deterministic_fallback"),
        "generation_events": events,
        "stopping_reason": reason,
        "error": error,
        "retrieved_ids": " ".join(state.evidence_ids),
        "state": state.to_dict()
    }
