"""Information-need analysis and LLM-backed candidate query generation."""
import re
from typing import Protocol
from src.retrieval.bm25 import tokenize

_QUESTION_WORDS = {"which", "what", "who", "where", "when", "why", "how", "the", "a", "an"}


class JsonCompleter(Protocol):
    def complete_json(self, prompt: str) -> dict: ...


def _entities(text: str) -> list[str]:
    phrases = re.findall(r"(?:[A-Z][A-Za-z0-9-]+(?:\s+(?:[A-Z][A-Za-z0-9-]+|of|the|and))*)", text)
    return [p.strip() for p in phrases if p.lower() not in _QUESTION_WORDS and len(p.strip()) > 2]


def _fallback_need(question: str, state) -> dict:
    observed = _entities(" ".join(d.text for d in state.accumulated_evidence))
    question_entities = _entities(question)
    targets = [token for token in tokenize(question) if token in {"country", "person", "organization", "city", "date", "year", "place"}]
    focus = sorted(dict.fromkeys(observed), key=lambda value: (-len(value.split()), value))[:3] or question_entities
    coverage = min(1.0, len(state.accumulated_evidence) / 2) if state.iteration else 0.0
    return {"known_facts": observed[-5:], "missing_information": f"Find the {' and '.join(targets) or 'answer'} linked to {'; '.join(focus[:2]) or 'the question'}.", "target_terms": focus[:3] + targets, "coverage": coverage, "source": "deterministic_fallback"}


def get_missing_information(question: str, state, provider: JsonCompleter | None = None) -> dict:
    """Use configured LLM structured output; fall back deterministically when unavailable."""
    if provider:
        prompt = ("Analyze remaining retrieval need for multi-hop QA. Return JSON with known_facts (list), "
                  "missing_information (string), target_terms (list), coverage (number 0..1).\n"
                  f"Question: {question}\nEvidence: {[d.text for d in state.accumulated_evidence]}")
        try:
            data = provider.complete_json(prompt)
            if isinstance(data.get("target_terms"), list) and isinstance(data.get("coverage"), (int, float)):
                return {"known_facts": [str(x) for x in data.get("known_facts", [])], "missing_information": str(data.get("missing_information", "")), "target_terms": [str(x) for x in data["target_terms"]], "coverage": max(0.0, min(1.0, float(data["coverage"]))), "source": "llm"}
        except (RuntimeError, ValueError, KeyError, TypeError):
            if getattr(provider, "strict", False):
                raise
    return _fallback_need(question, state)


def generate_candidates(question: str, state, need: dict, count: int, provider: JsonCompleter | None = None) -> list[str]:
    if provider:
        prompt = ("Generate concise, diverse retrieval queries for multi-hop QA. Return JSON {queries:[...]}; "
                  f"return at most {count} queries.\nQuestion: {question}\nEvidence: {[d.text for d in state.accumulated_evidence]}\nRemaining need: {need}")
        try:
            queries = [str(q).strip() for q in provider.complete_json(prompt).get("queries", []) if _valid_query(str(q))]
            if queries:
                return [q for q in dict.fromkeys(queries) if q not in set(state.query_history)][:count]
            if getattr(provider, "strict", False):
                raise ValueError("Model returned no valid candidate queries.")
        except (RuntimeError, ValueError, KeyError, TypeError):
            if getattr(provider, "strict", False):
                raise
    focus = " ".join(need.get("target_terms", [])[:2]) or " ".join(_entities(question)[:2]) or question
    target = " ".join(tokenize(need.get("missing_information", ""))[:4])
    candidates = [f"{focus} {target}".strip(), f"{focus} relationship", question]
    return [q for q in dict.fromkeys(candidates) if q and q not in set(state.query_history)][:count]


def _valid_query(value: str) -> bool:
    query = value.strip()
    return 2 <= len(query.split()) <= 20 and len(query) <= 200 and not any(marker in query for marker in ("{", "}", "\n"))
