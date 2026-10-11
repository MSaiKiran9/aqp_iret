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
    targets = [token for token in tokenize(question) if token in {
        "country", "person", "organization", "city", "date", "year", "place",
        "author", "director", "founder", "film", "team", "college", "school",
        "university", "state", "album", "band", "company", "location", "national", "agency"
    }]
    focus = sorted(dict.fromkeys(observed), key=lambda value: (-len(value.split()), value))[:3] or question_entities
    coverage = min(1.0, len(state.accumulated_evidence) / 2) if state.iteration else 0.0
    return {"known_facts": observed[-5:], "missing_information": f"Find the {' and '.join(targets) or 'answer'} linked to {'; '.join(focus[:2]) or 'the question'}.", "target_terms": focus[:3] + targets, "coverage": coverage, "source": "deterministic_fallback"}


def get_missing_information(question: str, state, provider: JsonCompleter | None = None) -> dict:
    """Use configured LLM structured output; fall back deterministically when unavailable."""
    if provider:
        prompt = (
            "Task: Analyze the information need for retrieving documents to answer the target question. "
            "Do NOT answer the question. Do NOT output prose or explanations.\n"
            "Output ONLY a raw JSON object matching this schema:\n"
            '{"known_facts":["facts from evidence"],"missing_information":"unknown entity or relation to search next","target_terms":["keywords"],"coverage":0.0}\n\n'
            f"Target Question: {question}\n"
            f"Current Evidence: {[d.text for d in state.accumulated_evidence]}\n\n"
            "JSON:"
        )
        try:
            data = provider.complete_json(prompt)
            if _valid_information_need(data):
                return {"known_facts": [str(x) for x in data["known_facts"]], "missing_information": str(data["missing_information"]), "target_terms": [str(x) for x in data["target_terms"]], "coverage": float(data["coverage"]), "source": "llm"}
            if getattr(provider, "strict", False):
                raise ValueError("Model JSON did not match the information-need schema.")
        except (RuntimeError, ValueError, KeyError, TypeError):
            if getattr(provider, "strict", False):
                raise
    return _fallback_need(question, state)


def generate_candidates(question: str, state, need: dict, count: int, provider: JsonCompleter | None = None) -> list[str]:
    if provider:
        prompt = (
            f"Task: Generate at most {count} search retrieval queries to find the missing information. "
            "These must be search engine search queries (2 to 8 words), NOT answers to the question. "
            "Do NOT answer the question. Do NOT output prose or explanations.\n"
            "Output ONLY a raw JSON object matching this schema:\n"
            '{"queries":["search query 1","search query 2"]}\n\n'
            f"Target Question: {question}\n"
            f"Current Evidence: {[d.text for d in state.accumulated_evidence]}\n"
            f"Missing Information Need: {need.get('missing_information', '')}\n"
            f"Target Search Terms: {need.get('target_terms', [])}\n\n"
            "JSON:"
        )
        try:
            res = provider.complete_json(prompt)
            raw_queries = res.get("queries", []) if isinstance(res, dict) else []
            if isinstance(raw_queries, str):
                raw_queries = [raw_queries]
            queries = [str(q).strip() for q in raw_queries if _valid_query(str(q))]
            if queries:
                return [q for q in dict.fromkeys(queries) if q not in set(state.query_history)][:count]
            if getattr(provider, "strict", False):
                raise ValueError("Model returned no valid candidate queries.")
        except (RuntimeError, ValueError, KeyError, TypeError):
            if getattr(provider, "strict", False):
                raise
    focus = " ".join(need.get("target_terms", [])[:2]) or " ".join(_entities(question)[:2]) or question
    target = " ".join(tokenize(need.get("missing_information", ""))[:4])
    candidates = [f"{focus} {target}".strip(), f"{focus} relationship", f"{focus}".strip(), question]
    return [q for q in dict.fromkeys(candidates) if q and q not in set(state.query_history)][:count]


def _valid_query(value: str) -> bool:
    query = value.strip()
    return 1 <= len(query.split()) <= 20 and len(query) <= 200 and not any(marker in query for marker in ("{", "}", "\n"))


def _valid_information_need(data: dict) -> bool:
    required = {"known_facts", "missing_information", "target_terms", "coverage"}
    if not isinstance(data, dict) or not (required <= set(data)):
        return False
    if not isinstance(data["missing_information"], str):
        return False
    if not isinstance(data["known_facts"], list) or not isinstance(data["target_terms"], list):
        return False
    try:
        cov = float(data["coverage"])
        return 0.0 <= cov <= 1.0
    except (ValueError, TypeError):
        return False
