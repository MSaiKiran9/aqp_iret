"""Explicit, serializable retrieval evidence state."""
from dataclasses import dataclass, field
from src.data.dataset import Document


@dataclass
class EvidenceState:
    question: str
    iteration: int = 0
    candidate_queries: list[str] = field(default_factory=list)
    selected_query: str | None = None
    candidate_scores: list[dict] = field(default_factory=list)
    retrieved_documents: list[Document] = field(default_factory=list)
    accumulated_evidence: list[Document] = field(default_factory=list)
    query_history: list[str] = field(default_factory=list)
    candidate_evaluation_cost: int = 0
    selected_retrieval_cost: int = 0
    missing_information: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    @property
    def evidence_ids(self) -> list[str]:
        return [d.id for d in self.accumulated_evidence]

    @property
    def retrieval_cost(self) -> int:
        return self.candidate_evaluation_cost + self.selected_retrieval_cost

    def add_retrieval(self, query: str, results: list[tuple[Document, float]], scores: list[dict], evaluation_cost: int):
        self.iteration += 1
        self.selected_query = query
        self.candidate_scores = scores
        self.query_history.append(query)
        self.retrieved_documents = [doc for doc, _ in results]
        seen = set(self.evidence_ids)
        new_docs = [doc for doc, _ in results if doc.id not in seen]
        self.accumulated_evidence.extend(new_docs)
        self.candidate_evaluation_cost += evaluation_cost
        self.selected_retrieval_cost += len(results)
        rationale = next((s.get("selection_rationale") for s in scores if s.get("query") == query), None)
        self.history.append({"iteration": self.iteration, "query": query, "selected_rationale": rationale, "retrieved_ids": [d.id for d, _ in results], "new_document_count": len(new_docs), "planner_scores": scores, "candidate_evaluation_cost": evaluation_cost, "selected_retrieval_cost": len(results), "missing_information": self.missing_information})

    def to_dict(self) -> dict:
        return {"question": self.question, "iteration": self.iteration, "selected_query": self.selected_query, "candidate_queries": self.candidate_queries, "candidate_scores": self.candidate_scores, "retrieved_document_ids": [d.id for d in self.retrieved_documents], "accumulated_evidence_ids": self.evidence_ids, "query_history": self.query_history, "missing_information": self.missing_information, "candidate_evaluation_cost": self.candidate_evaluation_cost, "selected_retrieval_cost": self.selected_retrieval_cost, "total_retrieval_cost": self.retrieval_cost, "history": self.history}
