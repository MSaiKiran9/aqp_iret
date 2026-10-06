"""Lightweight BM25 retrieval wrapper."""
import re
from rank_bm25 import BM25Okapi
from src.data.dataset import Document


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.index = BM25Okapi([tokenize(d.text) for d in documents])

    def search(self, query: str, top_k: int = 3, positive_scores_only: bool = True) -> list[tuple[Document, float]]:
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        if positive_scores_only:
            ranked = [item for item in ranked if item[1] > 0]
        ranked = ranked[:top_k]
        return [(self.documents[i], float(score)) for i, score in ranked]
