"""Dataset adapter and a small built-in multi-hop development set."""
from dataclasses import dataclass
from pathlib import Path
import json
import random


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    answer: str
    supporting_ids: list[str]


def demo_dataset() -> tuple[list[Document], list[QAExample]]:
    documents = [
        Document("d1", "Chandrayaan-3 was launched by the Indian Space Research Organisation, also called ISRO."),
        Document("d2", "The Indian Space Research Organisation is India's national space agency."),
        Document("d3", "Chandrayaan-3 was India's third lunar exploration mission."),
        Document("d4", "NASA is the civil space program of the United States."),
        Document("d5", "The James Webb Space Telescope is operated by NASA, ESA, and CSA."),
        Document("d6", "The European Space Agency is an intergovernmental space organisation."),
    ]
    examples = [
        QAExample("q1", "Which country operates the space agency that launched Chandrayaan-3?", "India", ["d1", "d2"]),
        QAExample("q2", "Which country has the space agency that operates NASA?", "United States", ["d4"]),
    ]
    return documents, examples


def load_json_dataset(path: str | Path, sample_size: int | None = None, seed: int = 42):
    """Load {'documents': [...], 'examples': [...]} with document ids and support ids."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = [Document(str(x["id"]), x["text"]) for x in raw["documents"]]
    examples = [QAExample(str(x["id"]), x["question"], x.get("answer", ""), list(x.get("supporting_ids", []))) for x in raw["examples"]]
    if sample_size is not None:
        examples = random.Random(seed).sample(examples, min(sample_size, len(examples)))
    return documents, examples


def load_hotpotqa(split: str = "validation", sample_size: int | None = None, seed: int = 42):
    """Load HotpotQA distractor data into the repository-neutral adapter types.

    The optional ``datasets`` dependency downloads data only when this function is called.
    Document ids are scoped to an example because HotpotQA context is question-local.
    """
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - dependency/setup error
        raise RuntimeError("Install requirements.txt to use HotpotQA support.") from error
    rows = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    indices = list(range(len(rows)))
    if sample_size is not None:
        indices = random.Random(seed).sample(indices, min(sample_size, len(indices)))
    documents, examples = [], []
    for index in indices:
        row = rows[index]
        qid = str(row["id"])
        context = row["context"]
        titles = context["title"] if isinstance(context, dict) else [item[0] for item in context]
        sentences = context["sentences"] if isinstance(context, dict) else [item[1] for item in context]
        ids_by_title = {}
        for title, parts in zip(titles, sentences):
            doc_id = f"{qid}::{title}"
            ids_by_title[title] = doc_id
            documents.append(Document(doc_id, f"{title}. {' '.join(parts)}"))
        supporting = row["supporting_facts"]
        support_titles = supporting["title"] if isinstance(supporting, dict) else [item[0] for item in supporting]
        examples.append(QAExample(qid, row["question"], row["answer"], [ids_by_title[t] for t in dict.fromkeys(support_titles) if t in ids_by_title]))
    return documents, examples
