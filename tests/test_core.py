from src.data.dataset import demo_dataset
from src.retrieval.bm25 import BM25Retriever
from src.evidence.state import EvidenceState
from src.query_generation.generator import generate_candidates, get_missing_information
from src.planner.adaptive_planner import AdaptiveQueryPlanner, PlannerConfig, jaccard
from src.stopping.stopping import StoppingConfig, can_afford_iteration, should_stop
from src.pipeline import run_adaptive
from src.evaluation.metrics import evidence_metrics
from baselines.one_shot_rag import run_one_shot
from baselines.react import run_react_style
from src.query_generation.providers import _json_object, build_provider, _patch_cache_compatibility
from src.query_generation.providers import HuggingFaceLocalProvider
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import pytest


class FakeProvider:
    def complete_json(self, prompt):
        if '"known_facts"' in prompt:
            return {"known_facts": [], "missing_information": "country relation", "target_terms": ["ISRO", "country"], "coverage": 0.2}
        return {"queries": ["ISRO country", "Indian Space Research Organisation country", "ISRO national agency"]}


def components():
    docs, examples = demo_dataset()
    retriever = BM25Retriever(docs)
    stop = StoppingConfig(max_iterations=3, retrieval_budget=24, min_iterations=2, min_new_documents=1, min_missing_coverage=.75, no_gain_limit=1)
    planner = AdaptiveQueryPlanner(retriever, PlannerConfig())
    return examples[0], retriever, stop, planner


def test_generation_bm25_and_normalized_planner_scores():
    example, retriever, _, planner = components()
    state = EvidenceState(example.question)
    need = get_missing_information(example.question, state, FakeProvider())
    candidates = generate_candidates(example.question, state, need, 3, FakeProvider())
    selected, scores = planner.select(candidates, state, 2)
    assert len(candidates) == 3 and selected["query"] in candidates
    assert all(0 <= row[key] <= 1 for row in scores for key in ("relevance", "missing", "redundancy", "cost"))
    assert retriever.search("Chandrayaan-3", 2)[0][0].id == "d3" or retriever.search("Chandrayaan-3", 2)[0][0].id == "d1"


def test_strict_information_need_rejects_schema_invalid_json():
    class InvalidSchemaProvider:
        strict = True
        def complete_json(self, prompt): return {"target_terms": [], "coverage": 0.5}

    example, _, _, _ = components()
    with pytest.raises(ValueError, match="schema"):
        get_missing_information(example.question, EvidenceState(example.question), InvalidSchemaProvider())


def test_state_redundancy_and_budget_accounting():
    example, retriever, stop, _ = components()
    state = EvidenceState(example.question)
    results = retriever.search(example.question, 2)
    state.add_retrieval(example.question, results, [], evaluation_cost=6)
    assert state.iteration == 1 and state.retrieval_cost == 8 and state.evidence_ids
    assert jaccard({"a"}, {"a"}) == 1.0 and jaccard({"a"}, {"b"}) == 0.0
    assert not can_afford_iteration(state, 3, 2, StoppingConfig(retrieval_budget=13))


def test_stopping_enforces_inference_only_conditions():
    example, retriever, stop, _ = components()
    state = EvidenceState(example.question)
    state.missing_information = {"coverage": 1.0}
    state.add_retrieval(example.question, retriever.search(example.question, 2), [], 0)
    assert should_stop(state, stop) == (False, "continue")


def test_baselines_metrics_and_adaptive_end_to_end():
    example, retriever, stop, planner = components()
    one = run_one_shot(example, retriever, 2)
    react = run_react_style(example, retriever, 2, 3, stop)
    adaptive = run_adaptive(example, retriever, planner, stop, 2, 3)
    assert one["steps"] == 1 and react["steps"] >= 1
    assert adaptive["steps"] >= 2 and adaptive["retrieval_cost"] <= stop.retrieval_budget
    assert adaptive["candidate_evaluation_cost"] > 0 and adaptive["state"]["history"]
    assert evidence_metrics([{"retrieved_ids": ["d1"]}], ["d1"], ["d1", "d2"], 1)["first_step_recall_at_k"] == .5


def test_local_provider_configuration_and_json_parser_without_model_download():
    assert _json_object('{"queries":["a"]}') == {"queries": ["a"]}
    assert _json_object("```json\n{\"queries\":[\"a\"]}\n```") == {"queries": ["a"]}
    assert _json_object("  \n {\"queries\":[\"a\"]} \n") == {"queries": ["a"]}
    assert _json_object("Answer: {\"queries\":[\"a\"]}") == {"queries": ["a"]}
    with pytest.raises(ValueError): _json_object('{"queries":}')
    with pytest.raises(ValueError): _json_object('{"queries":["a"]')
    with pytest.raises(ValueError): _json_object("")
    provider = build_provider({"enabled": True, "provider": "huggingface_local", "strict": True, "models": []}, {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "trust_remote_code": False})
    assert provider.model_id.endswith("Chat-v1.0") and provider.strict


def test_local_provider_passes_batch_encoding_as_generate_keywords(monkeypatch):
    class InputIds(list):
        shape = (1, 2)

    class BatchEncoding(dict):
        def to(self, device):
            self.device = device
            return self

    class Tokenizer:
        chat_template = "chat"
        eos_token_id = 0
        def apply_chat_template(self, *args, **kwargs):
            return BatchEncoding(input_ids=InputIds([1, 2]))
        def decode(self, tokens, skip_special_tokens=True):
            return '{"queries":["ISRO country"]}'

    class Model:
        def __init__(self): self.kwargs = None
        def parameters(self): return iter([SimpleNamespace(device="cpu")])
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return [[1, 2, 3, 4]]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=lambda: nullcontext()))
    provider = HuggingFaceLocalProvider("fake")
    provider.model, provider.tokenizer = Model(), Tokenizer()
    assert provider.complete_json("test") == {"queries": ["ISRO country"]}
    assert provider.model.kwargs["input_ids"] == [1, 2]


def test_local_provider_retains_empty_generation_diagnostics(monkeypatch):
    class InputIds(list):
        shape = (1, 2)

    class BatchEncoding(dict):
        def to(self, device): return self

    class Tokenizer:
        chat_template = "tinyllama-chat-template"
        eos_token_id = 0
        def apply_chat_template(self, *args, **kwargs): return BatchEncoding(input_ids=InputIds([1, 2]))
        def decode(self, tokens, skip_special_tokens=True): return ""

    class Model:
        def __init__(self): self.budgets = []
        def parameters(self): return iter([SimpleNamespace(device="cpu")])
        def generate(self, **kwargs):
            self.budgets.append(kwargs["max_new_tokens"])
            return [[1, 2]]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=lambda: nullcontext()))
    provider = HuggingFaceLocalProvider("tinyllama-fake")
    provider.model, provider.tokenizer = Model(), Tokenizer()
    with pytest.raises(RuntimeError, match="empty"):
        provider.complete_json("schema prompt")
    attempts = [event for event in provider.events if event["kind"] == "generation_attempt"]
    assert len(attempts) == 2
    assert provider.model.budgets == [provider.max_new_tokens, provider.max_new_tokens]
    assert all(event["output"]["raw_output_length"] == 0 and event["output"]["new_token_count"] == 0 for event in attempts)
    assert provider.events[-1]["raw_outputs"] == [event["output"] for event in attempts]


def test_cache_compatibility_patch_for_legacy_cache_models(monkeypatch):
    class FakeCacheBase:
        pass

    class FakeDynamicCache(FakeCacheBase):
        def __init__(self, length=5):
            self._len = length
        def get_seq_length(self):
            return self._len

    fake_cache_utils = SimpleNamespace(Cache=FakeCacheBase, DynamicCache=FakeDynamicCache)
    fake_transformers = SimpleNamespace(cache_utils=fake_cache_utils)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)

    _patch_cache_compatibility()

    cache_instance = FakeDynamicCache(length=12)
    assert hasattr(cache_instance, "seen_tokens")
    assert cache_instance.seen_tokens == 12
    assert hasattr(cache_instance, "get_max_length")
    assert cache_instance.get_max_length() is None


def test_prompt_instructions_and_structured_candidate_generation():
    recorded_calls = []

    class MockStructuredProvider:
        strict = True
        def complete_json(self, prompt: str, max_new_tokens: int | None = None) -> dict:
            recorded_calls.append((prompt, max_new_tokens))
            if "information need" in prompt:
                return {
                    "known_facts": ["Tokarev was a professor"],
                    "missing_information": "which university",
                    "target_terms": ["Tokarev", "university"],
                    "coverage": 0.5,
                }
            return {"queries": ["Tokarev university", "Sergei Tokarev professor"]}

    state = EvidenceState("Where was Tokarev a professor?")
    need = get_missing_information(state.question, state, MockStructuredProvider())
    assert need["source"] == "llm"
    assert need["missing_information"] == "which university"
    assert "Do NOT answer the question" in recorded_calls[0][0]
    assert recorded_calls[0][1] == 80

    candidates = generate_candidates(state.question, state, need, 2, MockStructuredProvider())
    assert candidates == ["Tokarev university", "Sergei Tokarev professor"]
    assert "search retrieval queries" in recorded_calls[1][0]
    assert "NOT answers to the question" in recorded_calls[1][0]
    assert recorded_calls[1][1] == 100


def test_provider_retry_on_malformed_json_and_token_limits(monkeypatch):
    class InputIds(list):
        shape = (1, 2)

    class BatchEncoding(dict):
        def to(self, device): return self

    class Tokenizer:
        chat_template = "chat"
        eos_token_id = 0
        def apply_chat_template(self, *args, **kwargs): return BatchEncoding(input_ids=InputIds([1, 2]))
        def decode(self, tokens, skip_special_tokens=True):
            return self.responses.pop(0)

    class Model:
        def __init__(self): self.budgets = []
        def parameters(self): return iter([SimpleNamespace(device="cpu")])
        def generate(self, **kwargs):
            self.budgets.append(kwargs["max_new_tokens"])
            return [[1, 2, 3]]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=lambda: nullcontext()))
    provider = HuggingFaceLocalProvider("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = Tokenizer()
    tokenizer.responses = ["Sorry, I cannot answer.", '{"queries": ["Qwen retrieval test"]}']
    provider.model, provider.tokenizer = Model(), tokenizer

    result = provider.complete_json("schema prompt", max_new_tokens=100)
    assert result == {"queries": ["Qwen retrieval test"]}
    assert provider.model.budgets == [100, 100]
    assert len(provider.events) == 2
    assert provider.events[0]["kind"] == "generation_attempt"
    assert provider.events[1]["kind"] == "generation" and provider.events[1]["success"] is True


def test_missing_information_reuses_cached_need_when_no_new_docs():
    state = EvidenceState("Where was Tokarev a professor?")
    state.missing_information = {"known_facts": ["fact"], "missing_information": "need", "target_terms": ["tokarev"], "coverage": 0.5}
    state.iteration = 1
    state.history = [{"iteration": 1, "new_document_count": 0}]

    class FailIfCalledProvider:
        strict = True
        def complete_json(self, *args, **kwargs):
            raise AssertionError("Provider should not be called when no new documents were found")

    cached_need = get_missing_information(state.question, state, FailIfCalledProvider())
    assert cached_need == state.missing_information


def test_end_to_end_call_path_tokens_and_placeholder_rejection(monkeypatch):
    class InputIds(list):
        shape = (1, 2)

    class BatchEncoding(dict):
        def to(self, device): return self

    class Tokenizer:
        chat_template = "chat"
        eos_token_id = 0
        def apply_chat_template(self, *args, **kwargs): return BatchEncoding(input_ids=InputIds([1, 2]))
        def decode(self, tokens, skip_special_tokens=True):
            return self.responses.pop(0)

    class Model:
        def __init__(self):
            self.generate_kwargs_history = []
        def parameters(self): return iter([SimpleNamespace(device="cpu")])
        def generate(self, **kwargs):
            self.generate_kwargs_history.append(dict(kwargs))
            return [[1, 2, 3]]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=lambda: nullcontext()))
    provider = HuggingFaceLocalProvider("Qwen/Qwen2.5-1.5B-Instruct", do_sample=False)
    tokenizer = Tokenizer()
    model = Model()
    provider.model, provider.tokenizer = model, tokenizer

    # 1. Missing information call should use 80 tokens
    tokenizer.responses = ['{"known_facts": ["Panchin is a researcher"], "missing_information": "university affiliation", "target_terms": ["Sergei Aleksandrovich Panchin", "university"], "coverage": 0.5}']
    state = EvidenceState("Where was Sergei Aleksandrovich Panchin a professor?")
    need = get_missing_information(state.question, state, provider)

    assert need["source"] == "llm"
    assert model.generate_kwargs_history[-1]["max_new_tokens"] == 80
    assert "temperature" not in model.generate_kwargs_history[-1]
    assert provider.events[-1]["settings"]["max_new_tokens"] == 80
    assert "temperature" not in provider.events[-1]["settings"]

    # 2. Candidate generation call should use 100 tokens and accept specific queries
    tokenizer.responses = ['{"queries": ["Sergei Aleksandrovich Panchin university affiliation", "Sergei Aleksandrovich Panchin academic professor", "Panchin Moscow State University"]}']
    candidates = generate_candidates(state.question, state, need, 3, provider)

    assert len(candidates) == 3
    assert "Sergei Aleksandrovich Panchin university affiliation" in candidates
    assert model.generate_kwargs_history[-1]["max_new_tokens"] == 100
    assert "temperature" not in model.generate_kwargs_history[-1]
    assert provider.events[-1]["settings"]["max_new_tokens"] == 100

    # 3. Verify generic placeholders are rejected in strict mode
    tokenizer.responses = ['{"queries": ["entity relation", "entity location"]}']
    with pytest.raises(ValueError, match="valid candidate queries"):
        generate_candidates(state.question, state, need, 3, provider)




