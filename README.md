# Adaptive Query Planning and Iterative Retrieval for Multi-Hop QA

This M.Tech CSE project studies whether an explicit query-level planner can retrieve multi-hop evidence more deliberately than one-shot or fixed iterative retrieval. The contribution is not an LLM choosing a query invisibly: the system generates candidate queries, scores each one transparently, retrieves with BM25, updates state, and repeats under a bounded budget.

## Architecture

`question -> information need -> candidate queries -> adaptive planner -> BM25 -> evidence state -> stopping`

For candidate query `q`, the planner computes normalized values in `[0, 1]`:

`score(q) = w_rel*relevance + w_missing*missing - w_red*redundancy - w_cost*cost`

- **Relevance:** candidate preview's maximum BM25 score divided by the maximum score over all candidates in that decision round.
- **Missing-information score:** token Jaccard overlap between query tokens and structured `target_terms` in the current information need.
- **Redundancy:** average token Jaccard overlap between preview documents and accumulated evidence.
- **Cost:** preview document count divided by `top_k`.

Planner previews are BM25 operations. Their returned-document cost is recorded separately from the selected retrieval cost, and both count toward the retrieval budget.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
python experiments/run_experiment.py
```

The default `demo` configuration is an offline two-question development set. Outputs are written to a timestamped folder under `results/` and include raw JSON, per-question CSVs, summaries, `comparison.csv`, and `comparison.png`.

## Local Hugging Face LLM Query Generation

Actual experiments use `transformers` and PyTorch locally in the runtime, with no hosted or paid inference API. The configured models are `Qwen/Qwen2.5-1.5B-Instruct`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`, and `microsoft/Phi-3.5-mini-instruct`. The provider loads a tokenizer and causal model from Hugging Face, applies the tokenizer chat template when provided, asks for JSON, validates extracted JSON, and records loading/generation events. Phi is configured with `trust_remote_code: true` because its model card documents that requirement; the others use the standard Transformers path. [Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct), and [Phi-3.5](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) publish Transformers loading guidance on their model cards.

The provider loads one model at a time and calls `release()` plus CUDA cache cleanup before the next. A Colab GPU is strongly recommended; Phi-3.5-mini is the largest selected model and may be impractical on a free T4 without sufficient VRAM. All model downloads are cached by Hugging Face but model weights are not committed.

`llm.enabled: false` keeps the deterministic fallback for offline tests and development only. With `llm.enabled: true` and `llm.strict: true`, a model-load, generation, or JSON-validation failure is recorded and fails that model run; it is never silently converted into a successful fallback experiment.

## Dataset Setup

The adapter supports a repository-neutral JSON schema and public HotpotQA distractor data. For HotpotQA, set the following in a copied experiment config:

```yaml
dataset:
  name: hotpotqa
  split: validation
  sample_size: 50
```

Then run `python experiments/run_experiment.py --config your_config.yaml`. The first call downloads HotpotQA through Hugging Face `datasets`; it is not committed to this repository. HotpotQA contexts are scoped per question, so document IDs are stored as `question_id::title` and supporting-fact titles become evidence annotations.

For custom data, pass `--dataset path/to/data.json`:

```json
{"documents":[{"id":"d1","text":"..."}],"examples":[{"id":"q1","question":"...","answer":"...","supporting_ids":["d1"]}]}
```

## Methods and Metrics

- **One-Shot RAG:** one BM25 call using the question.
- **Simplified ReAct-style retrieval:** one state-aware query per iteration. It is not an exact ReAct reproduction and never uses answer or support annotations to stop.
- **Adaptive Query Planning:** candidate previews, explicit weighted planner, selected BM25 retrieval, evidence-state update.

`Evidence Recall@k` is per-step support-document recall within the selected top-k results; the reported comparison uses the best step for iterative methods. `Accumulated Evidence Recall` is reported separately over unique evidence collected across all iterations. Average retrieval steps and retrieval cost are also reported. Cost is document-return units: `candidate_preview_documents + selected_retrieval_documents`.

Stopping is inference-only: it enforces the budget before another iteration, max iterations, minimum iterations, structured information-need coverage, and a configurable no-information-gain condition. It never uses gold supporting IDs or answers.

## Ablations and Configuration

All settings are in `config.yaml`: dataset/split/sample size, seed, BM25 `top_k`, candidate count, LLM provider/model, weights, enabled planner components, stopping thresholds, budget, and output path. Remove a component name from `planner.enabled` for ablations, for example `enabled: [relevance, missing, redundancy]`.

## Colab and Google Drive Results

Open [the existing notebook](notebooks/adaptive_query_planning_colab.ipynb) from `https://github.com/MSaiKiran9/aqp_iret`, select a GPU runtime, and run its cells in order. It clones the repository, mounts Drive, and creates this exact output root:

`/content/drive/MyDrive/AQP_IRET_Research_Results/`

It stores `configs`, `logs`, `model_metadata`, `raw_results`, `traces`, `summaries`, `comparisons`, `plots`, and `failures` beneath that folder. The default first run is a one-question actual-Hugging-Face smoke test with TinyLlama. Review its raw JSONL, trace, event log, and model metadata before setting `RUN_FULL_EVALUATION = True`.

The full cell invokes one model per process and releases memory after each. It uses a single `RUN_ID`; re-running a model with that same ID skips already written question records, making interrupted Colab work resumable. Comparison files are rebuilt from all raw records for the matching run ID. One-Shot RAG is retrieval-only, so it is saved once per run rather than redundantly executed once per LLM.

Local equivalent:

```powershell
python experiments/run_hf_experiments.py --config config_hotpotqa_dev.yaml --output drive_results --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 --smoke
```

Run all configured models sequentially by invoking the command once per model with a shared `--run-id`. It writes per-question JSONL incrementally, explicit failure JSONL records, model load metadata, `model_comparison.csv`, `method_comparison.csv`, `model_method_comparison.csv`, `per_question_results.csv`, `comparison_summary.json`, and `comparison_dashboard.png` from actual saved data.

## Project Structure

- `src/data`: dataset adapters and demo corpus
- `src/retrieval`: deterministic BM25
- `src/query_generation`: local Transformers provider and offline fallback
- `src/planner`: transparent scoring
- `src/evidence`, `src/stopping`: state and inference-only stopping
- `baselines`: one-shot and simplified ReAct-style methods
- `experiments`: reproducible runner
- `tests`: offline focused tests

## Limitations

The deterministic fallback is intentionally lightweight and not a substitute for LLM-based information-need analysis. Smaller models can produce invalid JSON or weak queries; such failures are logged instead of masked. BM25 candidate previews make adaptive planning costlier than one-shot retrieval; the cost is reported rather than hidden. Results on a small development subset are preliminary and must not be interpreted as evidence of superiority or statistical significance.
