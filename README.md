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

## LLM Query Generation

The normal configurable generator is an OpenAI-compatible JSON provider implemented in `src/query_generation/providers.py`. Set `llm.enabled: true` in `config.yaml` and configure credentials without committing them:

```powershell
$env:OPENAI_API_KEY = "your-key"
python experiments/run_experiment.py
```

The provider receives the question, accumulated evidence, and current structured information need; it returns JSON candidate queries and structured missing information. `llm.enabled: false` uses a deterministic evidence-aware fallback so tests and offline runs remain reproducible. This fallback is for development only, not the intended research configuration.

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

## Colab

Open `notebooks/adaptive_query_planning_colab.ipynb`. It prompts for your real GitHub repository URL, installs dependencies, optionally reads `OPENAI_API_KEY` from Colab Secrets, prepares a small HotpotQA validation subset, runs tests and all three methods, then writes tables and plots.

## Project Structure

- `src/data`: dataset adapters and demo corpus
- `src/retrieval`: deterministic BM25
- `src/query_generation`: LLM provider and offline fallback
- `src/planner`: transparent scoring
- `src/evidence`, `src/stopping`: state and inference-only stopping
- `baselines`: one-shot and simplified ReAct-style methods
- `experiments`: reproducible runner
- `tests`: offline focused tests

## Limitations

The deterministic fallback is intentionally lightweight and not a substitute for LLM-based information-need analysis. BM25 candidate previews make adaptive planning costlier than one-shot retrieval; the cost is reported rather than hidden. Results on a small development subset are preliminary and must not be interpreted as evidence of superiority or statistical significance.
