# Evaluation

## Local deterministic evaluation

Runs the full graph against `evals/dataset.jsonl` and measures:

- Routing accuracy
- Citation validity
- Refusal correctness
- Retrieval recall@5
- MRR
- Answer term coverage
- Latency

Run it with the agent environment:

```bash
FOUNDRY_PROJECT_ENDPOINT="https://e2e-azure-rag-resource.services.ai.azure.com/api/projects/e2e-azure-rag" \
AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-5.4-mini" \
TOOLBOX_NAME="rag-tools" \
src/rag_agent/.venv/bin/python evals/run_local.py
```

Gate the report:

```bash
uv run python evals/gate.py \
  --report evals/reports/local-latest.json \
  --thresholds evals/thresholds.yaml
```

## Foundry evaluation

The Foundry evaluation suite is declared in `evals/azure.eval.yaml` and uses
the same dataset with built-in evaluators:

- `builtin.groundedness`
- `builtin.relevance`
- `builtin.retrieval`
- `builtin.response_completeness`

Create the eval:

```bash
azd ai eval create rag-agent-smoke --path evals --no-prompt
```

Run the eval and gate on pass rate:

```bash
azd ai eval run start \
  --path evals \
  --wait \
  --fail-on pass-rate=0.60
```

Or record a run and gate it separately:

```bash
azd ai eval run show <run-id> --output json > evals/reports/foundry-latest.json
uv run python evals/foundry_gate.py \
  --run evals/reports/foundry-latest.json \
  --thresholds evals/thresholds.yaml
```

The local deterministic gate is the fast PR gate. The Foundry suite is the
staged quality gate and the source for continuous evaluation.

## Baseline

The first local baseline produced:

```text
routing_accuracy       0.8095
citation_validity      1.0000
refusal_correctness    0.9048
retrieval_recall_at_5  0.8095
mrr                    0.6667
answer_term_coverage   0.9087
p95_latency_ms         17645.6
```

The current Foundry run against agent version 11 passed 16 of 21 cases:

```text
foundry_pass_rate      0.762
```

