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

## Promotion and rollback

The Foundry evaluation target has no pinned version, so it evaluates the newest
immutable agent version. CD uses that behavior to gate promotion:

1. Read the currently promoted version with `promote.py --current`.
2. Pin the endpoint to that version so a new deploy receives no traffic.
3. Deploy and evaluate the new version.
4. Promote the evaluated version only after the gate passes.

```bash
# Show the currently promoted version
src/rag_agent/.venv/bin/python evals/promote.py --current

# Promote a version (100% of endpoint traffic)
src/rag_agent/.venv/bin/python evals/promote.py --version 12

# Roll back to a previous version
src/rag_agent/.venv/bin/python evals/promote.py --version 11
```

`promote.py` preserves the endpoint's protocol configuration and Entra
authorization schemes while replacing the fixed-ratio version selector.

## Continuous evaluation

Evaluation rules only support prompt agents, so hosted agents are monitored with
a recurring trace evaluation instead:

```bash
# Create or refresh the hourly schedule for the newest version
src/rag_agent/.venv/bin/python evals/continuous_eval.py

# Point the schedule at a specific promoted version
src/rag_agent/.venv/bin/python evals/continuous_eval.py --version 12

# Inspect the schedule
src/rag_agent/.venv/bin/python evals/continuous_eval.py --check
```

The schedule uses an `azure_ai_traces` data source scoped to
`rag-agent:<version>`, a one-hour lookback, and the
`rag-agent-continuous` eval group. It scores relevance, tool call accuracy, and
task adherence from Application Insights traces. The Foundry project managed
identity needs `Log Analytics Reader` and `Monitoring Reader` on Application
Insights and its Log Analytics workspace, plus `Foundry User` on the project.

## CI/CD

`.github/workflows/cd.yml` runs the batch gate in the `staging` environment and
promotes the same immutable version in the `production` environment. A threshold
miss leaves the endpoint on the previously promoted version, so a failing
evaluation never receives production traffic.

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

