# e2e-azure-rag

Ingestion CLI and Foundry-hosted LangGraph agent for an Azure AI Search RAG application.

## Project layout

```text
.
├─ azure.yaml              # Foundry azd project
├─ src/rag_agent/          # Foundry-hosted LangGraph agent
├─ src/rag_ingest/         # Azure AI Search ingestion CLI
├─ tools/rag-tools.yaml    # Foundry Toolbox definition
└─ plan/master_plan.md     # architecture and implementation plan
```

The Foundry `azd` project is at the repository root. Run agent commands from there:

```bash
azd ai agent run --no-client
azd ai agent invoke --local "your question"
azd deploy
```

## Ingestion

The ingestion CLI loads documents into the Azure AI Search `rag-docs` index.

## What it does

- Extracts text from Markdown, text, PDF, and HTML files
- Chunks documents with token-aware overlap
- Generates embeddings with a Foundry embedding deployment
- Creates or updates the Azure AI Search index
- Uploads documents with deterministic IDs for idempotent re-ingestion
- Supports a dry run that does not call Azure

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env` with your Azure AI Search and Foundry endpoints.

For local development, authenticate with Azure CLI:

```bash
az login
```

The identity needs:

- **Search Index Data Contributor** on the Azure AI Search resource
- **Search Service Contributor** if it creates or updates the index
- **Cognitive Services OpenAI User** on the Foundry/Azure OpenAI resource

## Run

Dry run first:

```bash
uv run --env-file .env python -m rag_ingest --source ./data --dry-run
```

Ingest documents:

```bash
uv run --env-file .env python -m rag_ingest --source ./data
```

Recreate the index:

```bash
uv run --env-file .env python -m rag_ingest --source ./data --recreate-index
```

Using the console script:

```bash
uv run --env-file .env rag-ingest --source ./data --index rag-docs
```

## Options

```text
--source PATH             File or directory of documents
--index NAME              Target index name
--chunk-tokens N          Maximum tokens per chunk
--chunk-overlap N         Overlap tokens between chunks
--url-prefix URL          Prefix for citation URLs
--dry-run                 Extract and chunk without calling Azure
--recreate-index          Delete and recreate the index before upload
--log-level LEVEL         DEBUG, INFO, WARNING, ERROR
```

## Verify

In Azure AI Search, open **Search Explorer** and run:

```json
{
  "search": "*",
  "count": true
}
```

Or query from Python:

```python
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

client = SearchClient(
    endpoint="https://<search-service>.search.windows.net",
    index_name="rag-docs",
    credential=DefaultAzureCredential(),
)

for result in client.search("refund policy", top=3, select=["title", "content"]):
    print(result["title"], result["content"][:200])
```

## Tests

```bash
uv run pytest
uv run ruff check .
uv run mypy src/rag_ingest
```

Agent graph tests and the web MCP contract tests run with their own lockfiles:

```bash
(cd src/rag_agent && uv run ruff check . && uv run mypy && uv run pytest)
(cd src/web_search_mcp && uv run ruff check . && uv run pytest)
```

## CI/CD

### Pull request CI

`.github/workflows/ci.yml` runs on pull requests and is also called by CD:

- Root ingestion: Ruff, mypy, pytest
- Agent: Ruff, mypy, deterministic LangGraph routing and toolbox tests
- Web MCP: Ruff, MCP contract tests, URL/SSRF tests
- Containers: Docker build for both images plus Trivy `HIGH,CRITICAL` scan

No Azure access is required for the PR path.

### Deployment

`.github/workflows/cd.yml` runs on pushes to `main` and on manual dispatch:

1. Azure login through GitHub OIDC. No Azure credentials are stored in GitHub.
2. Reconstruct the `rag-agent-dev` azd environment from
   `.github/azd-env/rag-agent-dev.env` (identifiers and endpoints only).
3. Read the currently promoted agent version and keep the endpoint pinned to it.
4. `azd provision` then `azd deploy rag-agent`, which creates a new immutable
   agent version with no production traffic.
5. Refresh the Azure AI Search index from `plan/master_plan.md`.
6. Run the Foundry evaluation `rag-agent-smoke` against the newest version and
   enforce `foundry_pass_rate` from `evals/thresholds.yaml`.
7. Promote the evaluated version with `evals/promote.py` only if the gate passes.
8. Smoke-test the promoted version and enable hourly trace-based continuous
   evaluation with `evals/continuous_eval.py`.

The `staging` and `production` GitHub environments gate the two jobs. Production
requires a reviewer approval and accepts deployments from `main` only.

To roll back, run the CD workflow manually with `rollback_to_version` set to a
previous immutable version. The rollback job promotes that version and smoke-tests
it.

### GitHub configuration

Repository variables (no secrets):

```text
AZURE_CLIENT_ID                  Client ID of the user-assigned managed identity
AZURE_TENANT_ID                  Microsoft Entra tenant ID
AZURE_SUBSCRIPTION_ID            Target subscription
AZURE_LOCATION                   southindia
AZURE_ENV_NAME                   rag-agent-dev
AZURE_AI_PROJECT_NAME            e2e-azure-rag
AZURE_AI_MODEL_DEPLOYMENT_NAME   gpt-5.4-mini
AZURE_RESOURCE_GROUP             rg-kuralamuthan300-9046
AZURE_FOUNDRY_RESOURCE_GROUP     rg-rag-agent-dev-foundry
FOUNDRY_PROJECT_ENDPOINT         https://<account>.services.ai.azure.com/api/projects/<project>
AZURE_AI_PROJECT_ID              Foundry project ARM ID
AZURE_AI_PROJECT_ACR_CONNECTION_NAME
AZURE_CONTAINER_REGISTRY_ENDPOINT
AZURE_CONTAINER_REGISTRY_RESOURCE_ID
AZURE_SEARCH_ENDPOINT
AZURE_SEARCH_INDEX_NAME
AZURE_OPENAI_ENDPOINT
AZURE_EMBEDDING_DEPLOYMENT
```

The OIDC identity needs:

- `Contributor` and `User Access Administrator` on `rg-kuralamuthan300-9046`
  and `rg-rag-agent-dev-foundry`
- `Foundry User` on the Foundry project
- `Search Index Data Contributor` and `Search Service Contributor` on the
  Azure AI Search service
- `Cognitive Services OpenAI User` on the Foundry account
