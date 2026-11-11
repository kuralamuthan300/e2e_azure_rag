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
