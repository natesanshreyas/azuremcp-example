# Azure MCP Subscription Assistant

Standalone app that answers subscription-wide Azure questions like:

- "Where is app X deployed?"
- "Which resources are missing tag CostCenter?"

This implementation is **Azure MCP only**. It will reject non-Azure MCP server commands.

## What it does

1. Starts Azure MCP server over stdio (`AZURE_MCP_SERVER_COMMAND`)
2. Lists available Azure MCP tools
3. Uses Azure OpenAI to plan iterative MCP tool calls
4. Returns a grounded final answer + tool-call trace

## Setup

```bash
cd azure-mcp-subscription-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set values in `.env`.

## Run

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8010 --reload
```

Open:

- `http://localhost:8010` for UI
- `http://localhost:8010/docs` for API

## API

`POST /api/query`

```json
{
  "question": "which resources are missing tag CostCenter?",
  "subscription_id": "00000000-0000-0000-0000-000000000000",
  "max_iterations": 6
}
```
