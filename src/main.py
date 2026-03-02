from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .azure_mcp_assistant import AzureMCPError, query_subscription
from .openai_client import OpenAIClientError, load_openai_settings


app = FastAPI(
    title="Azure MCP Subscription Assistant",
    version="0.1.0",
    description="Subscription-wide Azure Q&A using Azure MCP + Azure OpenAI",
)


class QueryRequest(BaseModel):
    question: str
    subscription_id: Optional[str] = None
    max_iterations: int = 6


class ToolCallResponse(BaseModel):
    name: str
    arguments: Dict
    result_preview: str


class QueryResponse(BaseModel):
    answer: str
    question: str
    subscription_id: Optional[str]
    iterations: int
    tool_calls: List[ToolCallResponse]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    ui_path = Path(__file__).resolve().parent / "ui.html"
    return ui_path.read_text(encoding="utf-8")


@app.post("/api/query", response_model=QueryResponse)
async def api_query(request: QueryRequest):
    if request.max_iterations < 1 or request.max_iterations > 12:
        raise HTTPException(status_code=400, detail="max_iterations must be between 1 and 12")

    try:
        settings = load_openai_settings()
        result = await asyncio.wait_for(
            asyncio.to_thread(
                query_subscription,
                openai_settings=settings,
                question=request.question,
                subscription_id=request.subscription_id,
                max_iterations=request.max_iterations,
            ),
            timeout=120,
        )
        return QueryResponse(
            answer=result.answer,
            question=request.question,
            subscription_id=request.subscription_id,
            iterations=result.iterations,
            tool_calls=[
                ToolCallResponse(
                    name=t.name,
                    arguments=t.arguments,
                    result_preview=t.result_preview,
                )
                for t in result.tool_calls
            ],
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "Query timed out after 120s. Verify Azure auth (run 'az login') "
                "and retry with a narrower question."
            ),
        ) from exc
    except AzureMCPError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OpenAIClientError as exc:
        message = str(exc)
        if "does not match resource tenant" in message.lower():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Azure OpenAI tenant mismatch. Either set AZURE_OPENAI_API_KEY and "
                    "AZURE_OPENAI_USE_AZURE_AD=false, or login to the tenant that owns "
                    "your AZURE_OPENAI_ENDPOINT resource."
                ),
            ) from exc
        if "400" in message and settings.use_azure_ad and not settings.api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Azure OpenAI authentication failed with Azure AD. "
                    "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_USE_AZURE_AD=false, "
                    "or login to the tenant that owns your AZURE_OPENAI_ENDPOINT."
                ),
            ) from exc
        raise HTTPException(status_code=400, detail=message) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc
