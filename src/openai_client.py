from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()


class OpenAIClientError(Exception):
    pass


@dataclass
class OpenAISettings:
    endpoint: str
    deployment_name: str
    api_version: str
    model_name: str
    api_key: str
    use_azure_ad: bool


_token_cache: Dict[str, Any] = {}


def load_openai_settings() -> OpenAISettings:
    return OpenAISettings(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        model_name=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4.1"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        use_azure_ad=os.getenv("AZURE_OPENAI_USE_AZURE_AD", "true").lower() == "true",
    )


def _get_azure_ad_token() -> str:
    if _token_cache.get("token") and _token_cache.get("expires_at", 0) > time.time() + 60:
        return _token_cache["token"]

    from azure.identity import DefaultAzureCredential

    token = DefaultAzureCredential().get_token("https://cognitiveservices.azure.com/.default")
    _token_cache["token"] = token.token
    _token_cache["expires_at"] = token.expires_on
    return token.token


def chat_completion(
    settings: OpenAISettings,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> str:
    if not settings.endpoint or not settings.deployment_name:
        raise OpenAIClientError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_DEPLOYMENT_NAME")
    if not settings.use_azure_ad and not settings.api_key:
        raise OpenAIClientError("Missing AZURE_OPENAI_API_KEY when AZURE_OPENAI_USE_AZURE_AD=false")

    url = f"{settings.endpoint}/openai/deployments/{settings.deployment_name}/chat/completions"
    params = {"api-version": settings.api_version}

    headers = {"Content-Type": "application/json"}
    if settings.use_azure_ad:
        headers["Authorization"] = f"Bearer {_get_azure_ad_token()}"
    else:
        headers["api-key"] = settings.api_key

    body = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model": settings.model_name,
    }

    response = requests.post(url, params=params, headers=headers, json=body, timeout=90)

    if response.status_code >= 400:
        tenant_mismatch = "does not match resource tenant" in response.text.lower()
        can_retry_with_key = settings.use_azure_ad and bool(settings.api_key)

        if tenant_mismatch and can_retry_with_key:
            retry_headers = {"Content-Type": "application/json", "api-key": settings.api_key}
            response = requests.post(url, params=params, headers=retry_headers, json=body, timeout=90)

        if response.status_code >= 400:
            raise OpenAIClientError(f"OpenAI API error {response.status_code}: {response.text}")

    payload = response.json()
    try:
        return payload["choices"][0]["message"]["content"]
    except Exception as exc:
        raise OpenAIClientError(f"Unexpected OpenAI response: {payload}") from exc
