from __future__ import annotations

import json
import os
from typing import Any

import httpx2


def get_llm_config() -> dict[str, str] | None:
    """Retrieve LLM provider, base URL, model (< 10B params), and API key."""
    # 1. Custom / Generic OpenAI-compatible
    if os.getenv("LLM_API_KEY"):
        return {
            "api_key": os.getenv("LLM_API_KEY", "").strip(),
            "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            "model": os.getenv("LLM_MODEL", "qwen/qwen-2.5-7b-instruct").strip(),
        }

    # 2. Groq (< 10B parameter model: llama-3.1-8b-instant)
    if os.getenv("GROQ_API_KEY"):
        return {
            "api_key": os.getenv("GROQ_API_KEY", "").strip(),
            "base_url": "https://api.groq.com/openai/v1",
            "model": os.getenv("LLM_MODEL", "llama-3.1-8b-instant").strip(),
        }

    # 3. OpenRouter (< 10B parameter model: qwen/qwen-2.5-7b-instruct)
    if os.getenv("OPENROUTER_API_KEY"):
        return {
            "api_key": os.getenv("OPENROUTER_API_KEY", "").strip(),
            "base_url": "https://openrouter.ai/api/v1",
            "model": os.getenv("LLM_MODEL", "qwen/qwen-2.5-7b-instruct").strip(),
        }

    # 4. Google Gemini (via OpenAI compatibility endpoint)
    if os.getenv("GEMINI_API_KEY"):
        return {
            "api_key": os.getenv("GEMINI_API_KEY", "").strip(),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": os.getenv("LLM_MODEL", "gemini-1.5-flash").strip(),
        }

    # 5. OpenAI (compact efficient model)
    if os.getenv("OPENAI_API_KEY"):
        return {
            "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
            "base_url": "https://api.openai.com/v1",
            "model": os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        }

    return None


async def call_llm_reasoning(
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float = 8.0,
) -> dict[str, Any] | None:
    """Call LLM using httpx2 with strict JSON parsing and timeout."""
    config = get_llm_config()
    if not config or not config["api_key"]:
        return None

    url = f"{config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    try:
        timeout = httpx2.Timeout(timeout_seconds, connect=3.0)
        async with httpx2.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                return None
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception:  # noqa: BLE001
        return None
