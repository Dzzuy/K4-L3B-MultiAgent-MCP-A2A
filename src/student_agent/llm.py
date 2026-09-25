from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx2

from .config import Settings

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class AdjudicationAudit:
    proposed_issue: str | None
    agreed: bool | None
    error_code: str | None


class OpenRouterAuditor:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.openrouter_enabled
        self.model = settings.openrouter_model
        self._client = (
            httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                timeout=httpx2.Timeout(20.0, connect=8.0),
            )
            if self.enabled
            else None
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def audit_issue(
        self,
        *,
        deterministic_issue: str,
        supported_issues: list[str],
        verified_facts: dict[str, Any],
    ) -> AdjudicationAudit:
        allowed = list(dict.fromkeys(supported_issues))
        if not self.enabled:
            return AdjudicationAudit(None, None, "DISABLED")
        if len(allowed) < 2:
            return AdjudicationAudit(None, None, "SINGLE_SIGNAL")
        if self._client is None:
            return AdjudicationAudit(None, None, "NO_CLIENT")
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 64,
            "reasoning": {"enabled": False},
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "Choose one issue only from supported_issues. Return JSON.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "supported_issues": allowed,
                            "verified_facts": verified_facts,
                        },
                        default=str,
                        separators=(",", ":"),
                    ),
                },
            ],
        }
        try:
            response = await self._client.post(OPENROUTER_ENDPOINT, json=body)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            proposed = json.loads(content).get("primary_issue")
            if proposed not in allowed:
                return AdjudicationAudit(None, None, "UNSUPPORTED_ISSUE")
            return AdjudicationAudit(proposed, proposed == deterministic_issue, None)
        except (
            httpx2.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return AdjudicationAudit(None, None, "AUDIT_FAILED")
