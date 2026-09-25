import asyncio
from pathlib import Path

import httpx2

from student_agent.config import Settings
from student_agent.llm import OpenRouterAuditor


def _settings(enabled: bool) -> Settings:
    return Settings(
        "https://example.com",
        "sk-team-abcdefghijklmnop",
        "https://example.com/mcp",
        Path("."),
        enabled,
        "key" if enabled else "",
        "qwen/qwen3.5-9b",
    )


def test_disabled_and_single_signal_do_not_need_network() -> None:
    disabled = OpenRouterAuditor(_settings(False))
    assert (
        asyncio.run(
            disabled.audit_issue(
                deterministic_issue="a", supported_issues=["a", "b"], verified_facts={}
            )
        ).error_code
        == "DISABLED"
    )
    enabled = OpenRouterAuditor(_settings(True))
    assert (
        asyncio.run(
            enabled.audit_issue(deterministic_issue="a", supported_issues=["a"], verified_facts={})
        ).error_code
        == "SINGLE_SIGNAL"
    )
    asyncio.run(enabled.aclose())


def test_disagreement_is_advisory_only() -> None:
    auditor = OpenRouterAuditor(_settings(True))

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"primary_issue":"b"}'}}]}

    class Client:
        async def post(self, *args, **kwargs):
            return Response()

        async def aclose(self):
            pass

    auditor._client = Client()
    audit = asyncio.run(
        auditor.audit_issue(deterministic_issue="a", supported_issues=["a", "b"], verified_facts={})
    )
    assert audit.proposed_issue == "b"
    assert audit.agreed is False


def test_unsupported_model_issue_is_rejected() -> None:
    auditor = OpenRouterAuditor(_settings(True))
    auditor._client = _client_with_content('{"primary_issue":"duplicate_charge"}')
    audit = asyncio.run(
        auditor.audit_issue(
            deterministic_issue="refund_failed",
            supported_issues=["refund_failed", "payment_mismatch"],
            verified_facts={},
        )
    )
    assert audit.proposed_issue is None and audit.error_code == "UNSUPPORTED_ISSUE"


def test_invalid_json_and_network_fail_open() -> None:
    auditor = OpenRouterAuditor(_settings(True))
    auditor._client = _client_with_content("not-json")
    assert (
        asyncio.run(
            auditor.audit_issue(
                deterministic_issue="a", supported_issues=["a", "b"], verified_facts={}
            )
        ).proposed_issue
        is None
    )

    class FailingClient:
        async def post(self, *args, **kwargs):
            raise httpx2.HTTPError("network")

        async def aclose(self):
            pass

    auditor._client = FailingClient()
    assert (
        asyncio.run(
            auditor.audit_issue(
                deterministic_issue="a", supported_issues=["a", "b"], verified_facts={}
            )
        ).error_code
        == "AUDIT_FAILED"
    )


def _client_with_content(content: str):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    class Client:
        async def post(self, *args, **kwargs):
            return Response()

        async def aclose(self):
            pass

    return Client()
