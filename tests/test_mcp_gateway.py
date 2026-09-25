import asyncio
from types import SimpleNamespace

import pytest

from student_agent.mcp_gateway import EvidenceGateway


class _Contracts:
    def validate_evidence(self, evidence, source):
        assert evidence["evidence_ref"] == "ev-1"
        assert source == "MCP tool get_order"


class _Session:
    def __init__(self, result):
        self.result = result

    async def call_tool(self, name, arguments):
        assert name == "get_order"
        assert arguments == {"case_id": "CASE_001", "order_id": "ORDER_001"}
        return self.result


def test_gateway_reads_current_mcp_result_fields() -> None:
    envelope = {"evidence_ref": "ev-1", "domain": "order", "data": {}}
    result = SimpleNamespace(
        is_error=False,
        structured_content=envelope,
        content=[],
    )
    gateway = EvidenceGateway(_Session(result), _Contracts())

    actual = asyncio.run(
        gateway.call("get_order", case_id="CASE_001", order_id="ORDER_001")
    )

    assert actual == envelope


def test_gateway_raises_current_mcp_tool_error() -> None:
    result = SimpleNamespace(
        is_error=True,
        structured_content=None,
        content=[SimpleNamespace(text="missing order")],
    )
    gateway = EvidenceGateway(_Session(result), _Contracts())

    with pytest.raises(RuntimeError, match="missing order"):
        asyncio.run(
            gateway.call("get_order", case_id="CASE_001", order_id="ORDER_001")
        )
