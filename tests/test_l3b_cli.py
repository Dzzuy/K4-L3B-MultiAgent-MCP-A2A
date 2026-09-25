import asyncio

import pytest
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED, INTERNAL_ERROR, REQUEST_TIMEOUT

from student_agent.agents.protocol import fetch, is_transport_failure
from student_agent.cli import _reconnect_delay


class _TransportGateway:
    async def call(self, tool, *, case_id, **arguments):
        raise MCPError(CONNECTION_CLOSED, "Connection closed")


class _State:
    case_id = "CASE_001"
    cache = {}

    def cache_key(self, tool, arguments):
        return tool


def test_transport_classifier() -> None:
    assert is_transport_failure(OSError())
    assert is_transport_failure(TimeoutError())
    assert not is_transport_failure(ValueError())
    assert not is_transport_failure(AssertionError())


def test_exception_group_requires_only_transport_errors() -> None:
    assert is_transport_failure(ExceptionGroup("network", [OSError(), TimeoutError()]))
    assert not is_transport_failure(ExceptionGroup("mixed", [OSError(), ValueError()]))


def test_mcp_connection_errors_trigger_reconnect() -> None:
    assert is_transport_failure(MCPError(CONNECTION_CLOSED, "Connection closed"))
    assert is_transport_failure(MCPError(REQUEST_TIMEOUT, "Request timed out"))
    assert is_transport_failure(MCPError(INTERNAL_ERROR, "Session not found"))
    assert not is_transport_failure(MCPError(INTERNAL_ERROR, "Tool failed"))


def test_reconnect_backoff_is_bounded() -> None:
    assert [_reconnect_delay(attempt) for attempt in range(1, 7)] == [
        1.0,
        2.0,
        4.0,
        8.0,
        8.0,
        8.0,
    ]


def test_fetch_propagates_transport_failure() -> None:
    with pytest.raises(MCPError, match="Connection closed"):
        asyncio.run(
            fetch(
                _TransportGateway(),
                object(),
                _State(),
                "entity-agent",
                "get_order",
                order_id="ORDER_001",
            )
        )
