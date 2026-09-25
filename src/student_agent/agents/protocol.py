from __future__ import annotations

import asyncio
from typing import Any

import httpx2
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT

from ..mcp_gateway import EvidenceGateway
from ..state import CaseState, Evidence
from ..trace import TraceWriter

MAX_ATTEMPTS = 2


def is_transport_failure(exc: BaseException) -> bool:
    if isinstance(exc, (OSError, TimeoutError, httpx2.HTTPError)):
        return True
    if isinstance(exc, MCPError):
        return exc.code in {CONNECTION_CLOSED, REQUEST_TIMEOUT} or (
            "session not found" in exc.message.lower()
        )
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(
            is_transport_failure(item) for item in exc.exceptions
        )
    name = type(exc).__name__.lower()
    module = type(exc).__module__.lower()
    return "mcp" in module and any(
        token in name for token in ("transport", "session", "connection")
    )


async def fetch(
    gateway: EvidenceGateway,
    trace: TraceWriter,
    state: CaseState,
    actor: str,
    tool: str,
    **arguments: Any,
) -> Evidence | None:
    key = state.cache_key(tool, arguments)
    evidence = state.cache.get(key)
    if evidence is None:
        for attempt in range(MAX_ATTEMPTS):
            try:
                envelope = await gateway.call(tool, case_id=state.case_id, **arguments)
                evidence = Evidence(
                    tool=tool,
                    domain=str(envelope["domain"]),
                    ref=str(envelope["evidence_ref"]),
                    data=envelope["data"],
                    arguments=dict(arguments),
                )
                state.cache[key] = evidence
                state.register_evidence(evidence)
                break
            except (RuntimeError, ValueError):
                return None
            except Exception as exc:
                if is_transport_failure(exc):
                    raise
                if attempt == MAX_ATTEMPTS - 1:
                    return None
                await asyncio.sleep(0.25 * (attempt + 1))
    if evidence is None:
        return None
    trace.emit(
        case_id=state.case_id,
        event_type="tool_result_consumed",
        actor=actor,
        tool_name=tool,
        evidence_refs=[evidence.ref],
    )
    return evidence


def assign(trace: TraceWriter, state: CaseState, target: str, code: str) -> None:
    trace.emit(
        case_id=state.case_id,
        event_type="task_assigned",
        actor="coordinator",
        target=target,
        decision_code=code,
    )


def handoff(
    trace: TraceWriter,
    state: CaseState,
    actor: str,
    target: str,
    code: str,
    refs: list[str] | None = None,
) -> None:
    trace.emit(
        case_id=state.case_id,
        event_type="handoff",
        actor=actor,
        target=target,
        decision_code=code,
        evidence_refs=list(dict.fromkeys(refs or []))[:20] or None,
    )
