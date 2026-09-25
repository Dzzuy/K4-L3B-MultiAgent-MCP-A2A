from __future__ import annotations

from typing import Any

from ..mcp_gateway import EvidenceGateway
from ..state import AgentResult, CaseState
from ..trace import TraceWriter
from .protocol import fetch

ACTOR = "entity-agent"


def _collect_order_ids(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"order_id", "order_ids", "related_order_ids"}:
                if isinstance(item, str):
                    if item and item not in out:
                        out.append(item)
                elif isinstance(item, list):
                    for child in item:
                        if (
                            isinstance(child, str)
                            and child
                            and child not in out
                        ):
                            out.append(child)

            _collect_order_ids(item, out)

    elif isinstance(value, list):
        for item in value:
            _collect_order_ids(item, out)


def _find_customer_unique_id(value: Any) -> str | None:
    if isinstance(value, dict):
        direct = value.get("customer_unique_id")
        if isinstance(direct, str) and direct:
            return direct

        for child in value.values():
            found = _find_customer_unique_id(child)
            if found:
                return found

    elif isinstance(value, list):
        for child in value:
            found = _find_customer_unique_id(child)
            if found:
                return found

    return None


def _order_payload_matches(
    requested_order_id: str,
    data: dict[str, Any],
) -> bool:
    returned = data.get("order_id")

    # If the payload names an order, it must match the requested candidate.
    return returned is None or str(returned) == requested_order_id


def _finish(
    state: CaseState,
    result: AgentResult,
) -> AgentResult:
    result.ok = state.selected_order_id is not None

    result.entities = {
        "order_ids": state.resolved_order_ids[:20],
    }

    result.findings = {
        "status": state.entity_status,
        "selected_order_id": state.selected_order_id,
        "resolved_order_ids": state.resolved_order_ids[:20],
        "rejected_candidates": state.rejected_candidates[:20],
        "confidence": state.entity_confidence,
        "customer_unique_id": state.customer_unique_id,
        "customer_unique_id_hint": state.customer_unique_id_hint,
        "related_order_ids": state.related_order_ids[:20],
    }

    if state.entity_status == "resolved":
        result.notes_code = "ENTITY_RESOLVED"
    elif state.entity_status == "ambiguous":
        result.notes_code = "ENTITY_AMBIGUOUS"
    else:
        result.notes_code = "ENTITY_NOT_FOUND"

    return result


async def run(
    state: CaseState,
    gateway: EvidenceGateway,
    trace: TraceWriter,
) -> AgentResult:
    result = AgentResult(actor=ACTOR)

    history_verified = False

    # The input field is explicitly a hint. Only promote it to an authoritative
    # customer_unique_id after MCP customer-history evidence succeeds.
    if (
        state.include_customer_history
        and state.customer_unique_id_hint
    ):
        history_ev = await fetch(
            gateway,
            trace,
            state,
            ACTOR,
            "get_customer_history",
            customer_unique_id=state.customer_unique_id_hint,
        )

        if history_ev and isinstance(history_ev.data, (dict, list)):
            history_verified = True
            result.evidence.append(history_ev)

            state.customer_unique_id = (
                _find_customer_unique_id(history_ev.data)
                or state.customer_unique_id_hint
            )

            _collect_order_ids(
                history_ev.data,
                state.related_order_ids,
            )

            state.related_order_ids = list(
                dict.fromkeys(state.related_order_ids)
            )[:20]

    candidates = list(dict.fromkeys(state.candidate_order_ids))

    valid: list[tuple[str, dict[str, Any]]] = []

    for order_id in candidates:
        order_ev = await fetch(
            gateway,
            trace,
            state,
            ACTOR,
            "get_order",
            order_id=order_id,
        )

        if (
            order_ev is None
            or not isinstance(order_ev.data, dict)
            or not _order_payload_matches(order_id, order_ev.data)
        ):
            if order_id not in state.rejected_candidates:
                state.rejected_candidates.append(order_id)
            continue

        result.evidence.append(order_ev)

        # When customer history gives us actual related order IDs, it is the
        # independent customer/order linkage required by L3B.
        if (
            history_verified
            and state.related_order_ids
            and order_id not in state.related_order_ids
        ):
            if order_id not in state.rejected_candidates:
                state.rejected_candidates.append(order_id)
            continue

        valid.append((order_id, order_ev.data))

    state.rejected_candidates = list(
        dict.fromkeys(state.rejected_candidates)
    )[:20]

    if len(valid) == 1:
        winner = valid[0][0]

        state.selected_order_id = winner
        state.resolved_order_ids = [winner]
        state.entity_status = "resolved"

        if history_verified and state.related_order_ids:
            state.entity_confidence = 0.97
        elif history_verified:
            state.entity_confidence = 0.85
        else:
            state.entity_confidence = 0.75

    elif len(valid) > 1:
        # Do NOT use "newest order wins".
        # Multiple independently valid candidates remain genuinely ambiguous.
        state.selected_order_id = None
        state.resolved_order_ids = [
            order_id
            for order_id, _ in valid
        ][:20]
        state.entity_status = "ambiguous"
        state.entity_confidence = 0.5

    else:
        state.selected_order_id = None
        state.resolved_order_ids = []
        state.entity_status = "not_found"
        state.entity_confidence = 0.2

    return _finish(state, result)
