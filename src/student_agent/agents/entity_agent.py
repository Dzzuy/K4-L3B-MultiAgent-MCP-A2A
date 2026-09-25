from __future__ import annotations

from typing import Any

from ..facts import parse_ts
from ..mcp_gateway import EvidenceGateway
from ..state import AgentResult, CaseState
from ..trace import TraceWriter
from .protocol import fetch

ACTOR = "entity-agent"


def _collect_order_ids(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"order_id", "order_ids", "related_order_ids"}:
                if isinstance(item, str) and item and item not in out:
                    out.append(item)
                elif isinstance(item, list):
                    for child in item:
                        if isinstance(child, str) and child and child not in out:
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


def _latest_order(valid_orders: list[tuple[str, dict[str, Any]]]) -> str:
    def key(item: tuple[str, dict[str, Any]]) -> float:
        moment = parse_ts(item[1].get("order_purchase_timestamp"))
        return moment.timestamp() if moment else 0.0

    return max(valid_orders, key=key)[0]


def _finish(state: CaseState, result: AgentResult) -> AgentResult:
    result.ok = state.selected_order_id is not None
    result.entities = {"order_ids": state.resolved_order_ids}
    result.findings = {
        "status": state.entity_status,
        "selected_order_id": state.selected_order_id,
        "resolved_order_ids": state.resolved_order_ids,
        "rejected_candidates": state.rejected_candidates,
        "confidence": state.entity_confidence,
        "customer_unique_id": state.customer_unique_id,
        "related_order_ids": state.related_order_ids[:20],
    }
    result.notes_code = "ENTITY_RESOLVED" if result.ok else "ENTITY_NOT_FOUND"
    return result


async def run(state: CaseState, gateway: EvidenceGateway, trace: TraceWriter) -> AgentResult:
    result = AgentResult(actor=ACTOR)

    if state.customer_unique_id:
        history_ev = await fetch(
            gateway,
            trace,
            state,
            ACTOR,
            "get_customer_history",
            customer_unique_id=state.customer_unique_id,
        )
        if history_ev:
            result.evidence.append(history_ev)
            _collect_order_ids(history_ev.data, state.related_order_ids)
            state.customer_unique_id = (
                _find_customer_unique_id(history_ev.data) or state.customer_unique_id
            )

    candidates = list(dict.fromkeys(state.candidate_order_ids))

    # Exact claimed ID is cheapest and strongest. Verify it first and stop on success.
    if state.claimed_order_id:
        claimed_ev = await fetch(
            gateway,
            trace,
            state,
            ACTOR,
            "get_order",
            order_id=state.claimed_order_id,
        )
        if claimed_ev and isinstance(claimed_ev.data, dict):
            result.evidence.append(claimed_ev)
            state.selected_order_id = state.claimed_order_id
            state.resolved_order_ids = [state.claimed_order_id]
            state.entity_status = "resolved"
            state.entity_confidence = 0.98
            return _finish(state, result)
        state.rejected_candidates.append(state.claimed_order_id)
        candidates = [item for item in candidates if item != state.claimed_order_id]

    if state.related_order_ids and candidates:
        narrowed = [order_id for order_id in candidates if order_id in state.related_order_ids]
        if narrowed:
            state.rejected_candidates.extend(
                order_id for order_id in candidates if order_id not in narrowed
            )
            candidates = narrowed

    valid: list[tuple[str, dict[str, Any]]] = []
    for order_id in candidates:
        order_ev = await fetch(gateway, trace, state, ACTOR, "get_order", order_id=order_id)
        if order_ev and isinstance(order_ev.data, dict):
            result.evidence.append(order_ev)
            valid.append((order_id, order_ev.data))
        elif order_id not in state.rejected_candidates:
            state.rejected_candidates.append(order_id)

    state.rejected_candidates = list(dict.fromkeys(state.rejected_candidates))[:20]
    if len(valid) == 1:
        state.selected_order_id = valid[0][0]
        state.resolved_order_ids = [valid[0][0]]
        state.entity_status = "resolved"
        state.entity_confidence = 0.92
    elif len(valid) > 1:
        state.selected_order_id = _latest_order(valid)
        state.resolved_order_ids = [order_id for order_id, _ in valid][:20]
        state.entity_status = "ambiguous"
        state.entity_confidence = 0.62
    else:
        state.selected_order_id = None
        state.resolved_order_ids = []
        state.entity_status = "not_found"
        state.entity_confidence = 0.2

    return _finish(state, result)
