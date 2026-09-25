from __future__ import annotations

from ..facts import extract_order
from ..mcp_gateway import EvidenceGateway
from ..state import AgentResult, CaseState
from ..trace import TraceWriter
from .protocol import fetch

ACTOR = "order-agent"


async def run(state: CaseState, gateway: EvidenceGateway, trace: TraceWriter) -> AgentResult:
    result = AgentResult(actor=ACTOR)
    order_id = state.selected_order_id
    if not order_id:
        result.notes_code = "SKIPPED_NO_RESOLVED_ORDER"
        return result

    order_ev = await fetch(gateway, trace, state, ACTOR, "get_order", order_id=order_id)
    if order_ev is None or not isinstance(order_ev.data, dict):
        result.notes_code = "ORDER_NOT_FOUND"
        return result
    result.evidence.append(order_ev)

    items_ev = await fetch(gateway, trace, state, ACTOR, "get_order_items", order_id=order_id)
    items = items_ev.data if items_ev and isinstance(items_ev.data, list) else []
    if items_ev:
        result.evidence.append(items_ev)

    needs_seller_evidence = state.claimed_issue in {
        "unavailable_order_paid",
        "late_delivery_seller",
        "late_delivery_logistics",
    }
    if needs_seller_evidence:
        sellers_ev = await fetch(gateway, trace, state, ACTOR, "get_sellers", order_id=order_id)
        if sellers_ev:
            result.evidence.append(sellers_ev)

    facts = extract_order(order_ev.data, items, state.opened_at)
    result.ok = True
    result.entities = {
        "order_ids": [facts.order_id or order_id],
        "item_ids": facts.item_ids[:20],
        "seller_ids": facts.seller_ids[:20],
    }
    result.findings = {"facts": facts}
    result.conflicts = facts.conflicts[:5]
    return result
