from __future__ import annotations

from ..facts import OrderFacts, extract_shipment, make_window
from ..mcp_gateway import EvidenceGateway
from ..state import AgentResult, CaseState
from ..trace import TraceWriter
from .protocol import fetch

ACTOR = "shipment-agent"


async def run(
    state: CaseState,
    gateway: EvidenceGateway,
    trace: TraceWriter,
    order: AgentResult,
) -> AgentResult:
    result = AgentResult(actor=ACTOR)
    facts = order.findings.get("facts")
    if not order.ok or not isinstance(facts, OrderFacts) or not state.selected_order_id:
        result.notes_code = "SKIPPED_NO_ORDER"
        return result

    ev = await fetch(
        gateway,
        trace,
        state,
        ACTOR,
        "get_shipment_summary",
        order_id=state.selected_order_id,
    )
    if ev is None or not isinstance(ev.data, dict):
        result.notes_code = "SHIPMENT_NOT_FOUND"
        result.findings = {
            "verdict": "insufficient_evidence",
            "late_seller_ids": [],
            "timeline_complete": False,
            "shipment_ids": [],
            "facts": None,
        }
        return result

    result.evidence.append(ev)
    shipment = extract_shipment(ev.data, facts, make_window(ev.data | {
        "order_purchase_timestamp": facts.purchased_at,
        "order_delivered_customer_date": facts.delivered_at,
    }, state.opened_at))
    result.ok = True
    result.entities = {"shipment_ids": shipment.shipment_ids}
    result.findings = {
        "verdict": shipment.verdict,
        "late_seller_ids": shipment.late_seller_ids,
        "timeline_complete": shipment.timeline_complete,
        "shipment_ids": shipment.shipment_ids,
        "facts": shipment,
    }
    result.conflicts = shipment.conflicts[:5]
    return result
