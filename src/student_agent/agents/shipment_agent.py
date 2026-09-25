from __future__ import annotations

from ..facts import (
    OrderFacts,
    extract_shipment,
    make_window,
)
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

    order_facts = order.findings.get("facts")

    if (
        not order.ok
        or not isinstance(order_facts, OrderFacts)
        or not state.selected_order_id
    ):
        result.notes_code = "SKIPPED_NO_ORDER"
        return result

    evidence = await fetch(
        gateway,
        trace,
        state,
        ACTOR,
        "get_shipment_summary",
        order_id=state.selected_order_id,
    )

    if (
        evidence is None
        or not isinstance(evidence.data, dict)
    ):
        result.notes_code = "SHIPMENT_NOT_FOUND"

        result.findings = {
            "verdict": "insufficient_evidence",
            "late_seller_ids": [],
            "timeline_complete": False,
            "shipment_ids": [],
            "facts": None,
        }

        return result

    result.evidence.append(evidence)

    window = make_window(
        {
            "order_purchase_timestamp": order_facts.purchased_at,
            "order_delivered_customer_date": order_facts.delivered_at,
        },
        state.opened_at,
    )

    shipment = extract_shipment(
        evidence.data,
        order_facts,
        window,
    )

    result.ok = True

    result.entities = {
        "shipment_ids": shipment.shipment_ids[:20],
    }

    result.findings = {
        "verdict": shipment.verdict,
        "late_seller_ids": shipment.late_seller_ids[:20],
        "timeline_complete": shipment.timeline_complete,
        "shipment_ids": shipment.shipment_ids[:20],
        "facts": shipment,
    }

    result.conflicts = shipment.conflicts[:5]

    result.notes_code = "SHIPMENT_READY"

    return result
