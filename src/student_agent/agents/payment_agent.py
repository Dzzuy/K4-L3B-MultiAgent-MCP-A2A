from __future__ import annotations

from ..facts import OrderFacts, extract_payments, make_window
from ..mcp_gateway import EvidenceGateway
from ..state import AgentResult, CaseState
from ..trace import TraceWriter
from .protocol import fetch

ACTOR = "payment-agent"


async def run(
    state: CaseState,
    gateway: EvidenceGateway,
    trace: TraceWriter,
    order: AgentResult,
) -> AgentResult:
    result = AgentResult(actor=ACTOR)
    order_facts = order.findings.get("facts")
    if not order.ok or not isinstance(order_facts, OrderFacts) or not state.selected_order_id:
        result.notes_code = "SKIPPED_NO_ORDER"
        return result

    pay_ev = await fetch(
        gateway,
        trace,
        state,
        ACTOR,
        "get_payment_timeline",
        order_id=state.selected_order_id,
    )
    if pay_ev is None or not isinstance(pay_ev.data, dict):
        result.notes_code = "PAYMENT_NOT_FOUND"
        result.findings = {
            "verdict": "insufficient_evidence",
            "captured_total_brl": None,
            "refunded_total_brl": None,
            "refundable_total_brl": None,
            "payment_references": [],
            "facts": None,
        }
        return result
    result.evidence.append(pay_ev)

    pay_events = pay_ev.data.get("events") or []
    topic_requires_refund = any("refund" in claim.topic for claim in state.claims)
    order_can_require_refund = order_facts.status in {"canceled", "unavailable"}
    timeline_mentions_refund = any(
        "refund" in str(event.get("event_type", "")) for event in pay_events
    )

    refund_ev = None
    if topic_requires_refund or order_can_require_refund or timeline_mentions_refund:
        refund_ev = await fetch(
            gateway,
            trace,
            state,
            ACTOR,
            "get_refund_timeline",
            order_id=state.selected_order_id,
        )
        if refund_ev:
            result.evidence.append(refund_ev)

    window = make_window(
        {
            "order_purchase_timestamp": order_facts.purchased_at,
            "order_delivered_customer_date": order_facts.delivered_at,
        },
        state.opened_at,
    )
    payment = extract_payments(
        pay_ev.data,
        refund_ev.data if refund_ev and isinstance(refund_ev.data, dict) else None,
        window,
        order_facts.total_brl,
    )

    if payment.refund_failed:
        verdict = "refund_failed"
    elif payment.refund_pending:
        verdict = "refund_pending"
    elif payment.duplicate:
        verdict = "duplicate_capture"
    elif payment.mismatch:
        verdict = "capture_mismatch"
    elif payment.refunded_total_brl > 0:
        verdict = "refunded"
    else:
        verdict = "reconciled"

    result.ok = True
    result.entities = {"payment_references": payment.payment_references}
    result.findings = {
        "verdict": verdict,
        "captured_total_brl": payment.captured_total_brl,
        "refunded_total_brl": payment.refunded_total_brl,
        "refundable_total_brl": 0.0,
        "payment_references": payment.payment_references,
        "facts": payment,
    }
    result.conflicts = payment.conflicts[:5]
    return result
