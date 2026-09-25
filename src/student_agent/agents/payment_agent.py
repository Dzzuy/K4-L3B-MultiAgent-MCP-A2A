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

    if (
        not order.ok
        or not isinstance(order_facts, OrderFacts)
        or not state.selected_order_id
    ):
        result.notes_code = "SKIPPED_NO_ORDER"
        return result

    order_id = state.selected_order_id

    rows_ev = await fetch(
        gateway,
        trace,
        state,
        ACTOR,
        "get_order_payments",
        order_id=order_id,
    )

    payment_rows = None

    if rows_ev is not None:
        result.evidence.append(rows_ev)

        if isinstance(rows_ev.data, list):
            payment_rows = rows_ev.data

    timeline_ev = await fetch(
        gateway,
        trace,
        state,
        ACTOR,
        "get_payment_timeline",
        order_id=order_id,
    )

    if (
        timeline_ev is None
        or not isinstance(timeline_ev.data, dict)
    ):
        result.notes_code = "PAYMENT_TIMELINE_NOT_FOUND"

        result.findings = {
            "verdict": "insufficient_evidence",
            "captured_total_brl": None,
            "refunded_total_brl": None,
            "refundable_total_brl": None,
            "payment_references": [],
            "facts": None,
        }

        return result

    result.evidence.append(timeline_ev)

    pay_events = timeline_ev.data.get("events") or []

    topic_requires_refund = any(
        "refund" in claim.topic
        for claim in state.claims
    )

    order_can_require_refund = order_facts.status in {
        "canceled",
        "unavailable",
    }

    timeline_mentions_refund = any(
        "refund" in str(event.get("event_type", "")).lower()
        for event in pay_events
        if isinstance(event, dict)
    )

    refund_ev = None

    if (
        topic_requires_refund
        or order_can_require_refund
        or timeline_mentions_refund
    ):
        refund_ev = await fetch(
            gateway,
            trace,
            state,
            ACTOR,
            "get_refund_timeline",
            order_id=order_id,
        )

        if refund_ev is not None:
            result.evidence.append(refund_ev)

    window = make_window(
        {
            "order_purchase_timestamp": order_facts.purchased_at,
            "order_delivered_customer_date": order_facts.delivered_at,
        },
        state.opened_at,
    )

    payment = extract_payments(
        payment_rows,
        timeline_ev.data,
        (
            refund_ev.data
            if refund_ev
            and isinstance(refund_ev.data, dict)
            else None
        ),
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

    result.entities = {
        "payment_references": payment.payment_references[:20],
    }

    result.findings = {
        "verdict": verdict,
        "captured_total_brl": payment.captured_total_brl,
        "refunded_total_brl": payment.refunded_total_brl,
        "refundable_total_brl": 0.0,
        "payment_references": payment.payment_references[:20],
        "refund_pending_brl": payment.refund_pending_brl,
        "refund_failed_brl": payment.refund_failed_brl,
        "capture_amounts_brl": payment.capture_amounts_brl,
        "payment_rows_available": payment_rows is not None,
        "facts": payment,
    }

    result.conflicts = payment.conflicts[:5]

    if payment_rows is None:
        result.notes_code = "PAYMENT_READY_ROWS_MISSING"
    else:
        result.notes_code = "PAYMENT_READY"

    return result
