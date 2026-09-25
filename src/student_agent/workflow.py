from __future__ import annotations

import asyncio
from typing import Any

from . import OUTPUT_SCHEMA_VERSION
from .agents import entity_agent, order_agent, payment_agent, policy_agent, shipment_agent, verifier
from .agents.protocol import assign, handoff
from .mcp_gateway import EvidenceGateway
from .state import AgentResult, CaseState
from .trace import TraceWriter


def _empty_result(actor: str, code: str) -> AgentResult:
    return AgentResult(actor=actor, ok=False, notes_code=code)


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    state = CaseState.from_case(case)

    assign(trace, state, "entity-agent", "RESOLVE_ENTITY_AND_CUSTOMER")
    entity = await entity_agent.run(state, gateway, trace)
    handoff(trace, state, "entity-agent", "order-agent", entity.notes_code, entity.evidence_refs)

    if state.selected_order_id:
        assign(trace, state, "order-agent", "VERIFY_ORDER_AND_ITEMS")
        order = await order_agent.run(state, gateway, trace)
    else:
        order = _empty_result("order-agent", "SKIPPED_NO_RESOLVED_ORDER")

    if order.ok:
        handoff(trace, state, "order-agent", "shipment-agent", "ORDER_READY", order.evidence_refs)
        handoff(trace, state, "order-agent", "payment-agent", "ORDER_READY", order.evidence_refs)
        assign(trace, state, "shipment-agent", "ANALYZE_SHIPMENT")
        assign(trace, state, "payment-agent", "ANALYZE_PAYMENT_REFUND")
        shipment, payment = await asyncio.gather(
            shipment_agent.run(state, gateway, trace, order),
            payment_agent.run(state, gateway, trace, order),
        )
    else:
        shipment = _empty_result("shipment-agent", "SKIPPED_NO_ORDER")
        payment = _empty_result("payment-agent", "SKIPPED_NO_ORDER")

    handoff(
        trace,
        state,
        "shipment-agent",
        "policy-agent",
        shipment.notes_code if not shipment.ok else "SHIPMENT_READY",
        shipment.evidence_refs,
    )
    handoff(
        trace,
        state,
        "payment-agent",
        "policy-agent",
        payment.notes_code if not payment.ok else "PAYMENT_READY",
        payment.evidence_refs,
    )

    assign(trace, state, "policy-agent", "ARBITRATE_WITH_POLICY")
    decision = await policy_agent.run(state, gateway, trace, order, shipment, payment)
    handoff(trace, state, "policy-agent", "verifier", "RESOLUTION_PROPOSED", decision["evidence_refs"])

    shipment_analysis = {
        "verdict": shipment.findings.get("verdict", "insufficient_evidence"),
        "late_seller_ids": shipment.findings.get("late_seller_ids", []),
        "timeline_complete": bool(shipment.findings.get("timeline_complete", False)),
    }
    payment_analysis = {
        "verdict": payment.findings.get("verdict", "insufficient_evidence"),
        "captured_total_brl": payment.findings.get("captured_total_brl"),
        "refunded_total_brl": payment.findings.get("refunded_total_brl"),
        "refundable_total_brl": decision["refundable_total_brl"],
    }

    refund = float(decision["refund"])
    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": state.case_id,
        "assessment": {
            "primary_issue": decision["issue"],
            "secondary_issues": decision["secondary_issues"],
            "case_status": decision["case_status"],
            "confidence": decision["confidence"],
        },
        "affected_entities": {
            "order_ids": order.entities.get("order_ids", state.resolved_order_ids)[:20],
            "item_ids": order.entities.get("item_ids", [])[:20],
            "seller_ids": order.entities.get("seller_ids", [])[:20],
            "payment_references": payment.entities.get("payment_references", [])[:20],
            "shipment_ids": shipment.entities.get("shipment_ids", [])[:20],
        },
        "claim_assessments": decision["claim_assessments"],
        "entity_resolution": {
            "status": state.entity_status,
            "resolved_order_ids": state.resolved_order_ids[:20],
            "rejected_candidates": state.rejected_candidates[:20],
            "confidence": state.entity_confidence,
        },
        "customer_context": {
            "customer_unique_id": state.customer_unique_id,
            "related_order_ids": state.related_order_ids[:20],
        },
        "shipment_analysis": shipment_analysis,
        "payment_analysis": payment_analysis,
        "root_cause_analysis": {
            "ranked_causes": decision["ranked_causes"],
            "responsible_parties": decision["responsible_parties"],
        },
        "evidence_refs": decision["evidence_refs"],
        "data_conflicts": decision["conflicts"],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": round(refund, 2),
            "refund_lines": (
                [
                    {
                        "reason_code": decision["actions"][0],
                        "amount_brl": round(refund, 2),
                        "entity_id": state.selected_order_id,
                    }
                ]
                if refund > 0
                else []
            ),
        },
        "resolution_actions": decision["actions"],
    }
    return verifier.run(state, trace, output)
