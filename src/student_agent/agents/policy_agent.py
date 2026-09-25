from __future__ import annotations

from typing import Any

from ..facts import OrderFacts, PaymentFacts, ShipmentFacts, candidate_issues, money
from ..mcp_gateway import EvidenceGateway
from ..state import AgentResult, CaseState, Evidence
from ..trace import TraceWriter
from .protocol import fetch

ACTOR = "policy-agent"

CAUSE_CODES = {
    "canceled_order_paid": "ORDER_CANCELED_AFTER_CAPTURE",
    "unavailable_order_paid": "ORDER_UNAVAILABLE_AFTER_CAPTURE",
    "late_delivery_seller": "SELLER_SHIPPING_LIMIT_BREACH",
    "late_delivery_logistics": "CARRIER_TRANSIT_DELAY",
    "valid_split_payment": "SPLIT_PAYMENT_RECONCILED",
    "payment_mismatch": "PAYMENT_SUM_MISMATCH",
    "duplicate_charge": "DUPLICATE_PAYMENT_CAPTURE",
    "refund_pending": "REFUND_NOT_SETTLED",
    "refund_failed": "REFUND_ATTEMPT_FAILED",
    "unsupported_claim": "CLAIM_CONTRADICTS_EVIDENCE",
    "insufficient_evidence": "MISSING_REQUIRED_EVIDENCE",
}

ISSUE_DOMAINS = {
    "canceled_order_paid": {"order", "payment", "policy"},
    "unavailable_order_paid": {"order", "item", "seller", "payment", "policy"},
    "late_delivery_seller": {"order", "item", "seller", "shipment", "policy"},
    "late_delivery_logistics": {"order", "shipment", "policy"},
    "valid_split_payment": {"order", "payment", "policy"},
    "payment_mismatch": {"order", "payment", "policy"},
    "duplicate_charge": {"order", "payment", "policy"},
    "refund_pending": {"order", "payment", "refund", "policy"},
    "refund_failed": {"order", "payment", "refund", "policy"},
    "unsupported_claim": {"order", "payment", "shipment", "policy"},
    "insufficient_evidence": {"order", "customer", "policy"},
}


def _dedupe_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row not in out:
            out.append(row)
    return out


def _select_refs(state: CaseState, issue: str) -> list[str]:
    wanted = ISSUE_DOMAINS.get(issue, {"order", "policy"})
    if issue == "unsupported_claim" and state.claimed_issue in ISSUE_DOMAINS:
        wanted = ISSUE_DOMAINS[state.claimed_issue] | {"policy"}
    refs = [
        evidence.ref
        for evidence in state.evidence_by_ref.values()
        if evidence.domain in wanted
    ]
    if state.customer_unique_id:
        refs.extend(
            evidence.ref
            for evidence in state.evidence_by_ref.values()
            if evidence.domain == "customer"
        )
    return list(dict.fromkeys(refs))[:30]


def _confidence(
    state: CaseState,
    issue: str,
    signals: list[str],
    conflicts: list[dict[str, Any]],
    shipment: ShipmentFacts | None,
) -> float:
    if state.entity_status == "not_found":
        return 0.2
    if state.entity_status == "ambiguous":
        return 0.55
    if issue == "insufficient_evidence":
        return 0.35
    if issue.startswith("late_delivery") and shipment and not shipment.timeline_complete:
        return 0.6
    if conflicts or len(signals) > 1:
        return 0.78
    if state.claimed_issue and state.claimed_issue != issue:
        return 0.82
    return 0.92


async def run(
    state: CaseState,
    gateway: EvidenceGateway,
    trace: TraceWriter,
    order: AgentResult,
    shipment: AgentResult,
    payment: AgentResult,
) -> dict[str, Any]:
    policy_ev = await fetch(
        gateway,
        trace,
        state,
        ACTOR,
        "get_policy",
        policy_version=state.policy_version,
    )
    rules = (
        policy_ev.data.get("rules", {})
        if policy_ev and isinstance(policy_ev.data, dict)
        else {}
    )

    order_facts = order.findings.get("facts")
    shipment_facts = shipment.findings.get("facts")
    payment_facts = payment.findings.get("facts")
    order_facts = order_facts if isinstance(order_facts, OrderFacts) else None
    shipment_facts = shipment_facts if isinstance(shipment_facts, ShipmentFacts) else None
    payment_facts = payment_facts if isinstance(payment_facts, PaymentFacts) else None

    payment_claims = {
        "canceled_order_paid",
        "unavailable_order_paid",
        "valid_split_payment",
        "payment_mismatch",
        "duplicate_charge",
        "refund_pending",
        "refund_failed",
    }
    shipment_claims = {"late_delivery_seller", "late_delivery_logistics"}
    missing_claim_evidence = (
        state.claimed_issue in payment_claims and payment_facts is None
    ) or (state.claimed_issue in shipment_claims and shipment_facts is None)

    if order_facts is None or missing_claim_evidence:
        signals: list[str] = []
        issue = "insufficient_evidence"
    else:
        signals = candidate_issues(order_facts, payment_facts, shipment_facts)
        issue = state.claimed_issue if state.claimed_issue in signals else (
            signals[0] if signals else "unsupported_claim"
        )

    secondary_issues = [candidate for candidate in signals if candidate != issue][:10]
    rule = rules.get(issue) or {}
    if issue == "insufficient_evidence":
        case_status = "needs_investigation"
        refund = 0.0
        action = "escalate_missing_evidence"
        parties = [{"party_type": "unknown", "party_id": None}]
    else:
        case_status = str(rule.get("case_status") or (
            "no_action"
            if issue in {"unsupported_claim", "valid_split_payment"}
            else "needs_investigation"
        ))
        refund = money(rule.get("refund_brl"))
        action = str(rule.get("recommended_action") or (
            "document_no_action" if case_status == "no_action" else "review_case"
        ))
        seller_ids = order.entities.get("seller_ids", [])
        parties = []
        for raw in rule.get("responsible_parties") or [{"party_type": "unknown"}]:
            party_type = str(raw.get("party_type") or "unknown")
            if party_type == "seller" and seller_ids:
                parties.extend(
                    {"party_type": "seller", "party_id": seller_id}
                    for seller_id in seller_ids[:5]
                )
            else:
                parties.append(
                    {"party_type": party_type, "party_id": raw.get("party_id")}
                )

    if case_status == "no_action":
        refund = 0.0

    conflicts = _dedupe_dicts([
        *order.conflicts,
        *shipment.conflicts,
        *payment.conflicts,
    ])[:5]
    confidence = _confidence(state, issue, signals, conflicts, shipment_facts)
    cited_refs = _select_refs(state, issue)

    captured = payment_facts.captured_total_brl if payment_facts else 0.0
    refunded = payment_facts.refunded_total_brl if payment_facts else 0.0
    if payment_facts is None:
        refundable = None
    else:
        if issue in {"refund_pending", "refund_failed"} and refund <= 0:
            entitled = captured
        else:
            entitled = min(refund, captured) if captured > 0 else refund
        refundable = round(max(entitled - refunded, 0.0), 2)

    claim_assessments: list[dict[str, Any]] = []
    for claim in state.claims:
        if issue == "insufficient_evidence":
            verdict = "insufficient_evidence"
        elif claim.topic == "requested_full_refund":
            if refund <= 0:
                verdict = "unsupported"
            elif captured > 0 and refund + 0.05 >= captured:
                verdict = "supported"
            else:
                verdict = "partially_supported"
        else:
            verdict = "supported" if claim.topic == issue else "unsupported"
        claim_assessments.append(
            {
                "claim_id": claim.claim_id,
                "verdict": verdict,
                "confidence": confidence,
                "evidence_refs": cited_refs,
            }
        )

    decision = {
        "issue": issue,
        "secondary_issues": secondary_issues,
        "case_status": case_status,
        "confidence": confidence,
        "ranked_causes": [{"cause_code": CAUSE_CODES[issue], "rank": 1}],
        "responsible_parties": parties[:5],
        "refund": round(refund, 2),
        "refundable_total_brl": refundable,
        "actions": [action],
        "evidence_refs": cited_refs,
        "claim_assessments": claim_assessments[:5],
        "conflicts": conflicts,
    }
    trace.emit(
        case_id=state.case_id,
        event_type="policy_decided",
        actor=ACTOR,
        decision_code=issue,
        evidence_refs=cited_refs[:20] or None,
        attributes={"case_status": case_status, "refund_brl": round(refund, 2)},
    )
    return decision
