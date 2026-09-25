from __future__ import annotations

from typing import Any

from ..facts import OrderFacts, PaymentFacts, ShipmentFacts, candidate_issues, money
from ..llm import OpenRouterAuditor
from ..mcp_gateway import EvidenceGateway
from ..state import AgentResult, CaseState
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

ISSUE_TOOLS = {
    "canceled_order_paid": {
        "get_order",
        "get_order_payments",
        "get_payment_timeline",
        "get_policy",
    },
    "unavailable_order_paid": {
        "get_order",
        "get_order_items",
        "get_sellers",
        "get_product_context",
        "get_order_payments",
        "get_payment_timeline",
        "get_policy",
    },
    "late_delivery_seller": {
        "get_order",
        "get_order_items",
        "get_sellers",
        "get_shipment_summary",
        "get_policy",
    },
    "late_delivery_logistics": {
        "get_order",
        "get_order_items",
        "get_shipment_summary",
        "get_policy",
    },
    "valid_split_payment": {
        "get_order",
        "get_order_payments",
        "get_payment_timeline",
        "get_policy",
    },
    "payment_mismatch": {
        "get_order",
        "get_order_payments",
        "get_payment_timeline",
        "get_policy",
    },
    "duplicate_charge": {
        "get_order",
        "get_order_payments",
        "get_payment_timeline",
        "get_policy",
    },
    "refund_pending": {
        "get_order",
        "get_order_payments",
        "get_payment_timeline",
        "get_refund_timeline",
        "get_policy",
    },
    "refund_failed": {
        "get_order",
        "get_order_payments",
        "get_payment_timeline",
        "get_refund_timeline",
        "get_policy",
    },
    "unsupported_claim": {
        "get_order",
        "get_policy",
    },
    "insufficient_evidence": {
        "get_customer_history",
        "get_order",
        "get_policy",
    },
}

ORDER_SCOPED_TOOLS = {
    "get_order",
    "get_order_items",
    "get_order_payments",
    "get_payment_timeline",
    "get_product_context",
    "get_refund_timeline",
    "get_sellers",
    "get_shipment_summary",
}

FULL_REFUND_ACTION_VERDICT = {
    "issue_refund": "supported",
    "retry_refund": "supported",
    "refund_freight": "partially_supported",
    "refund_duplicate_charge": "partially_supported",
    "reconcile_payment": "partially_supported",
    "monitor_refund": "partially_supported",
    "document_no_action": "unsupported",
}


def _dedupe_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row not in out:
            out.append(row)
    return out


def _select_refs(
    state: CaseState,
    issue: str,
) -> list[str]:
    wanted_tools = set(
        ISSUE_TOOLS.get(
            issue,
            {
                "get_order",
                "get_policy",
            },
        )
    )

    if (
        issue == "unsupported_claim"
        and state.claimed_issue in ISSUE_TOOLS
    ):
        wanted_tools |= ISSUE_TOOLS[state.claimed_issue]

    if (
        state.require_independent_verification
        and state.customer_unique_id is not None
    ):
        wanted_tools.add("get_customer_history")

    refs: list[str] = []

    for evidence in state.evidence_by_ref.values():
        if evidence.tool not in wanted_tools:
            continue

        if evidence.tool in ORDER_SCOPED_TOOLS:
            evidence_order_id = evidence.arguments.get("order_id")

            if state.selected_order_id is not None:
                if evidence_order_id != state.selected_order_id:
                    continue

            elif state.entity_status == "ambiguous":
                if evidence_order_id not in state.resolved_order_ids:
                    continue

            else:
                continue

        refs.append(evidence.ref)

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


def _claim_issue_verdict(
    topic: str,
    issue: str,
    supported_signals: list[str],
) -> str:
    if topic == issue:
        return "supported"
    if topic in supported_signals:
        return "supported"
    return "unsupported"


def _full_refund_verdict(
    action: str,
    captured_total_brl: float,
    recommended_refund_brl: float,
) -> str:
    mapped = FULL_REFUND_ACTION_VERDICT.get(action)
    if mapped is not None:
        return mapped
    if recommended_refund_brl <= 0:
        return "unsupported"
    if (
        captured_total_brl > 0
        and recommended_refund_brl + 0.05 >= captured_total_brl
    ):
        return "supported"
    return "partially_supported"


def _compute_refundable_total(
    issue: str,
    payment: PaymentFacts | None,
    recommended_refund_brl: float,
) -> float | None:
    if payment is None:
        return None

    captured = payment.captured_total_brl
    refunded = payment.refunded_total_brl
    outstanding_capture = max(captured - refunded, 0.0)

    if issue == "refund_pending" and payment.refund_pending_brl > 0:
        return round(min(payment.refund_pending_brl, outstanding_capture), 2)
    if issue == "refund_failed" and payment.refund_failed_brl > 0:
        return round(min(payment.refund_failed_brl, outstanding_capture), 2)

    entitlement = (
        min(recommended_refund_brl, captured)
        if captured > 0
        else recommended_refund_brl
    )
    return round(max(entitlement - refunded, 0.0), 2)


def _responsible_parties(
    issue: str,
    rule: dict[str, Any],
    order: AgentResult,
    shipment: AgentResult,
) -> list[dict[str, Any]]:
    seller_ids = (
        shipment.findings.get("late_seller_ids", [])
        if issue == "late_delivery_seller"
        else order.entities.get("seller_ids", [])
    )
    parties: list[dict[str, Any]] = []

    for raw in rule.get("responsible_parties") or [{"party_type": "unknown"}]:
        party_type = str(raw.get("party_type") or "unknown")
        if party_type == "seller":
            if seller_ids:
                parties.extend(
                    {"party_type": "seller", "party_id": seller_id}
                    for seller_id in seller_ids[:5]
                )
            else:
                parties.append({"party_type": "unknown", "party_id": None})
        else:
            parties.append(
                {"party_type": party_type, "party_id": raw.get("party_id")}
            )
    return parties[:5]


async def run(
    state: CaseState,
    gateway: EvidenceGateway,
    trace: TraceWriter,
    order: AgentResult,
    shipment: AgentResult,
    payment: AgentResult,
    auditor: OpenRouterAuditor | None = None,
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
        policy_ev.data.get("rules")
        if policy_ev and isinstance(policy_ev.data, dict)
        else None
    )
    policy_available = isinstance(rules, dict) and bool(rules)

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

    if not policy_available or order_facts is None or missing_claim_evidence:
        signals: list[str] = []
        issue = "insufficient_evidence"
    else:
        signals = candidate_issues(order_facts, payment_facts, shipment_facts)
        issue = (
            state.claimed_issue
            if state.claimed_issue in signals
            else signals[0] if signals else "unsupported_claim"
        )

    secondary_issues = [candidate for candidate in signals if candidate != issue][:10]
    audit = None
    if auditor is not None:
        audit = await auditor.audit_issue(
            deterministic_issue=issue,
            supported_issues=signals,
            verified_facts={
                "order_status": order_facts.status if order_facts else None,
                "shipment_verdict": shipment.findings.get("verdict"),
                "payment_verdict": payment.findings.get("verdict"),
            },
        )
    rule = rules.get(issue) if policy_available else {}
    rule = rule if isinstance(rule, dict) else {}

    if issue == "insufficient_evidence":
        case_status = "needs_investigation"
        refund = 0.0
        action = "escalate_missing_evidence"
        parties = [{"party_type": "unknown", "party_id": None}]
    else:
        case_status = str(
            rule.get("case_status")
            or (
                "no_action"
                if issue in {"unsupported_claim", "valid_split_payment"}
                else "needs_investigation"
            )
        )
        refund = money(rule.get("refund_brl"))
        action = str(
            rule.get("recommended_action")
            or (
                "document_no_action"
                if case_status == "no_action"
                else "review_case"
            )
        )
        parties = _responsible_parties(issue, rule, order, shipment)

    if case_status == "no_action":
        refund = 0.0

    conflicts = _dedupe_dicts(
        [
            *order.conflicts,
            *shipment.conflicts,
            *payment.conflicts,
        ]
    )[:5]
    confidence = _confidence(state, issue, signals, conflicts, shipment_facts)
    cited_refs = _select_refs(state, issue)
    refundable = _compute_refundable_total(issue, payment_facts, refund)
    captured = payment_facts.captured_total_brl if payment_facts else 0.0

    claim_assessments: list[dict[str, Any]] = []
    for claim in state.claims:
        if issue == "insufficient_evidence":
            verdict = "insufficient_evidence"
        elif claim.topic == "requested_full_refund":
            verdict = _full_refund_verdict(action, captured, refund)
        else:
            verdict = _claim_issue_verdict(claim.topic, issue, signals)
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
        "responsible_parties": parties,
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
        attributes={
            "case_status": case_status,
            "refund_brl": round(refund, 2),
            "llm_enabled": auditor.enabled if auditor else False,
            "llm_agreed": audit.agreed if audit else None,
            "llm_issue": audit.proposed_issue if audit else None,
            "llm_error": audit.error_code if audit else None,
        },
    )
    return decision
