from __future__ import annotations

from typing import Any

from ..state import CaseState
from ..trace import TraceWriter

ACTOR = "verifier"

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


def _ref_is_in_scope(state: CaseState, ref: str) -> bool:
    evidence = state.evidence_by_ref.get(ref)
    if evidence is None:
        return False
    if evidence.tool not in ORDER_SCOPED_TOOLS:
        return True

    order_id = evidence.arguments.get("order_id")
    if state.selected_order_id is not None:
        return order_id == state.selected_order_id
    if state.entity_status == "ambiguous":
        return order_id in state.resolved_order_ids
    return False


def _filter_refs(state: CaseState, refs: list[str]) -> list[str]:
    return list(dict.fromkeys(ref for ref in refs if _ref_is_in_scope(state, ref)))[:30]


def _downgrade_missing_policy(output: dict[str, Any]) -> None:
    refs = output["evidence_refs"]
    assessment = output["assessment"]
    assessment["primary_issue"] = "insufficient_evidence"
    assessment["secondary_issues"] = []
    assessment["case_status"] = "needs_investigation"
    assessment["confidence"] = min(float(assessment["confidence"]), 0.30)

    output["root_cause_analysis"] = {
        "ranked_causes": [{"cause_code": "MISSING_REQUIRED_EVIDENCE", "rank": 1}],
        "responsible_parties": [{"party_type": "unknown", "party_id": None}],
    }
    output["financial_resolution"]["recommended_refund_brl"] = 0.0
    output["financial_resolution"]["refund_lines"] = []
    output["resolution_actions"] = ["escalate_missing_evidence"]

    payment = output["payment_analysis"]
    if payment["refundable_total_brl"] is not None:
        payment["refundable_total_brl"] = 0.0

    for claim in output.get("claim_assessments", []):
        claim["verdict"] = "insufficient_evidence"
        claim["confidence"] = min(float(claim["confidence"]), 0.30)
        claim["evidence_refs"] = refs[:30]


def _has_policy_evidence(state: CaseState, refs: list[str]) -> bool:
    return any(
        state.evidence_by_ref[ref].tool == "get_policy"
        for ref in refs
        if ref in state.evidence_by_ref
    )


def run(state: CaseState, trace: TraceWriter, output: dict[str, Any]) -> dict[str, Any]:
    fixes: list[str] = []

    original_refs = output["evidence_refs"]
    output["evidence_refs"] = _filter_refs(state, original_refs)
    if output["evidence_refs"] != original_refs:
        fixes.append("DROP_UNCONSUMED_OR_OUT_OF_SCOPE_EVIDENCE")

    for claim in output.get("claim_assessments", []):
        original_claim_refs = claim["evidence_refs"]
        claim["evidence_refs"] = [
            ref for ref in original_claim_refs if ref in output["evidence_refs"]
        ][:30]
        if claim["evidence_refs"] != original_claim_refs:
            fixes.append("DROP_OUT_OF_SCOPE_CLAIM_EVIDENCE")

    entity = output["entity_resolution"]
    if entity["status"] == "resolved" and not entity["resolved_order_ids"]:
        entity["status"] = "not_found"
        entity["confidence"] = min(entity["confidence"], 0.25)
        fixes.append("EMPTY_RESOLUTION_DOWNGRADED")

    assessment = output["assessment"]
    if (
        assessment["primary_issue"] != "insufficient_evidence"
        and not _has_policy_evidence(state, output["evidence_refs"])
    ):
        _downgrade_missing_policy(output)
        fixes.append("MISSING_POLICY_EVIDENCE_DOWNGRADED")

    finance = output["financial_resolution"]
    if output["assessment"]["case_status"] == "no_action":
        if finance["recommended_refund_brl"] or finance["refund_lines"]:
            fixes.append("NO_ACTION_ZERO_REFUND")
        finance["recommended_refund_brl"] = 0.0
        finance["refund_lines"] = []
    else:
        total = round(sum(line["amount_brl"] for line in finance["refund_lines"]), 2)
        if abs(total - finance["recommended_refund_brl"]) > 0.01:
            finance["recommended_refund_brl"] = total
            fixes.append("REFUND_TOTAL_REBALANCED")

    payment = output["payment_analysis"]
    captured = payment["captured_total_brl"]
    refunded = payment["refunded_total_brl"]
    refundable = payment["refundable_total_brl"]
    if captured is not None and refunded is not None and refunded > captured:
        payment["refunded_total_brl"] = captured
        refunded = captured
        fixes.append("REFUNDED_CLAMPED_TO_CAPTURED")
    if captured is not None and refundable is not None:
        max_outstanding = max(captured - (refunded or 0.0), 0.0)
        if refundable > max_outstanding:
            payment["refundable_total_brl"] = round(max_outstanding, 2)
            fixes.append("REFUNDABLE_CLAMPED")

    sellers = set(output["affected_entities"]["seller_ids"])
    late_sellers = set(output["shipment_analysis"]["late_seller_ids"])
    for party in output["root_cause_analysis"]["responsible_parties"]:
        seller_out_of_scope = (
            party["party_type"] == "seller" and party["party_id"] not in sellers
        )
        seller_not_late = (
            output["assessment"]["primary_issue"] == "late_delivery_seller"
            and party["party_type"] == "seller"
            and party["party_id"] not in late_sellers
        )
        if seller_out_of_scope or seller_not_late:
            party["party_type"] = "unknown"
            party["party_id"] = None
            fixes.append("SELLER_OUT_OF_SCOPE_OR_NOT_LATE")

    output["resolution_actions"] = list(dict.fromkeys(output["resolution_actions"]))[:8]
    output["assessment"]["confidence"] = min(
        max(float(output["assessment"]["confidence"]), 0.0), 1.0
    )
    entity["confidence"] = min(max(float(entity["confidence"]), 0.0), 1.0)
    for claim in output.get("claim_assessments", []):
        claim["confidence"] = min(max(float(claim["confidence"]), 0.0), 1.0)
        claim["evidence_refs"] = [
            ref for ref in claim["evidence_refs"] if ref in output["evidence_refs"]
        ][:30]

    trace.contracts.validate_output(output, f"verifier:output:{state.case_id}")
    trace.emit(
        case_id=state.case_id,
        event_type="verification_completed",
        actor=ACTOR,
        decision_code="PASS" if not fixes else "FIXED",
        evidence_refs=output["evidence_refs"][:20] or None,
        attributes={"fixes": ",".join(fixes) or None},
    )
    return output
