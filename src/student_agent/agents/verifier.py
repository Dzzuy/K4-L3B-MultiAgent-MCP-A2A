from __future__ import annotations

from typing import Any

from ..state import CaseState
from ..trace import TraceWriter

ACTOR = "verifier"


def run(state: CaseState, trace: TraceWriter, output: dict[str, Any]) -> dict[str, Any]:
    fixes: list[str] = []
    allowed_refs = set(state.evidence_by_ref)

    refs = [ref for ref in output["evidence_refs"] if ref in allowed_refs]
    if refs != output["evidence_refs"]:
        fixes.append("DROP_UNCONSUMED_EVIDENCE")
    output["evidence_refs"] = list(dict.fromkeys(refs))[:30]
    for claim in output.get("claim_assessments", []):
        claim["evidence_refs"] = [
            ref for ref in claim["evidence_refs"] if ref in output["evidence_refs"]
        ][:30]

    entity = output["entity_resolution"]
    if entity["status"] == "resolved" and not entity["resolved_order_ids"]:
        entity["status"] = "not_found"
        entity["confidence"] = min(entity["confidence"], 0.25)
        fixes.append("EMPTY_RESOLUTION_DOWNGRADED")

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
    for party in output["root_cause_analysis"]["responsible_parties"]:
        if party["party_type"] == "seller" and party["party_id"] not in sellers:
            party["party_type"] = "unknown"
            party["party_id"] = None
            fixes.append("SELLER_OUT_OF_SCOPE")

    output["resolution_actions"] = list(dict.fromkeys(output["resolution_actions"]))[:8]
    output["assessment"]["confidence"] = min(
        max(float(output["assessment"]["confidence"]), 0.0), 1.0
    )
    entity["confidence"] = min(max(float(entity["confidence"]), 0.0), 1.0)

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
