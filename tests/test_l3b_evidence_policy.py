from pathlib import Path

from student_agent.agents.policy_agent import (
    _claim_issue_verdict,
    _compute_refundable_total,
    _full_refund_verdict,
    _select_refs,
)
from student_agent.agents.verifier import run as verify_output
from student_agent.contracts import Contracts
from student_agent.facts import PaymentFacts
from student_agent.state import CaseState, Evidence
from student_agent.trace import TraceWriter


def _state() -> CaseState:
    state = CaseState.from_case(
        {
            "case_id": "L3B_CASE_001",
            "customer_request": {
                "claims": [{"claim_id": "claim-1", "topic": "canceled_order_paid"}],
            },
            "candidate_order_ids": ["O_REAL", "O_BAD"],
            "investigation_scope": {
                "require_independent_verification": True,
            },
        }
    )
    state.selected_order_id = "O_REAL"
    state.resolved_order_ids = ["O_REAL"]
    state.entity_status = "resolved"
    return state


def _ref(letter: str) -> str:
    return f"ev_{letter * 20}"


def _payment(
    *,
    captured: float,
    refunded: float,
    pending: float = 0.0,
    failed: float = 0.0,
) -> PaymentFacts:
    return PaymentFacts(
        captured_total_brl=captured,
        refunded_total_brl=refunded,
        duplicate=False,
        mismatch=False,
        split_valid=False,
        refund_pending=pending > 0,
        refund_failed=failed > 0,
        refund_pending_brl=pending,
        refund_failed_brl=failed,
        capture_amounts_brl=[],
        payment_references=[],
    )


def test_rejected_candidate_evidence_is_excluded() -> None:
    state = _state()
    real_ref = _ref("a")
    bad_ref = _ref("b")
    policy_ref = _ref("c")
    state.register_evidence(
        Evidence("get_order", "order", real_ref, {}, {"order_id": "O_REAL"})
    )
    state.register_evidence(
        Evidence("get_order", "order", bad_ref, {}, {"order_id": "O_BAD"})
    )
    state.register_evidence(Evidence("get_policy", "policy", policy_ref, {}))

    refs = _select_refs(state, "canceled_order_paid")

    assert real_ref in refs
    assert policy_ref in refs
    assert bad_ref not in refs


def test_independent_customer_history_evidence_is_retained() -> None:
    state = _state()
    state.customer_unique_id = "customer-1"
    order_ref = _ref("a")
    policy_ref = _ref("b")
    history_ref = _ref("c")
    state.register_evidence(
        Evidence("get_order", "order", order_ref, {}, {"order_id": "O_REAL"})
    )
    state.register_evidence(Evidence("get_policy", "policy", policy_ref, {}))
    state.register_evidence(
        Evidence("get_customer_history", "customer", history_ref, {})
    )

    assert history_ref in _select_refs(state, "canceled_order_paid")


def test_secondary_signal_claim_remains_supported() -> None:
    assert (
        _claim_issue_verdict(
            "payment_mismatch",
            "refund_failed",
            ["refund_failed", "payment_mismatch"],
        )
        == "supported"
    )


def test_requested_full_refund_action_mapping() -> None:
    assert _full_refund_verdict("issue_refund", 100, 0) == "supported"
    assert _full_refund_verdict("refund_freight", 100, 0) == "partially_supported"
    assert _full_refund_verdict("document_no_action", 100, 100) == "unsupported"


def test_refund_pending_uses_exact_outstanding_amount() -> None:
    assert _compute_refundable_total(
        "refund_pending", _payment(captured=100, refunded=20, pending=50), 100
    ) == 50


def test_refund_pending_is_capped_by_outstanding_capture() -> None:
    assert _compute_refundable_total(
        "refund_pending", _payment(captured=100, refunded=80, pending=50), 100
    ) == 20


def test_verifier_removes_rejected_order_evidence(tmp_path: Path) -> None:
    state = _state()
    real_ref = _ref("a")
    bad_ref = _ref("b")
    policy_ref = _ref("c")
    state.register_evidence(
        Evidence("get_order", "order", real_ref, {}, {"order_id": "O_REAL"})
    )
    state.register_evidence(
        Evidence("get_order", "order", bad_ref, {}, {"order_id": "O_BAD"})
    )
    state.register_evidence(Evidence("get_policy", "policy", policy_ref, {}))

    output = {
        "schema_version": "day09-l3b-output-v2",
        "case_id": state.case_id,
        "assessment": {
            "primary_issue": "canceled_order_paid",
            "secondary_issues": [],
            "case_status": "needs_investigation",
            "confidence": 0.9,
        },
        "affected_entities": {
            "order_ids": ["O_REAL"],
            "item_ids": [],
            "seller_ids": [],
            "payment_references": [],
            "shipment_ids": [],
        },
        "claim_assessments": [
            {
                "claim_id": "claim-1",
                "verdict": "supported",
                "confidence": 0.9,
                "evidence_refs": [real_ref, bad_ref, policy_ref],
            }
        ],
        "entity_resolution": {
            "status": "resolved",
            "resolved_order_ids": ["O_REAL"],
            "rejected_candidates": ["O_BAD"],
            "confidence": 0.9,
        },
        "customer_context": {"customer_unique_id": None, "related_order_ids": []},
        "shipment_analysis": {
            "verdict": "insufficient_evidence",
            "late_seller_ids": [],
            "timeline_complete": False,
        },
        "payment_analysis": {
            "verdict": "insufficient_evidence",
            "captured_total_brl": None,
            "refunded_total_brl": None,
            "refundable_total_brl": None,
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": "ORDER_CANCELED_AFTER_CAPTURE", "rank": 1}],
            "responsible_parties": [{"party_type": "unknown", "party_id": None}],
        },
        "evidence_refs": [real_ref, bad_ref, policy_ref],
        "data_conflicts": [],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": 0.0,
            "refund_lines": [],
        },
        "resolution_actions": ["review_case"],
    }
    trace = TraceWriter(
        tmp_path / "trace.jsonl",
        Contracts(Path("contracts/schemas")),
    )

    verified = verify_output(state, trace, output)

    assert bad_ref not in verified["evidence_refs"]
    assert bad_ref not in verified["claim_assessments"][0]["evidence_refs"]
