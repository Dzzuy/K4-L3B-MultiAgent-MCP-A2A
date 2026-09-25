from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from student_agent.contracts import ContractError, Contracts


@pytest.fixture
def contracts() -> Contracts:
    root = Path(__file__).resolve().parents[1]
    return Contracts(root / "contracts" / "schemas")


def make_valid_l3b_output(case_id: str = "CASE_001") -> dict[str, Any]:
    return {
        "schema_version": "day09-l3b-output-v2",
        "case_id": case_id,
        "assessment": {
            "primary_issue": "canceled_order_paid",
            "secondary_issues": ["seller_dispatch_delay"],
            "case_status": "action_required",
            "confidence": 0.95,
        },
        "affected_entities": {
            "order_ids": [case_id],
            "item_ids": ["ITEM_1"],
            "seller_ids": ["SELLER_1"],
            "payment_references": ["PAY_1"],
            "shipment_ids": ["SHIP_1"],
        },
        "entity_resolution": {
            "status": "resolved",
            "resolved_order_ids": [case_id],
            "rejected_candidates": ["CASE_OLD"],
            "confidence": 0.98,
        },
        "customer_context": {
            "customer_unique_id": "CUST_123",
            "related_order_ids": [case_id, "CASE_OLD"],
        },
        "shipment_analysis": {
            "verdict": "seller_delay",
            "late_seller_ids": ["SELLER_1"],
            "timeline_complete": True,
        },
        "payment_analysis": {
            "verdict": "reconciled",
            "captured_total_brl": 150.0,
            "refunded_total_brl": 0.0,
            "refundable_total_brl": 150.0,
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": "ORDER_CANCELED_BEFORE_FULFILLMENT", "rank": 1}],
            "responsible_parties": [{"party_type": "platform", "party_id": "platform"}],
        },
        "evidence_refs": ["ev_1234567890abcdef1234567890"],
        "data_conflicts": [],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": 150.0,
            "refund_lines": [
                {
                    "reason_code": "FULL_REFUND_ORDER_CANCELLATION",
                    "amount_brl": 150.0,
                    "entity_id": case_id,
                }
            ],
        },
        "resolution_actions": ["process_customer_refund"],
    }


def test_valid_l3b_output_passes(contracts: Contracts) -> None:
    output = make_valid_l3b_output()
    contracts.validate_output(output, "test_output")


def test_extra_property_in_output_is_rejected(contracts: Contracts) -> None:
    output = make_valid_l3b_output()
    output["unauthorized_extra_field"] = "malicious"
    with pytest.raises(ContractError, match="Additional properties are not allowed"):
        contracts.validate_output(output, "test_output")


def test_extra_property_in_nested_assessment_is_rejected(contracts: Contracts) -> None:
    output = make_valid_l3b_output()
    output["assessment"]["unauthorized_nested"] = 123
    with pytest.raises(ContractError, match="Additional properties are not allowed"):
        contracts.validate_output(output, "test_output")


def test_invalid_primary_issue_is_rejected(contracts: Contracts) -> None:
    output = make_valid_l3b_output()
    output["assessment"]["primary_issue"] = "not_a_valid_issue"
    with pytest.raises(ContractError):
        contracts.validate_output(output, "test_output")


def test_invalid_evidence_ref_format_is_rejected(contracts: Contracts) -> None:
    output = make_valid_l3b_output()
    output["evidence_refs"] = ["invalid_ref"]
    with pytest.raises(ContractError):
        contracts.validate_output(output, "test_output")


def test_valid_trace_event_passes(contracts: Contracts) -> None:
    event = {
        "schema_version": "day09-trace-event-v1",
        "event_id": "evt_1234567890abcdef123456",
        "case_id": "CASE_001",
        "event_type": "task_assigned",
        "occurred_at": "2026-09-25T10:00:00Z",
        "actor": "coordinator",
        "target": "order_agent",
        "attributes": {"task": "order_investigation"},
    }
    contracts.validate_trace(event, "test_trace")


def test_trace_event_with_extra_property_is_rejected(contracts: Contracts) -> None:
    event = {
        "schema_version": "day09-trace-event-v1",
        "event_id": "evt_1234567890abcdef123456",
        "case_id": "CASE_001",
        "event_type": "task_assigned",
        "occurred_at": "2026-09-25T10:00:00Z",
        "actor": "coordinator",
        "forbidden_extra": "rejected",
    }
    with pytest.raises(ContractError, match="Additional properties are not allowed"):
        contracts.validate_trace(event, "test_trace")


def test_valid_mcp_evidence_response_passes(contracts: Contracts) -> None:
    evidence = {
        "schema_version": "day09-mcp-evidence-v1",
        "evidence_ref": "ev_1234567890abcdef1234567890",
        "result_hash": f"sha256:{'a' * 64}",
        "domain": "order",
        "data": {"order_id": "CASE_001", "order_status": "delivered"},
    }
    contracts.validate_evidence(evidence, "test_evidence")


def test_mcp_evidence_with_extra_property_is_rejected(contracts: Contracts) -> None:
    evidence = {
        "schema_version": "day09-mcp-evidence-v1",
        "evidence_ref": "ev_1234567890abcdef1234567890",
        "result_hash": f"sha256:{'a' * 64}",
        "domain": "order",
        "data": {},
        "extra_field": "disallowed",
    }
    with pytest.raises(ContractError, match="Additional properties are not allowed"):
        contracts.validate_evidence(evidence, "test_evidence")
