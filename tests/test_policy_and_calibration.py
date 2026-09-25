from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import MockEvidenceGateway
from student_agent.contracts import Contracts
from student_agent.trace import TraceWriter
from student_agent.workflow import solve_case


@pytest.fixture
def contracts() -> Contracts:
    root = Path(__file__).resolve().parents[1]
    return Contracts(root / "contracts" / "schemas")


@pytest.fixture
def trace_writer(tmp_path: Path, contracts: Contracts) -> TraceWriter:
    trace_file = tmp_path / "traces" / "test_trace.jsonl"
    return TraceWriter(trace_file, contracts)


@pytest.mark.anyio
async def test_policy_logistics_delay_consistency(
    contracts: Contracts, trace_writer: TraceWriter
) -> None:
    """Verify that late_delivery_logistics blames carrier and never seller."""
    case_id = "CASE_LOGISTICS_DELAY_01"
    raw_case = {
        "case_id": case_id,
        "order_id": "ORD_LOGISTICS_01",
        "complaint": "Parcel arrived late after estimated date.",
    }

    gateway = MockEvidenceGateway(
        tool_responses={
            "get_order": {
                "order_status": "delivered",
                "order_delivered_carrier_date": "2026-02-01 10:00:00",
                "order_delivered_customer_date": "2026-02-15 10:00:00",
                "order_estimated_delivery_date": "2026-02-10 10:00:00",
            },
            "get_order_items": [
                {
                    "order_item_id": "ITM_1",
                    "seller_id": "SEL_ONTIME_1",
                    "shipping_limit_date": "2026-02-03 10:00:00",
                    "price": 60.0,
                }
            ],
            "get_order_payments": [{"payment_reference": "PAY_1", "payment_value": 60.0}],
            "get_shipment_details": {
                "order_delivered_carrier_date": "2026-02-01 10:00:00",
                "order_delivered_customer_date": "2026-02-15 10:00:00",
                "order_estimated_delivery_date": "2026-02-10 10:00:00",
            },
        }
    )

    output = await solve_case(raw_case, gateway, trace_writer)
    contracts.validate_output(output, "logistics_case")

    assert output["assessment"]["primary_issue"] == "late_delivery_logistics"
    assert output["shipment_analysis"]["verdict"] == "logistics_delay"
    assert output["shipment_analysis"]["late_seller_ids"] == []

    # Consistency invariant: carrier is blamed, NOT seller
    parties = output["root_cause_analysis"]["responsible_parties"]
    assert any(p["party_type"] == "logistics_provider" for p in parties)
    assert not any(p["party_type"] == "seller" for p in parties)


@pytest.mark.anyio
async def test_policy_unsupported_claim_consistency(
    contracts: Contracts, trace_writer: TraceWriter
) -> None:
    """Verify that unsupported claims have no refund, no_action, and calibrated confidence."""
    case_id = "CASE_UNSUPPORTED_01"
    raw_case = {
        "case_id": case_id,
        "order_id": "ORD_ONTIME_01",
        "complaint": "Delivery was late!",
    }

    gateway = MockEvidenceGateway(
        tool_responses={
            "get_order": {
                "order_status": "delivered",
                "order_delivered_carrier_date": "2026-01-01 10:00:00",
                "order_delivered_customer_date": "2026-01-03 10:00:00",
                "order_estimated_delivery_date": "2026-01-05 10:00:00",
            },
            "get_order_items": [
                {
                    "order_item_id": "ITM_1",
                    "seller_id": "SEL_1",
                    "shipping_limit_date": "2026-01-02 10:00:00",
                    "price": 120.0,
                }
            ],
            "get_order_payments": [{"payment_reference": "PAY_1", "payment_value": 120.0}],
            "get_shipment_details": {
                "order_delivered_carrier_date": "2026-01-01 10:00:00",
                "order_delivered_customer_date": "2026-01-03 10:00:00",
                "order_estimated_delivery_date": "2026-01-05 10:00:00",
            },
        }
    )

    output = await solve_case(raw_case, gateway, trace_writer)
    contracts.validate_output(output, "unsupported_case")

    assert output["assessment"]["primary_issue"] == "unsupported_claim"
    assert output["assessment"]["case_status"] == "no_action"
    assert output["financial_resolution"]["recommended_refund_brl"] == 0.0
    assert output["financial_resolution"]["refund_lines"] == []
    # Calibrated confidence: bounded strictly < 1.0
    assert 0.85 <= output["assessment"]["confidence"] <= 0.95


@pytest.mark.anyio
async def test_confidence_calibration_penalizes_conflicts(
    contracts: Contracts, trace_writer: TraceWriter
) -> None:
    """Verify that source conflicts degrade confidence."""
    case_id = "CASE_CONFLICT_01"
    raw_case = {
        "case_id": case_id,
        "order_id": "ORD_CONFLICT_01",
    }

    # Order says delivered, but carrier says returned/lost -> conflict!
    gateway = MockEvidenceGateway(
        tool_responses={
            "get_order": {"order_status": "delivered"},
            "get_order_items": [{"order_item_id": "ITM_1", "seller_id": "SEL_1", "price": 50.0}],
            "get_order_payments": [{"payment_reference": "PAY_1", "payment_value": 50.0}],
            "get_shipment_details": {"status": "lost"},
        }
    )

    output = await solve_case(raw_case, gateway, trace_writer)
    contracts.validate_output(output, "conflict_case")

    assert len(output["data_conflicts"]) > 0
    # Confidence is penalized due to conflict
    assert output["assessment"]["confidence"] < 0.85


@pytest.mark.anyio
async def test_full_lifecycle_events_progression(
    contracts: Contracts, trace_writer: TraceWriter
) -> None:
    """Verify trace lifecycle events include all required events in correct order."""
    case_id = "CASE_LIFECYCLE_01"
    raw_case = {"case_id": case_id, "order_id": "ORD_LIFE_01"}

    gateway = MockEvidenceGateway(
        tool_responses={"get_order": {"order_status": "delivered"}}
    )

    # Simulate cli.py emission of case_received before solve_case
    trace_writer.emit(case_id=case_id, event_type="case_received", actor="coordinator")

    output = await solve_case(raw_case, gateway, trace_writer)
    contracts.validate_output(output, "lifecycle_case")

    # Simulate cli.py emission of case_finalized after solve_case
    trace_writer.emit(case_id=case_id, event_type="case_finalized", actor="coordinator")

    lines = trace_writer.path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    event_types = [e["event_type"] for e in events]

    # Required events in scoring policy
    required = [
        "case_received",
        "task_assigned",
        "tool_result_consumed",
        "handoff",
        "policy_decided",
        "verification_completed",
        "case_finalized",
    ]
    for req in required:
        assert req in event_types, f"Missing required trace event: {req}"

    # Verify first and last events
    assert event_types[0] == "case_received"
    assert event_types[-1] == "case_finalized"
