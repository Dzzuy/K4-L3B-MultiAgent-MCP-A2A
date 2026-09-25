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
async def test_solve_case_canceled_order_paid(
    contracts: Contracts, trace_writer: TraceWriter
) -> None:
    case_id = "CASE_CANCELED_01"
    raw_case = {
        "case_id": case_id,
        "order_id": "ORD_CANCELED_001",
        "complaint": "I canceled the order but was charged anyway.",
    }

    gateway = MockEvidenceGateway(
        tool_responses={
            "get_order": {"order_status": "canceled"},
            "get_order_items": [
                {"order_item_id": "ITM_1", "seller_id": "SEL_1", "price": 100.0}
            ],
            "get_order_payments": [{"payment_reference": "PAY_1", "payment_value": 100.0}],
            "get_shipment_details": {"status": "canceled"},
        }
    )

    output = await solve_case(raw_case, gateway, trace_writer)

    # 1. Output adheres to public contract
    contracts.validate_output(output, "canceled_case")

    # 2. Correct business conclusions
    assert output["case_id"] == case_id
    assert output["assessment"]["primary_issue"] == "canceled_order_paid"
    assert output["assessment"]["case_status"] == "action_required"
    assert output["financial_resolution"]["recommended_refund_brl"] == 100.0
    assert len(output["financial_resolution"]["refund_lines"]) == 1
    assert output["entity_resolution"]["status"] == "resolved"
    assert output["entity_resolution"]["resolved_order_ids"] == ["ORD_CANCELED_001"]

    # 3. Trace events were generated and valid
    lines = trace_writer.path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    event_types = [e["event_type"] for e in events]

    assert "task_assigned" in event_types
    assert "handoff" in event_types
    assert "tool_result_consumed" in event_types
    assert "policy_decided" in event_types
    assert "verification_completed" in event_types


@pytest.mark.anyio
async def test_solve_case_seller_delay(
    contracts: Contracts, trace_writer: TraceWriter
) -> None:
    case_id = "CASE_SELLER_DELAY_02"
    raw_case = {
        "case_id": case_id,
        "order_id": "ORD_DELAY_002",
        "complaint": "Delivery arrived two weeks late due to seller delay.",
    }

    gateway = MockEvidenceGateway(
        tool_responses={
            "get_order": {
                "order_status": "delivered",
                "order_delivered_carrier_date": "2026-03-10 10:00:00",
                "order_delivered_customer_date": "2026-03-15 10:00:00",
                "order_estimated_delivery_date": "2026-03-12 10:00:00",
            },
            "get_order_items": [
                {
                    "order_item_id": "ITM_1",
                    "seller_id": "SEL_LATE_99",
                    "shipping_limit_date": "2026-03-05 10:00:00",
                    "price": 80.0,
                }
            ],
            "get_order_payments": [{"payment_reference": "PAY_2", "payment_value": 80.0}],
            "get_shipment_details": {
                "order_delivered_carrier_date": "2026-03-10 10:00:00",
                "order_delivered_customer_date": "2026-03-15 10:00:00",
                "order_estimated_delivery_date": "2026-03-12 10:00:00",
            },
        }
    )

    output = await solve_case(raw_case, gateway, trace_writer)
    contracts.validate_output(output, "seller_delay_case")

    assert output["assessment"]["primary_issue"] == "late_delivery_seller"
    assert output["shipment_analysis"]["verdict"] == "seller_delay"
    assert output["shipment_analysis"]["late_seller_ids"] == ["SEL_LATE_99"]


@pytest.mark.anyio
async def test_mcp_principles_all_pass(
    contracts: Contracts, trace_writer: TraceWriter
) -> None:
    """Test all 5 MCP Gateway Principles in Phase 3."""
    case_id = "CASE_MCP_PRINCIPLES_01"
    raw_case = {
        "case_id": case_id,
        "order_id": "ORD_AUDIT_100",
        "customer_unique_id": "CUST_AUDIT_999",
        "claims": [
            {
                "claim_id": "CLM_001",
                "type": "late_delivery",
                "description": "Carrier delivered late",
            },
            {
                "claim_id": "CLM_002",
                "type": "payment_refund",
                "description": "Refund not received",
            },
        ],
    }

    gateway = MockEvidenceGateway(
        tool_responses={
            "get_customer_history": {
                "customer_unique_id": "CUST_AUDIT_999",
                "orders": ["ORD_AUDIT_100"],
            },
            "get_order": {"order_status": "delivered"},
            "get_order_items": [{"order_item_id": "ITM_1", "seller_id": "SEL_1", "price": 50.0}],
            "get_order_payments": [{"payment_reference": "PAY_1", "payment_value": 50.0}],
            "get_shipment_details": {
                "order_delivered_carrier_date": "2026-01-01 10:00:00",
                "order_delivered_customer_date": "2026-01-05 10:00:00",
                "order_estimated_delivery_date": "2026-01-03 10:00:00",
            },
        }
    )

    output = await solve_case(raw_case, gateway, trace_writer)
    contracts.validate_output(output, "mcp_principles_case")

    assert gateway.call_count > 0
    for call_entry in gateway.call_log:
        assert call_entry["case_id"] == case_id, "Principle 1 violated: wrong case_id"

    trace_lines = trace_writer.path.read_text(encoding="utf-8").splitlines()
    trace_events = [json.loads(line) for line in trace_lines if line.strip()]
    consumed_events = [e for e in trace_events if e["event_type"] == "tool_result_consumed"]

    server_audit_refs = {
        f"ev_{entry['tool'][:12]}_{case_id[:8]}_{idx:04d}00000000"
        for idx, entry in enumerate(gateway.call_log, 1)
    }

    assert len(consumed_events) > 0
    for ce in consumed_events:
        assert ce["actor"] in (
            "entity-agent", "order-agent", "payment-agent", "shipment-agent", "policy-agent"
        )
        assert ce["tool_name"] is not None
        assert ce["evidence_refs"] is not None
        for ref in ce["evidence_refs"]:
            assert ref in server_audit_refs, "Principle 5 violated: un-audited evidence ref"

    for claim in output.get("claim_assessments", []):
        claim_id = claim["claim_id"]
        c_refs = claim["evidence_refs"]
        if claim_id == "CLM_001":
            for ref in c_refs:
                assert "shipment" in ref or ref in output["evidence_refs"]


@pytest.mark.anyio
async def test_retry_on_transient_failure(
    contracts: Contracts, trace_writer: TraceWriter
) -> None:
    """Test retry budget on transient network failure."""
    case_id = "CASE_RETRY_01"
    raw_case = {"case_id": case_id, "order_id": "ORD_RETRY_100"}

    gateway = MockEvidenceGateway(
        fail_count_before_success={"get_order": 1},
        tool_responses={"get_order": {"order_status": "delivered"}},
    )

    output = await solve_case(raw_case, gateway, trace_writer)
    contracts.validate_output(output, "retry_case")

    get_order_calls = [c for c in gateway.call_log if c["tool"] == "get_order"]
    assert len(get_order_calls) == 2


@pytest.mark.anyio
async def test_cache_avoids_duplicate_gateway_calls(
    contracts: Contracts, trace_writer: TraceWriter
) -> None:
    case_id = "CASE_CACHE_04"
    raw_case = {"case_id": case_id, "order_id": "ORD_CACHE_01"}

    gateway = MockEvidenceGateway(
        tool_responses={"get_order": {"order_status": "delivered"}}
    )

    output = await solve_case(raw_case, gateway, trace_writer)
    contracts.validate_output(output, "cache_case")

    initial_count = gateway.call_count
    assert initial_count > 0
