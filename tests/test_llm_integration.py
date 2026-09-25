from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
async def test_llm_reasoning_applied_when_valid(
    contracts: Contracts, trace_writer: TraceWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that when an LLM (< 10B) produces valid reasoning, it is applied."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

    mock_llm_response = {
        "primary_issue": "late_delivery_seller",
        "secondary_issues": ["seller_delayed_fulfillment"],
        "case_status": "action_required",
        "confidence": 0.94,
        "cause_code": "SELLER_DISPATCH_SLA_BREACH",
        "party_type": "seller",
        "party_id": "SEL_001",
        "recommended_refund_brl": 30.0,
        "refund_reason_code": "SELLER_LATE_PENALTY_REFUND",
        "resolution_actions": [
            "penalize_seller_late_fulfillment", "notify_customer_delivery_apology"
        ],
    }

    with patch("student_agent.agents.call_llm_reasoning", return_value=mock_llm_response):
        case_id = "CASE_LLM_001"
        raw_case = {
            "case_id": case_id,
            "order_id": "ORD_LLM_001",
            "customer_request": {
                "claimed_order_id": "ORD_LLM_001",
                "claims": [{"claim_id": "c1", "topic": "late_delivery_seller"}],
            },
        }
        gateway = MockEvidenceGateway(
            tool_responses={
                "get_order": {"order_status": "delivered"},
                "get_order_items": [{"order_item_id": "i1", "seller_id": "SEL_001", "price": 30.0}],
                "get_order_payments": [{"payment_reference": "p1", "payment_value": 30.0}],
                "get_shipment_summary": {"status": "delivered"},
            }
        )

        output = await solve_case(raw_case, gateway, trace_writer)
        contracts.validate_output(output, "llm_case")

        assert output["assessment"]["primary_issue"] == "late_delivery_seller"
        assert output["assessment"]["confidence"] == 0.94
        assert output["financial_resolution"]["recommended_refund_brl"] == 30.0
        assert output["root_cause_analysis"]["responsible_parties"][0]["party_type"] == "seller"
