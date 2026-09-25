from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from conftest import MockEvidenceGateway
from student_agent import VARIANT_ID
from student_agent.cases import load_case_set
from student_agent.contracts import Contracts
from student_agent.submission import package_submission, validate_artifacts
from student_agent.trace import TraceWriter
from student_agent.workflow import solve_case


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def contracts() -> Contracts:
    root = Path(__file__).resolve().parents[1]
    return Contracts(root / "contracts" / "schemas")


@pytest.mark.anyio
async def test_full_100_case_batch_run_and_validation(
    tmp_path: Path, contracts: Contracts
) -> None:
    """Simulate Phase 5: 100 cases batch execution, validation, and packaging."""
    case_ids = [f"L3B_CASE_{i:03d}" for i in range(1, 101)]

    # Copy contracts to tmp_path so CLI package_submission finds them
    shutil.copytree(contracts.root, tmp_path / "contracts" / "schemas")

    # 1. Create simulated 100-case input bundle
    manifest_data = {
        "case_set_version": "v2.1",
        "variant_id": VARIANT_ID,
        "case_ids": case_ids,
    }
    write_json(tmp_path / "case-set.json", manifest_data)

    categories = [
        ("canceled_order_paid", "canceled", 100.0, "delivered_late"),
        ("late_delivery_seller", "delivered", 50.0, "seller_delay"),
        ("late_delivery_logistics", "delivered", 75.0, "carrier_delay"),
        ("duplicate_charge", "delivered", 120.0, "duplicate"),
        ("unsupported_claim", "delivered", 80.0, "ontime"),
    ]

    for idx, case_id in enumerate(case_ids):
        cat_type, order_status, price, issue_type = categories[idx % len(categories)]
        case_payload: dict[str, Any] = {
            "case_id": case_id,
            "order_id": f"ORD_{case_id}",
            "customer_unique_id": f"CUST_{idx:03d}",
            "complaint": f"Complaint for {cat_type}",
        }
        write_json(tmp_path / "inputs" / f"{case_id}.json", case_payload)

    # 2. Test input validation (day09 validate-inputs)
    loaded_set = load_case_set(tmp_path, expected_count=100)
    assert len(loaded_set.case_ids) == 100
    assert loaded_set.variant_id == VARIANT_ID

    # 3. Batch Run 100 cases (day09 run)
    outputs_dir = tmp_path / "outputs"
    trace_path = tmp_path / "traces" / "trace.jsonl"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    trace_writer = TraceWriter(trace_path, contracts)

    for idx, case_id in enumerate(loaded_set.case_ids):
        case = loaded_set.cases[case_id]
        cat_type, order_status, price, issue_type = categories[idx % len(categories)]

        tool_responses = {
            "get_order": {
                "order_status": order_status,
                "order_delivered_carrier_date": "2026-03-10 10:00:00",
                "order_delivered_customer_date": "2026-03-15 10:00:00",
                "order_estimated_delivery_date": "2026-03-12 10:00:00",
            },
            "get_order_items": [
                {
                    "order_item_id": f"ITM_{case_id}",
                    "seller_id": f"SEL_{idx:03d}",
                    "shipping_limit_date": (
                        "2026-03-05 10:00:00"
                        if cat_type == "late_delivery_seller"
                        else "2026-03-12 10:00:00"
                    ),
                    "price": price,
                }
            ],
            "get_order_payments": [
                {"payment_reference": f"PAY_A_{case_id}", "payment_value": price}
            ],
            "get_shipment_details": {
                "order_delivered_carrier_date": "2026-03-10 10:00:00",
                "order_delivered_customer_date": "2026-03-15 10:00:00",
                "order_estimated_delivery_date": "2026-03-12 10:00:00",
            },
        }

        if cat_type == "duplicate_charge":
            tool_responses["get_order_payments"].append(
                {"payment_reference": f"PAY_B_{case_id}", "payment_value": price}
            )

        gateway = MockEvidenceGateway(tool_responses=tool_responses)

        trace_writer.emit(case_id=case_id, event_type="case_received", actor="coordinator")
        output = await solve_case(case, gateway, trace_writer)
        contracts.validate_output(output, f"outputs/{case_id}.json")
        write_json(outputs_dir / f"{case_id}.json", output)
        trace_writer.emit(case_id=case_id, event_type="case_finalized", actor="coordinator")

    # 4. Thẩm định kết quả trước khi đóng gói (day09 validate)
    validated_outputs, trace_lines = validate_artifacts(tmp_path, loaded_set, contracts)
    assert len(validated_outputs) == 100
    assert len(trace_lines) >= 600

    # 5. Đóng gói nộp bài (day09 package)
    zip_dest = tmp_path / "dist" / "submission.zip"
    packaged = package_submission(tmp_path, zip_dest)
    assert packaged.exists()
    assert packaged.stat().st_size > 0
