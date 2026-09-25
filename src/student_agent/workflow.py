from __future__ import annotations

from typing import Any

from .agents import (
    CaseContext,
    CoordinatorAgent,
    OrderItemAgent,
    PaymentAgent,
    PolicyAgent,
    ShipmentAgent,
    VerifierAgent,
)
from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Execute the L3B Multi-Agent A2A workflow.

    Workflow topology:
    1. Coordinator: Ingests case, performs entity resolution, dispatches specialists.
    2. Order/Item Agent: Investigates order status, items, products, and sellers.
    3. Payment Agent: Reconciles payments, detects duplicate charges, audits refunds.
    4. Shipment Agent: Analyzes carrier telemetry, shipping SLA deadlines, delays.
    5. Policy Agent: Arbitrates conflicts, assigns root cause, decides resolution.
    6. Verifier Agent: Enforces invariants, checks JSON Schema, finalizes output.
    """
    contracts = trace.contracts
    context = CaseContext(
        case_id=case["case_id"],
        raw_case=case,
        gateway=gateway,
        trace=trace,
        contracts=contracts,
    )

    # 1. Coordinator: Entity Resolution & Specialist Dispatch
    coordinator = CoordinatorAgent()
    await coordinator.run(context)

    # 2. Specialist Agents (Order, Payment, Shipment)
    order_agent = OrderItemAgent()
    payment_agent = PaymentAgent()
    shipment_agent = ShipmentAgent()

    await order_agent.run(context)
    await payment_agent.run(context)
    await shipment_agent.run(context)

    # 3. Policy Agent: Conflict Resolution, Root Cause & Financial Decision
    policy_agent = PolicyAgent()
    await policy_agent.run(context)

    # 4. Verifier Agent: Invariant Enforcement & Final Validation
    verifier = VerifierAgent()
    return await verifier.verify_and_build_output(context)
