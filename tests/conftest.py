from __future__ import annotations

from typing import Any


class MockEvidenceGateway:
    """Mock Gateway simulating MCP tools and evidence envelopes."""

    def __init__(
        self,
        tools: list[str] | None = None,
        tool_responses: dict[str, Any] | None = None,
        fail_count_before_success: dict[str, int] | None = None,
    ):
        self._tools = tools or [
            "get_order",
            "get_order_details",
            "get_order_items",
            "get_order_payments",
            "get_payment_timeline",
            "get_policy",
            "get_product_context",
            "get_refund_timeline",
            "get_sellers",
            "get_shipment_summary",
            "get_shipment_details",
            "get_customer_history",
        ]
        self._responses = tool_responses or {}
        self._fails = fail_count_before_success or {}
        self.call_count = 0
        self.call_log: list[dict[str, Any]] = []

    async def list_tools(self) -> list[str]:
        return sorted(self._tools)

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        self.call_count += 1
        self.call_log.append({"tool": tool_name, "case_id": case_id, "args": arguments})

        if self._fails.get(tool_name, 0) > 0:
            self._fails[tool_name] -= 1
            raise RuntimeError(f"Transient network error calling {tool_name}")

        custom_data = self._responses.get(tool_name)
        if custom_data is None:
            # Fallback alias matching
            if tool_name == "get_shipment_summary":
                custom_data = self._responses.get("get_shipment_details", {})
            elif tool_name == "get_shipment_details":
                custom_data = self._responses.get("get_shipment_summary", {})
            elif tool_name == "get_refund_timeline":
                custom_data = self._responses.get("get_refund_status", {})
            elif tool_name == "get_refund_status":
                custom_data = self._responses.get("get_refund_timeline", {})
            else:
                custom_data = {}

        domain = "order"
        if "payment" in tool_name:
            domain = "payment"
        elif "refund" in tool_name:
            domain = "refund"
        elif "ship" in tool_name:
            domain = "shipment"
        elif "item" in tool_name:
            domain = "item"
        elif "customer" in tool_name:
            domain = "customer"
        elif "policy" in tool_name:
            domain = "policy"

        return {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": f"ev_{tool_name[:12]}_{case_id[:8]}_{self.call_count:04d}00000000",
            "result_hash": f"sha256:{'f' * 64}",
            "domain": domain,
            "data": custom_data,
        }
