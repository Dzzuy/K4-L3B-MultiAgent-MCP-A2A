from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

MONEY_TOLERANCE = 0.05
ISSUE_PRIORITY = (
    "canceled_order_paid",
    "unavailable_order_paid",
    "refund_failed",
    "refund_pending",
    "payment_mismatch",
    "duplicate_charge",
    "late_delivery_seller",
    "late_delivery_logistics",
    "valid_split_payment",
)


def parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def money(value: Any) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def close(a: float, b: float) -> bool:
    return abs(a - b) <= MONEY_TOLERANCE


def _dedupe(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        seen.setdefault(json.dumps(row, sort_keys=True, default=str), row)
    return list(seen.values()), len(rows) - len(seen)


@dataclass(frozen=True)
class Window:
    start: datetime | None
    end: datetime | None

    def contains(self, value: Any) -> bool:
        moment = parse_ts(value)
        if moment is None:
            return False
        if self.start is not None and moment < self.start:
            return False
        return self.end is None or moment <= self.end


@dataclass
class OrderFacts:
    order_id: str
    status: str
    customer_id: str | None
    purchased_at: str | None
    carrier_at: str | None
    delivered_at: str | None
    estimated_at: str | None
    shipping_limit_at: str | None
    item_ids: list[str]
    seller_ids: list[str]
    total_brl: float
    conflicts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PaymentFacts:
    captured_total_brl: float
    refunded_total_brl: float
    duplicate: bool
    mismatch: bool
    split_valid: bool
    refund_pending: bool
    refund_failed: bool
    payment_references: list[str]
    conflicts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ShipmentFacts:
    verdict: str
    late_seller_ids: list[str]
    timeline_complete: bool
    shipment_ids: list[str]
    conflicts: list[dict[str, Any]] = field(default_factory=list)


def make_window(order: dict[str, Any], opened_at: str) -> Window:
    start = parse_ts(order.get("order_purchase_timestamp"))
    opened = parse_ts(opened_at)
    delivered = parse_ts(order.get("order_delivered_customer_date"))
    end = max(opened, delivered) if opened and delivered else opened or delivered
    return Window(start=start, end=end)


def extract_order(
    order: dict[str, Any], items: list[dict[str, Any]], opened_at: str
) -> OrderFacts:
    window = make_window(order, opened_at)
    order_id = str(order.get("order_id") or "")
    own = [item for item in items if item.get("order_id") in (None, order_id)]
    scoped = [
        item
        for item in own
        if not item.get("shipping_limit_date") or window.contains(item.get("shipping_limit_date"))
    ]
    unique, duplicates = _dedupe(scoped)
    conflicts: list[dict[str, Any]] = []
    if len(scoped) < len(own):
        conflicts.append(
            {
                "field": "order_items.shipping_limit_date",
                "sources": ["order", "item"],
                "selected_source": "order",
                "resolution_code": "EXCLUDED_OUTSIDE_ORDER_TIMELINE",
            }
        )
    if duplicates:
        conflicts.append(
            {
                "field": "order_items",
                "sources": ["item", "item_duplicate"],
                "selected_source": "item",
                "resolution_code": "DEDUPLICATED_IDENTICAL_ROWS",
            }
        )
    limits = [
        item.get("shipping_limit_date")
        for item in unique
        if parse_ts(item.get("shipping_limit_date")) is not None
    ]
    shipping_limit = (
        max(
            limits,
            key=lambda value: (
                parse_ts(value).timestamp() if parse_ts(value) else float("-inf")
            ),
        )
        if limits
        else None
    )
    return OrderFacts(
        order_id=order_id,
        status=str(order.get("order_status") or "unknown"),
        customer_id=order.get("customer_id"),
        purchased_at=order.get("order_purchase_timestamp"),
        carrier_at=order.get("order_delivered_carrier_date"),
        delivered_at=order.get("order_delivered_customer_date"),
        estimated_at=order.get("order_estimated_delivery_date"),
        shipping_limit_at=shipping_limit,
        item_ids=list(
            dict.fromkeys(
                str(item["order_item_id"])
                for item in unique
                if item.get("order_item_id")
            )
        ),
        seller_ids=list(
            dict.fromkeys(str(item["seller_id"]) for item in unique if item.get("seller_id"))
        ),
        total_brl=round(
            sum(money(item.get("price")) + money(item.get("freight_value")) for item in unique),
            2,
        ),
        conflicts=conflicts,
    )


def extract_payments(
    timeline: dict[str, Any],
    refund_timeline: dict[str, Any] | None,
    window: Window,
    order_total_brl: float,
) -> PaymentFacts:
    events = timeline.get("events") or []
    in_window = [event for event in events if window.contains(event.get("event_at"))]
    captures = [
        event
        for event in in_window
        if event.get("event_type") == "captured"
        and event.get("status") in (None, "confirmed", "succeeded")
    ]
    amounts = [money(event.get("amount_brl")) for event in captures]
    captured_total = round(sum(amounts), 2)
    split_valid = (
        len(amounts) >= 2
        and order_total_brl > 0
        and close(captured_total, order_total_brl)
    )
    duplicate = (
        not split_valid
        and any(count >= 2 for count in Counter(amounts).values())
        and order_total_brl > 0
        and captured_total > order_total_brl + MONEY_TOLERANCE
    )
    mismatch = any(
        event.get("event_type") == "reconciliation_mismatch"
        and event.get("status") not in ("resolved", "closed")
        for event in in_window
    )

    refund_events = (refund_timeline or {}).get("events") or []
    refund_in_window = [event for event in refund_events if window.contains(event.get("event_at"))]
    refund_in_window, _ = _dedupe(refund_in_window)
    refunded_statuses = {"completed", "succeeded", "refunded", "confirmed", "settled"}
    refunded_total = round(
        sum(
            money(event.get("amount_brl"))
            for event in refund_in_window
            if str(event.get("status") or "").lower() in refunded_statuses
            or str(event.get("event_type") or "").lower() in {"refunded", "refund_completed"}
        ),
        2,
    )
    statuses = {str(event.get("status") or "").lower() for event in refund_in_window}
    pending = bool(statuses & {"pending", "processing", "queued"})
    failed = bool(statuses & {"failed", "rejected", "error"})

    rows = timeline.get("payments") or []
    refs: list[str] = []
    for row in rows:
        for key in ("payment_reference", "payment_id", "transaction_id", "payment_sequential"):
            value = row.get(key)
            if value is not None:
                text = str(value)
                if text and text not in refs:
                    refs.append(text)
                break

    conflicts: list[dict[str, Any]] = []
    if len(in_window) < len(events):
        conflicts.append(
            {
                "field": "payment_events.event_at",
                "sources": ["order", "payment"],
                "selected_source": "order",
                "resolution_code": "EXCLUDED_OUTSIDE_ORDER_TIMELINE",
            }
        )
    return PaymentFacts(
        captured_total_brl=captured_total,
        refunded_total_brl=refunded_total,
        duplicate=duplicate,
        mismatch=mismatch,
        split_valid=split_valid,
        refund_pending=pending,
        refund_failed=failed,
        payment_references=refs[:20],
        conflicts=conflicts,
    )


def extract_shipment(summary: dict[str, Any], order: OrderFacts, window: Window) -> ShipmentFacts:
    events = summary.get("events") or []
    in_window = [event for event in events if window.contains(event.get("event_at"))]
    unique, _ = _dedupe(in_window)
    event_types = {str(event.get("event_type") or "").lower() for event in unique}
    statuses = {str(event.get("status") or "").lower() for event in unique}
    event_types.add(str(summary.get("event_type") or "").lower())
    statuses.add(str(summary.get("status") or "").lower())

    delivered = parse_ts(order.delivered_at)
    estimated = parse_ts(order.estimated_at)
    carrier = parse_ts(order.carrier_at)
    shipping_limit = parse_ts(order.shipping_limit_at)

    timeline_complete = estimated is not None and (
        delivered is not None or order.status in ("canceled", "unavailable")
    )
    if "lost" in event_types or "lost" in statuses:
        verdict = "lost"
    elif event_types & {"returned", "return_completed"} or "returned" in statuses:
        verdict = "returned"
    elif delivered and estimated and delivered > estimated:
        verdict = (
            "seller_delay"
            if carrier and shipping_limit and carrier > shipping_limit
            else "logistics_delay"
        )
    elif timeline_complete:
        verdict = "on_time"
    else:
        verdict = "insufficient_evidence"

    shipment_ids: list[str] = []
    for row in [summary, *unique]:
        if not isinstance(row, dict):
            continue
        for key in ("shipment_id", "tracking_id", "tracking_code"):
            value = row.get(key)
            if value is not None:
                text = str(value)
                if text and text not in shipment_ids:
                    shipment_ids.append(text)

    conflicts: list[dict[str, Any]] = []
    if len(in_window) < len(events):
        conflicts.append(
            {
                "field": "shipment_events.event_at",
                "sources": ["order", "shipment"],
                "selected_source": "order",
                "resolution_code": "EXCLUDED_OUTSIDE_ORDER_TIMELINE",
            }
        )
    late_sellers = order.seller_ids if verdict == "seller_delay" else []
    return ShipmentFacts(
        verdict=verdict,
        late_seller_ids=late_sellers[:20],
        timeline_complete=timeline_complete,
        shipment_ids=shipment_ids[:20],
        conflicts=conflicts,
    )


def candidate_issues(
    order: OrderFacts, payment: PaymentFacts | None, shipment: ShipmentFacts | None
) -> list[str]:
    supported: set[str] = set()
    if payment:
        if order.status == "canceled" and payment.captured_total_brl > 0:
            supported.add("canceled_order_paid")
        if order.status == "unavailable" and payment.captured_total_brl > 0:
            supported.add("unavailable_order_paid")
        if payment.refund_failed:
            supported.add("refund_failed")
        if payment.refund_pending:
            supported.add("refund_pending")
        if payment.mismatch:
            supported.add("payment_mismatch")
        if payment.duplicate:
            supported.add("duplicate_charge")
        if payment.split_valid:
            supported.add("valid_split_payment")
    if shipment:
        if shipment.verdict == "seller_delay":
            supported.add("late_delivery_seller")
        if shipment.verdict == "logistics_delay":
            supported.add("late_delivery_logistics")
    return [issue for issue in ISSUE_PRIORITY if issue in supported]
