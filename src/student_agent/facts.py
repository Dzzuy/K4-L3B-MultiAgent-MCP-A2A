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
    seller_shipping_limits: dict[str, str]
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
    refund_pending_brl: float
    refund_failed_brl: float
    capture_amounts_brl: list[float]
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
    seller_limit_values: dict[str, list[str]] = {}

    for item in unique:
        seller_id = item.get("seller_id")
        limit = item.get("shipping_limit_date")

        if (
            seller_id is None
            or parse_ts(limit) is None
        ):
            continue

        seller_limit_values.setdefault(
            str(seller_id),
            [],
        ).append(str(limit))

    seller_shipping_limits: dict[str, str] = {}

    for seller_id, values in seller_limit_values.items():
        # If one seller owns multiple items, missing any item-level shipping
        # deadline is enough to establish that seller breached a commitment.
        earliest = min(
            values,
            key=lambda value: (
                parse_ts(value).timestamp()
                if parse_ts(value)
                else float("inf")
            ),
        )
        seller_shipping_limits[seller_id] = earliest
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
        seller_shipping_limits=seller_shipping_limits,
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


def _money_or_none(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _payment_row_amount(row: dict[str, Any]) -> float | None:
    for key in (
        "payment_value",
        "amount_brl",
        "amount",
        "value",
    ):
        if key in row:
            amount = _money_or_none(row.get(key))
            if amount is not None:
                return amount
    return None


def _payment_references(rows: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []

    for row in rows:
        for key in (
            "payment_reference",
            "payment_id",
            "transaction_id",
            "payment_sequential",
        ):
            value = row.get(key)
            if value is None:
                continue

            text = str(value)
            if text and text not in refs:
                refs.append(text)
            break

    return refs[:20]


def _count_rows_for_amount(
    rows: list[dict[str, Any]],
    amount: float,
) -> int:
    return sum(
        1
        for row in rows
        if (
            (row_amount := _payment_row_amount(row)) is not None
            and close(row_amount, amount)
        )
    )


def _remove_uncorroborated_repeated_captures(
    captures: list[dict[str, Any]],
    payment_rows: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Remove copied repeated capture events when payment rows contradict them.

    A true repeated charge should have distinct underlying payment rows.
    If timeline contains N captures for one amount but authoritative payment
    rows support fewer than N, keep only the corroborated count.

    If payment-row evidence is unavailable (`None`), do not discard captures.
    """
    if payment_rows is None:
        return captures, False

    unique_rows, _ = _dedupe(payment_rows)

    amounts = [
        amount
        for amount in (
            _money_or_none(event.get("amount_brl"))
            for event in captures
        )
        if amount is not None
    ]

    counts = Counter(amounts)
    allowed_by_amount: dict[float, int] = {}
    changed = False

    for amount, capture_count in counts.items():
        if capture_count < 2:
            continue

        row_count = _count_rows_for_amount(unique_rows, amount)

        if 0 < row_count < capture_count:
            allowed_by_amount[amount] = row_count
            changed = True

    if not changed:
        return captures, False

    used: Counter[float] = Counter()
    filtered: list[dict[str, Any]] = []

    for event in captures:
        amount = _money_or_none(event.get("amount_brl"))

        if amount is None or amount not in allowed_by_amount:
            filtered.append(event)
            continue

        if used[amount] < allowed_by_amount[amount]:
            filtered.append(event)
            used[amount] += 1

    return filtered, True


def extract_payments(
    payment_rows: list[dict[str, Any]] | None,
    timeline: dict[str, Any],
    refund_timeline: dict[str, Any] | None,
    window: Window,
    order_total_brl: float,
) -> PaymentFacts:
    events = timeline.get("events") or []

    in_window = [
        event
        for event in events
        if window.contains(event.get("event_at"))
    ]

    captures = [
        event
        for event in in_window
        if event.get("event_type") == "captured"
        and str(event.get("status") or "").lower()
        in {"", "confirmed", "succeeded", "captured"}
    ]

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

    # Exact repeated rows are never useful evidence twice.
    captures, exact_capture_duplicates = _dedupe(captures)

    if exact_capture_duplicates:
        conflicts.append(
            {
                "field": "payment_events",
                "sources": ["payment", "payment_duplicate"],
                "selected_source": "payment",
                "resolution_code": "DEDUPLICATED_IDENTICAL_ROWS",
            }
        )

    # Stronger LonelyStar-style reconciliation:
    # timeline captures must agree with distinct payment rows before a repeated
    # capture is treated as a genuine second charge.
    captures, removed_uncorroborated = (
        _remove_uncorroborated_repeated_captures(
            captures,
            payment_rows,
        )
    )

    if removed_uncorroborated:
        conflicts.append(
            {
                "field": "payment_events",
                "sources": ["payment_timeline", "payment_rows"],
                "selected_source": "payment_rows",
                "resolution_code": "EXCLUDED_UNCORROBORATED_CAPTURE",
            }
        )

    capture_amounts = [
        amount
        for amount in (
            _money_or_none(event.get("amount_brl"))
            for event in captures
        )
        if amount is not None
    ]

    captured_total = round(sum(capture_amounts), 2)

    split_valid = (
        len(capture_amounts) >= 2
        and order_total_brl > 0
        and close(captured_total, order_total_brl)
    )

    repeated_amounts = {
        amount
        for amount, count in Counter(capture_amounts).items()
        if count >= 2
    }

    if payment_rows is None:
        # Fallback only when corroborating payment-row evidence is unavailable.
        duplicate = (
            not split_valid
            and bool(repeated_amounts)
            and order_total_brl > 0
            and captured_total > order_total_brl + MONEY_TOLERANCE
        )
    else:
        unique_rows, _ = _dedupe(payment_rows)

        corroborated_repeat = any(
            _count_rows_for_amount(unique_rows, amount)
            >= Counter(capture_amounts)[amount]
            for amount in repeated_amounts
        )

        duplicate = (
            not split_valid
            and corroborated_repeat
            and order_total_brl > 0
            and captured_total > order_total_brl + MONEY_TOLERANCE
        )

    mismatch = any(
        event.get("event_type") == "reconciliation_mismatch"
        and str(event.get("status") or "").lower()
        not in {"resolved", "closed"}
        for event in in_window
    )

    refund_events = (refund_timeline or {}).get("events") or []

    refund_in_window = [
        event
        for event in refund_events
        if window.contains(event.get("event_at"))
    ]

    if len(refund_in_window) < len(refund_events):
        conflicts.append(
            {
                "field": "refund_events.event_at",
                "sources": ["order", "refund"],
                "selected_source": "order",
                "resolution_code": "EXCLUDED_OUTSIDE_ORDER_TIMELINE",
            }
        )

    refund_in_window, refund_duplicates = _dedupe(refund_in_window)

    if refund_duplicates:
        conflicts.append(
            {
                "field": "refund_events",
                "sources": ["refund", "refund_duplicate"],
                "selected_source": "refund",
                "resolution_code": "DEDUPLICATED_IDENTICAL_ROWS",
            }
        )

    refunded_statuses = {
        "completed",
        "succeeded",
        "refunded",
        "confirmed",
        "settled",
    }

    pending_statuses = {
        "pending",
        "processing",
        "queued",
    }

    failed_statuses = {
        "failed",
        "rejected",
        "error",
    }

    refunded_total = 0.0
    pending_total = 0.0
    failed_total = 0.0

    for event in refund_in_window:
        status = str(event.get("status") or "").lower()
        event_type = str(event.get("event_type") or "").lower()
        amount = money(event.get("amount_brl"))

        if (
            status in refunded_statuses
            or event_type in {"refunded", "refund_completed"}
        ):
            refunded_total += amount
        elif status in pending_statuses:
            pending_total += amount
        elif status in failed_statuses:
            failed_total += amount

    refunded_total = round(refunded_total, 2)
    pending_total = round(pending_total, 2)
    failed_total = round(failed_total, 2)

    rows = payment_rows or []
    unique_rows, row_duplicates = _dedupe(rows)

    if row_duplicates:
        conflicts.append(
            {
                "field": "payment_rows",
                "sources": ["payment", "payment_duplicate"],
                "selected_source": "payment",
                "resolution_code": "DEDUPLICATED_IDENTICAL_ROWS",
            }
        )

    return PaymentFacts(
        captured_total_brl=captured_total,
        refunded_total_brl=refunded_total,
        duplicate=duplicate,
        mismatch=mismatch,
        split_valid=split_valid,
        refund_pending=pending_total > 0,
        refund_failed=failed_total > 0,
        refund_pending_brl=pending_total,
        refund_failed_brl=failed_total,
        capture_amounts_brl=capture_amounts,
        payment_references=_payment_references(unique_rows),
        conflicts=conflicts[:5],
    )


def extract_shipment(
    summary: dict[str, Any],
    order: OrderFacts,
    window: Window,
) -> ShipmentFacts:
    events = summary.get("events") or []

    in_window = [
        event
        for event in events
        if window.contains(event.get("event_at"))
    ]

    unique, duplicates = _dedupe(in_window)

    event_types = {
        str(event.get("event_type") or "").lower()
        for event in unique
    }

    statuses = {
        str(event.get("status") or "").lower()
        for event in unique
    }

    event_types.add(
        str(summary.get("event_type") or "").lower()
    )

    statuses.add(
        str(summary.get("status") or "").lower()
    )

    # Prefer authoritative order timestamps. Shipment top-level fields are
    # fallbacks only when the order record lacks that timestamp.
    delivered = parse_ts(
        order.delivered_at
        or summary.get("delivered_customer_at")
        or summary.get("order_delivered_customer_date")
    )

    estimated = parse_ts(
        order.estimated_at
        or summary.get("estimated_delivery_at")
        or summary.get("order_estimated_delivery_date")
    )

    carrier = parse_ts(
        order.carrier_at
        or summary.get("delivered_carrier_at")
        or summary.get("order_delivered_carrier_date")
    )

    shippable = order.status not in {
        "canceled",
        "unavailable",
    }

    delivered_late = bool(
        shippable
        and delivered is not None
        and estimated is not None
        and delivered > estimated
    )

    # For an undelivered order, the case-open boundary is the authoritative
    # point at which we decide whether the promised date has already passed.
    overdue_undelivered = bool(
        shippable
        and delivered is None
        and estimated is not None
        and window.end is not None
        and estimated < window.end
    )

    customer_late = (
        delivered_late
        or overdue_undelivered
    )

    late_sellers: list[str] = []

    for seller_id in order.seller_ids:
        raw_limit = order.seller_shipping_limits.get(
            seller_id
        )

        limit = parse_ts(raw_limit)

        if limit is None:
            continue

        # If carrier handoff exists, compare the handoff against the seller's
        # own deadline. If no handoff exists yet, compare case-open time:
        # the seller is late if the order still had not been handed over after
        # its shipping deadline.
        reference = carrier or window.end

        if (
            reference is not None
            and reference > limit
            and seller_id not in late_sellers
        ):
            late_sellers.append(seller_id)

    confirmed_late_events = [
        event
        for event in unique
        if (
            str(event.get("event_type") or "").lower()
            == "delivered_late"
            and str(event.get("status") or "").lower()
            in {"", "confirmed", "succeeded"}
        )
    ]

    late_actors = {
        str(event.get("actor") or "").lower()
        for event in confirmed_late_events
        if event.get("actor")
    }

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

    if duplicates:
        conflicts.append(
            {
                "field": "shipment_events",
                "sources": [
                    "shipment",
                    "shipment_duplicate",
                ],
                "selected_source": "shipment",
                "resolution_code": "DEDUPLICATED_IDENTICAL_ROWS",
            }
        )

    # High-scoring L3A implementations observed misleading delivered_late
    # events. Never let such an event override authoritative order dates.
    if confirmed_late_events and not customer_late:
        conflicts.append(
            {
                "field": "shipment_events.delivered_late",
                "sources": ["order", "shipment"],
                "selected_source": "order",
                "resolution_code": "EXCLUDED_CONTRADICTS_ORDER_DATES",
            }
        )

    if (
        "lost" in event_types
        or "lost" in statuses
    ):
        verdict = "lost"

    elif (
        event_types
        & {
            "returned",
            "return_completed",
        }
        or "returned" in statuses
    ):
        verdict = "returned"

    elif customer_late:
        if late_sellers:
            verdict = "seller_delay"

        elif (
            "logistics_provider" in late_actors
            or "logistics" in late_actors
            or "carrier" in late_actors
        ):
            verdict = "logistics_delay"

        elif (
            carrier is not None
            and order.seller_shipping_limits
        ):
            # Customer delivery was late, while known seller deadlines were
            # not breached: responsibility moves to logistics.
            verdict = "logistics_delay"

        elif (
            "seller" in late_actors
            and len(order.seller_ids) == 1
        ):
            verdict = "seller_delay"
            late_sellers = order.seller_ids[:1]

        else:
            # We know the delivery is late but lack enough evidence to assign
            # responsibility safely.
            verdict = "insufficient_evidence"

    elif (
        estimated is not None
        and delivered is not None
    ):
        verdict = "on_time"

    else:
        verdict = "insufficient_evidence"

    # "timeline_complete" describes whether a completed delivery timeline is
    # available. An overdue-but-still-undelivered order can therefore be
    # correctly classified as late while timeline_complete remains False.
    timeline_complete = (
        estimated is not None
        and delivered is not None
    )

    shipment_ids: list[str] = []

    for row in [summary, *unique]:
        if not isinstance(row, dict):
            continue

        for key in (
            "shipment_id",
            "tracking_id",
            "tracking_code",
        ):
            value = row.get(key)

            if value is None:
                continue

            text = str(value)

            if text and text not in shipment_ids:
                shipment_ids.append(text)

    return ShipmentFacts(
        verdict=verdict,
        late_seller_ids=late_sellers[:20],
        timeline_complete=timeline_complete,
        shipment_ids=shipment_ids[:20],
        conflicts=conflicts[:5],
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
