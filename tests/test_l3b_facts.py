from student_agent.facts import (
    extract_order,
    extract_payments,
    extract_shipment,
    make_window,
)


def _base_order():
    return {
        "order_id": "O1",
        "order_status": "delivered",
        "order_purchase_timestamp": "2018-01-01T00:00:00+00:00",
        "order_delivered_carrier_date": "2018-01-06T00:00:00+00:00",
        "order_delivered_customer_date": "2018-01-12T00:00:00+00:00",
        "order_estimated_delivery_date": "2018-01-10T00:00:00+00:00",
    }


def test_only_late_seller_is_reported():
    order = _base_order()

    items = [
        {
            "order_id": "O1",
            "order_item_id": "1",
            "seller_id": "S1",
            "price": 50,
            "freight_value": 5,
            "shipping_limit_date": "2018-01-04T00:00:00+00:00",
        },
        {
            "order_id": "O1",
            "order_item_id": "2",
            "seller_id": "S2",
            "price": 40,
            "freight_value": 5,
            "shipping_limit_date": "2018-01-08T00:00:00+00:00",
        },
    ]

    facts = extract_order(
        order,
        items,
        "2018-01-15T00:00:00+00:00",
    )

    shipment = extract_shipment(
        {"events": []},
        facts,
        make_window(
            order,
            "2018-01-15T00:00:00+00:00",
        ),
    )

    assert shipment.verdict == "seller_delay"
    assert shipment.late_seller_ids == ["S1"]


def test_undelivered_order_can_already_be_late():
    order = {
        "order_id": "O1",
        "order_status": "shipped",
        "order_purchase_timestamp": "2018-01-01T00:00:00+00:00",
        "order_delivered_carrier_date": "2018-01-03T00:00:00+00:00",
        "order_delivered_customer_date": None,
        "order_estimated_delivery_date": "2018-01-10T00:00:00+00:00",
    }

    items = [
        {
            "order_id": "O1",
            "order_item_id": "1",
            "seller_id": "S1",
            "price": 90,
            "freight_value": 10,
            "shipping_limit_date": "2018-01-04T00:00:00+00:00",
        }
    ]

    facts = extract_order(
        order,
        items,
        "2018-01-15T00:00:00+00:00",
    )

    shipment = extract_shipment(
        {"events": []},
        facts,
        make_window(
            order,
            "2018-01-15T00:00:00+00:00",
        ),
    )

    assert shipment.verdict == "logistics_delay"
    assert shipment.timeline_complete is False
    assert shipment.late_seller_ids == []


def test_uncorroborated_duplicate_capture_is_removed():
    order = _base_order()

    window = make_window(
        order,
        "2018-01-15T00:00:00+00:00",
    )

    payment_rows = [
        {
            "payment_sequential": 1,
            "payment_value": 100,
        }
    ]

    timeline = {
        "events": [
            {
                "event_type": "captured",
                "status": "confirmed",
                "amount_brl": 100,
                "event_at": "2018-01-02T00:00:00+00:00",
                "event_id": "A",
            },
            {
                "event_type": "captured",
                "status": "confirmed",
                "amount_brl": 100,
                "event_at": "2018-01-02T01:00:00+00:00",
                "event_id": "B",
            },
        ]
    }

    facts = extract_payments(
        payment_rows,
        timeline,
        None,
        window,
        100,
    )

    assert facts.captured_total_brl == 100
    assert facts.duplicate is False


def test_real_duplicate_requires_two_payment_rows():
    order = _base_order()

    window = make_window(
        order,
        "2018-01-15T00:00:00+00:00",
    )

    payment_rows = [
        {
            "payment_sequential": 1,
            "payment_value": 100,
        },
        {
            "payment_sequential": 2,
            "payment_value": 100,
        },
    ]

    timeline = {
        "events": [
            {
                "event_type": "captured",
                "status": "confirmed",
                "amount_brl": 100,
                "event_at": "2018-01-02T00:00:00+00:00",
                "event_id": "A",
            },
            {
                "event_type": "captured",
                "status": "confirmed",
                "amount_brl": 100,
                "event_at": "2018-01-02T01:00:00+00:00",
                "event_id": "B",
            },
        ]
    }

    facts = extract_payments(
        payment_rows,
        timeline,
        None,
        window,
        100,
    )

    assert facts.captured_total_brl == 200
    assert facts.duplicate is True


def test_true_split_payment_is_not_duplicate():
    order = _base_order()

    window = make_window(
        order,
        "2018-01-15T00:00:00+00:00",
    )

    payment_rows = [
        {
            "payment_sequential": 1,
            "payment_value": 40,
        },
        {
            "payment_sequential": 2,
            "payment_value": 60,
        },
    ]

    timeline = {
        "events": [
            {
                "event_type": "captured",
                "status": "confirmed",
                "amount_brl": 40,
                "event_at": "2018-01-02T00:00:00+00:00",
            },
            {
                "event_type": "captured",
                "status": "confirmed",
                "amount_brl": 60,
                "event_at": "2018-01-02T01:00:00+00:00",
            },
        ]
    }

    facts = extract_payments(
        payment_rows,
        timeline,
        None,
        window,
        100,
    )

    assert facts.split_valid is True
    assert facts.duplicate is False
