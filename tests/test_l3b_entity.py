import asyncio

from student_agent.agents import entity_agent
from student_agent.state import CaseState, Evidence


def _case(candidates: list[str]) -> dict:
    return {
        "case_id": "L3B_TEST_001",
        "customer_request": {"claimed_order_id": "O_BAD", "claims": []},
        "candidate_order_ids": candidates,
        "customer_unique_id_hint": "CUSTOMER_1",
        "investigation_scope": {
            "include_customer_history": True,
            "include_product_context": True,
            "require_independent_verification": True,
        },
    }


def test_real_l3b_state_fields() -> None:
    state = CaseState.from_case(_case(["O_BAD", "O_REAL"]))
    assert state.claimed_order_id == "O_BAD"
    assert state.candidate_order_ids == ["O_BAD", "O_REAL"]
    assert state.customer_unique_id_hint == "CUSTOMER_1"
    assert (
        state.include_customer_history
        and state.include_product_context
        and state.require_independent_verification
    )


def test_history_rejects_existing_claimed_order(monkeypatch) -> None:
    async def fake_fetch(gateway, trace, state, actor, tool, **kwargs):
        if tool == "get_customer_history":
            return Evidence(
                tool,
                "customer",
                "ev_" + "a" * 20,
                {"customer_unique_id": "CUSTOMER_1", "related_order_ids": ["O_REAL"]},
                kwargs,
            )
        order_id = kwargs["order_id"]
        return Evidence(
            tool, "order", "ev_" + order_id[-1].lower() * 20, {"order_id": order_id}, kwargs
        )

    monkeypatch.setattr(entity_agent, "fetch", fake_fetch)
    state = CaseState.from_case(_case(["O_BAD", "O_REAL"]))
    result = asyncio.run(entity_agent.run(state, None, None))
    assert (
        result.ok and state.selected_order_id == "O_REAL" and "O_BAD" in state.rejected_candidates
    )


def test_multiple_valid_candidates_stay_ambiguous(monkeypatch) -> None:
    async def fake_fetch(gateway, trace, state, actor, tool, **kwargs):
        return (
            Evidence(
                tool,
                "order",
                "ev_" + kwargs["order_id"][-1] * 20,
                {"order_id": kwargs["order_id"]},
                kwargs,
            )
            if tool == "get_order"
            else None
        )

    monkeypatch.setattr(entity_agent, "fetch", fake_fetch)
    state = CaseState.from_case(_case(["O1", "O2"]))
    state.include_customer_history = False
    asyncio.run(entity_agent.run(state, None, None))
    assert state.entity_status == "ambiguous" and state.selected_order_id is None


def test_one_valid_candidate_resolves_without_history(monkeypatch) -> None:
    async def fake_fetch(gateway, trace, state, actor, tool, **kwargs):
        if tool == "get_customer_history":
            return None
        order_id = kwargs["order_id"]
        if order_id == "O_BAD":
            return None
        return Evidence(tool, "order", "ev_" + "r" * 20, {"order_id": order_id}, kwargs)

    monkeypatch.setattr(entity_agent, "fetch", fake_fetch)
    state = CaseState.from_case(_case(["O_BAD", "O_REAL"]))
    asyncio.run(entity_agent.run(state, None, None))
    assert state.entity_status == "resolved"
    assert state.selected_order_id == "O_REAL"
    assert state.resolved_order_ids == ["O_REAL"]
    assert 0 < state.entity_confidence < 0.97
