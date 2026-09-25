from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Claim:
    claim_id: str
    topic: str


@dataclass(frozen=True)
class Evidence:
    tool: str
    domain: str
    ref: str
    data: Any


@dataclass
class AgentResult:
    actor: str
    ok: bool = False
    evidence: list[Evidence] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    findings: dict[str, Any] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    notes_code: str = "OK"

    @property
    def evidence_refs(self) -> list[str]:
        return [item.ref for item in self.evidence]


def _add_strings(target: list[str], value: Any) -> None:
    if isinstance(value, str) and value and value not in target:
        target.append(value)
    elif isinstance(value, list):
        for item in value:
            _add_strings(target, item)
    elif isinstance(value, dict):
        for key in ("order_id", "candidate_order_id", "id"):
            if key in value:
                _add_strings(target, value[key])


def extract_candidate_order_ids(case: dict[str, Any]) -> list[str]:
    req = case.get("customer_request") or {}
    found: list[str] = []
    _add_strings(found, req.get("claimed_order_id"))

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = key.lower()
                if key in {
                    "candidate_order_ids",
                    "order_candidates",
                    "candidate_orders",
                    "entity_candidates",
                    "candidates",
                } or ("candidate" in lowered and "order" in lowered):
                    _add_strings(found, child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(case)
    return found


def extract_customer_unique_id(case: dict[str, Any]) -> str | None:
    def visit(value: Any) -> str | None:
        if isinstance(value, dict):
            direct = value.get("customer_unique_id")
            if isinstance(direct, str) and direct:
                return direct
            for child in value.values():
                found = visit(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = visit(child)
                if found:
                    return found
        return None

    return visit(case)


@dataclass
class CaseState:
    case: dict[str, Any]
    case_id: str
    opened_at: str
    policy_version: str
    claimed_order_id: str | None
    candidate_order_ids: list[str]
    customer_unique_id: str | None
    claims: tuple[Claim, ...]

    selected_order_id: str | None = None
    entity_status: str = "not_found"
    entity_confidence: float = 0.0
    resolved_order_ids: list[str] = field(default_factory=list)
    rejected_candidates: list[str] = field(default_factory=list)
    related_order_ids: list[str] = field(default_factory=list)

    cache: dict[str, Evidence] = field(default_factory=dict)
    evidence_by_ref: dict[str, Evidence] = field(default_factory=dict)

    @classmethod
    def from_case(cls, case: dict[str, Any]) -> CaseState:
        req = case.get("customer_request") or {}
        claims = tuple(
            Claim(str(item["claim_id"]), str(item["topic"]))
            for item in req.get("claims") or []
            if isinstance(item, dict) and item.get("claim_id") and item.get("topic")
        )
        claimed = req.get("claimed_order_id")
        return cls(
            case=case,
            case_id=str(case["case_id"]),
            opened_at=str(case.get("opened_at") or ""),
            policy_version=str(case.get("policy_version") or "EC_POLICY_V1"),
            claimed_order_id=str(claimed) if claimed else None,
            candidate_order_ids=extract_candidate_order_ids(case),
            customer_unique_id=extract_customer_unique_id(case),
            claims=claims,
        )

    def cache_key(self, tool: str, arguments: dict[str, Any]) -> str:
        return f"{tool}:{json.dumps(arguments, sort_keys=True, default=str)}"

    def register_evidence(self, evidence: Evidence) -> None:
        self.evidence_by_ref[evidence.ref] = evidence

    @property
    def claimed_issue(self) -> str | None:
        return next(
            (claim.topic for claim in self.claims if claim.topic != "requested_full_refund"),
            None,
        )
