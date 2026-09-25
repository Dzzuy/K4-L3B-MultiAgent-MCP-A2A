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
    arguments: dict[str, Any] = field(default_factory=dict)


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


@dataclass
class CaseState:
    case: dict[str, Any]
    case_id: str
    opened_at: str
    policy_version: str

    claimed_order_id: str | None
    candidate_order_ids: list[str]

    customer_unique_id_hint: str | None
    customer_unique_id: str | None

    include_customer_history: bool
    include_product_context: bool
    require_independent_verification: bool

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
        scope = case.get("investigation_scope") or {}

        claims = tuple(
            Claim(
                claim_id=str(item["claim_id"]),
                topic=str(item["topic"]),
            )
            for item in req.get("claims") or []
            if isinstance(item, dict)
            and item.get("claim_id")
            and item.get("topic")
        )

        claimed_raw = req.get("claimed_order_id")
        claimed = str(claimed_raw) if claimed_raw else None

        candidates: list[str] = []
        for value in case.get("candidate_order_ids") or []:
            if isinstance(value, str) and value and value not in candidates:
                candidates.append(value)

        # Keep private-set compatibility if a claimed order is provided but is
        # accidentally omitted from candidate_order_ids.
        if claimed and claimed not in candidates:
            candidates.insert(0, claimed)

        hint_raw = case.get("customer_unique_id_hint")
        customer_hint = str(hint_raw) if hint_raw else None

        return cls(
            case=case,
            case_id=str(case["case_id"]),
            opened_at=str(case.get("opened_at") or ""),
            policy_version=str(case.get("policy_version") or "EC_POLICY_V1"),
            claimed_order_id=claimed,
            candidate_order_ids=candidates,
            customer_unique_id_hint=customer_hint,
            customer_unique_id=None,
            include_customer_history=bool(
                scope.get("include_customer_history", False)
            ),
            include_product_context=bool(
                scope.get("include_product_context", False)
            ),
            require_independent_verification=bool(
                scope.get("require_independent_verification", False)
            ),
            claims=claims,
        )

    def cache_key(self, tool: str, arguments: dict[str, Any]) -> str:
        return f"{tool}:{json.dumps(arguments, sort_keys=True, default=str)}"

    def register_evidence(self, evidence: Evidence) -> None:
        self.evidence_by_ref[evidence.ref] = evidence

    @property
    def claimed_issue(self) -> str | None:
        return next(
            (
                claim.topic
                for claim in self.claims
                if claim.topic != "requested_full_refund"
            ),
            None,
        )
