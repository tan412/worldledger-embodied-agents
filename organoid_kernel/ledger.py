"""Claim Ledger(阶段 4):逐 claim 六态裁决 + 修复分级,取代全局 pass/fail。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ACCEPTED = "accepted"
REJECTED = "rejected"
INCONCLUSIVE = "inconclusive"
NOT_EVALUATED = "not_evaluated"
NOT_APPLICABLE = "not_applicable"
ERROR = "error"
STATES = {ACCEPTED, REJECTED, INCONCLUSIVE, NOT_EVALUATED, NOT_APPLICABLE, ERROR}


@dataclass
class Claim:
    name: str
    status: str
    reason: str = ""
    receipt: str = ""            # 该 claim 的 receipt 文件名
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.status in STATES, self.status


@dataclass
class Ledger:
    episode_id: str
    policy: str
    claims: list = field(default_factory=list)
    grade: str = ""              # accepted / accepted_repaired / accepted_with_warnings /
                                 # rejected / not_evaluable / not_applicable
    grade_reason: str = ""
    repairs: dict = field(default_factory=dict)
    warnings: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)

    def add(self, claim: Claim) -> None:
        self.claims.append(claim)

    def status_of(self, name: str) -> str:
        for c in self.claims:
            if c.name == name:
                return c.status
        return NOT_EVALUATED

    def to_json(self) -> dict:
        return {
            "schema": "organoid-kernel.claim-ledger.v1",
            "episode_id": self.episode_id,
            "policy": self.policy,
            "identity": self.identity,
            "claims": [{"claim": c.name, "status": c.status,
                        "reason": c.reason or None, "receipt": c.receipt or None,
                        **({"detail": c.detail} if c.detail else {})}
                       for c in self.claims],
            "grade": self.grade,
            "grade_reason": self.grade_reason,
            "repairs": self.repairs or None,
            "warnings": self.warnings or None,
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1),
                              encoding="utf-8")
