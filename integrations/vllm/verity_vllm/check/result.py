"""One result for every per-run check, and the four outcomes a verdict folds.

A check returns a `CheckResult`: its `name`, its `code` and the evidence the code rests on.  The code is core's
`VerificationCode`: `ACCEPTED` when the check holds, a rejection code saying why it does not, `None` when it did not run.
A verdict reads codes, never the prose in the evidence.

The verdict documents spell the code as one of four outcomes: `ACCEPTED` is PASS, `None` is NOT_RUN, a missing input (the
record lacks a part the check reads, `MISSING_INPUT`) is INSUFFICIENT_EVIDENCE, and any other rejection is FAIL.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable

from verity.verification.codes import VerificationCode

PASS, FAIL, INSUFFICIENT, NOT_RUN = "PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "NOT_RUN"
OUTCOMES = (PASS, FAIL, INSUFFICIENT, NOT_RUN)
SEVERITY = {FAIL: 3, INSUFFICIENT: 2, PASS: 1, NOT_RUN: 0}
MISSING_INPUT = VerificationCode.MALFORMED_TRANSCRIPT
EVIDENCE = ("claim", "inputs", "scope", "differences", "compat", "group", "rule", "notes", "reason")


def worst(outcomes: Iterable[str]) -> str:
    """FAIL > INSUFFICIENT_EVIDENCE > PASS > NOT_RUN; nothing is NOT_RUN."""
    return max(outcomes, key=lambda o: SEVERITY[o], default=NOT_RUN)


def outcome_of(code: VerificationCode | None) -> str:
    if code is None:
        return NOT_RUN
    if code is VerificationCode.ACCEPTED:
        return PASS
    return INSUFFICIENT if code is MISSING_INPUT else FAIL


def code_of(outcome: str, reject: VerificationCode | None = None) -> VerificationCode | None:
    """The code of `outcome` for a check whose findings against its claim are `reject` (read only on FAIL)."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome {outcome!r} not in {OUTCOMES}")
    if outcome != FAIL:
        return {PASS: VerificationCode.ACCEPTED, NOT_RUN: None, INSUFFICIENT: MISSING_INPUT}[outcome]
    if reject is None or reject in (VerificationCode.ACCEPTED, MISSING_INPUT):
        raise ValueError(f"FAIL needs a finding against the claim, not {reject!r}")
    return reject


@dataclass
class CheckResult:
    """`name` is `<group>` or `<group>/<rule>`; the fields after `code` are the evidence: `claim` (what ACCEPTED establishes),
    `inputs` (digests and ids of what the check read), `scope` (what it established, never more than the record shows),
    `differences` (named, machine-readable findings), `compat` (the record fields it was read from), `rule` (None on a group),
    and `notes` / `reason` (prose)."""
    name: str
    code: VerificationCode | None = None
    claim: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    differences: list[dict[str, Any]] = field(default_factory=list)
    compat: list[str] = field(default_factory=list)
    group: str | None = None
    rule: str | None = None
    notes: list[str] = field(default_factory=list)
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.group is None:
            self.group = self.name.split("/", 1)[0]

    @property
    def outcome(self) -> str:
        return outcome_of(self.code)

    @property
    def accepted(self) -> bool:
        return self.code is VerificationCode.ACCEPTED

    @property
    def evidence(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in EVIDENCE}

    def to_dict(self) -> dict[str, Any]:
        """The result as an entry of a verdict document: the code spelled as its outcome, after `scope`."""
        ev = copy.deepcopy(self.evidence)
        return {"name": self.name, **{k: ev[k] for k in EVIDENCE[:3]}, "outcome": self.outcome, **{k: ev[k] for k in EVIDENCE[3:]}}

    @classmethod
    def from_dict(cls, d: dict[str, Any], reject: VerificationCode | None = None) -> "CheckResult":
        """An entry of a verdict document; `reject` is the code its FAIL stands for."""
        kw = {k: d[k] for k in EVIDENCE if k in d}
        return cls(name=d["name"], code=code_of(d.get("outcome", NOT_RUN), reject), **kw)
