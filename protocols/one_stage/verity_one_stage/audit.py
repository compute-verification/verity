"""What a one-stage audit established: the `IntegrityProfile`, and the audit record that cites what the profile is about.

The profile has one level, `unit` (the partition's units, `sizes = (population,)`), drawn by the verifier's law
(`Subset(k)` or `Bernoulli(p)`), with the count drawn and delta. It counts wrong units: under `Subset(k)` a transcript with
m wrong units passes with probability C(N - m, k) / C(N, k), and except with probability delta an accepted transcript has at
most `wrong_units_bound` wrong units (`worst_case` with utility m, the linear relaxation).

The audit record carries what the profile can't: the registration and its digest, the draw (its law and which units were
proved), the statement's digests, and each verifier's verdict. A verdict is `{"accepted": true | false | null, ...}`;
null means that verifier did not run, which leaves the record incomplete. The profile is present only when every verdict
that ran accepted and the draw and the proved units passed their checks.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping

from verity.ir.partition import Level
from verity.proofs.profile import Bernoulli, IntegrityProfile, Subset

from .draw import check_draw, check_proved, law_of
from .registration import digest as registration_digest

FORMAT = "verity/one-stage/audit/v0"
LEVEL = "unit"


def profile(registration: Mapping[str, Any], draw: Mapping[str, Any], delta: float) -> IntegrityProfile:
    law = Subset(draw["k"]) if draw["law"] == "subset" else Bernoulli(Fraction(draw["p"]))
    return IntegrityProfile(levels=(Level(LEVEL, registration["partition_digest"]),), sizes=(registration["population"],),
                            draws={LEVEL: law}, delta=delta, programs=(registration["partition"]["program"],),
                            drawn={LEVEL: len(draw["units"])})


def wrong_units_bound(p: IntegrityProfile) -> float:
    return p.worst_case(lambda m: float(m))


def audit_record(registration: Mapping[str, Any], draw: Mapping[str, Any], law: Mapping[str, Any], proved_units: list[int],
                 instances: int, statement: Mapping[str, Any], verdicts: Mapping[str, Mapping[str, Any]],
                 delta: float) -> dict[str, Any]:
    checks = {"draw": check_draw(draw, registration, law).to_json(),
              "proved": check_proved(draw, proved_units, instances).to_json()}
    ran = {k: v for k, v in verdicts.items() if v.get("accepted") is not None}
    accepted = bool(ran) and all(v["accepted"] is True for v in ran.values()) and all(c["ok"] for c in checks.values())
    rec: dict[str, Any] = {
        "format": FORMAT, "registration": dict(registration), "registration_digest": registration_digest(registration),
        "draw": dict(draw), "law": law_of(draw), "proved_units": list(proved_units), "statement": dict(statement),
        "checks": checks, "verdicts": {k: dict(v) for k, v in verdicts.items()},
        "accepted": accepted, "complete": len(ran) == len(verdicts), "profile": None, "wrong_units": None}
    if accepted:
        p = profile(registration, draw, delta)
        bound = wrong_units_bound(p)
        rec["profile"] = p.to_json()
        rec["wrong_units"] = {"delta": delta, "bound": bound, "of": registration["population"],
                              "rule": "except with probability delta, an accepted transcript has at most `bound` wrong units"}
    return rec


__all__ = ["FORMAT", "LEVEL", "audit_record", "profile", "wrong_units_bound"]
