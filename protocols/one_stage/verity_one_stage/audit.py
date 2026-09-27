"""What a one-stage audit established: the integrity profile, and the audit record that cites what it is about.

Two forms of the profile, from the same facts:
* the core `IntegrityProfile`: one level `unit` (the partition's units, `sizes = (population,)`), drawn by the verifier's
  law, with the count drawn and delta. It counts wrong units: under `Subset(k)` a transcript with m wrong units passes with
  probability C(N - m, k) / C(N, k), and except with probability delta an accepted transcript has at most
  `wrong_units_bound` wrong units (`worst_case` with utility m, the linear relaxation);
* `verity/integrity-profile/v1` (`docs/audit-protocols.md` §0.2): the partition, the registration it counts against, the
  law and which units were drawn and proved, and the evidence by digest. Its non-sampling terms (ε) come from the
  soundness analysis, not from the run, so the run leaves them null.

The audit record carries the registration and its receipt, the draw, the proved units, the statement's digests and each
verifier's verdict. A verdict is `{"accepted": true | false | null, ...}`; null means that verifier did not run, which leaves
the record incomplete. The profiles are present only when every verdict that ran accepted and the draw and the proved
units passed their checks.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Any, Mapping, Sequence

from verity.ir.codec import canonical_json
from verity.ir.partition import Level
from verity.proofs.profile import Bernoulli, IntegrityProfile, Subset

from .draw import check_draw, check_proved, law_of
from .registration import digest as registration_digest

FORMAT = "verity/one-stage/audit/v0"
PROFILE = "verity/integrity-profile/v1"
LEVEL = "unit"


def _sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


def profile(registration: Mapping[str, Any], draw: Mapping[str, Any], delta: float) -> IntegrityProfile:
    law = Subset(draw["k"]) if draw["law"] == "subset" else Bernoulli(Fraction(draw["p"]))
    return IntegrityProfile(levels=(Level(LEVEL, registration["partition"]),), sizes=(registration["population"],),
                            draws={LEVEL: law}, delta=delta, programs=(registration["program"],),
                            drawn={LEVEL: len(draw["units"])})


def wrong_units_bound(p: IntegrityProfile) -> float:
    return p.worst_case(lambda m: float(m))


def integrity_profile(registration: Mapping[str, Any], receipt: Mapping[str, Any], draw: Mapping[str, Any],
                      sessions: Sequence[str], verdict: str | None) -> dict[str, Any]:
    """`verity/integrity-profile/v1`; `sessions` are the session records' SHA-512, `verdict` the verifier of record's."""
    return {"format": PROFILE, "protocol": registration["protocol"], "program": registration["program"],
            "query": registration["query"], "partition": registration["partition"], "population": registration["population"],
            "registration": registration_digest(registration), "scheme": registration["scheme"],
            "roots": [{"port": r["port"], "root": r["root"], "leaves": r["leaves"]} for r in registration["roots"]],
            "law": registration["law"], "drawn": list(draw["units"]),
            "inputs": {"anchored": list(registration["anchors"]), "linked": []},
            "budget_log2_t": None, "terms": None, "epsilon_log2": None,
            "evidence": {"receipt": _sha512(dict(receipt)), "draw": _sha512(dict(draw)), "sessions": list(sessions),
                         "verdict": verdict}}


def audit_record(registration: Mapping[str, Any], receipt: Mapping[str, Any], draw: Mapping[str, Any],
                 law: Mapping[str, Any], proved_units: list[int], instances: int, statement: Mapping[str, Any],
                 verdicts: Mapping[str, Mapping[str, Any]], delta: float, sessions: Sequence[str] = ()) -> dict[str, Any]:
    checks = {"draw": check_draw(draw, registration, law).to_json(),
              "proved": check_proved(draw, proved_units, instances).to_json()}
    if receipt.get("registration") != registration_digest(registration):
        checks["receipt"] = {"ok": False, "codes": ["receipt"], "detail": {"receipt": receipt.get("registration")}}
    ran = {k: v for k, v in verdicts.items() if v.get("accepted") is not None}
    accepted = bool(ran) and all(v["accepted"] is True for v in ran.values()) and all(c["ok"] for c in checks.values())
    rec: dict[str, Any] = {
        "format": FORMAT, "registration": dict(registration), "registration_digest": registration_digest(registration),
        "receipt": dict(receipt), "draw": dict(draw), "law": law_of(draw), "proved_units": list(proved_units),
        "statement": dict(statement), "checks": checks, "verdicts": {k: dict(v) for k, v in verdicts.items()},
        "accepted": accepted, "complete": len(ran) == len(verdicts), "profile": None, "integrity_profile": None,
        "wrong_units": None}
    if accepted:
        p = profile(registration, draw, delta)
        rec["profile"] = p.to_json()
        rec["integrity_profile"] = integrity_profile(registration, receipt, draw, sessions,
                                                     next((v.get("digest") for v in ran.values() if v.get("of_record")), None))
        rec["wrong_units"] = {"delta": delta, "bound": wrong_units_bound(p), "of": registration["population"],
                              "rule": "except with probability delta, an accepted transcript has at most `bound` wrong units"}
    return rec


__all__ = ["FORMAT", "LEVEL", "PROFILE", "audit_record", "integrity_profile", "profile", "wrong_units_bound"]
