"""What a one-stage audit established: its integrity profile, and the audit record that cites what it is about.

The profile (`profile`) is core's `IntegrityProfile` of one level `unit`, the partition's proof blocks (`sizes = (population,)`),
drawn by the verifier's sampling design (`Subset`, `Bernoulli`, or a `Stratified` sampling design with one stratum per template), with the count
drawn and delta. It bounds faulty proof blocks and weights nothing: under `Subset(k)` a trace with m faulty proof blocks passes with
probability C(N - m, k) / C(N, k), and under the stratified sampling design with prod_s C(n_s - m_s, k_s) / C(n_s, k_s). Except with
probability delta, an accepted trace has at most `bound()` faulty proof blocks, and at most `stratum_bounds()[t]` in template
t: exact for `subset` and `stratified`, the linear relaxation for `bernoulli`. Applications read it through `consumers`.

The record states the profile three ways, in the forms its recorded runs hold: `profile`, the profile's JSON (a stratified
sampling design's in its v0 form: the sampling design's JSON with delta and the count drawn per template); `wrong_units`, its bounds; and
`integrity_profile`, `verity/integrity-profile/v1` (README "The audit record and the profile"): the partition, the registration it
counts against, the sampling design and which proof blocks were drawn and proved, and the evidence by digest. The v1 form's non-sampling
terms (ε) come from the soundness analysis, not from the run, so the run leaves them null.

The audit record carries the registration and its receipt, the draw, the proved proof blocks, the statement's digests and each
verifier's verdict. A verdict is `{"accepted": true | false | null, ...}`; null means that verifier did not run, which leaves
the record incomplete. The record is accepted, and the profiles are present, only when there is a verifier of record
(`of_record`) and every one of them ran and accepted, every other verdict that ran accepted, and the draw, the proved
proof blocks, the served draw, the sessions' roots and, for registered values, the roots their reads were opened against passed
their checks.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Any, Mapping, Sequence

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.partition import Level
from verity.core.protocols.profile import Bernoulli, IntegrityProfile, Stratified, Stratum, Subset

from .draw import check_draw, check_proved, law_of
from .registration import digest as registration_digest

FORMAT = "verity/one-stage/audit/v0"
PROFILE = "verity/integrity-profile/v1"
LEVEL = "unit"


def _sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


def profile(registration: Mapping[str, Any], draw: Mapping[str, Any], delta: float) -> IntegrityProfile:
    """The audit's integrity profile under any of the four sampling designs. The work sampling design is the stratified sampling design at its sizes (Lean
    `Law.work`), so its profile is the same `Stratified` one, counting the drawn proof blocks (a closure proves more, and only
    helps)."""
    if draw["law"] == "subset":
        law = Subset(draw["k"])
    elif draw["law"] == "bernoulli":
        law = Bernoulli(Fraction(draw["p"]))
    elif draw["law"] in ("stratified", "work"):
        law = core_law(law_of(draw), registration["population"])
    else:
        raise ValueError(f"no profile for a {draw['law']!r} draw")
    return IntegrityProfile(levels=(Level(LEVEL, registration["partition"]),), sizes=(registration["population"],),
                            draws={LEVEL: law}, delta=delta, programs=(registration["program"],),
                            drawn={LEVEL: len(draw["units"])})


def core_law(law: Mapping[str, Any], population: int) -> Stratified | None:
    """The sampling design as core's `Stratified`: the uniform k-subset is one stratum, and a stratified or work sampling design has one per
    template (named by its descriptor id). None for `bernoulli`."""
    if law.get("law") == "subset":
        return Stratified.uniform(population, law["k"])
    if law.get("law") in ("stratified", "work"):
        return Stratified(tuple(Stratum(s["template"], sum(hi - lo for lo, hi in s["units"]), s["k"]) for s in law["strata"]))
    return None


def wrong_units(registration: Mapping[str, Any], draw: Mapping[str, Any], delta: float) -> dict[str, Any]:
    """The profile's bounds on faulty proof blocks at delta, as the record states them: `bound`, and for `stratified` each
    template's own (`per_template`)."""
    p = profile(registration, draw, delta)
    out: dict[str, Any] = {"delta": delta, "of": registration["population"], "bound": p.bound(),
                           "rule": "except with probability delta, an accepted transcript has at most `bound` wrong units"}
    if draw["law"] in ("stratified", "work"):
        out["per_template"] = p.stratum_bounds()
        out["rule"] += ", and at most `per_template[t]` of them in template t"
    return out


def integrity_profile(registration: Mapping[str, Any], receipt: Mapping[str, Any], draw: Mapping[str, Any],
                      sessions: Sequence[str], verdict: str | list[str] | None) -> dict[str, Any]:
    """`verity/integrity-profile/v1`; `sessions` are the session records' SHA-512, `verdict` the verifier of record's (a list
    when the drawn proof blocks were proved in several sessions)."""
    return {"format": PROFILE, "protocol": registration["protocol"], "program": registration["program"],
            "query": registration["query"], "partition": registration["partition"], "population": registration["population"],
            "registration": registration_digest(registration), "scheme": registration["scheme"],
            "roots": [{"port": r["port"], "root": r["root"], "leaves": r["leaves"]} for r in registration["roots"]],
            "law": registration["law"], "drawn": list(draw["units"]),
            "inputs": {"anchored": list(registration["anchors"]), "linked": [],
                       **({"registered": [{k: e[k] for k in ("value", "root", "leaves", "domain")} for e in registration["registered"]]}
                          if "registered" in registration else {})},
            "budget_log2_t": None, "terms": None, "epsilon_log2": None,
            "evidence": {"receipt": _sha512(dict(receipt)), "draw": _sha512(dict(draw)), "sessions": list(sessions),
                         "verdict": verdict}}


def session_registered(verdicts: Mapping[str, Mapping[str, Any]],
                       registration: Mapping[str, Any] | None = None) -> dict[str, str | None] | None:
    """`audit_record`'s `session_registered` from the verdicts alone: the `{value: root}` that the verifiers of record report
    (the Lean verdict's `registered`, `{port: {"value", "root"}}`) on the sessions they accepted, a value two of them
    report under different roots mapped to None. None when no accepted verdict of record reports registered reads. With
    `registration`, only its own values (its roots' ports and its registered values)."""
    out: dict[str, str | None] | None = None
    for v in verdicts.values():
        if v.get("of_record") and v.get("accepted") is True and isinstance(v.get("registered"), Mapping):
            out = {} if out is None else out
            for e in v["registered"].values():
                out[e["value"]] = e["root"] if out.get(e["value"], e["root"]) == e["root"] else None
    if out is None or registration is None:
        return out
    named = {r["port"] for r in registration["roots"]} | {e["value"] for e in registration.get("registered", [])}
    return {k: r for k, r in out.items() if k in named}


def audit_record(registration: Mapping[str, Any], receipt: Mapping[str, Any], draw: Mapping[str, Any],
                 law: Mapping[str, Any], proved_units: list[int], instances: int, statement: Mapping[str, Any],
                 verdicts: Mapping[str, Mapping[str, Any]], delta: float, sessions: Sequence[str] = (),
                 served: Sequence[int] | None = None, session_roots: Mapping[str, str] | None = None,
                 session_registered: Mapping[str, str] | None = None) -> dict[str, Any]:
    """`draw` is the verifier's own draw. `served`: the proof blocks the sessions were served, when the draw was handed to them
    (each must serve exactly the draw). `session_roots`: `{registered port: root}` from the sessions' statements (each
    must be the registered root). `session_registered`: `{value: root}`, the registered roots the verifier of record
    opened the sessions' reads against (`session_registered(verdicts)`, never the registration's own); a registration with
    registered values needs it, each the registered root."""
    checks = {"draw": check_draw(draw, registration, law).to_json(),
              "proved": check_proved(draw, proved_units, instances).to_json()}
    if receipt.get("registration") != registration_digest(registration):
        checks["receipt"] = {"ok": False, "codes": ["receipt"], "detail": {"receipt": receipt.get("registration")}}
    if served is not None and sorted(served) != sorted(draw["units"]):
        checks["served"] = {"ok": False, "codes": ["served"], "detail": {"served": sorted(served)[:16], "drawn": list(draw["units"])[:16]}}
    if session_roots is not None:
        registered = {r["port"]: r["root"] for r in registration["roots"]}
        wrong = sorted(p for p, r in session_roots.items() if registered.get(p) != r)
        if not session_roots or wrong:
            checks["roots"] = {"ok": False, "codes": ["roots"], "detail": {"ports": wrong}}
    if "registered" in registration or session_registered is not None:
        want = {e["value"]: e["root"] for e in registration.get("registered", [])}
        if dict(session_registered or {}) != want:
            checks["registered"] = {"ok": False, "codes": ["registered"],
                                    "detail": {"registered": sorted(want), "opened": sorted(session_registered or {}),
                                               "wrong": sorted(v for v in want if (session_registered or {}).get(v) != want[v])}}
    ran = {k: v for k, v in verdicts.items() if v.get("accepted") is not None}
    of_record = [v for v in verdicts.values() if v.get("of_record")]
    accepted = (bool(of_record) and all(v.get("accepted") is True for v in of_record)
                and all(v["accepted"] is True for v in ran.values()) and all(c["ok"] for c in checks.values()))
    rec: dict[str, Any] = {
        "format": FORMAT, "registration": dict(registration), "registration_digest": registration_digest(registration),
        "receipt": dict(receipt), "draw": dict(draw), "law": law_of(draw), "proved_units": list(proved_units),
        "statement": dict(statement), "checks": checks, "verdicts": {k: dict(v) for k, v in verdicts.items()},
        "accepted": accepted, "complete": len(ran) == len(verdicts), "profile": None, "integrity_profile": None,
        "wrong_units": None}
    if accepted:
        p = profile(registration, draw, delta)
        if draw["law"] in ("stratified", "work"):
            drawn = {s["template"]: sum(1 for u in draw["units"] if any(lo <= u < hi for lo, hi in s["units"]))
                     for s in draw["strata"]}
            rec["profile"] = {**p.draws[LEVEL].to_json(), "delta": p.delta, "drawn": drawn}
        else:
            rec["profile"] = p.to_json()
        digests = [v.get("digest") for v in of_record]
        rec["integrity_profile"] = integrity_profile(registration, receipt, draw, sessions,
                                                     digests[0] if len(digests) == 1 else digests)
        rec["wrong_units"] = wrong_units(registration, draw, delta)
    return rec


__all__ = ["FORMAT", "LEVEL", "PROFILE", "audit_record", "core_law", "integrity_profile", "profile", "session_registered",
           "wrong_units"]
