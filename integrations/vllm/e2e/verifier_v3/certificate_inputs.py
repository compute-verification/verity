"""[A3 contract 2026-09-10] ``verify/record_v3.json`` (written by ``verifier_v3/reverify.py``) -> the inputs of A7's
``e2e.certificate_v2.build_certificate`` (``Epoch``, ``Budget``, ``Law``, ``OutputDomain``, ``Sampled``), so A9's runner feeds the
verifier's record directly.  The population is NOT here (A1's modules build it; A9 passes it).  Nothing is re-interpreted: a record whose
status is not ``VERIFIED`` is refused (no certificate for a refused run), ``B_ex`` must be an RU count with provenance, the epoch must
name the exact requests.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from e2e import certificate_v2 as cv2

SCHEMA = "veritor/verifier-record/v3"


class RecordRefused(ValueError):
    """The verifier's record cannot feed a certificate."""


def load(path: Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SCHEMA:
        raise RecordRefused(f"{path}: schema {doc.get('schema')!r} is not {SCHEMA}")
    return doc


def certificate_inputs(doc: dict[str, Any], *, require_verified: bool = True) -> dict[str, Any]:
    """``dict(epoch=, budget=, law=, output_domain=, sampled=)`` for ``cv2.build_certificate(population=..., **inputs)``."""
    if require_verified and doc.get("status") != "VERIFIED":
        raise RecordRefused(f"record status {doc.get('status')!r}: {doc.get('status_line')} -- no certificate is built for a refused run")
    e = doc.get("epoch") or {}
    if not e.get("identifier") or not e.get("requests"):
        raise RecordRefused("record epoch lacks an identifier or the exact request ids (R16)")
    if int(e.get("replay_units", -1)) != len(e["requests"]) and not e.get("ru_indices"):
        raise RecordRefused("record epoch: replay_units does not match the requests and no ru_indices are given")
    b = doc.get("B_ex")
    if not isinstance(b, dict) or b.get("unit") != "RU" or not b.get("provenance"):
        raise RecordRefused(f"record B_ex {b!r} is not {{value, unit: 'RU', provenance}} (a VU count or a bare number is refused)")
    law = doc["law"]
    epoch = cv2.Epoch(str(e["identifier"]), tuple(str(r) for r in e["requests"]), int(e["replay_units"]),
                      separately_analysed_audits=bool(e.get("separately_analysed_audits", False)))
    budget = cv2.Budget(int(b["value"]), "RU", str(b["provenance"]), per_request=False)
    lw = cv2.Law(Fraction(str(law["q"])), Fraction(str(law["s"])), Fraction(str(law["delta"])), str(law.get("provenance", "protocol_instance.json")))
    od = doc["output_domain"]
    output_domain = cv2.OutputDomain(int(od["W_declared_bits"]), str(od.get("provenance", "verifier record")), tokens=od.get("tokens"), vocabulary=od.get("vocabulary"))
    s = doc.get("sampled") or {}
    sampled = cv2.Sampled(len(s["J"]) if s.get("J") is not None else None, len(s["F"]) if s.get("F") is not None else None, s.get("Q"),
                          str(s.get("provenance", "verifier record")), tuple(str(x) for x in s.get("J_identities", ())), tuple(str(x) for x in s.get("F_identities", ())))
    return {"epoch": epoch, "budget": budget, "law": lw, "output_domain": output_domain, "sampled": sampled}


def completion_risk_inputs(doc: dict[str, Any]) -> dict[str, Any] | None:
    """The R18 companion as recorded by ``ru_fields`` (per retained RU ``n`` from the actual configuration, the sampled unsupported
    identities) as keyword arguments of ``cv2.completion_risk(s, per_retained_RU, ...)``; ``None`` when any ``n`` is unknown (the
    runner reports UNAVAILABLE rather than an estimate)."""
    cr = doc.get("completion_risk") or {}
    per = cr.get("per_retained_ru") or {}
    if not per or any(v.get("n") is None for v in per.values()):
        return None
    return {"s": Fraction(str(cr["s"])), "per_retained_RU": {str(k): int(v["n"]) for k, v in per.items()}, "aggregate_N": cr.get("N_distinct_unsupported"),
            "sampled_unsupported": [str(x) for x in (doc.get("sampled_unsupported") or [])],
            "provenance": "verify/ru_fields.json: population kind counts x the configured unsupported-kind set (actual configuration)"}
