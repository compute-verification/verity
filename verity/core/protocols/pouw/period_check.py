"""A PoUW period as a core protocol, ``verity/pouw/period/v1``: the Python the Lean `PouwCheck.Period` (``pouw-period``,
PoUW's verifier of record for a period and onsite's work component) is held to by ``lean/check/tests/test_period_difftest.py``.

Its parameters are the verifier's own, ``{key, main, name}``: the challenger key it pins, its copy of main's
first-parent history and the period it credits. Its commitment is the auditor's ledger (the lines the challenger signed),
its digest SHA-512(`COMMITMENT_TAG` ‖ 0 ‖ its bytes). Its evidence is ``{pins, windows}``: the challenger host's pins log
and each window's evidence in the close's order (`credit.PROVED_KEYS`). The check is `credit._period_outcomes`, all or
nothing: it accepts with the period's credit (``{windows, work, k, delta, eps}``) and the profile ``{period, windows}``,
or rejects with the first refusal, ``code: why``. Each drawn unit's and each round's proof outcome is the verifier of
record's verdict, which the evidence carries."""
from __future__ import annotations

import hashlib
from typing import Any

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as Cr

ID = "verity/pouw/period/v1"
COMMITMENT_TAG = "verity/pouw/period/ledger/v1"
PARAMS_KEYS = ("key", "main", "name")
EVIDENCE_KEYS = ("pins", "windows")


def params(j: Any) -> dict:
    if not isinstance(j, dict) or sorted(j) != list(PARAMS_KEYS):
        raise CC.Refused(f"keys are not {', '.join(PARAMS_KEYS)}")
    if not isinstance(j["name"], str):
        raise CC.Refused("name is not a string")
    return j


def commitment(b: bytes) -> dict:
    ledger = CC.canonical(b)
    if not isinstance(ledger, list):
        raise CC.Refused("the auditor's ledger is not a list of its lines")
    return {"ledger": ledger, "bytes": b}


def digest(c: dict) -> bytes:
    return hashlib.sha512(COMMITMENT_TAG.encode() + b"\0" + c["bytes"]).digest()


def evidence(b: bytes) -> dict:
    j = CC.canonical(b)
    if not isinstance(j, dict) or sorted(j) != list(EVIDENCE_KEYS):
        raise CC.Refused(f"keys are not {', '.join(EVIDENCE_KEYS)}")
    return j


def check(p: dict, c: dict, e: dict) -> tuple[str, str | None, tuple[Any, Any] | None]:
    """`PouwCheck.Period.check`: the period's credit and profile, or the first refusal."""
    got, refused = Cr._period_outcomes(p["key"], c["ledger"], e["pins"], p["main"], p["name"], e["windows"],
                                       CC.lms_verify)
    if refused is not None:
        return "reject", f"{refused[0]}: {refused[1]}", None
    return "accept", None, (got, {"period": p["name"], "windows": len(got["windows"])})


def answer(params_b: bytes, commitment_b: bytes, evidence_b: bytes) -> dict:
    """`Core.Protocol.answer`: every input gets a verdict, a decoding failure being a rejection naming its input."""
    def out(verdict: str, reason: str | None, d: bytes | None, pub: Any = None, prof: Any = None) -> dict:
        return {"protocol": ID, "verdict": verdict, "reason": reason, "commitment": None if d is None else d.hex(),
                "public": pub, "profile": prof}
    try:
        p = params(CC.canonical(params_b))
    except CC.Refused as e:
        return out("reject", f"params: {e}", None)
    try:
        c = commitment(commitment_b)
    except CC.Refused as e:
        return out("reject", f"commitment: {e}", None)
    d = digest(c)
    try:
        ev = evidence(evidence_b)
    except CC.Refused as e:
        return out("reject", f"evidence: {e}", d)
    verdict, reason, acc = check(p, c, ev)
    return out(verdict, reason, d, *(acc or (None, None)))


def case(j: Any) -> dict:
    """`Core.Protocol.case` on one `serve` case, ``{"commitment": hex, "evidence": hex, "params": JSON}``: a malformed one
    is rejected with no commitment."""
    try:
        params_j = CC._get(j, "params", "the case")
        c = CC._unhex(CC._str_at(j, "commitment", "the case"))
        e = CC._unhex(CC._str_at(j, "evidence", "the case"))
        params_b = CC._canon(params_j).encode()
    except CC.Refused as err:
        return {"protocol": ID, "verdict": "reject", "reason": f"case: {err}", "commitment": None, "public": None,
                "profile": None}
    return answer(params_b, bytes.fromhex(c), bytes.fromhex(e))


__all__ = ["COMMITMENT_TAG", "EVIDENCE_KEYS", "ID", "PARAMS_KEYS", "answer", "case", "check", "commitment", "digest",
           "evidence", "params"]
