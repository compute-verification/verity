"""PoUB-P2's core case: a session over PoUS's P2 store (`p2_party.session`) in the core interface's form, which
`RemoteAudit.P2Protocol` (`remote-audit-p2-protocol`, in `lean/`) replays, and the store it names, which onsite holds
to PoUS's.

A case (`case_of`) is ``{"params", "commitment", "evidence", "store_root"}``, the commitment and evidence canonical JSON
in hex:

* ``params``: ``{"not_before_ns": n, "request": R}``, the verifier's clock floor and the prover's request R;
* ``commitment``: the auditor's registration file, its fields and its plan (`p2_party.registration_json`), whose digest
  (`digest`) is SHA-512 of `DIGEST_DOMAIN` and those bytes, so it binds the plan with the registration;
* ``evidence``: ``{"log": [...]}``, every request the verifier's Lean process answered of the session after it opened,
  in order (`p2_party.Twin` records them), each line's bytes as the prover sent them in hex:
  ``{"op": "p2_event", "event", "coin"}``, ``{"op": "p2_root", "t", "i", "now", "line"}``, ``{"op": "p2_openings", "t",
  "line"}`` and ``{"op": "p2_fail"}``;
* ``store_root``: the store the session audits, its registration's root, 128 lowercase hex characters. That root is the
  root of `C*` PoUS's setup published (`pous.run.Setup.root`), so `same_store` holds it to PoUS's run case. No digest
  binds the field itself: it counts only as the root the commitment registers (`store_root`).

The evidence holds `C*`'s opened blocks and the verifier's coins and clock: it is witness, and `write` puts a case only
in a private directory (0700, its files 0600). The case's answer (`answer`) is the core form of the session's own
answer (`P2Session.answer`), which the Lean's replay of the case is held to; `judged` is the Lean's, given its
executable. A session the Python's books vetoed, or one refused before its first coin, has no case.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.ml.commitments.identity import canonical_json_bytes

from . import p2_party

PROTOCOL = p2_party.PROTOCOL
EXE = "remote-audit-p2-protocol"
DIGEST_DOMAIN = b"verity/remote-bandwidth/p2/commitment/v0\0"
STORE_ROOT = re.compile(r"[0-9a-f]{128}")
CASE, ANSWER = "case.json", "answer.json"
#: what an accepted case's `eps` sums, as `RemoteAudit.P2Protocol.certify` does
EPS_TERMS = p2_party.EPS_TERMS


def case_of(registration: Mapping[str, Any], request: Mapping[str, Any], not_before_ns: int,
            log: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The case of a session over the store `registration` (`p2_party.registration_json`, with its plan): the
    prover's `request`, the verifier's clock floor and its process's `log` (module docstring)."""
    commitment = canonical_json_bytes(dict(registration))
    return {"commitment": commitment.hex(), "evidence": canonical_json_bytes({"log": [dict(x) for x in log]}).hex(),
            "params": {"not_before_ns": not_before_ns, "request": dict(request)}, "store_root": registration["root"]}


def digest(case: Mapping[str, Any]) -> bytes:
    """`P2Protocol.Commitment.digest`: SHA-512 of `DIGEST_DOMAIN` and the case's commitment bytes."""
    return hashlib.sha512(DIGEST_DOMAIN + bytes.fromhex(case["commitment"])).digest()


def store_root(case: Mapping[str, Any]) -> bytes:
    """The store PoUB-P2's case names (`store_root`): its registration's root, 128 lowercase hex characters, refused
    (ValueError) when the case names none, or one its commitment doesn't register. A generic PoUB case is refused: its
    commitment is no P2 registration, and its payload root is its own salted tree's, never PoUS's `C*`."""
    s = case.get("store_root")
    if type(s) is not str or not STORE_ROOT.fullmatch(s):
        raise ValueError("the case names no store: store_root is not 128 lowercase hex characters")
    try:
        registered = p2_party.registration_of(json.loads(bytes.fromhex(case["commitment"]))).root
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"the case's commitment is no PoUB-P2 registration: {type(e).__name__}: {e}") from None
    if registered.hex() != s:
        raise ValueError("the case's store_root is not the root its commitment registers")
    return registered


def same_store(case: Mapping[str, Any], pous_case: Mapping[str, Any]) -> str | None:
    """Onsite's one store: None when PoUB-P2's case `case` names (`store_root`) the store PoUS's run case `pous_case` is
    committed to, the root of `C*` its setup published (`pous.run.Setup.root`, in the run case's commitment), else why
    not."""
    from verity.core.protocols.pous.run import Setup

    try:
        ours = store_root(case)
    except ValueError as e:
        return str(e)
    try:
        theirs = Setup.from_json(json.loads(bytes.fromhex(pous_case["commitment"]))).root
    except (KeyError, TypeError, ValueError) as e:
        return f"PoUS's case names no setup: {type(e).__name__}: {e}"
    if ours != theirs:
        return f"bandwidth's store {ours.hex()[:16]}… is not PoUS's {theirs.hex()[:16]}…"
    return None


def answer(case: Mapping[str, Any], session: Mapping[str, Any]) -> dict[str, Any]:
    """The case's answer in the core interface's form from the session's own answer `session` (`p2_party.answer`'s,
    `P2Session.answer`), as `RemoteAudit.P2Protocol` answers the case: accepted, `{"eps", "q_bytes", "span_ns"}` and the
    rest of the answer as its profile; rejected with the failure, its window and its draw; or undecided."""
    out: dict[str, Any] = {"commitment": digest(case).hex(), "profile": None, "protocol": PROTOCOL, "public": None,
                           "reason": None}
    verdict = session.get("verdict")
    if verdict == "accept":
        draws = session["draws"]
        public = {"eps": str(Fraction(2**24 + sum(draws), 2**64)), "q_bytes": session["q_bytes"],
                  "span_ns": session["span_ns"]}
        profile = {"bound": session["bound"], "delta": session["delta"], "draws": draws,
                   "plan_units": session["plan_units"], "receipts": session["receipts"],
                   "registration": session["registration"], "unit_bytes": session["unit_bytes"],
                   "units": session["units"], "windows": len(session["windows"]), "wrong": session["wrong_at_most"]}
        return out | {"profile": profile, "public": public, "verdict": "accept"}
    if verdict == "reject":
        at = session.get("at")
        draw = f", draw [{at[0]}, {at[1]}]" if at is not None else ""
        return out | {"reason": f"the session was rejected: {session['failure']} at window {session['window']}{draw}",
                      "verdict": "reject"}
    return out | {"reason": "the session has a window not yet verified", "verdict": "undecided"}


def serve(exe: str | os.PathLike, cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`EXE serve` on `cases`, one canonical JSON line each, the served keys only: the Lean's answers."""
    lines = "".join(canonical_json_bytes({k: c[k] for k in ("commitment", "evidence", "params") if k in c}).decode()
                    + "\n" for c in cases)
    p = subprocess.run([str(exe), "serve"], input=lines, capture_output=True, text=True, check=True)
    return [json.loads(x) for x in p.stdout.splitlines()]


def judged(case: Mapping[str, Any], session: Mapping[str, Any], exe: str | os.PathLike | None = None) -> dict[str, Any]:
    """The case's answer: the Lean's (`serve`) with `exe`, else `answer`'s of the session's own."""
    return serve(exe, [case])[0] if exe is not None else answer(case, session)


def write(private: Path, case: Mapping[str, Any], answer_: Mapping[str, Any]) -> Path:
    """``case.json`` and ``answer.json`` into ``private`` (0700, each file 0600): the piece composite.py reads."""
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(private, 0o700)
    for name, doc in ((CASE, case), (ANSWER, answer_)):
        path = private / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(canonical_json_bytes(dict(doc)) + b"\n")
        os.chmod(path, 0o600)
    return private


__all__ = ["ANSWER", "CASE", "DIGEST_DOMAIN", "EXE", "PROTOCOL", "STORE_ROOT", "answer", "case_of", "digest", "judged",
           "same_store", "serve", "store_root", "write"]
