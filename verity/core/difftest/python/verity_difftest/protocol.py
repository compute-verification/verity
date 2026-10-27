"""The core protocol interface's framing in Python, for a `ProtocolPort`'s reference: `Core.Protocol.case` and
`Core.Protocol.answer` (`verity/core/protocols/interface`, `CoreProtocol/Cli.lean` and `Protocol.lean`), with
the protocol's own decoders and check passed in.

`answer` follows the Lean order, which decides the outcome's `commitment`: a malformed case (not an object, a field missing,
a commitment or evidence that isn't lowercase hex) and parameters that don't decode are rejected with no commitment; a
commitment that doesn't decode too; evidence that doesn't decode, and the check's own verdicts, carry the commitment's
digest. A decoder refuses by raising `verity_difftest.Reject`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from verity_difftest import Reject, canonical

HEX = frozenset("0123456789abcdef")


def unhex(s: str) -> bytes:
    """`Flock.unhex`: even-length lowercase hex only (`bytes.fromhex` also takes uppercase and spaces)."""
    if len(s) % 2:
        raise Reject("odd-length hex")
    if not HEX.issuperset(s):
        raise Reject("not lowercase hex")
    return bytes.fromhex(s)


def canonical_json(b: bytes) -> Any:
    """`Core.canonicalJson`: UTF-8 that is JSON and that `canonical` writes back byte for byte."""
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        raise Reject("not UTF-8") from None
    try:
        j = json.loads(s, parse_constant=lambda c: float(c))
        same = canonical(j) == s
    except (ValueError, TypeError, RecursionError) as e:
        raise Reject(f"not JSON, or not canonical JSON ({type(e).__name__}: {e})") from None
    if not same:
        raise Reject("not canonical JSON (compact, keys sorted, integers only)")
    return j


def nat(j: Any, key: str) -> int:
    """`j.getObjValAs? Nat key`: a non-negative integer (not a boolean) under `key` of an object."""
    v = j.get(key) if isinstance(j, dict) else None
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise Reject(f"{key} is missing or not a natural number")
    return v


def outcome(protocol: str, verdict: str, reason: str | None = None, digest: bytes | None = None,
            public: Any = None, profile: Any = None) -> dict:
    """`Core.Answer.toJson`."""
    return {"commitment": None if digest is None else digest.hex(), "profile": profile, "protocol": protocol,
            "public": public, "reason": reason, "verdict": verdict}


def answer(protocol: str, case: Any, *, params: Callable[[Any], Any], commitment: Callable[[bytes], Any],
           evidence: Callable[[bytes], Any], check: Callable[[Any, Any, Any], tuple],
           digest: Callable[[Any], bytes] = bytes) -> dict:
    """The outcome of one `serve` case. `check(params, commitment, evidence)` returns `("accept", public JSON, profile
    JSON)`, `("reject", reason)` or `("undecided", reason)`."""
    try:
        if not isinstance(case, dict):
            raise Reject("the case is not an object")
        missing = [k for k in ("params", "commitment", "evidence") if k not in case]
        if missing:
            raise Reject(f"the case: {missing[0]} missing")
        hexes = [case[k] for k in ("commitment", "evidence")]
        if not all(isinstance(h, str) for h in hexes):
            raise Reject("the case: commitment and evidence are hex strings")
        c_bytes, e_bytes = (unhex(h) for h in hexes)
    except Reject as e:
        return outcome(protocol, "reject", f"case: {e}")
    try:
        p = params(case["params"])
    except Reject as e:
        return outcome(protocol, "reject", f"params: {e}")
    try:
        c = commitment(c_bytes)
    except Reject as e:
        return outcome(protocol, "reject", f"commitment: {e}")
    d = digest(c)
    try:
        ev = evidence(e_bytes)
    except Reject as e:
        return outcome(protocol, "reject", f"evidence: {e}", d)
    verdict, *rest = check(p, c, ev)
    if verdict == "accept":
        return outcome(protocol, "accept", None, d, *rest)
    return outcome(protocol, verdict, rest[0], d)
