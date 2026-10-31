"""`Core.Answer` in Python: the outcome every core protocol gives (`verity/core/protocols/interface`,
`CoreProtocol/Protocol.lean`), its one canonical JSON and its decoder. A Python verifier that answers as a core protocol
builds its answer here, and whatever reads another protocol's answer decodes it here.

* `Verdict`: `Core.Verdict`. `undecided` does not reject the prover: the evidence doesn't settle the question, and
  nothing is made public.
* `Answer`: `Core.Answer`, in the shapes `Protocol.answer` returns (`Answer.accept`, `reject`, `undecided`): an accept
  has no reason and both its public outputs and its profile, which are never ``None``; a reject or undecided has a
  reason and neither. `Answer.to_json` is `Answer.toJson`, `Answer.canonical` is `Answer.canonical`.
* `outcome`: `Answer.toJson` of the fields, the dict most callers write.
* `canonical_json`: `Core.canonicalJson`, ASCII without DEL that `canon` (`Flock.canon`) writes back byte for byte.
* `of_json`: `Answer.ofJson`, which accepts only what `to_json` writes, in those shapes; its refusals carry `Flock.J`'s
  messages.

Stdlib only: a verifier imports no `verity.ml` code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    """`Core.Verdict`, by `Verdict.name`."""

    accept = "accept"
    reject = "reject"
    undecided = "undecided"


VERDICTS = tuple(v.value for v in Verdict)
#: `Answer.toJson`'s keys, in canonical order
KEYS = ("commitment", "profile", "protocol", "public", "reason", "verdict")


class Refused(ValueError):
    """A decoder's refusal (the message is the reason)."""


def canon(x: Any) -> str:
    """`Flock.canon`: compact, keys sorted, every character outside printable ASCII escaped, integers only."""
    if isinstance(x, float):
        raise Refused("canonical JSON: a non-integer number")
    if isinstance(x, dict):
        return "{" + ",".join(json.dumps(k) + ":" + canon(v) for k, v in sorted(x.items())) + "}"
    if isinstance(x, list):
        return "[" + ",".join(canon(v) for v in x) + "]"
    return json.dumps(x)


def canonical_json(b: bytes) -> Any:
    """`Core.canonicalJson`: the JSON value of `b`, which must be ASCII without DEL that `canon` writes back byte for
    byte."""
    if any(c >= 0x7f for c in b):
        raise Refused("not ASCII without DEL (a byte at or above 0x7f)")
    try:
        s = b.decode()
    except UnicodeDecodeError:
        raise Refused("not UTF-8") from None
    try:
        x = json.loads(s)
    except ValueError as e:
        raise Refused(f"not JSON ({e})") from None
    if canon(x) != s:
        raise Refused("not canonical JSON (compact, keys sorted, integers only)")
    return x


def unhex(s: str) -> bytes:
    """`Flock.unhex`: even-length lowercase hex only (`bytes.fromhex` also takes uppercase and spaces)."""
    if len(s) % 2:
        raise Refused("odd-length hex")
    if any(c not in "0123456789abcdef" for c in s):
        raise Refused("not lowercase hex")
    return bytes.fromhex(s)


def outcome(protocol: str, verdict: str, reason: str | None = None, digest: bytes | None = None, public: Any = None,
            profile: Any = None) -> dict[str, Any]:
    """`Answer.toJson` of the fields: the commitment's digest as lowercase hex, and ``None`` for each field absent."""
    return {"commitment": None if digest is None else digest.hex(), "profile": profile, "protocol": protocol,
            "public": public, "reason": reason, "verdict": Verdict(verdict).value}


@dataclass(frozen=True)
class Answer:
    """`Core.Answer`: the verdict, the reason when it isn't `accept`, the commitment's digest once it decoded, and the
    public outputs and profile on acceptance."""

    protocol: str
    verdict: Verdict
    reason: str | None = None
    digest: bytes | None = None
    public: Any = None
    profile: Any = None

    @classmethod
    def accept(cls, protocol: str, digest: bytes | None, public: Any, profile: Any) -> Answer:
        if public is None or profile is None:
            raise ValueError("an accept has both its public outputs and its profile")
        return cls(protocol, Verdict.accept, None, digest, public, profile)

    @classmethod
    def reject(cls, protocol: str, reason: str, digest: bytes | None = None) -> Answer:
        return cls(protocol, Verdict.reject, reason, digest)

    @classmethod
    def undecided(cls, protocol: str, reason: str, digest: bytes | None = None) -> Answer:
        return cls(protocol, Verdict.undecided, reason, digest)

    def to_json(self) -> dict[str, Any]:
        """`Answer.toJson`."""
        return outcome(self.protocol, self.verdict, self.reason, self.digest, self.public, self.profile)

    def canonical(self) -> str:
        """`Answer.canonical`: the one form a caller or a difftest reads."""
        return canon(self.to_json())


def _get(j: Any, k: str, ctx: str) -> Any:
    if not isinstance(j, dict) or k not in j:
        raise Refused(f"{ctx}: {k} missing")
    return j[k]


def _as_str(v: Any, ctx: str) -> str:
    if not isinstance(v, str):
        raise Refused(f"{ctx}: not a string")
    return v


def of_json(j: Any, ctx: str = "answer") -> Answer:
    """`Answer.ofJson`: only what `to_json` writes, and only the shapes `Protocol.answer` returns."""
    name = _as_str(_get(j, "verdict", ctx), f"{ctx}.verdict")
    if name not in VERDICTS:
        raise Refused(f"{ctx}.verdict: not accept, reject or undecided")
    verdict = Verdict(name)
    protocol = _as_str(_get(j, "protocol", ctx), f"{ctx}.protocol")
    reason = _get(j, "reason", ctx)
    if reason is not None:
        _as_str(reason, f"{ctx}.reason")
    com = _get(j, "commitment", ctx)
    digest = None if com is None else unhex(_as_str(com, f"{ctx}.commitment"))
    public, profile = _get(j, "public", ctx), _get(j, "profile", ctx)
    if verdict == Verdict.accept:
        if not (reason is None and public is not None and profile is not None):
            raise Refused(f"{ctx}: an accept has no reason, and both its public outputs and its profile")
    elif not (reason is not None and public is None and profile is None):
        raise Refused(f"{ctx}: a {verdict.value} has a reason, and neither public outputs nor a profile")
    if sorted(j) != list(KEYS):
        raise Refused(f"{ctx}: not in the form Answer.toJson writes")
    return Answer(protocol, verdict, reason, digest, public, profile)


__all__ = ["KEYS", "VERDICTS", "Answer", "Refused", "Verdict", "canon", "canonical_json", "of_json", "outcome", "unhex"]
