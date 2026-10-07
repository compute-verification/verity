"""The Python the composition's Lean (`CoreComposition`, `CoreOnsite`, `CoreRemote`) is held to: the composition rule
vendored from #1036's trial, and mirrors of what it composes over (the core interface's `answer`, the stub components).

The rule is the trial's onsite and remote over their components' statements, from danielreuter/verity#1036, branch
`cursor/protocol-interfaces-c3b2` at 46820ebcb, which no branch this one builds on carries:

* `trial/verity/protocols/onsite.py` and `remote.py`: a product's fields are its components, and its verifier is the list
  of their verifiers' actors;
* `trial/verity/run.py`: an actor that raises records `Raised`, and what it would have established is missing: "whatever
  reads it waits for it forever: a missing statement fails closed";
* `trial/verity/primitives/events.py` `gather`, which `Reconcile.verifier` (`accounting/reconcile.py`) uses to wait for
  each resource's statement and `Nci.verifier` (`compliance/nci.py`) for the stream and the balance: a statement with
  reasons explains nothing, and the epoch complies only with no reasons.

Reduced to answers: a component that accepts records a statement with no reasons, one that rejects a statement with
reasons, and one that is undecided no statement yet. `trial_verdict` is that reduction, with one change already made:
every component's reasons fail the product. Of the components' statements the trial's NCI fails the epoch on work's
reasons alone (integrity's reach it through them), and space's or communication's reasons only zero their explained usage
in the balance. The port follows top's rule (Oct 7, binding), `verdict`, which changes the reduction in one place more:

* **reject dominates**: in the trial a missing statement leaves the product with no verdict even when another component's
  statement has failed (`gather` waits on), so a prover could hide a rejection by leaving another component undecided;
  here any rejection rejects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from verity_difftest import canonical

ACCEPT, REJECT, UNDECIDED = "accept", "reject", "undecided"


def trial_verdict(verdicts: list[str]) -> str:
    """The trial's product: no verdict while any statement is missing, else it complies iff no statement has reasons."""
    if UNDECIDED in verdicts:
        return UNDECIDED
    return REJECT if REJECT in verdicts else ACCEPT


def verdict(verdicts: list[str]) -> str:
    """Top's rule, which the port follows: any rejection rejects, then any undecided component leaves it undecided."""
    if REJECT in verdicts:
        return REJECT
    return UNDECIDED if UNDECIDED in verdicts else ACCEPT


class Refused(Exception):
    """A decoder refuses its input; `answer` prefixes the message and rejects."""


@dataclass(frozen=True)
class Answer:
    """`Core.Answer`: the verdict, its reason unless it accepts, the commitment's digest once it decoded, and the public
    outputs and profile as JSON on acceptance."""

    protocol: str
    verdict: str
    reason: str | None
    digest: bytes | None
    public: Any = None
    profile: Any = None

    def json(self, reason: bool = True) -> dict:
        """`Answer.toJson`, with `"reason": null` when `reason` is false (the difftest doesn't compare reasons)."""
        return {"commitment": None if self.digest is None else self.digest.hex(), "profile": self.profile,
                "protocol": self.protocol, "public": self.public, "reason": self.reason if reason else None,
                "verdict": self.verdict}


def canonical_json(b: bytes) -> Any:
    """`Core.canonicalJson`: UTF-8 that canonical JSON writes back byte for byte."""
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        raise Refused("not UTF-8") from None
    try:
        j = json.loads(s)
    except ValueError:
        raise Refused("not JSON") from None
    try:
        same = canonical(j) == s
    except TypeError:
        same = False
    if not same:
        raise Refused("not canonical JSON (compact, keys sorted, integers only)")
    return j


def unhex(s: str) -> bytes:
    """`Flock.unhex`: lowercase hex digits, an even number of them."""
    if len(s) % 2 or any(c not in "0123456789abcdef" for c in s):
        raise Refused("not lowercase hex")
    return bytes.fromhex(s)


def answer(protocol, params: bytes, commitment: bytes, evidence: bytes) -> Answer:
    """`Core.Protocol.answer`: decode params (canonical JSON), commitment and evidence in that order, a refusal rejecting
    with the commitment's digest once it decoded, then the check."""
    def refuse(reason: str, digest: bytes | None) -> Answer:
        return Answer(protocol.id, REJECT, reason, digest)

    try:
        p = protocol.params(canonical_json(params))
    except Refused as e:
        return refuse(f"params: {e}", None)
    try:
        c = protocol.commitment(commitment)
    except Refused as e:
        return refuse(f"commitment: {e}", None)
    d = protocol.digest(c)
    try:
        ev = protocol.evidence(evidence)
    except Refused as e:
        return refuse(f"evidence: {e}", d)
    kind, a, b = protocol.check(p, c, ev)
    if kind == ACCEPT:
        return Answer(protocol.id, ACCEPT, None, d, protocol.public_json(a), protocol.profile_json(b))
    return Answer(protocol.id, kind, a, d)


def _nat(j: Any, k: str) -> int:
    """`j.getObjValAs? Nat k`."""
    v = j.get(k) if isinstance(j, dict) else None
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise Refused(f"{k} is missing or not a natural number")
    return v


@dataclass(frozen=True)
class Stub:
    """`Core.Example.protocol` under the id `id` (`Core.Composition.Stub.stub`): params `{"k": k}`, a 32-byte root that is
    its own digest, evidence `{"drawn": n, "passed": m}`; rejects when a drawn unit failed, is undecided when fewer than k
    were drawn, and accepts with `{"passed": m}` and an integrity profile otherwise."""

    id: str

    def params(self, j):
        return _nat(j, "k")

    def commitment(self, b: bytes) -> bytes:
        if len(b) != 32:
            raise Refused(f"a root is 32 bytes, not {len(b)}")
        return b

    def digest(self, root: bytes) -> bytes:
        return root

    def evidence(self, b: bytes) -> tuple[int, int]:
        j = canonical_json(b)
        return _nat(j, "drawn"), _nat(j, "passed")

    def check(self, k: int, root: bytes, e: tuple[int, int]):
        drawn, passed = e
        if passed < drawn:
            return REJECT, f"{drawn - passed} of {drawn} drawn units failed", None
        if drawn < k:
            return UNDECIDED, f"{drawn} of the {k} units drawn", None
        profile = {"covered": [], "delta": "1/1000000", "drawn": {"unit": drawn},
                   "levels": [{"draw": {"subset": k}, "name": "unit", "query": None, "size": drawn}], "linked": True,
                   "prescribed": [], "programs": [root.hex()]}
        return ACCEPT, passed, profile

    def public_json(self, passed: int):
        return {"passed": passed}

    def profile_json(self, profile):
        return profile


def bind(id: str, digests: list[tuple[str, bytes]]) -> bytes:
    """`Core.Composition.bind`: SHA-512 over the composition's id, a zero byte and the canonical JSON of the components'
    digests in order."""
    body = canonical({"components": [{"commitment": d.hex(), "id": i} for i, d in digests]})
    return hashlib.sha512(id.encode() + b"\0" + body.encode()).digest()


def _exactly(j: Any, keys: list[str], ctx: str) -> None:
    if not isinstance(j, dict):
        raise Refused(f"{ctx}: not an object")
    if sorted(j) != keys:
        raise Refused(f"{ctx}: the keys must be {keys}, not {sorted(j)}")


def _entries(j: Any, ctx: str) -> list:
    _exactly(j, ["components"], ctx)
    if not isinstance(j["components"], list):
        raise Refused(f"{ctx}.components: not a list")
    return j["components"]


def _str(x: dict, k: str) -> str:
    if not isinstance(x[k], str):
        raise Refused(f"{k}: not a string")
    return x[k]


@dataclass(frozen=True)
class Product:
    """`Core.Composition.product id components`."""

    id: str
    components: tuple

    def params(self, j) -> list[tuple[str, str]]:
        out = []
        for x in _entries(j, "the composition's params"):
            _exactly(x, ["id", "params"], "a component's params")
            out.append((_str(x, "id"), canonical(x["params"])))
        return out

    def commitment(self, b: bytes) -> bytes:
        if len(b) != 64:
            raise Refused(f"the binding of the components' digests is 64 bytes, not {len(b)}")
        return b

    def digest(self, c: bytes) -> bytes:
        return c

    def evidence(self, b: bytes) -> list[tuple[str, str, bytes, bytes]]:
        out = []
        for x in _entries(canonical_json(b), "the composition's evidence"):
            _exactly(x, ["commitment", "evidence", "id", "params"], "a component's case")
            out.append((_str(x, "id"), canonical(x["params"]), unhex(_str(x, "commitment")), unhex(_str(x, "evidence"))))
        return out

    def run(self, ps, ev) -> list[tuple[str, Answer]] | str:
        """`Core.Composition.run`: the components' answers, or why the cases don't fit the composition."""
        want = [P.id for P in self.components]
        if [i for i, _ in ps] != want:
            return "params: not the composition's components, once each, in order"
        if [k[0] for k in ev] != want:
            return "evidence: not the composition's components, once each, in order"
        if any(k[1] != n[1] for n, k in zip(ps, ev)):
            return "evidence: a case carries params other than the composition's"
        return [(P.id, answer(P, n[1].encode(), k[2], k[3])) for P, n, k in zip(self.components, ps, ev)]

    def check(self, ps, c: bytes, ev):
        answers = self.run(ps, ev)
        if isinstance(answers, str):
            return REJECT, answers, None
        rule = verdict([a.verdict for _, a in answers])
        if rule == REJECT:
            return REJECT, "a component rejected", None
        if bind(self.id, [(i, a.digest) for i, a in answers]) != c:
            return REJECT, "the commitment does not bind the components' digests", None
        if rule == UNDECIDED:
            return UNDECIDED, "a component is undecided", None
        return ACCEPT, {i: a.public for i, a in answers}, {i: a.profile for i, a in answers}

    def public_json(self, keyed):
        return keyed

    def profile_json(self, keyed):
        return keyed


def stub_ids(*names: str) -> tuple[Stub, ...]:
    return tuple(Stub(f"verity/stub/{n}/v0") for n in names)


#: `Core.Remote.protocol` and `Core.Onsite.protocol`, over the stubs that stand in for their components.
REMOTE = Product("verity/remote/v0", stub_ids("pouw", "sampled-proofs"))
ONSITE = Product("verity/onsite/v0", stub_ids("certifier", "pous", "pouw", "sampled-proofs"))
