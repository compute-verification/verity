"""The Python the composition's Lean (`CoreComposition`, `CoreOnsite`, `CoreRemote`) is held to: the composition rule
vendored from #1036's trial, over the harness's framing of the core interface (`verity_difftest.protocol.answer`, as
`Core.Protocol.case` and `answer` decode a case) and mirrors of the stub components.

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

from verity_difftest import Reject, canonical
from verity_difftest import protocol as P

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


@dataclass(frozen=True)
class Stub:
    """`Core.Example.protocol` under the id `id` (`Core.Composition.Stub.stub`): params `{"k": k}`, a 32-byte root that is
    its own digest, evidence `{"drawn": n, "passed": m}`; rejects when a drawn unit failed, is undecided when fewer than k
    were drawn, and accepts with `{"passed": m}` and an integrity profile otherwise."""

    id: str

    @staticmethod
    def _root(b: bytes) -> bytes:
        if len(b) != 32:
            raise Reject(f"a root is 32 bytes, not {len(b)}")
        return b

    @staticmethod
    def _evidence(b: bytes) -> tuple[int, int]:
        j = P.canonical_json(b)
        return P.nat(j, "drawn"), P.nat(j, "passed")

    @staticmethod
    def _check(k: int, root: bytes, e: tuple[int, int]) -> tuple:
        drawn, passed = e
        if passed < drawn:
            return REJECT, f"{drawn - passed} of {drawn} drawn units failed"
        if drawn < k:
            return UNDECIDED, f"{drawn} of the {k} units drawn"
        profile = {"covered": [], "delta": "1/1000000", "drawn": {"unit": drawn},
                   "levels": [{"draw": {"subset": k}, "name": "unit", "query": None, "size": drawn}], "linked": True,
                   "prescribed": [], "programs": [root.hex()]}
        return ACCEPT, {"passed": passed}, profile

    def answer(self, case: Any) -> dict:
        return P.answer(self.id, case, params=lambda j: P.nat(j, "k"), commitment=self._root, evidence=self._evidence,
                        check=self._check)


def bind(id: str, digests: list[tuple[str, bytes]]) -> bytes:
    """`Core.Composition.bind`: SHA-512 over the composition's id, a zero byte and the canonical JSON of the components'
    digests in order."""
    body = canonical({"components": [{"commitment": d.hex(), "id": i} for i, d in digests]})
    return hashlib.sha512(id.encode() + b"\0" + body.encode()).digest()


def _exactly(j: Any, keys: list[str], ctx: str) -> None:
    if not isinstance(j, dict):
        raise Reject(f"{ctx}: not an object")
    if sorted(j) != keys:
        raise Reject(f"{ctx}: the keys must be {keys}, not {sorted(j)}")


def _entries(j: Any, ctx: str) -> list:
    _exactly(j, ["components"], ctx)
    if not isinstance(j["components"], list):
        raise Reject(f"{ctx}.components: not a list")
    return j["components"]


def _str(x: dict, k: str) -> str:
    if not isinstance(x[k], str):
        raise Reject(f"{k}: not a string")
    return x[k]


@dataclass(frozen=True)
class Product:
    """`Core.Composition.product id components`."""

    id: str
    components: tuple

    def params(self, j: Any) -> list[tuple[str, str]]:
        out = []
        for x in _entries(j, "the composition's params"):
            _exactly(x, ["id", "params"], "a component's params")
            out.append((_str(x, "id"), canonical(x["params"])))
        return out

    @staticmethod
    def commitment(b: bytes) -> bytes:
        if len(b) != 64:
            raise Reject(f"the binding of the components' digests is 64 bytes, not {len(b)}")
        return b

    @staticmethod
    def evidence(b: bytes) -> list[tuple[str, str, bytes, bytes]]:
        out = []
        for x in _entries(P.canonical_json(b), "the composition's evidence"):
            _exactly(x, ["commitment", "evidence", "id", "params"], "a component's case")
            out.append((_str(x, "id"), canonical(x["params"]), P.unhex(_str(x, "commitment")),
                        P.unhex(_str(x, "evidence"))))
        return out

    def run(self, ps, ev) -> list[tuple[str, dict]] | str:
        """`Core.Composition.run`: each component's outcome on its case, or why the cases don't fit the composition."""
        want = [C.id for C in self.components]
        if [i for i, _ in ps] != want:
            return "params: not the composition's components, once each, in order"
        if [k[0] for k in ev] != want:
            return "evidence: not the composition's components, once each, in order"
        if any(k[1] != n[1] for n, k in zip(ps, ev)):
            return "evidence: a case carries params other than the composition's"
        return [(C.id, C.answer({"commitment": k[2].hex(), "evidence": k[3].hex(), "params": json.loads(n[1])}))
                for C, n, k in zip(self.components, ps, ev)]

    def binds(self, answers: list[tuple[str, dict]], c: bytes) -> bool:
        """`Core.Composition.binding id answers = some c`: every component's digest is there, and they bind to `c`."""
        if any(a["commitment"] is None for _, a in answers):
            return False
        return bind(self.id, [(i, bytes.fromhex(a["commitment"])) for i, a in answers]) == c

    def check(self, ps, c: bytes, ev) -> tuple:
        answers = self.run(ps, ev)
        if isinstance(answers, str):
            return REJECT, answers
        rule = verdict([a["verdict"] for _, a in answers])
        if rule == REJECT:
            return REJECT, "a component rejected"
        if not self.binds(answers, c):
            return REJECT, "the commitment does not bind the components' digests"
        if rule == UNDECIDED:
            return UNDECIDED, "a component is undecided"
        return ACCEPT, {i: a["public"] for i, a in answers}, {i: a["profile"] for i, a in answers}

    def answer(self, case: Any) -> dict:
        """`Core.Protocol.case` of the product: the outcome of one `serve` line."""
        return P.answer(self.id, case, params=self.params, commitment=self.commitment, evidence=self.evidence,
                        check=self.check)

    def verdicts(self, case: Any) -> list[str]:
        """The components' verdicts on a case that fits the composition, else none."""
        try:
            answers = self.run(self.params(case["params"]), self.evidence(P.unhex(case["evidence"])))
        except (Reject, KeyError, TypeError):
            return []
        return [] if isinstance(answers, str) else [a["verdict"] for _, a in answers]


def stub_ids(*names: str) -> tuple[Stub, ...]:
    return tuple(Stub(f"verity/stub/{n}/v0") for n in names)


#: `Core.Remote.protocol` and `Core.Onsite.protocol`, over the stubs that stand in for their components.
REMOTE = Product("verity/remote/v0", stub_ids("pouw", "sampled-proofs"))
ONSITE = Product("verity/onsite/v0", stub_ids("certifier", "pous", "pouw", "sampled-proofs"))
