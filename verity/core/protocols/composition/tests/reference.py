"""The Python the composition's Lean (`CoreComposition`, `CoreOnsite`, `CoreRemote`) is held to: the composition rule
vendored from #1036's trial, over the harness's framing of the core interface (`verity_difftest.protocol.answer`, as
`Core.Protocol.case` and `answer` decode a case), and each component's own Python: PoUW's `core_check.case`
(`verity/pouw/v1`), the network certifier's period check's difftest's `reference` (`verity/certifier/v0`, its Python
mirror over the certifier's certificates and one-stage's answers,
`verity/core/protocols/network/tests/test_network_period_protocol.py`), NCI's protocol difftest's `reference`
(`verity/nci/v1`, onsite's registration check, `nci_reference`'s `check_baseline`,
`verity/core/protocols/network/tests/test_nci_protocol.py`), for
sampled proofs' one-stage audit (`verity/sampled-proofs/one-stage/v1`) `one_stage` behind the protocol's decoders, its
difftest's `answer` (`verity/core/service/sampled_proofs/tests/test_difftest.py`), and for space (`verity/pous/p2/v4/run`) and bandwidth
(`verity/remote-bandwidth/v0`) their Python verifiers' answers on small periods of their hooks (`Recorded`,
`periods.py`). Each component's own difftest holds its Lean to that Python; this one holds the composition of them.

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
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from verity_difftest import Reject, canonical
from verity_difftest import protocol as P

from verity.core.protocols.pouw import core_check

ACCEPT, REJECT, UNDECIDED = "accept", "reject", "undecided"

CORE = Path(__file__).resolve().parents[3]


def module(name: str, path: Path) -> ModuleType:
    """Another suite's test module, loaded from its file under `name`, with its directory on the path for the siblings it
    imports."""
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


#: Sampled proofs' one-stage difftest: `answer` is `one_stage` behind the decoders `OneStage` reads, and its strategies
#: draw audits.
ONE_STAGE = module("one_stage_difftest", CORE / "service/sampled_proofs/tests/test_difftest.py")
#: PoUW's check difftest's cases: honest windows and one mutation each.
POUW_CASES = module("pouw_check_cases", CORE / "protocols/pouw/lean/check/tests/pouw_check_cases.py")
#: The network certifier's period difftest: `reference` is the period check's Python mirror, and `build` makes a signed
#: period with one change.
PERIOD = module("network_period_difftest", CORE / "protocols/network/tests/test_network_period_protocol.py")
#: NCI's protocol difftest: `reference` is `check_baseline` behind the protocol's decoders, and `cases` draws
#: declarations, honest or with one change.
NCI = module("nci_protocol_difftest", CORE / "protocols/network/tests/test_nci_protocol.py")


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
class Ported:
    """A component whose port has replaced its stub: its id and its Python's answer to a `serve` case."""

    id: str
    answer: Callable[[Any], dict]


@dataclass(frozen=True)
class Recorded:
    """A component on the cases recorded for it: `cases` holds each kind's case and the answer its Python verifier gave
    it (`periods.build` fills it), and `answer` is that answer. The composition hands a component the case it was drawn,
    or one whose commitment is no longer JSON (`bad-hex` under a `0`), which both recorded components refuse with no
    digest, as they do params that don't decode; any other case is a difference."""

    id: str
    cases: dict[str, tuple[dict, dict]] = field(default_factory=dict, compare=False, hash=False)

    def answer(self, case: Any) -> dict:
        for c, a in self.cases.values():
            if canonical(c) == canonical(case):
                return a
        try:
            P.canonical_json(P.unhex(case["commitment"]))
        except Reject as e:
            return P.outcome(self.id, REJECT, f"commitment: {e}")
        raise KeyError(f"{self.id}: no answer recorded for this case")


CERTIFIER = Ported("verity/certifier/v0", PERIOD.reference)
NCI_CHECK = Ported("verity/nci/v1", NCI.reference)
POUW = Ported("verity/pouw/v1", core_check.case)
SAMPLED_PROOFS = Ported("verity/sampled-proofs/one-stage/v1", ONE_STAGE.answer)
#: Space and bandwidth, `PousCheck.RunProtocol.protocol` and `RemoteAudit.Protocol.protocol`.
POUS = Recorded("verity/pous/p2/v4/run")
BANDWIDTH = Recorded("verity/remote-bandwidth/v0")


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


@dataclass(frozen=True)
class Registered:
    """`Core.Composition.registered id R s components`: the registration check ``registration`` on the registration's
    case first, a refusal refusing the whole, then ``product`` of the components on the rest, under the commitment's
    first 64 bytes; the rest of the commitment is the registration's digest, which the component ``committer`` must have
    committed to once the cases fit the composition (one registration digest in both)."""

    id: str
    registration: Ported
    committer: str
    product: Product

    def params(self, j: Any) -> tuple[tuple[str, str], list[tuple[str, str]]]:
        _exactly(j, ["components", "registration"], "the composition's params")
        r = j["registration"]
        _exactly(r, ["id", "params"], "the registration's params")
        return (_str(r, "id"), canonical(r["params"])), self.product.params({"components": j["components"]})

    @staticmethod
    def commitment(b: bytes) -> tuple[bytes, bytes]:
        if len(b) < 64:
            raise Reject(f"the binding of the components' digests is 64 bytes, then the registration's digest: not {len(b)}")
        return b[:64], b[64:]

    def evidence(self, b: bytes) -> tuple[tuple[str, str, bytes, bytes], list]:
        j = P.canonical_json(b)
        _exactly(j, ["components", "registration"], "the composition's evidence")
        r = j["registration"]
        _exactly(r, ["commitment", "evidence", "id", "params"], "the registration's case")
        reg = (_str(r, "id"), canonical(r["params"]), P.unhex(_str(r, "commitment")), P.unhex(_str(r, "evidence")))
        return reg, self.product.evidence(canonical({"components": j["components"]}).encode())

    def check(self, p, c, e) -> tuple:
        (n, ps), (b, d), (k, ev) = p, c, e
        R = self.registration
        if n[0] != R.id:
            return REJECT, f"params: the registration is {R.id}'s, not {n[0]}'s"
        if k[0] != R.id:
            return REJECT, f"evidence: the registration is {R.id}'s, not {k[0]}'s"
        if k[1] != n[1]:
            return REJECT, "evidence: the registration's case carries params other than the composition's"
        a = R.answer({"commitment": k[2].hex(), "evidence": k[3].hex(), "params": json.loads(n[1])})
        if a["verdict"] != ACCEPT:
            return REJECT, f"registration: {R.id} {a['verdict']}: {a.get('reason') or ''}"
        if a["commitment"] != d.hex():
            return REJECT, "the commitment does not bind the registration's digest"
        answers = self.product.run(ps, ev)
        if isinstance(answers, str):
            return REJECT, answers
        # A component's answer carries its digest exactly when its commitment decodes and it doesn't reject before;
        # where `committedBy` has a digest the answer doesn't, the component rejects, and so does the composition.
        got = next((x["commitment"] for i, x in answers if i == self.committer), None)
        if got != d.hex():
            return REJECT, f"registration: {R.id} checked {d.hex()}, {self.committer} committed {got or 'no digest'}"
        return self.product.check(ps, b, ev)

    def answer(self, case: Any) -> dict:
        """`Core.Protocol.case` of the registered composition: the outcome of one `serve` line."""
        return P.answer(self.id, case, params=self.params, commitment=self.commitment, evidence=self.evidence,
                        check=self.check, digest=lambda c: c[0] + c[1])

    def verdicts(self, case: Any) -> list[str]:
        """The components' verdicts on a case whose components fit the composition, else none."""
        try:
            _, ps = self.params(case["params"])
            _, ev = self.evidence(P.unhex(case["evidence"]))
            answers = self.product.run(ps, ev)
        except (Reject, KeyError, TypeError):
            return []
        return [] if isinstance(answers, str) else [a["verdict"] for _, a in answers]


#: `Core.Remote.protocol` and `Core.Onsite.protocol`, over their components in the order of their ids, onsite's after its
#: registration check, NCI's.
REMOTE = Product("verity/remote/v0", (POUW, SAMPLED_PROOFS))
ONSITE = Registered("verity/onsite/v0", NCI_CHECK, SAMPLED_PROOFS.id,
                    Product("verity/onsite/v0", (CERTIFIER, POUS, POUW, BANDWIDTH, SAMPLED_PROOFS)))
