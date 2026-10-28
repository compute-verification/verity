"""The Python the composition's Lean (`CoreComposition`, `CoreOnsite`, `CoreRemote`) is held to: the composition rule
of onsite and remote, over the harness's framing of the core interface (`verity_difftest.protocol.answer`, as
`Core.Protocol.case` and `answer` decode a case), and each component's own Python: PoUW's `core_check.case`
(`verity/pouw/v1`, remote's work) and `period_check.case` (`verity/pouw/period/v1`, onsite's work), the network certifier's period check's difftest's `reference` (`verity/certifier/v0`, its Python
mirror over the certifier's certificates and one-stage's answers,
`verity/core/protocols/network/tests/test_network_period_protocol.py`), for sampled proofs' one-stage audit on the
proofs themselves (`verity/sampled-proofs/one-stage/v2`) `one_stage_zk`: v2's framing over v1's `one_stage` and decoders
(`verity/core/service/sampled_proofs/tests/test_difftest.py`), on the cases v2 decides without a proof, the wipe's
(`verity/wipe/v0`) `wipe_protocol_case.answer`, for onsite's registration check NCI's protocol difftest's `reference`
(`verity/nci/v1`, `nci_reference`'s `check_baseline`, `verity/core/protocols/network/tests/test_nci_protocol.py`), and for space (`verity/pous/p2/v4/run`) and bandwidth
(`verity/remote-bandwidth/v0`) their Python verifiers' answers on small periods of their hooks (`Recorded`,
`periods.py`). Each component's own difftest holds its Lean to that Python; this one holds the composition of them.

The trial's rule, onsite's and remote's over their components' statements:

* a product's fields are its components, and its verifier is the list of their verifiers' actors;
* an actor that raises records `Raised`, and what it would have established is missing: whatever reads it waits for it
  forever, so a missing statement fails closed;
* the product waits for each component's statement (`gather`): a statement with reasons explains nothing, and the epoch
  complies only with no reasons.

Reduced to answers: a component that accepts records a statement with no reasons, one that rejects a statement with
reasons, and one that is undecided no statement yet. `trial_verdict` is that reduction, with one change already made:
every component's reasons fail the product. Of the components' statements the trial's NCI fails the epoch on work's
reasons alone (integrity's reach it through them), and space's or communication's reasons only zero their explained usage
in the balance. The port follows the composition's rule, `verdict`, which changes the reduction in one place more:

* **reject dominates**: in the trial a missing statement leaves the product with no verdict even when another component's
  statement has failed (`gather` waits on), so a prover could hide a rejection by leaving another component undecided;
  here any rejection rejects.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from verity_difftest import Reject, canonical
from verity_difftest import protocol as P

from verity.core.protocols.pouw import core_check, period_check

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
#: PoUW's period difftest's cases: periods of the auditor's ledger and their windows, honest and with one mutation each.
POUW_PERIOD_CASES = module("pouw_period_cases", CORE / "protocols/pouw/lean/check/tests/pouw_period_cases.py")
#: The network certifier's period difftest: `reference` is the period check's Python mirror, and `build` makes a signed
#: period with one change.
PERIOD = module("network_period_difftest", CORE / "protocols/network/tests/test_network_period_protocol.py")
#: The wipe protocol in Python: `answer`, its mirror under S1's LMS, and `case`, a signed wipe aimed at a kind (`KINDS`).
WIPE_CASES = module("wipe_protocol_case", CORE / "devices/memory_challenger/lean/tests/wipe_protocol_case.py")
#: NCI's protocol difftest: `reference` is `check_baseline` behind the protocol's decoders, and `split` and `seeds` give
#: its declarations.
NCI = module("nci_protocol_difftest", CORE / "protocols/network/tests/test_nci_protocol.py")


def trial_verdict(verdicts: list[str]) -> str:
    """The trial's product: no verdict while any statement is missing, else it complies iff no statement has reasons."""
    if UNDECIDED in verdicts:
        return UNDECIDED
    return REJECT if REJECT in verdicts else ACCEPT


def verdict(verdicts: list[str]) -> str:
    """The composition's rule, which the port follows: any rejection rejects, then any undecided component leaves it undecided."""
    if REJECT in verdicts:
        return REJECT
    return UNDECIDED if UNDECIDED in verdicts else ACCEPT


def fitted_verdicts(answers: Callable[[], Any]) -> list[str]:
    """The components' verdicts in ``answers()``, else none when the case doesn't fit the composition."""
    try:
        out = answers()
    except (Reject, KeyError, TypeError):
        return []
    return [] if isinstance(out, str) else [a["verdict"] for _, a in out]


@dataclass(frozen=True)
class Ported:
    """A component whose port has replaced its stub: its id and its Python's answer to a `serve` case. One that reads
    blobs (`Core.BlobProtocol`) gets the product's under the case's ``blobs``, as bytes."""

    id: str
    answer: Callable[[Any], dict]
    blobs: bool = False


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


V2 = "verity/sampled-proofs/one-stage/v2"
DIGEST = re.compile(r"[0-9a-f]{128}")
REGISTERED_INPUT = re.compile(r"RegisteredInput[1-9][0-9]*_v1")


def _v2_params(j: Any) -> dict:
    """`OneStageZk.paramsOfJson`: v1's parameters with ``proofs``, exactly the auditor's circuit digest and the verifier's
    registered file, and the verifier's own partition object."""
    pr = j.get("proofs") if isinstance(j, dict) else None
    if not isinstance(pr, dict) or sorted(pr) != ["circuit", "registered"]:
        raise Reject("the parameters name no proofs, or other keys under them")
    if not isinstance(pr["circuit"], str) or not DIGEST.fullmatch(pr["circuit"]):
        raise Reject("proofs: circuit is not a SHA-512 digest")
    if pr["registered"] is not None and not isinstance(pr["registered"], dict):
        raise Reject("proofs: registered is not an object or null")
    base = ONE_STAGE.decode_params({k: v for k, v in j.items() if k != "proofs"})
    if base["partition"] is None:
        raise Reject("partition is null")
    return {"base": base, "program": j.get("program"), **pr}


def _v2_evidence(b: bytes) -> dict:
    """`OneStageZk.evidenceOf`: v1's evidence with no verdicts and with ``proofs``, the public file's digest and the
    sessions' files, named as the evidence's sessions."""
    j = P.canonical_json(b)
    if not isinstance(j, dict) or "verdicts" in j or "proofs" not in j:
        raise Reject("the evidence is not an object naming proofs and no verdicts")
    base = ONE_STAGE.decode_evidence(canonical({**{k: v for k, v in j.items() if k != "proofs"}, "verdicts": {}}).encode())
    pr = j["proofs"]
    if not isinstance(pr, dict) or sorted(pr) != ["public", "sessions"] or not isinstance(pr["public"], str) \
            or not isinstance(pr["sessions"], list):
        raise Reject("the evidence's proofs")
    for s in pr["sessions"]:
        if not isinstance(s, dict) or sorted(s) != ["name", "record", "reps"] or not isinstance(s["name"], str) \
                or not isinstance(s["record"], str) or not isinstance(s["reps"], dict) \
                or not all(isinstance(d, str) for d in s["reps"].values()):
            raise Reject("a session of the proofs")
    if [s["name"] for s in pr["sessions"]] != base["sessions"]:
        raise Reject("proofs: the sessions are not the evidence's sessions")
    return {"base": base, "public": pr["public"], "sessions": pr["sessions"]}


def _v2_check(blobs: list[bytes]):
    """`OneStageZk.checkBlobs` on `blobs`, for evidence whose sessions' records are in no blob, where it needs no proof:
    the circuit and the public file are blobs, a session whose record is in none is refused, and with no session the
    evidence proves no unit and v1 decides on no verdict (undecided at best: v2 accepts only on a session's proof)."""
    held = {hashlib.sha512(b).hexdigest() for b in blobs}

    def check(p: dict, c: tuple, e: dict) -> tuple:
        if p["registered"] is not None:
            raise ValueError("outside the mirror: the verifier's registered file (`OneStageZk.readsOf`)")
        definitions = p["program"].get("definitions") if isinstance(p["program"], dict) else None
        if not isinstance(definitions, dict) or any(REGISTERED_INPUT.fullmatch(k) for k in definitions):
            return REJECT, "registered: the program reads registered values"
        if p["circuit"] not in held or not DIGEST.fullmatch(e["public"]) or e["public"] not in held:
            return REJECT, "proofs: the circuit or the public file is in no blob"
        if e["sessions"]:
            if e["sessions"][0]["record"] in held:
                raise ValueError("outside the mirror: a session's record is a blob")
            return REJECT, "proofs: a session's record is in no blob"
        if e["base"]["proved_units"]:
            return REJECT, "proofs: the units the sessions proved are not the evidence's proved_units"
        return ONE_STAGE.check(p["base"], c, e["base"])
    return check


def one_stage_zk(case: Any) -> dict:
    """`SampledProofs.OneStageZk.protocol` on a case and its ``blobs`` (`_v2_check`'s cases)."""
    return P.answer(V2, case, params=_v2_params, commitment=ONE_STAGE.decode_commitment, evidence=_v2_evidence,
                    digest=lambda c: c[1], check=_v2_check(case.get("blobs", []) if isinstance(case, dict) else []))


CERTIFIER = Ported("verity/certifier/v0", PERIOD.reference)
POUW = Ported("verity/pouw/v1", core_check.case)
#: The one-stage audit on the proofs themselves, v2 (`SampledProofs.OneStageZk`).
SAMPLED_PROOFS = Ported(V2, one_stage_zk, blobs=True)


@functools.cache
def _period_answer(case: str) -> str:
    return canonical(period_check.case(json.loads(case)))


#: A period's answer is slow to make (its windows' ledger signatures and layouts), and a composition's case asks for each
#: component's twice, so it is made once a case.
POUW_PERIOD = Ported(period_check.ID, lambda case: json.loads(_period_answer(canonical(case))))
#: Space and bandwidth, `PousCheck.RunProtocol.protocol` and `RemoteAudit.Protocol.protocol`.
POUS = Recorded("verity/pous/p2/v4/run")
BANDWIDTH = Recorded("verity/remote-bandwidth/v0")
#: The wipe, `MemoryChallenger.WipeProtocol.protocol`.
WIPE = Ported("verity/wipe/v0", WIPE_CASES.answer)
#: Onsite's registration check, `NciCheck.Protocol.protocol`: not a component.
NCI_CHECK = Ported("verity/nci/v1", NCI.reference)


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

    def run(self, ps, ev, blobs: list[bytes] = ()) -> list[tuple[str, dict]] | str:
        """`Core.Composition.run`: each component's outcome on its case, those that read blobs on the caller's `blobs`
        (`productB`), or why the cases don't fit the composition."""
        want = [C.id for C in self.components]
        if [i for i, _ in ps] != want:
            return "params: not the composition's components, once each, in order"
        if [k[0] for k in ev] != want:
            return "evidence: not the composition's components, once each, in order"
        if any(k[1] != n[1] for n, k in zip(ps, ev)):
            return "evidence: a case carries params other than the composition's"
        return [(C.id, C.answer({"commitment": k[2].hex(), "evidence": k[3].hex(), "params": json.loads(n[1]),
                                 **({"blobs": list(blobs)} if getattr(C, "blobs", False) else {})}))
                for C, n, k in zip(self.components, ps, ev)]

    def binds(self, answers: list[tuple[str, dict]], c: bytes) -> bool:
        """`Core.Composition.binding id answers = some c`: every component's digest is there, and they bind to `c`."""
        if any(a["commitment"] is None for _, a in answers):
            return False
        return bind(self.id, [(i, bytes.fromhex(a["commitment"])) for i, a in answers]) == c

    def check(self, ps, c: bytes, ev, blobs: list[bytes] = ()) -> tuple:
        answers = self.run(ps, ev, blobs)
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

    @staticmethod
    def split(case: Any) -> tuple[Any, list[bytes]]:
        """`Core.BlobProtocol.case`'s split: the case without ``blobs``, and the blobs (absent: none)."""
        if not isinstance(case, dict) or "blobs" not in case:
            return case, []
        bs = case["blobs"]
        if not isinstance(bs, list) or not all(isinstance(b, str) for b in bs):
            raise Reject("the case's blobs are not a list of hex strings")
        return {k: v for k, v in case.items() if k != "blobs"}, [P.unhex(b) for b in bs]

    def answer(self, case: Any) -> dict:
        """`Core.BlobProtocol.case` of the product: the outcome of one `serve` line."""
        try:
            case, blobs = self.split(case)
        except Reject as e:
            return P.outcome(self.id, REJECT, f"case: {e}")
        return P.answer(self.id, case, params=self.params, commitment=self.commitment, evidence=self.evidence,
                        check=lambda ps, c, ev: self.check(ps, c, ev, blobs))

    def verdicts(self, case: Any) -> list[str]:
        """The components' verdicts on a case that fits the composition, else none."""
        def answers():
            c, blobs = self.split(case)
            return self.run(self.params(c["params"]), self.evidence(P.unhex(c["evidence"])), blobs)
        return fitted_verdicts(answers)


@dataclass(frozen=True)
class Registered:
    """`Core.Composition.registeredB id R s components`: the registration check ``registration`` on the registration's
    case first, a refusal refusing the whole, then ``product`` of the components on the rest and the caller's blobs,
    under the commitment's first 64 bytes; the rest of the commitment is the registration's digest, which the component
    ``committer`` must have committed to once the cases fit the composition (one registration digest in both)."""

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

    def check(self, p, c, e, blobs: list[bytes] = ()) -> tuple:
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
        answers = self.product.run(ps, ev, blobs)
        if isinstance(answers, str):
            return REJECT, answers
        # A component's answer carries its digest exactly when its commitment decodes and it doesn't reject before;
        # where `committedBy` has a digest the answer doesn't, the component rejects, and so does the composition.
        got = next((x["commitment"] for i, x in answers if i == self.committer), None)
        if got != d.hex():
            return REJECT, f"registration: {R.id} checked {d.hex()}, {self.committer} committed {got or 'no digest'}"
        return self.product.check(ps, b, ev, blobs)

    def answer(self, case: Any) -> dict:
        """`Core.BlobProtocol.case` of the registered composition: the outcome of one `serve` line."""
        try:
            case, blobs = Product.split(case)
        except Reject as e:
            return P.outcome(self.id, REJECT, f"case: {e}")
        return P.answer(self.id, case, params=self.params, commitment=self.commitment, evidence=self.evidence,
                        check=lambda p, c, e: self.check(p, c, e, blobs), digest=lambda c: c[0] + c[1])

    def verdicts(self, case: Any) -> list[str]:
        """The components' verdicts on a case whose components fit the composition, else none."""
        def answers():
            c, blobs = Product.split(case)
            _, ps = self.params(c["params"])
            _, ev = self.evidence(P.unhex(c["evidence"]))
            return self.product.run(ps, ev, blobs)
        return fitted_verdicts(answers)


#: `Core.Remote.protocol` and `Core.Onsite.protocol`, over their components in the order of their ids, onsite's after its
#: registration check, NCI's, on the registration integrity (v2) commits to.
REMOTE = Product("verity/remote/v0", (POUW, SAMPLED_PROOFS))
ONSITE = Registered("verity/onsite/v0", NCI_CHECK, SAMPLED_PROOFS.id,
                    Product("verity/onsite/v0", (CERTIFIER, POUS, POUW_PERIOD, BANDWIDTH, SAMPLED_PROOFS, WIPE)))
