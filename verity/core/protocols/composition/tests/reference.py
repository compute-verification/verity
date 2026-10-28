"""The Python the composition's Lean (`CoreComposition`, `CoreOnsite`, `CoreRemote`) is held to: the composition rule
of onsite and remote, over the harness's framing of the core interface (`verity_difftest.protocol.answer`, as
`Core.Protocol.case` and `answer` decode a case), and each subprotocol's own Python: PoUW's `core_check.case`
(`verity/pouw/v1`, remote's work) and `period_check.case` (`verity/pouw/period/v1`, onsite's work), the network
certifier's period check's difftest's `reference` (`verity/certifier/v0`, its Python mirror over the certifier's
certificates and one-stage v2 on each sampling block,
`verity/core/protocols/network/tests/test_network_period_protocol.py`), the
inference-only policy's difftest's `reference` (`verity/inference-only/v0`, the policy's Python check,
`nci_reference.check_baseline`, behind the protocol's decoders,
`verity/core/protocols/network/tests/test_inference_only_protocol.py`), for sampled proofs' one-stage audit on the
proofs themselves (`verity/sampled-proofs/one-stage/v2`) `one_stage_zk_case.answer`: v2's framing over v1's `one_stage`
and decoders (`verity/core/service/sampled_proofs/tests/`), on the cases v2 decides without a proof, the wipe's
(`verity/wipe/v0`) `wipe_protocol_case.answer`, and for space (`verity/pous/p2/v4/run`) and bandwidth
(`verity/remote-bandwidth/v0`) their Python verifiers' answers on small periods of their hooks (`Recorded`,
`periods.py`). Each subprotocol's own difftest holds its Lean to that Python; this one holds the composition of them.

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


#: One-stage v2 in Python on the cases it decides without a proof (`answer`), over sampled proofs' one-stage difftest
#: (`ONE_STAGE`: `one_stage` behind the decoders `OneStage` reads, whose strategies draw audits).
ZK = module("one_stage_zk_case", CORE / "service/sampled_proofs/tests/one_stage_zk_case.py")
ONE_STAGE = ZK.ONE_STAGE
#: PoUW's check difftest's cases: honest windows and one mutation each.
POUW_CASES = module("pouw_check_cases", CORE / "protocols/pouw/lean/check/tests/pouw_check_cases.py")
#: PoUW's period difftest's cases: periods of the auditor's ledger and their windows, honest and with one mutation each.
POUW_PERIOD_CASES = module("pouw_period_cases", CORE / "protocols/pouw/lean/check/tests/pouw_period_cases.py")
#: The network certifier's period difftest: `reference` is the period check's Python mirror, and `build` makes a signed
#: period with one change.
PERIOD = module("network_period_difftest", CORE / "protocols/network/tests/test_network_period_protocol.py")
#: The inference-only policy's difftest: `reference` is the policy's Python check (`check_baseline`) behind the
#: protocol's decoders, and `drawn` draws a declaration's case with one change.
INFERENCE = module("inference_only_protocol_difftest", CORE / "protocols/network/tests/test_inference_only_protocol.py")
#: The wipe protocol in Python: `answer`, its mirror under S1's LMS, and `case`, a signed wipe aimed at a kind (`KINDS`).
WIPE_CASES = module("wipe_protocol_case", CORE / "devices/memory_challenger/lean/tests/wipe_protocol_case.py")


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


def fitted_answers(answers: Callable[[], Any]) -> list[tuple[str, dict]]:
    """The components' answers in ``answers()``, else none when the case doesn't fit the composition."""
    try:
        out = answers()
    except (Reject, KeyError, TypeError):
        return []
    return [] if isinstance(out, str) else out


def fitted_verdicts(answers: Callable[[], Any]) -> list[str]:
    """The components' verdicts in ``answers()``, else none when the case doesn't fit the composition."""
    return [a["verdict"] for _, a in fitted_answers(answers)]


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


#: Communication: the period check, which runs one-stage v2 on each sampling block's proof on the product's blobs.
CERTIFIER = Ported("verity/certifier/v0", PERIOD.reference, blobs=True)
INFERENCE_ONLY = Ported("verity/inference-only/v0", INFERENCE.reference)
POUW = Ported("verity/pouw/v1", core_check.case)


@functools.cache
def _period_answer(case: str) -> str:
    return canonical(period_check.case(json.loads(case)))


#: A period's answer is slow to make (its windows' ledger signatures and layouts), and a composition's case asks for each
#: component's twice, so it is made once a case.
POUW_PERIOD = Ported(period_check.ID, lambda case: json.loads(_period_answer(canonical(case))))
#: Integrity: one-stage v2, the audit on the proofs themselves (`SampledProofs.OneStageZk`).
SAMPLED_PROOFS = Ported(ZK.ID, ZK.answer, blobs=True)
#: Space and bandwidth, `PousCheck.RunProtocol.protocol` and `RemoteAudit.Protocol.protocol`.
POUS = Recorded("verity/pous/p2/v4/run")
BANDWIDTH = Recorded("verity/remote-bandwidth/v0")
#: The wipe, `MemoryChallenger.WipeProtocol.protocol`.
WIPE = Ported("verity/wipe/v0", WIPE_CASES.answer)


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
        return self.judge(answers, c)

    def judge(self, answers: list[tuple[str, dict]], c: bytes) -> tuple:
        """`Core.Composition.judge`: reject first, then the binding, then undecided."""
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
class Same(Product):
    """`Core.Composition.sameB id ss components`: `product`, except that once the cases fit, the components ``same``
    (each committing to the run's record) must all have committed to the first one's digest."""

    same: tuple = ()

    def strays(self, answers: list[tuple[str, dict]]) -> str | None:
        """`Core.Composition.strays`. A component's answer carries its digest exactly when its commitment decodes and it
        doesn't reject before; where `committedBy` has a digest the answer doesn't, the component rejects, and so does
        the composition."""
        got = [(i, a["commitment"]) for i, a in answers if i in self.same]
        if not got or any(d is None for _, d in got):
            return None
        (s, d), rest = got[0], got[1:]
        return next((f"record: {s} committed {d}, {t} committed {e}" for t, e in rest if e != d), None)

    def check(self, ps, c: bytes, ev, blobs: list[bytes] = ()) -> tuple:
        answers = self.run(ps, ev, blobs)
        if isinstance(answers, str):
            return REJECT, answers
        if stray := self.strays(answers):
            return REJECT, stray
        return self.judge(answers, c)

    def committed(self, case: Any) -> list[str | None]:
        """The digests the components ``same`` committed to on a case that fits the composition, else none."""
        def answers():
            c, blobs = self.split(case)
            return self.run(self.params(c["params"]), self.evidence(P.unhex(c["evidence"])), blobs)
        return [a["commitment"] for i, a in fitted_answers(answers) if i in self.same]


#: `Core.Remote.protocol` and `Core.Onsite.protocol`, over their subprotocols in the order of their ids: remote's three
#: committing to one window record, onsite's policy and integrity (v2) to one registration record.
REMOTE = Same("verity/remote/v0", (INFERENCE_ONLY, POUW, SAMPLED_PROOFS), (INFERENCE_ONLY.id, POUW.id, SAMPLED_PROOFS.id))
ONSITE = Same("verity/onsite/v0", (CERTIFIER, INFERENCE_ONLY, POUS, POUW_PERIOD, BANDWIDTH, SAMPLED_PROOFS, WIPE),
              (INFERENCE_ONLY.id, SAMPLED_PROOFS.id))
