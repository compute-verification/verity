"""The Python the composition's Lean (`CoreComposition`, `CoreOnsite`, `CoreRemote`) is held to: the composition rule
of onsite and remote, over the harness's framing of the core interface (`verity_difftest.protocol.answer`, as
`Core.Protocol.case` and `answer` decode a case), and each subprotocol's own Python: PoUW's `period_check.case`
(`verity/pouw/period/v1`, onsite's and remote's work, with `credit.commit_of` reading the record each window of a period
names), the network
certifier's period check's difftest's `reference` (`verity/certifier/v0`, its Python mirror over the certifier's
certificates and one-stage's answers, `verity/protocols/certifier/verifier/network/tests/test_network_period_protocol.py`), the
inference-only policy's difftest's `reference` (`verity/inference-only/v0`, the policy's Python check,
`nci_reference.check_baseline`, behind the protocol's decoders,
`verity/protocols/certifier/verifier/network/tests/test_inference_only_protocol.py`), for sampled proofs (`verity/sampled-proofs/v1`)
`spot`: its form's split over one-stage v2's audit (`verity/sampled-proofs/one-stage/v2`, one-stage v1
refused by name, `V1_REFUSED`), its decoders `OneStageZk` reads (behind the decoders `OneStage` reads, v1's difftest's,
`verity/protocols/spot/verifier/lean/tests/test_difftest.py`) and its check up to its one session's coin tree, and the
recursive form's params refused by name (`RECURSIVE_REFUSED`), the wipe's
(`verity/wipe/v0`) `wipe_protocol_case.answer`, and for space (`verity/pous/p2/v4/run`) and bandwidth (PoUB-P2,
`verity/remote-bandwidth/p2/v0`) their Python verifiers' answers on small periods of their hooks (`Recorded`,
`periods.py`). Each subprotocol's own difftest holds its Lean to that Python; this one holds the composition of them.

The trial's rule, onsite's and remote's over their subprotocols' statements:

* a product's fields are its subprotocols, and its verifier is the list of their verifiers' actors;
* an actor that raises records `Raised`, and what it would have established is missing: whatever reads it waits for it
  forever, so a missing statement fails closed;
* the product waits for each subprotocol's statement (`gather`): a statement with reasons explains nothing, and the epoch
  complies only with no reasons.

Reduced to answers: a subprotocol that accepts records a statement with no reasons, one that rejects a statement with
reasons, and one that is undecided no statement yet. `trial_verdict` is that reduction, with one change already made:
every subprotocol's reasons fail the product. Of the subprotocols' statements the trial's inference-only check fails the epoch on work's
reasons alone (integrity's reach it through them), and space's or communication's reasons only zero their explained usage
in the balance. The port follows the composition's rule, `verdict`, which changes the reduction in one place more:

* **any rejection rejects the whole**: in the trial a missing statement leaves the product with no verdict even when another subprotocol's
  statement has failed (`gather` waits on), so a prover could hide a rejection by leaving another subprotocol undecided;
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

from verity.protocols.pouw.verifier import core_check, credit, period_check
from verity.protocols.spot.device import challenger as CH
from verity.protocols.spot.verifier.one_stage import registration as REG

ACCEPT, REJECT, UNDECIDED = "accept", "reject", "undecided"

ROOT = Path(__file__).resolve().parents[4]


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


#: Sampled proofs' one-stage difftest: `answer` is `one_stage` behind the decoders `OneStage` reads, `decode_params_v2`
#: and `_challenger_draw` v2's parameters and its challenger's coins, and its strategies sample audits.
ONE_STAGE = module("one_stage_difftest", ROOT / "verity/protocols/spot/verifier/lean/tests/test_difftest.py")
#: PoUW's period difftest's cases: periods of the auditor's ledger and their windows, honest and with one mutation each.
POUW_PERIOD_CASES = module("pouw_period_cases", ROOT / "verity/protocols/pouw/verifier/lean/check/tests/pouw_period_cases.py")
#: The network certifier's period difftest: `reference` is the period check's Python mirror, and `build` makes a signed
#: period with one change.
PERIOD = module("network_period_difftest", ROOT / "verity/protocols/certifier/verifier/network/tests/test_network_period_protocol.py")
#: The inference-only policy's difftest: `reference` is the policy's Python check (`check_baseline`) behind the
#: protocol's decoders, and `drawn` samples a declaration's case with one change.
INFERENCE = module("inference_only_protocol_difftest", ROOT / "verity/protocols/certifier/verifier/network/tests/test_inference_only_protocol.py")
#: The wipe protocol in Python: `answer`, its mirror under S1's LMS, and `case`, a signed wipe aimed at a kind (`KINDS`).
WIPE_CASES = module("wipe_protocol_case", ROOT / "verity/protocols/pous/device/lean/tests/wipe_protocol_case.py")


def trial_verdict(verdicts: list[str]) -> str:
    """The trial's product: no verdict while any statement is missing, else it complies iff no statement has reasons."""
    if UNDECIDED in verdicts:
        return UNDECIDED
    return REJECT if REJECT in verdicts else ACCEPT


def verdict(verdicts: list[str]) -> str:
    """The composition's rule, which the port follows: any rejection rejects, then any undecided subprotocol leaves it undecided."""
    if REJECT in verdicts:
        return REJECT
    return UNDECIDED if UNDECIDED in verdicts else ACCEPT


def fitted_answers(answers: Callable[[], Any]) -> list[tuple[str, dict]]:
    """The subprotocols' answers in ``answers()``, else none when the case doesn't fit the composition."""
    try:
        out = answers()
    except (Reject, KeyError, TypeError):
        return []
    return [] if isinstance(out, str) else out


def fitted_verdicts(answers: Callable[[], Any]) -> list[str]:
    """The subprotocols' verdicts in ``answers()``, else none when the case doesn't fit the composition."""
    return [a["verdict"] for _, a in fitted_answers(answers)]


@dataclass(frozen=True)
class Ported:
    """A subprotocol whose port has replaced its stub: its id and its Python's answer to a `serve` case. One that reads
    blobs (`Core.BlobProtocol`) gets the product's under the case's ``blobs``, as bytes."""

    id: str
    answer: Callable[[Any], dict]
    blobs: bool = False


@dataclass(frozen=True)
class Recorded:
    """A subprotocol on the cases recorded for it: `cases` holds each kind's case and the answer its Python verifier gave
    it (`periods.build` fills it), and `answer` is that answer. The composition hands a subprotocol the case it was sampled,
    or one whose commitment is no longer JSON (`bad-hex` under a `0`), which both recorded subprotocols refuse with no
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


SAMPLED_PROOFS_V1 = "verity/sampled-proofs/v1"


def _form(j: Any, what: str) -> tuple[str, dict]:
    """`VStar.Recursive.Sampled`'s split: the object's ``form`` and the rest."""
    if not isinstance(j, dict):
        raise Reject(f"the {what} is not an object")
    if not isinstance(j.get("form"), str):
        raise Reject(f"the {what} names no form")
    return j["form"], {k: v for k, v in j.items() if k != "form"}


def _recursive(rest: dict, what: str):
    """The recursive form on a case that names no ``recursive`` (`VStar.Recursive.paramsOfJson`, `evidenceOf`), which
    it refuses; any other is outside the mirror."""
    if "recursive" in rest:
        raise ValueError(f"outside the mirror: the recursive audit's {what}")
    raise Reject(f"the {what} name no recursive")


#: `VStar.Recursive.Sampled.V1_REFUSED`: a one-stage statement naming no ``proofs`` is one-stage v1's.
V1_REFUSED = ("one-stage v1 (verity/sampled-proofs/one-stage/v1) is refused: SPOT's one-stage form is v2 "
              "(verity/sampled-proofs/one-stage/v2), whose verdicts are its own run of the verifier on the proofs")
#: `VStar.Recursive.Sampled.RECURSIVE_REFUSED`: the recursive form's params, refused before their rest is read.
RECURSIVE_REFUSED = ("the recursive audit (verity/sampled-proofs/recursive/v1) is refused: SPOT's soundness "
                     "covers one-stage v2 (verity/sampled-proofs/one-stage/v2) only")


def _keys(j: Any, keys: list[str], what: str) -> None:
    if not isinstance(j, dict) or sorted(j) != keys:
        raise Reject(f"the keys of {what} are not {', '.join(keys)}")


def _one_stage_params(rest: dict) -> dict:
    """`OneStageZk.paramsOfJson` behind `VStar.Recursive.Sampled`'s refusal of v1: ``proofs``, exactly the circuit and
    public files' SHA-512s and the verifier's registered file or null, then v1's params (`decode_params_v2`, which
    refuses them naming no challenger) with a partition object."""
    if "proofs" not in rest:
        raise Reject(f"the parameters name no proofs: {V1_REFUSED}")
    pr = rest["proofs"]
    _keys(pr, ["circuit", "public", "registered"], "proofs")
    for k in ("circuit", "public"):
        if not (isinstance(pr[k], str) and len(pr[k]) == 128 and all(c in "0123456789abcdef" for c in pr[k])):
            raise Reject(f"proofs: {k} is not a SHA-512 digest (128 lowercase hex)")
    if pr["registered"] is not None and not isinstance(pr["registered"], dict):
        raise Reject("proofs: registered is not the verifier's registered file (an object) or null")
    p = ONE_STAGE.decode_params_v2(rest)
    if p["partition"] is None:
        raise Reject("partition is null: the sessions are verified against the verifier's own partition object")
    return {**p, "proofs": pr}


def _one_stage_evidence(rest: dict) -> dict:
    """`OneStageZk.evidenceOf` behind `VStar.Recursive.Sampled`'s refusal of v1: no ``verdicts``, v1's evidence with
    none (`covered_evidence`, `OneStage.evidenceOf`), and ``proofs``, exactly the sessions: for each of v1's sessions in
    order, its name, its record's digest and its rep files' digests. The public file is the parameters'."""
    if "proofs" not in rest:
        raise Reject(f"the evidence names no proofs: {V1_REFUSED}")
    if "verdicts" in rest:
        raise Reject("the evidence names verdicts: this audit's verdicts are its own run of the verifier on the proofs")
    base = ONE_STAGE.covered_evidence(canonical({**{k: v for k, v in rest.items() if k != "proofs"},
                                                 "verdicts": {}}).encode())
    pr = rest["proofs"]
    _keys(pr, ["sessions"], "the evidence's proofs")
    if not isinstance(pr["sessions"], list):
        raise Reject("proofs: sessions is not a list")
    for s in pr["sessions"]:
        _keys(s, ["name", "record", "reps"], "a session of the proofs")
        if not (isinstance(s["name"], str) and isinstance(s["record"], str) and isinstance(s["reps"], dict)
                and all(isinstance(d, str) for d in s["reps"].values())):
            raise Reject("a session of the proofs is not a name, its record's digest and its rep files' digests")
    if [s["name"] for s in pr["sessions"]] != base["sessions"]:
        raise Reject("proofs: the sessions are not the evidence's sessions, in order")
    return {**base, "proofs": pr}


def _one_stage_check(p: dict, c: tuple[dict, bytes], e: dict, blobs: list[bytes]) -> tuple:
    """`OneStageZk.checkBlobs` on `blobs` up to its one session's coin tree: at the coins v1 reads from the challenger's
    release (`_coins_of`), the evidence's one session (`oneSession`: the sampled proof blocks are proved in one session), its
    record on the blobs and held to the coin trees that release log shows (`challenger.sessions_tree_reason`). So every
    case the mirror decides is refused: a session on a tree the challenger logged goes on to the population, the public
    file and `flock-verify`, outside the mirror."""
    try:
        if ONE_STAGE._coins_of(p, c, e) is None:
            return REJECT, "coins: no challenger"
    except (CH.CoinError, KeyError, TypeError):
        return REJECT, "coins"
    sessions = e["proofs"]["sessions"]
    if len(sessions) != 1:
        return REJECT, f"proofs: the evidence names {len(sessions)} sessions: the drawn units are proved in one session"
    try:
        trees = CH.coin_trees_of(e["coins"]["session"]["log"])
    except (CH.CoinError, KeyError, TypeError):
        return REJECT, "coins"
    (s,) = sessions
    held = {hashlib.sha512(b).hexdigest(): b for b in blobs}
    try:
        held[s["record"]].decode()
    except KeyError:
        return REJECT, f"proofs: session {s['name']}'s record: no blob has its SHA-512"
    except UnicodeDecodeError:
        return REJECT, f"proofs: session {s['name']}'s record is not UTF-8"
    if trees:
        raise ValueError("outside the mirror: a session on the challenger's coin trees")
    return REJECT, f"proofs: {CH.sessions_tree_reason(trees, [(s['name'], {'circuit': None}, None)])}"


def _form_params(j: Any) -> dict:
    form, rest = _form(j, "parameters")
    if form == "one-stage":
        return _one_stage_params(rest)
    if form == "recursive":
        raise Reject(RECURSIVE_REFUSED)
    raise Reject(f"the parameters' form {form} is not one-stage or recursive")


def _covered(form_params: Callable[[Any], dict]) -> Callable[[Any], dict]:
    """`OneStage.paramsOfJson` behind the form's split: `form_params`, and every bernoulli law refused at any p, as
    `covered_params` refuses it (`service.bernoulli`)."""
    def params(j: Any) -> dict:
        p = form_params(j)
        if ONE_STAGE.service.bernoulli(p["law"]):
            raise Reject(ONE_STAGE.service.BERNOULLI)
        return p
    return params


_form_params = _covered(_form_params)


def _form_evidence(b: bytes) -> dict:
    form, rest = _form(P.canonical_json(b), "evidence")
    if form == "one-stage":
        return _one_stage_evidence(rest)
    if form == "recursive":
        _recursive(rest, "evidence")
    raise Reject(f"the evidence's form {form} is not one-stage or recursive")


def spot(case: Any) -> dict:
    """`VStar.Recursive.Sampled.protocol` on a case and its ``blobs``: one-stage v2 (`Spot.OneStageZk`,
    `_one_stage_check`, at the coins the challenger's release gives) when its params and evidence name the form
    ``one-stage`` and ``proofs``, a refusal of one-stage v1 (naming no ``proofs``) and of the recursive form's params
    (`RECURSIVE_REFUSED`) by name, and of any other form on the cases the mirror decides."""
    blobs = case.get("blobs", []) if isinstance(case, dict) else []
    return P.answer(SAMPLED_PROOFS_V1, case, params=_form_params, commitment=ONE_STAGE.decode_commitment,
                    evidence=_form_evidence, digest=lambda c: c[1],
                    check=lambda p, c, e: _one_stage_check(p, c, e, blobs))


CERTIFIER = Ported("verity/certifier/v0", PERIOD.reference)
INFERENCE_ONLY = Ported("verity/inference-only/v0", INFERENCE.reference)


@functools.cache
def _period_answer(case: str) -> str:
    return canonical(period_check.case(json.loads(case)))


#: A period's answer is slow to make (its windows' ledger signatures and layouts), and a composition's case asks for each
#: subprotocol's twice, so it is made once a case.
POUW_PERIOD = Ported(period_check.ID, lambda case: json.loads(_period_answer(canonical(case))))
#: Integrity: SPOT (`VStar.Recursive.Sampled`), one-stage v2's audit (`Spot.OneStageZk`); one-stage v1 and the
#: recursive audit refused.
SAMPLED_PROOFS = Ported(SAMPLED_PROOFS_V1, spot, blobs=True)
#: Space and bandwidth, `PousCheck.RunProtocol.protocol` and `RemoteAudit.Protocol.protocol`.
POUS = Recorded("verity/pous/p2/v4/run")
#: Pearl+'s run verdict, remote's work in its composite of record: on the served run a test records for it
#: (`pearl_plus_answer`), and else it refuses a commitment that isn't JSON, as Pearl+'s decoder does, and has no answer
#: to any other case
PEARL_PLUS = Recorded("verity/pouw/pearl-plus/v1")
#: `PouwCheck.PearlPlus.Protocol.commitmentTag`
PEARL_PLUS_TAG = "verity/pouw/pearl-plus/manifests/v1"


def pearl_plus_answer(commitment: bytes, verified: dict) -> dict:
    """Pearl+'s outcome on a served run whose Python verdict ``served_passes.py verify`` recorded (`verified`, its
    ``verify.json``), framed as `PouwCheck.PearlPlus.Protocol` frames the check's: its digest SHA-512(`PEARL_PLUS_TAG` ‖
    0 ‖ the commitment's bytes), and on acceptance its public the credit at δ (`publicJson`) and its profile the draws
    asked for, the tiles drawn and the passes (`profileJson`)."""
    a, digest = verified["answer"], hashlib.sha512(PEARL_PLUS_TAG.encode() + b"\0" + commitment).digest()
    if a["verdict"] != ACCEPT:
        return P.outcome(PEARL_PLUS.id, a["verdict"], a["reason"], digest)
    w, pr = a["public"]["work"], a["profile"]
    public = {"credited": w["credited"], "credited_tiles": w["credited_tiles"], "delta": pr["delta"],
              "epsilon": w["epsilon"], "macs": w["macs"], "rho": w["rho"]}
    profile = {"draws": verified["draws"], "drawn": len(pr["tiles"]) - pr["exclusion_tiles"],
               "passes": len(a["public"]["passes"])}
    return P.outcome(PEARL_PLUS.id, ACCEPT, None, digest, public, profile)


BANDWIDTH = Recorded("verity/remote-bandwidth/p2/v0")
#: The wipe, `MemoryChallenger.WipeProtocol.protocol`.
WIPE = Ported("verity/wipe/v0", WIPE_CASES.answer)


def bind(id: str, digests: list[tuple[str, bytes]]) -> bytes:
    """`Core.Composition.bind`: SHA-512 over the composition's id, a zero byte and the canonical JSON of the subprotocols'
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
        """`Core.Composition.run`: each subprotocol's outcome on its case, those that read blobs on the caller's `blobs`
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
        """`Core.Composition.binding id answers = some c`: every subprotocol's digest is there, and they bind to `c`."""
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
        """The subprotocols' verdicts on a case that fits the composition, else none."""
        def answers():
            c, blobs = self.split(case)
            return self.run(self.params(c["params"]), self.evidence(P.unhex(c["evidence"])), blobs)
        return fitted_verdicts(answers)


@dataclass(frozen=True)
class Same(Product):
    """`Core.Composition.sameB id ss components`: `product`, except that once the cases fit, the subprotocols ``same``
    (each committing to the run's record) must all have committed to the first one's digest."""

    same: tuple = ()

    def strays(self, answers: list[tuple[str, dict]]) -> str | None:
        """`Core.Composition.strays`. A subprotocol's answer carries its digest exactly when its commitment decodes and it
        doesn't reject before; where `committedBy` has a digest the answer doesn't, the subprotocol rejects, and so does
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
        """The digests the subprotocols ``same`` committed to on a case that fits the composition, else none."""
        def answers():
            c, blobs = self.split(case)
            return self.run(self.params(c["params"]), self.evidence(P.unhex(c["evidence"])), blobs)
        return [a["commitment"] for i, a in fitted_answers(answers) if i in self.same]


@dataclass(frozen=True)
class Tie:
    """`Core.Composition.Tie`: every value that each subprotocol of ``names`` names must be one value, ``what``; each
    reads the values it names off its case's commitment and evidence bytes with its function (None: its case doesn't
    decode)."""

    what: str
    names: tuple = ()

    def values(self, ev: list) -> list[tuple[str, bytes]] | None:
        """Each value the subprotocols of ``names`` name on the cases ``ev``, in their order; None when one of them has no
        case or names no list of values (its case doesn't decode, where it rejects as a subprotocol)."""
        cases: dict = {}
        for k in ev:
            cases.setdefault(k[0], k)
        out: list[tuple[str, bytes]] = []
        for s, f in self.names:
            ds = f(cases[s][2], cases[s][3]) if s in cases else None
            if not isinstance(ds, list):
                return None
            out += [(s, d) for d in ds]
        return out

    def strays(self, ev: list) -> str | None:
        """`Core.Composition.Tie.strays`: the first value that isn't the first one, with both; none when they all named
        it, or when a subprotocol names none."""
        got = self.values(ev)
        if not got:
            return None
        (s, d), *rest = got
        return next((f"{self.what}: {s} committed {d.hex()}, {t} committed {e.hex()}" for t, e in rest if e != d), None)


def _commitment_hex(key: str) -> Callable[[bytes, bytes], list[bytes]]:
    """A subprotocol's function naming one value: the hex field ``key`` of its JSON commitment."""
    return lambda c, e: [bytes.fromhex(json.loads(c)[key])]


def policy_record(commitment: bytes, evidence: bytes) -> list[bytes] | Exception:
    """`Core.Remote.policyRecord`: the digest of the registration record the policy's commitment is, as the policy
    decodes it; else what refused the commitment, which reads as no value."""
    try:
        return [bytes.fromhex(INFERENCE.commitment(commitment)[1])]
    except (Reject, ValueError, KeyError, TypeError) as e:
        return e


def window_records(commitment: bytes, evidence: bytes) -> list[bytes] | core_check.Refused | None:
    """`Core.Remote.windowRecords`: the digests of each window's Commit record (`credit.commit_of`) in a period's
    evidence, in the close's order; else the period's refusal of the evidence, or None when a window names no Commit
    record (which the period refuses), either of which reads as no value."""
    try:
        j = period_check.evidence(evidence)
    except core_check.Refused as e:
        return e
    if not isinstance(j["windows"], list):
        return None
    out = []
    for w in j["windows"]:
        got, _ = credit.commit_of(w.get("commit") if isinstance(w, dict) else None)
        if got is None:
            return None
        out.append(got[0])
    return out


def _nats(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in x)


def pearl_commitment(commitment: bytes) -> tuple[bytes, bytes] | Reject | None:
    """The registration record's digest and the window record W's digest Pearl+'s commitment names, as Pearl+ decodes
    it (`PouwCheck.PearlPlus.Protocol.commitmentOf`: exactly ``manifests``, a nonempty list, ``record``, a SHA-512
    digest in lowercase hex, and ``window``, exactly ``dropped``, a list of nonnegative integers, ``record``, W, an
    object, and ``rounds``, null or such a list); else what refused it, or None when it doesn't decode."""
    try:
        j = P.canonical_json(commitment)
        if not isinstance(j, dict) or sorted(j) != ["manifests", "record", "window"]:
            return None
        if not isinstance(j["manifests"], list) or not j["manifests"] or not isinstance(j["record"], str):
            return None
        record = P.unhex(j["record"])
    except Reject as e:
        return e
    w = j["window"]
    if len(record) != 64 or not isinstance(w, dict) or sorted(w) != ["dropped", "record", "rounds"]:
        return None
    if not isinstance(w["record"], dict) or not (w["rounds"] is None or _nats(w["rounds"])) or not _nats(w["dropped"]):
        return None
    return record, bytes.fromhex(REG.digest(w["record"]))


def run_record(commitment: bytes, evidence: bytes) -> list[bytes] | Reject | None:
    """`Core.Remote.V1.runRecord`: the digest of the registration record Pearl+'s run is on, its commitment's
    ``record``, as Pearl+ decodes the commitment (`pearl_commitment`); else what refused it, or None when it doesn't
    decode, either of which reads as no value."""
    got = pearl_commitment(commitment)
    return [got[0]] if isinstance(got, tuple) else got


def run_window(commitment: bytes, evidence: bytes) -> list[bytes] | Reject | None:
    """`Core.Remote.V1.runWindow`: the digest of the window record W Pearl+'s check holds the run's passes to, as
    Pearl+ decodes its commitment (`pearl_commitment`); else what refused it, or None, either of which reads as no
    value."""
    got = pearl_commitment(commitment)
    return [got[1]] if isinstance(got, tuple) else got


def policy_previous(commitment: bytes, evidence: bytes) -> list[bytes] | Exception:
    """`Core.Remote.V1.policyPrevious`: the record the policy's record is chained right after, its ``previous``, as the
    policy decodes its record; the empty bytes when it names none (no record's digest, so a stray); else what refused
    the commitment, which reads as no value."""
    try:
        record, _ = INFERENCE.commitment(commitment)
    except (Reject, ValueError, KeyError, TypeError) as e:
        return e
    prev = record.get("previous")
    try:
        return [P.unhex(prev) if isinstance(prev, str) else b""]
    except Reject:
        return [b""]


#: `Core.Onsite.store`: PoUS's setup's root of `C*` and the root PoUB-P2's registration names (`p2_case.store_root`).
STORE = Tie("store", (("verity/pous/p2/v4/run", _commitment_hex("root")),
                      ("verity/remote-bandwidth/p2/v0", _commitment_hex("root"))))
#: `Core.Remote.windows`: every window of PoUW's period on the policy's record, remote's tie and onsite's.
WINDOWS = Tie("record", ((INFERENCE_ONLY.id, policy_record), (POUW_PERIOD.id, window_records)))
#: `Core.Remote.V1.run`: Pearl+'s run on the policy's record.
RUN = Tie("record", ((INFERENCE_ONLY.id, policy_record), (PEARL_PLUS.id, run_record)))
#: `Core.Remote.V1.window`: Pearl+'s passes on the window record the policy's record is chained right after.
WINDOW = Tie("window", ((INFERENCE_ONLY.id, policy_previous), (PEARL_PLUS.id, run_window)))


@dataclass(frozen=True)
class TiedSame(Same):
    """`Core.Composition.tiedSameB id ss ties components`: `same`, except that when it doesn't reject, the subprotocols
    of a tie of ``ties`` that didn't all commit to one value reject (`holdTies`)."""

    ties: tuple = ()

    def check(self, ps, c: bytes, ev, blobs: list[bytes] = ()) -> tuple:
        answers = self.run(ps, ev, blobs)
        if isinstance(answers, str):
            return REJECT, answers
        if stray := self.strays(answers):
            return REJECT, stray
        out = self.judge(answers, c)
        if out[0] != REJECT and (stray := next(filter(None, (t.strays(ev) for t in self.ties)), None)):
            return REJECT, stray
        return out

    def tied(self, case: Any) -> list[tuple[str, bool]]:
        """On a case whose subprotocols fit the composition and whose tied subprotocols' commitments all decode, each tie
        by what it ties, and whether they committed to one value."""
        ev: list = []

        def answers():
            c, blobs = self.split(case)
            ev[:] = self.evidence(P.unhex(c["evidence"]))
            return self.run(self.params(c["params"]), ev, blobs)
        digests = dict(fitted_answers(answers))
        return [(t.what, len({d for _, d in got}) <= 1) for t in self.ties
                if all(digests.get(s, {}).get("commitment") is not None for s, _ in t.names)
                and (got := t.values(ev)) is not None]


#: `Core.Remote.V1.protocol`, `Core.Remote.protocol` and `Core.Onsite.protocol`, over their subprotocols in the order
#: of their ids: each one's policy and integrity committing to one registration record, remote's composite of record's
#: Pearl+ run naming the policy's record, every window of its development composite's PoUW period and of onsite's naming
#: it too, and onsite's PoUS and PoUB-P2 on one store (`Core.Onsite.ties`, in its order).
REMOTE_V1 = TiedSame("verity/remote/v1", (INFERENCE_ONLY, PEARL_PLUS, SAMPLED_PROOFS),
                     (INFERENCE_ONLY.id, SAMPLED_PROOFS.id), (RUN, WINDOW))
REMOTE = TiedSame("verity/remote/v0", (INFERENCE_ONLY, POUW_PERIOD, SAMPLED_PROOFS), (INFERENCE_ONLY.id, SAMPLED_PROOFS.id),
                  (WINDOWS,))
ONSITE = TiedSame("verity/onsite/v0", (CERTIFIER, INFERENCE_ONLY, POUS, POUW_PERIOD, BANDWIDTH, SAMPLED_PROOFS, WIPE),
                  (INFERENCE_ONLY.id, SAMPLED_PROOFS.id), (WINDOWS, STORE))
