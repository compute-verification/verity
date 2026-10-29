"""The Python the composition's Lean (`CoreComposition`, `CoreOnsite`, `CoreRemote`) is held to: the composition rule
of onsite and remote, over the harness's framing of the core interface (`verity_difftest.protocol.answer`, as
`Core.Protocol.case` and `answer` decode a case), and each component's own Python: PoUW's `core_check.case`
(`verity/pouw/v1`), the network certifier's period check's difftest's `reference` (`verity/certifier/v0`, its Python
mirror over the certifier's certificates and one-stage's answers,
`verity/core/protocols/network/tests/test_network_period_protocol.py`), for sampled proofs' one-stage audit
(`verity/sampled-proofs/one-stage/v1`) `one_stage` behind the protocol's decoders, its difftest's `answer`
(`verity/core/service/sampled_proofs/tests/test_difftest.py`), and for space (`verity/pous/p2/v4/run`) and bandwidth
(`verity/poub/v0`) their Python verifiers' answers on small periods of their hooks (`Recorded`,
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
POUW = Ported("verity/pouw/v1", core_check.case)
SAMPLED_PROOFS = Ported("verity/sampled-proofs/one-stage/v1", ONE_STAGE.answer)
#: Space and bandwidth, `PousCheck.RunProtocol.protocol` and `PoubCheck.Protocol.protocol`.
POUS = Recorded("verity/pous/p2/v4/run")
BANDWIDTH = Recorded("verity/poub/v0")


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


#: `Core.Remote.protocol` and `Core.Onsite.protocol`, over their components in the order of their ids.
REMOTE = Product("verity/remote/v0", (POUW, SAMPLED_PROOFS))
ONSITE = Product("verity/onsite/v0", (CERTIFIER, POUS, POUW, BANDWIDTH, SAMPLED_PROOFS))
