"""The wipe protocol (`MemoryChallenger.WipeProtocol`, `verity/wipe/v0`) in Python, for every difftest that runs it: its
mirror, `answer` (and `reference(id, schemes)`), and its cases, `build`, `case` and `cases`.

`reference(id, schemes)(case)` is the outcome of one `serve` line in the core interface's answer format
(`verity_difftest.protocol.answer`): it decodes the protocol's formats as `MemoryChallenger/WipeProtocol.lean` does, each
input exactly the canonical JSON of what it decodes to (the commitment and the signed session as `wipe_cases` decodes
them), and checks with the period hook's own Python, `wipe_period.outcome` over `streamed_wipe.verdict`. Its verdict,
public, profile and digest are the Lean's exactly, and past decoding its reasons are the Lean's word for word. `answer`
is `verity/wipe/v0`'s, under S1's LMS; `test_answer` the test signers' protocol's.

`build(m, seed)` makes one wipe with the change `m` (`EXPECT`), signed by the memory challenger's tests' key at S1's
deployed parameters (`mc_cases.deployed_signer`) so that `verity/wipe/v0` itself takes it: an honest wipe at a small fill
and `READS` reads, one 32 MiB segment at `READS` reads and one at `REMOTE_READS` with a MiB declared past the fill (each
accepted and certified at `StreamedWipeMeetsReads`' threshold for its `k`, its egress counted), a prover keeping one
label past that threshold of foreign content, a read past its deadline, a session the prover left before it ended
(unsigned), a commitment of another server or session, an untrusted key, a record edited after it was signed, and
malformed params, commitment and evidence. `case(kind, seed)` is a wipe aimed at a verdict or at a decoding failure
(`KINDS`), and `cases(seed)` one of each with its answer, as a composition's difftest takes a real component's cases.

This is test-side Python (it imports `verity_difftest` and the memory challenger's tests' cases), so it lives beside the
wipe protocol's difftest (`test_wipe_protocol.py`), and other suites load it by path."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mc_cases as M
import verity_difftest as D
import wipe_cases as WC
from verity_difftest import protocol as P
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw
from verity.core.protocols.pous import wipe_period as WP

ID = WP.PROTOCOL
TEST_ID = mc.TEST_PREFIX + ID
ACCEPT, REJECT, UNDECIDED = "accept", "reject", "undecided"
SERVER = b"vy-nebius-cpu-4"
REMOTE_READS = 5231
"""The reads of `StreamedWipeMeets141GB5`, the H200's wipe at 5 GB (`r = 59`)."""

# --- the mirror --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    server: bytes
    session: bytes
    trusted: tuple[bytes, ...]

    def to_json(self) -> dict:
        return WP.params_json(self.server, self.session, self.trusted)


def _exactly(decode: Callable[[Any], Any], enc: Callable[[Any], Any], j: Any) -> Any:
    a = decode(j)
    if D.canonical(enc(a)) != D.canonical(j):
        raise D.Reject("not the canonical JSON of what it holds (a field extra, or a constant not the protocol's)")
    return a


def decode_params(j: Any) -> Params:
    return _exactly(lambda j: Params(M.hex_(j, "server"), M.hex_(j, "session"), tuple(M.hexes(j, "trusted"))),
                    Params.to_json, j)


def decode_commitment(b: bytes) -> sw.WipeCommitment:
    return _exactly(WC.commitment_of, sw.WipeCommitment.to_json, P.canonical_json(b))


def _evidence_json(s: sw.SignedWipe | None) -> dict:
    return {"signed": None if s is None else s.to_json()}


def decode_evidence(b: bytes) -> sw.SignedWipe | None:
    def of(j: Any) -> sw.SignedWipe | None:
        s = M._get(j, "signed")
        return None if s is None else WC.signed_of(s)
    return _exactly(of, _evidence_json, P.canonical_json(b))


def digest(c: sw.WipeCommitment) -> bytes:
    return hashlib.sha512(D.canonical(c.to_json()).encode()).digest()


def reference(protocol_id: str, schemes) -> Callable[[Any], dict]:
    """`MemoryChallenger.WipeProtocol.mkProtocol protocol_id schemes` in Python."""

    def check(p: Params, c: sw.WipeCommitment, signed: sw.SignedWipe | None) -> tuple:
        return WP.outcome(p.server, p.session, p.trusted, c, signed, schemes)

    return lambda case: P.answer(protocol_id, case, params=decode_params, commitment=decode_commitment,
                                 evidence=decode_evidence, check=check, digest=digest)


answer = reference(ID, mc.SCHEMES)
test_answer = reference(TEST_ID, M.ts.SCHEMES)

# --- wipes -------------------------------------------------------------------------------------------------------------


def wipe(params: sw.WipeParams, signer: M.AtLeaf, *, prover: Callable[[sw.WipeParams], sw.Prover] = sw.Prover,
         spends: Callable[[int], int] = lambda b: 1, stop_after: int | None = None, egress_bits: int = 1234,
         coins: bytes = b"coins", session_id: bytes = bytes(32), server: bytes = SERVER,
         start: int = 1_000) -> tuple[sw.WipeCommitment, sw.SignedWipe | None]:
    """One wipe of `Challenger` against `prover(params)` on an injected clock: read `b` takes `spends(b)` ns, and with
    `stop_after` the prover leaves after that many reads, so the session is never signed."""
    clock = M.Clock(start)
    c = sw.Challenger(WC.SECRET, server, session_id, params, signer, log=M.ts.session_log(), clock=clock,
                      coin_secret=coins)
    p = prover(params)
    for first, data in c.fill(max(1, params.labels // 3)):
        p.receive(first, data)
        clock.now += 7
    while c.ended is None and (stop_after is None or len(c.batches) < stop_after):
        idx = c.challenge()
        clock.now += spends(len(c.batches))
        c.receive(p.answer(idx))
    return c.commitment, None if c.ended is None else c.close(egress_bits)


def wipe_case(commitment: sw.WipeCommitment, signed: sw.SignedWipe | None, trusted: tuple[bytes, ...], *,
              server: bytes | None = None, session: bytes | None = None) -> dict:
    """The period hook's case for a wipe (`wipe_period.case`), the params' server and session the commitment's unless
    given."""
    return WP.case(commitment.server if server is None else server,
                   commitment.session if session is None else session, trusted, commitment, signed)


EXPECT = {
    "honest": ACCEPT, "segment": ACCEPT, "reads": ACCEPT,
    "foreign": REJECT, "late": REJECT, "other-server": REJECT, "other-session": REJECT, "untrusted": REJECT,
    "tampered": REJECT,
    "unsigned": UNDECIDED,
    "params": REJECT, "commitment": REJECT, "evidence": REJECT,
}


def _small(r: random.Random) -> sw.WipeParams:
    labels = r.randint(16, 96)
    return sw.WipeParams(labels, WP.READS, 1, r.choice([1_000, 10 ** 6]),
                         WC.tiers_of(labels, [r.randint(1, labels - 1)], ["gpu", "host-dram"]))


@lru_cache(maxsize=None)
def _build(m: str, seed: int) -> dict:
    r = random.Random(f"wipe-protocol {m} {seed}")
    signer = M.deployed_signer()
    trusted = (signer.public_key,)
    params = {"segment": WP.params_for(),
              "reads": WP.params_for(k=REMOTE_READS, tier_bytes=WP.SEGMENT_BYTES + (1 << 20))}.get(m) or _small(r)
    kw: dict[str, Any] = {"coins": r.randbytes(16), "session_id": r.randbytes(32), "egress_bits": r.randrange(1, 4 << 23)}
    if m == "foreign":
        kw["prover"] = WP.keeping(WP.foreign_labels("fM", params))
    elif m == "late":
        at = r.randrange(params.k)
        kw["spends"] = lambda b: params.limit_ns + 1 if b == at else 1
    elif m == "unsigned":
        kw["stop_after"] = r.randrange(params.k)
    commitment, signed = wipe(params, signer, **kw)
    if m == "other-server":
        return wipe_case(commitment, signed, trusted, server=b"another node")
    if m == "other-session":
        return wipe_case(commitment, signed, trusted, session=M._flip(commitment.session, r.randrange(256)))
    if m == "untrusted":
        return wipe_case(commitment, signed, (M.test_signer(M.STRANGER, 1, 0).public_key,))
    if m == "tampered":
        assert signed is not None
        signed = dataclasses.replace(signed, egress_bits=signed.egress_bits + 1)
    c = wipe_case(commitment, signed, trusted)
    if m == "params":
        c["params"] = [c["params"] | {"extra": 1}, {k: v for k, v in c["params"].items() if k != "trusted"},
                       c["params"] | {"server": c["params"]["server"].upper()}][r.randrange(3)]
    elif m == "commitment":
        j = json.loads(bytes.fromhex(c["commitment"]))
        c["commitment"] = [json.dumps(j).encode().hex(), D.canonical(j | {"labels": 0}).encode().hex(),
                           D.canonical({k: v for k, v in j.items() if k != "clock"}).encode().hex()][r.randrange(3)]
    elif m == "evidence":
        j = json.loads(bytes.fromhex(c["evidence"]))
        c["evidence"] = [D.canonical(j | {"extra": None}).encode().hex(), D.canonical({}).encode().hex(),
                         D.canonical({"signed": 1}).encode().hex(), b"\xff".hex()][r.randrange(4)]
    return c


def build(m: str, seed: int = 0) -> dict:
    """A wipe's case with the change `m` (`EXPECT`), every choice sampled from `seed`."""
    return copy.deepcopy(_build(m, seed))


KINDS = (ACCEPT, REJECT, UNDECIDED, "broken-params", "broken-commitment", "broken-evidence")
#: Each kind's changes (`build`'s `m`): a rejection by the check itself, not by decoding, for `reject`.
AIMS: dict[str, tuple[str, ...]] = {
    ACCEPT: ("honest", "segment", "reads"),
    REJECT: tuple(m for m, v in EXPECT.items() if v == REJECT and m not in ("params", "commitment", "evidence")),
    UNDECIDED: ("unsigned",),
    "broken-params": ("params",), "broken-commitment": ("commitment",), "broken-evidence": ("evidence",),
}


def verdict_of(kind: str) -> str:
    """The verdict a case of `kind` gets."""
    return REJECT if kind.startswith("broken-") else kind


def case(kind: str, seed: int = 0) -> dict:
    """A wipe aimed at `kind`, its change and every choice sampled from `seed`."""
    ms = AIMS[kind]
    return build(ms[random.Random(f"wipe-protocol {kind} {seed}").randrange(len(ms))], seed)


def cases(seed: int = 0) -> dict[str, tuple[dict, dict]]:
    """One wipe of each kind with its answer, `{kind: (case, answer)}`."""
    return {k: (c, answer(c)) for k in KINDS for c in [case(k, seed)]}
