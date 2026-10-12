"""The memory challenger: PoUS's timed rounds, run by an auditor-trusted device beside the network certifier, which signs
its rounds.  It is the auditor's hardware, on a machine the prover doesn't control at the prover's site (the auditor's
own VM or box, under the auditor's account), never a process on the prover's node.  On the deployed path it replaces
`audit.UnsignedRounds`, the same rounds unsigned.

A session has four steps:

1. **Commit.** Before the session the memory challenger draws its `k` indices from its own secret
   (`verity.ml.randomness`, `COIN_DOMAIN`) and commits to them in a coin tree.  It publishes the coin tree's
   root and hm96 key together with the session's parameters (`SessionCommitment`).
2. **Rounds.** Round `j` reveals index `j` (`challenge`), receives the whole answer block (`receive`), reads its clock
   for the receipt before it hashes anything, then hashes the received bytes itself (`block_digest`).  Its record is
   `(j, index, send time, receipt time, H(block))` (`Round`).  The first late round ends the session.  No round is
   retried, and no index is redrawn.
3. **Sign.** One signature per session (`close`) over `SHA-512(SESSION_TAG ‖ header ‖ records root)`: the header is
   the commitment's fields and how the session ended, and the records root is the frame-v3-sha512 root of the round
   records.  The records are witness: the prover proves things about them, and they never reach the auditor in the
   clear.
4. **Open.** After the session the memory challenger opens every coin it revealed (`openings`).

`verdict` is what the prover will prove about a signed session, here checked in the clear.  It checks the signature
under a trusted key and an accepted scheme, the header against the commitment, every revealed index against its coin
opening, every `H(block)` against the expected digest of that block of `C`, and every round on time within `Δ + RTT`.
Its public output is the verdict alone (`Verdict.public`); the per-round on-time bits and the reasons stay in the
signed record.  The guarantee of the sequential draw is the coin tree's: an index is fixed before the session
(binding) and unknown to the server until its round reveals it (hiding).

**Coin tree** (`COIN_TREE`): hm96-sha512/v1 hiding leaves in a frame-v3-sha512 tree, core's helpers
(`verity.ml.commitments.hm96`, `.merkle`), with integers big-endian:

~~~text
x_j     = SHA-512(COIN_TAG ‖ session ‖ u32 j ‖ u64 index_j)          round j's coin, private until opened
leaf_j  = frame leaf (rank j, position j, COIN_SCHEMA) of hm96.SHA512.tree_leaf(K, b ‖ c)
          with b ‖ c = hm96.SHA512.commit_string(K, x_j, y_j)        y_j a fresh 192-byte salt, K the session's fresh key
root    = frame-v3-sha512 root over leaf_0..leaf_{k-1}, domain (binding(COIN_BINDING, session), owner -1, range(k))
~~~

An opening is `(j, index_j, y_j, path)`.  Binding is SHA-512's collision resistance; hiding is hm96's statistical
bound, `k · 2^-256` under the fresh key.  The domain's 32-byte binding is the first 32 bytes of
`SHA-512(tag ‖ session)`, since a frame-v3 binding is 32 bytes until the identity re-baseline.

**Round records** (`RECORDS_TREE`): record `j` is `u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)` (92
bytes), the value of frame leaf `(rank j, position j, RECORD_SCHEMA)`, in a frame-v3-sha512 tree over the session's
rounds (binding(RECORD_BINDING, session), owner -1).  `H(block)` is the block's `sha512/row/v2` digest
`SHA-512(P(8N) ‖ block)` for an N-byte block (`verity.ml.commitments.rowleaf.sha512_row_v2_digest`), the inner
digest of C's `hm96-sha512/row/v2` leaf: `answer_circuit.PousAnswerOpens_v1` proves it equal to `h` and opens it against
C's leaf with no second hash of the block.  The digest costs the same 17 compressions as a plain SHA-512 of the block,
P's block a constant midstate, and is taken after the receipt time.

**Signatures.** A `Signer` signs a session's record (`message`, 64 bytes) under its session id; `SCHEMES` maps an
accepted scheme to its `verify(public_key, session, record, signature)`.  The one accepted scheme is S1's, `LMS`:
LMS/LM-OTS over SHA-512 at `lms.DEPLOYED` (`verity.ml.crypto.lms`, claim `euf-cma/lms-sha512-n64`), which signs
the memory challenger role's digest `lms.memory_challenger_message(I, session, record)`; `LmsSigner` is its signer.  A
test signer's scheme starts with `TEST_PREFIX`; `register` never adds one, so a test-signed record is refused outside
tests, as is an unsigned one.

Times are the challenger's monotonic clock in ns (`time.perf_counter_ns`, Linux's CLOCK_MONOTONIC); the challenger
refuses a clock coarser than a microsecond (`CLOCK_RESOLUTION_NS`).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from verity.ml.commitments import hm96
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, Opening, tree_levels, verify_opening
from verity.ml.commitments.rowleaf import sha512_row_v2_digest
from verity.ml.crypto import lms
from verity.ml.randomness import derive

from .protocol import Certificate, Scheme

PROTOCOL = "verity-pous/memory-challenger/v1"
COIN_DOMAIN = "verity/pous/memory-challenger-coins/v1"
COIN_TREE = "verity-pous/coin-tree/hm96-sha512/v1"
RECORDS_TREE = "verity-pous/round-records/frame-v3-sha512/v1"
COIN_TAG = b"verity-pous/memory-challenger/coin/v1\0"
COIN_BINDING = b"verity-pous/memory-challenger/coin-tree/v1\0"
RECORD_BINDING = b"verity-pous/memory-challenger/round-records/v1\0"
SESSION_TAG = b"verity-pous/memory-challenger/session/v1\0"
COIN_SCHEMA = "pous/coin/hm96-sha512/v1"
RECORD_SCHEMA = "pous/round-record/v2"
TEST_PREFIX = "test/"
CLOCK_RESOLUTION_NS = 1_000
"""The coarsest clock the memory challenger runs on: a microsecond (the certifier's 100 ms buckets are far too coarse)."""
COMPLETE, LATE = "complete", "late"
HM96 = hm96.SHA512
LIMITS = VerificationLimits()


def _sha512(*parts: bytes) -> bytes:
    h = hashlib.sha512()
    for p in parts:
        h.update(p)
    return h.digest()


def block_digest(block: bytes) -> bytes:
    """`H(block)`: the block's `sha512/row/v2` digest, `SHA-512(P(8N) ‖ block)` (module docstring)."""
    return sha512_row_v2_digest(block, 8 * len(block))


def _domain(tag: bytes, session: bytes, count: int) -> CommitmentDomain:
    return CommitmentDomain(_sha512(tag, session)[:32], -1, RangeIndexedDomain(0, count), hash="sha512")


def _root(domain: CommitmentDomain, leaves: Sequence[bytes]) -> bytes:
    levels = tree_levels(domain, leaves)
    return levels[-1][0] if levels else domain.empty_root()


def _path(levels: tuple[tuple[bytes, ...], ...], rank: int) -> tuple[bytes, ...]:
    out = []
    for level in levels[:-1]:
        out.append(level[rank ^ 1])
        rank >>= 1
    return tuple(out)


def coin_digest(session: bytes, j: int, index: int) -> bytes:
    """`x_j`, round `j`'s coin before hiding."""
    return _sha512(COIN_TAG, session, j.to_bytes(4, "big"), index.to_bytes(8, "big"))


def coin_leaf(key: bytes, session: bytes, j: int, index: int, salt: bytes) -> bytes:
    """The hm96-sha512 leaf of round `j`'s coin under the session's key `key` and the coin's salt."""
    return HM96.tree_leaf(key, HM96.commit_string(key, coin_digest(session, j, index), salt))


@dataclass(frozen=True, slots=True)
class CoinOpening:
    """Round `j`'s coin: its index, its hm96 salt, and the leaf's path to the coin tree's root."""

    j: int
    index: int
    salt: bytes
    path: tuple[bytes, ...]

    def to_json(self) -> dict[str, Any]:
        return {"j": self.j, "index": self.index, "salt": self.salt.hex(), "path": [p.hex() for p in self.path]}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> CoinOpening:
        return cls(int(doc["j"]), int(doc["index"]), bytes.fromhex(doc["salt"]), tuple(bytes.fromhex(p) for p in doc["path"]))


class CoinTree:
    """The session's `k` coins over `blocks` blocks, committed before the session (module docstring).  `draw` takes
    the challenger's own secret, never a public beacon or the server's; the hm96 key and the salts are fresh from the OS
    unless given (the tests)."""

    def __init__(self, session: bytes, blocks: int, indices: Sequence[int], key: bytes, salts: Sequence[bytes]) -> None:
        if len(session) != 32:
            raise ValueError("a session id is 32 bytes")
        if blocks < 1 or not indices or len(salts) != len(indices) or any(not 0 <= i < blocks for i in indices):
            raise ValueError("a coin tree has one salt per index, every index a block")
        self.session, self.blocks, self.indices, self.key = session, blocks, tuple(indices), HM96.check_key(key)
        self.salts = tuple(salts)
        self.domain = _domain(COIN_BINDING, session, len(self.indices))
        hm = [coin_leaf(self.key, session, j, i, s) for j, (i, s) in enumerate(zip(self.indices, self.salts, strict=True))]
        self._levels = tree_levels(self.domain, [self.domain.leaf(j, j, COIN_SCHEMA, h) for j, h in enumerate(hm)])
        self.root = self._levels[-1][0]

    @classmethod
    def draw(cls, source: bytes, session: bytes, blocks: int, k: int, *, key: bytes | None = None,
             salts: Sequence[bytes] | None = None) -> CoinTree:
        coins = derive(source, COIN_DOMAIN, {"session": session, "blocks": blocks, "k": k})
        return cls(session, blocks, [coins.uniform(blocks, j) for j in range(k)], HM96.fresh_key() if key is None else key,
                   [HM96.fresh_salt() for _ in range(k)] if salts is None else salts)

    @property
    def k(self) -> int:
        return len(self.indices)

    def opening(self, j: int) -> CoinOpening:
        return CoinOpening(j, self.indices[j], self.salts[j], _path(self._levels, j))


def verify_coin(commitment: SessionCommitment, opening: CoinOpening) -> bool:
    """`opening` opens round `opening.j`'s coin under the commitment's coin tree, to a block index."""
    if not (0 <= opening.j < commitment.k and 0 <= opening.index < commitment.blocks):
        return False
    try:
        leaf = coin_leaf(commitment.coin_key, commitment.session, opening.j, opening.index, opening.salt)
        domain = _domain(COIN_BINDING, commitment.session, commitment.k)
        return verify_opening(domain, Commitment(commitment.coin_root, commitment.k), Opening(opening.j, leaf, opening.path),
                              COIN_SCHEMA, LIMITS)
    except Exception:  # noqa: BLE001 - a malformed opening opens nothing
        return False


@dataclass(frozen=True, slots=True)
class SessionCommitment:
    """What the memory challenger publishes before the session: the session id, the coin tree (root and hm96 key),
    the number of blocks and rounds, the limit `Δ + RTT`, the answer's size, and `binding`, what the session audits
    (for example the salted roots of the encoding).  Every field is in the signed header."""

    session: bytes
    coin_root: bytes
    coin_key: bytes
    blocks: int
    k: int
    limit_ns: int
    block_bytes: int
    binding: bytes = b""

    def header(self) -> dict[str, Any]:
        return {"protocol": PROTOCOL, "coin_tree": COIN_TREE, "records_tree": RECORDS_TREE, "session": self.session.hex(),
                "coin_root": self.coin_root.hex(), "coin_key_sha512": _sha512(self.coin_key).hex(), "blocks": self.blocks,
                "k": self.k, "limit_ns": self.limit_ns, "block_bytes": self.block_bytes, "binding": self.binding.hex(),
                "clock": "CLOCK_MONOTONIC ns"}

    def to_json(self) -> dict[str, Any]:
        return {**self.header(), "coin_key": self.coin_key.hex()}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SessionCommitment:
        return cls(bytes.fromhex(doc["session"]), bytes.fromhex(doc["coin_root"]), bytes.fromhex(doc["coin_key"]),
                   int(doc["blocks"]), int(doc["k"]), int(doc["limit_ns"]), int(doc["block_bytes"]),
                   bytes.fromhex(doc["binding"]))


@dataclass(frozen=True, slots=True)
class Round:
    """One round's record: its index, the challenge's send time, the answer's receipt time and `block_digest` of the
    whole answer as received."""

    j: int
    index: int
    send_ns: int
    receipt_ns: int
    digest: bytes

    @property
    def elapsed_ns(self) -> int:
        return self.receipt_ns - self.send_ns

    def value(self) -> bytes:
        return (self.j.to_bytes(4, "big") + self.index.to_bytes(8, "big") + self.send_ns.to_bytes(8, "big")
                + self.receipt_ns.to_bytes(8, "big") + self.digest)

    def to_json(self) -> dict[str, Any]:
        return {"j": self.j, "index": self.index, "send_ns": self.send_ns, "receipt_ns": self.receipt_ns,
                "digest": self.digest.hex()}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Round:
        return cls(int(doc["j"]), int(doc["index"]), int(doc["send_ns"]), int(doc["receipt_ns"]), bytes.fromhex(doc["digest"]))


def records_root(session: bytes, rounds: Sequence[Round]) -> bytes:
    """The frame-v3-sha512 root over the round records (module docstring)."""
    domain = _domain(RECORD_BINDING, session, len(rounds))
    return _root(domain, [domain.leaf(r, r, RECORD_SCHEMA, x.value()) for r, x in enumerate(rounds)])


def message(commitment: SessionCommitment, ended: str, rounds: Sequence[Round]) -> bytes:
    """What the session's one signature signs: `SHA-512(SESSION_TAG ‖ u64 len ‖ header ‖ records root)`."""
    head = json.dumps({**commitment.header(), "ended": ended, "rounds": len(rounds)}, sort_keys=True,
                      separators=(",", ":")).encode()
    return _sha512(SESSION_TAG, len(head).to_bytes(8, "big"), head, records_root(commitment.session, rounds))


class Signer(Protocol):
    """A signer of session records: its scheme's id, its public key, and `sign(session, record)`, which a stateful
    hash-based scheme refuses past its last one-time key."""

    scheme: str
    public_key: bytes

    def sign(self, session: bytes, record: bytes) -> bytes: ...


type Verify = Callable[[bytes, bytes, bytes, bytes], bool]
"""`verify(public_key, session, record, signature)`."""

LMS = "lms-sha512/n64-w1-h20/v0"
"""S1's scheme at its deployed parameters (`lms.DEPLOYED`)."""


def lms_verify(params: lms.Params) -> Verify:
    """S1's verification at `params`, of the memory challenger role's digest of `(session, record)`."""

    def verify(public_key: bytes, session: bytes, record: bytes, signature: bytes) -> bool:
        if len(public_key) != 16 + params.N:
            return False
        sig = lms.Signature.decode(params, signature)
        return lms.verify(params, public_key, lms.memory_challenger_message(public_key[:16], session, record), sig)

    return verify


class LmsSigner:
    """A memory challenger's S1 key (`lms.PrivateKey`) and its leaf state (`lms.LeafState`), which refuses a used or
    missing leaf; `scheme` is `LMS` at `lms.DEPLOYED`, else a test's own id for other parameters."""

    def __init__(self, key: lms.PrivateKey, state: lms.LeafState, scheme: str = LMS) -> None:
        if (scheme == LMS) != (key.params == lms.DEPLOYED):
            raise ValueError(f"{LMS} is LMS at {lms.DEPLOYED}, not {key.params}")
        self.key, self.state, self.scheme, self.public_key = key, state, scheme, lms.public_key(key)

    def sign(self, session: bytes, record: bytes) -> bytes:
        return lms.sign(self.key, self.state, lms.memory_challenger_message(self.key.I, session, record)).encode()


_SCHEMES: dict[str, Verify] = {LMS: lms_verify(lms.DEPLOYED)}
SCHEMES: Mapping[str, Verify] = MappingProxyType(_SCHEMES)
"""The signature schemes a session is accepted under: S1's (`LMS`) and nothing else."""


def register(scheme: str, verify: Verify) -> None:
    """Accept sessions signed under `scheme`; a test signer's scheme (`TEST_PREFIX`) is never accepted here."""
    if scheme.startswith(TEST_PREFIX):
        raise ValueError(f"{scheme} is a test signer's scheme: it is accepted only by a test's own mapping")
    _SCHEMES[scheme] = verify


@dataclass(frozen=True, slots=True)
class SignedSession:
    """A closed session: its commitment, how it ended, its round records, and the one signature over them."""

    commitment: SessionCommitment
    ended: str
    rounds: tuple[Round, ...]
    scheme: str
    public_key: bytes
    signature: bytes

    def message(self) -> bytes:
        return message(self.commitment, self.ended, self.rounds)

    def to_json(self) -> dict[str, Any]:
        return {"commitment": self.commitment.to_json(), "ended": self.ended, "rounds": [r.to_json() for r in self.rounds],
                "scheme": self.scheme, "public_key": self.public_key.hex(), "signature": self.signature.hex()}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SignedSession:
        return cls(SessionCommitment.from_json(doc["commitment"]), doc["ended"], tuple(Round.from_json(r) for r in doc["rounds"]),
                   doc["scheme"], bytes.fromhex(doc["public_key"]), bytes.fromhex(doc["signature"]))


def covering_certificate(scheme: Scheme, k: int, limit_ns: int, *, experimental: bool = False) -> Certificate:
    """The first of `scheme`'s certificates that covers `k` rounds within `limit_ns`, as `protocol.TimedAudit` picks it;
    `ValueError` when none does, or when the scheme is experimental and `experimental` is not set."""
    if scheme.experimental and not experimental:
        raise ValueError(f"{scheme.name} is experimental ({scheme.experimental}): pass experimental=True to audit it")
    for cert in scheme.certificates:
        if not cert.refusals(k, limit_ns):
            return cert
    why = "; ".join(f"{c.theorem}: {'; '.join(c.refusals(k, limit_ns))}" for c in scheme.certificates) or "no certificate"
    raise ValueError(f"no certificate of {scheme.name} covers k = {k} within {limit_ns} ns ({why})")


class MemoryChallenger:
    """One session of the memory challenger over `coins`, judged against `limit_ns`, signed by `signer` (module
    docstring).  `challenge()` reveals the next index and reads the send time; `receive(block)` reads the receipt time
    first, then hashes the block.  A late round ends the session; `close()` signs it once."""

    def __init__(self, coins: CoinTree, signer: Signer, *, limit_ns: int, block_bytes: int, binding: bytes = b"",
                 clock: Callable[[], int] = time.perf_counter_ns, resolution_ns: int | None = None) -> None:
        if limit_ns <= 0 or block_bytes <= 0:
            raise ValueError("a session's limit and answer size are positive")
        if resolution_ns is None:
            resolution_ns = time.clock_getres(time.CLOCK_MONOTONIC) * 1e9 if clock is time.perf_counter_ns else 1
        if resolution_ns > CLOCK_RESOLUTION_NS:
            raise ValueError(f"the clock ticks every {resolution_ns} ns, coarser than {CLOCK_RESOLUTION_NS} ns")
        self.coins, self.signer, self.clock = coins, signer, clock
        self.commitment = SessionCommitment(coins.session, coins.root, coins.key, coins.blocks, coins.k, limit_ns,
                                            block_bytes, binding)
        self.rounds: list[Round] = []
        self.ended: str | None = None
        self.signed: SignedSession | None = None
        self._open: tuple[int, int, int] | None = None

    @property
    def done(self) -> bool:
        return self.ended is not None

    def challenge(self) -> int:
        """Reveal round `j`'s index and read its send time; refused with a round open or the session ended."""
        if self.ended is not None:
            raise RuntimeError(f"the session has ended ({self.ended}): no round is retried and no index redrawn")
        if self._open is not None:
            raise RuntimeError("the previous round is unanswered")
        j = len(self.rounds)
        i = self.coins.indices[j]
        self._open = (j, i, self.clock())
        return i

    def receive(self, block: bytes) -> Round:
        """The open round's answer: its receipt time, then `block_digest` of all of `block`, which must be `block_bytes`."""
        receipt = self.clock()
        if self._open is None:
            raise RuntimeError("no round is open")
        if len(block) != self.commitment.block_bytes:
            raise ValueError(f"an answer is the whole {self.commitment.block_bytes}-byte block, not {len(block)} bytes")
        j, i, send = self._open
        self._open = None
        r = Round(j, i, send, receipt, block_digest(block))
        self.rounds.append(r)
        if not 0 <= r.elapsed_ns <= self.commitment.limit_ns:
            self.ended = LATE
        elif len(self.rounds) == self.coins.k:
            self.ended = COMPLETE
        return r

    def run(self, respond: Callable[[int], bytes]) -> SignedSession:
        """Drive the session against `respond(index)` and close it."""
        while not self.done:
            self.receive(respond(self.challenge()))
        return self.close()

    def close(self) -> SignedSession:
        """Sign the ended session, once."""
        if self.ended is None:
            raise RuntimeError("the session has not ended")
        if self.signed is None:
            rounds = tuple(self.rounds)
            sig = self.signer.sign(self.commitment.session, message(self.commitment, self.ended, rounds))
            self.signed = SignedSession(self.commitment, self.ended, rounds, self.signer.scheme, self.signer.public_key, sig)
        return self.signed

    def openings(self) -> tuple[CoinOpening, ...]:
        """The coins of every revealed round, opened once the session is signed."""
        if self.signed is None:
            raise RuntimeError("the coins are opened after the session, once it is signed")
        return tuple(self.coins.opening(r.j) for r in self.rounds)


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether the signed session passes.  `reasons` (empty when it passes) and `late`, the late rounds, are witness:
    only `public()` is published."""

    accepted: bool
    reasons: tuple[str, ...]
    late: int = 0
    wrong: int = 0

    def public(self) -> dict[str, Any]:
        return {"accepted": self.accepted}


def verdict(signed: SignedSession, commitment: SessionCommitment, openings: Sequence[CoinOpening],
            expected: Callable[[int], bytes], *, trusted: frozenset[bytes] | set[bytes],
            schemes: Mapping[str, Verify] = SCHEMES) -> Verdict:
    """The check of a signed session against the commitment published before it (module docstring): `expected(i)` is
    `block_digest` of block `i` of `C`, `trusted` the memory challengers' public keys, `schemes` the accepted signature schemes
    (S1's; a test's own mapping in tests).  It fails closed: an unknown scheme, an untrusted key or a bad signature
    rejects, whatever the rounds."""
    reasons: list[str] = []
    verify = schemes.get(signed.scheme)
    if verify is None:
        reasons.append(f"signature scheme {signed.scheme!r} is not accepted")
    elif signed.public_key not in trusted:
        reasons.append("the session is signed by a key that is not a trusted memory challenger's")
    else:
        try:
            ok = bool(verify(signed.public_key, signed.commitment.session, signed.message(), signed.signature))
        except Exception:  # noqa: BLE001 - a malformed signature verifies nothing
            ok = False
        if not ok:
            reasons.append("the signature does not verify")
    if signed.commitment != commitment:
        reasons.append("the signed header is not the session's commitment")
    if signed.ended != COMPLETE or len(signed.rounds) != commitment.k:
        reasons.append(f"the session ended {signed.ended} after {len(signed.rounds)} of {commitment.k} rounds")
    by_j = {o.j: o for o in openings}
    late = wrong = 0
    for j, r in enumerate(signed.rounds):
        o = by_j.get(j)
        if r.j != j or o is None or o.index != r.index or not verify_coin(commitment, o):
            reasons.append(f"round {j}'s index is not its committed coin")
        if not 0 <= r.index < commitment.blocks or r.digest != expected(r.index):
            wrong += 1
        if not 0 <= r.elapsed_ns <= commitment.limit_ns:
            late += 1
    if wrong:
        reasons.append(f"{wrong} answers are not their blocks of C")
    if late:
        reasons.append(f"{late} answers are past Δ + RTT = {commitment.limit_ns} ns")
    return Verdict(not reasons, tuple(reasons), late, wrong)


__all__ = ["CLOCK_RESOLUTION_NS", "COIN_DOMAIN", "COIN_SCHEMA", "COIN_TREE", "COMPLETE", "LATE", "LMS", "PROTOCOL",
           "RECORDS_TREE", "RECORD_SCHEMA", "SCHEMES", "TEST_PREFIX", "CoinOpening", "CoinTree", "LmsSigner", "MemoryChallenger",
           "Round", "SessionCommitment", "SignedSession", "Signer", "Verdict", "Verify", "block_digest", "coin_digest", "coin_leaf",
           "covering_certificate", "lms_verify", "message", "records_root", "register", "verdict", "verify_coin"]
