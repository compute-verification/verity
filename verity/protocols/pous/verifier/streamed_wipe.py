"""The streamed-label wipe (experimental): a proof of secure erasure for a memory wipe, run by the memory challenger.
`PROTOCOL.md` ("The streamed-label wipe") is its specification and states the guarantee.

The memory challenger, the auditor's device on a machine the developer doesn't control, fills every memory tier of one
server with labels it derives itself, then reveals `k` indices sampled on live coins after the fill and checks each answer
against its own derivation, under a deadline per batch.  The labels are the auditor's randomness: the answers are
checked in the clear, and nothing of the prover's is opened.

~~~text
K     = derive(secret, KEY_DOMAIN, {server, session})            32 bytes, one key per server and wipe
L_i   = SHA-512(LABEL_TAG ‖ K ‖ u64 i)                           64 bytes, i in [0, N): one compression a label
c_j   = derive(coin_secret, COIN_DOMAIN, {server, session, n}).uniform(N, j)    sampled once the fill is complete
~~~

`secret` is the auditor's for this server and wipe, `coin_secret` is fresh from the OS when the fill completes.  A
session has five steps:

1. **Open.** The challenger claims the session id in its `SessionLog` (one session an id) and publishes the
   `WipeCommitment`: the server, the session, `N`, `k`, the batch size, the deadline a batch and the declared memory tiers,
   whose sizes add up to at least the fill.
2. **Mark.** A challenger with a mark source hands it `start_message(commitment)`, which the source takes to the
   network certifier, and gets the start certificate's bytes back.  It keeps them, and its start mark is their SHA-512.
   It does not verify them: bytes that are no certificate cost the session, never the count.  It holds the mark before
   it reads its clock for the fill, so no label is streamed before it.  With no source the start mark is empty.
3. **Fill.** `fill(chunk)` streams `L_0 … L_{N-1}` in order; the challenger reads its clock at the first and last chunk.
   `fill_by(send)` is the same fill streamed by `send`, a streamer on the challenger's own host given `K`.  Its clock
   is the raw monotonic clock (`raw_clock_ns`), which the header names (`CLOCK`).
4. **Challenge.** Once the fill is complete the challenger flips its coins, then reveals them a batch at a time
   (`challenge`), reads the receipt time before it hashes anything (`receive`), and checks every 64-byte answer against
   `L_{c_j}`.  The first late or wrong batch ends the session: no batch is retried, and no index is redrawn.
5. **Sign.** One signature over `SHA-512(SESSION_TAG ‖ u64 len ‖ header ‖ records root)`, where the header is the
   commitment, how the session ended, the start mark and the fill's times (`close`), and the records root is the
   frame-v3-sha512 root of the batch records.  `end_message(signed)` is the end message, what the challenger hands the
   network certifier after its last read.  No egress count is signed: the count of what left between the two marks
   exists only after the signature, and the challenger never sees it.

The two marks are 64 bytes each, and neither is signed:

~~~text
start = SHA-512(START_TAG ‖ u64 len ‖ commitment header)              the commitment's header alone, as sorted JSON
end   = SHA-512(END_TAG ‖ u64 len ‖ message ‖ u64 len ‖ signature)    the session's signed message and its signature
~~~

A batch record is `u32 b ‖ u32 first ‖ u32 count ‖ u64 send_ns ‖ u64 receipt_ns ‖ u32 wrong ‖ SHA-512(indices ‖
answers)` (96 bytes).  The guarantee (`PROTOCOL.md`; Lean `Pous.Guarantees.StreamedWipe`) covers the sequential reveal,
batch size 1 (`WipeParams.covered`); a larger batch lets an answer depend on the whole batch, which the proof doesn't
cover yet, so `verdict` refuses it.  `verdict` is what a verifier checks of a signed session; its public output is the
verdict alone.  Its Lean port decides it (`MemoryChallenger.StreamedWipe` and `wipe_verdict.judge` in
`verity/protocols/pous/device/lean/`, reading the records' `to_json`), and `verdict` here is the reference that port is
difftested against.  The egress the bound reads is not the session's: the wipe's check takes it as evidence beside the
signed session, in whole MiB (`egress_units`, 15 bits), and publishes it (`wipe_period`).  `retained_bits` and
`challenges_for` are the guarantee's bound, which says how much of the prior content can survive an accepted wipe.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain, tree_levels
from verity.commitments.scheme import DIGEST_BYTES
from verity.commitments.randomness import derive

from verity.protocols.pous.device.memory_challenger import CLOCK_RESOLUTION_NS, SCHEMES, SessionLog, Signer, Verify

PROTOCOL = "verity-pous/streamed-wipe/v2"
KEY_DOMAIN = "verity/pous/streamed-wipe-labels/v1"
COIN_DOMAIN = "verity/pous/streamed-wipe-coins/v1"
LABEL_TAG = b"verity-pous/streamed-wipe/label/v1\0"
SESSION_TAG = b"verity-pous/streamed-wipe/session/v2\0"
START_TAG = b"verity-pous/streamed-wipe/start/v1\0"
END_TAG = b"verity-pous/streamed-wipe/end/v1\0"
RECORD_BINDING = b"verity-pous/streamed-wipe/batch-records/v1\0"
RECORDS_TREE = "verity-pous/streamed-wipe/batch-records/frame-v3-sha512/v1"
RECORD_SCHEMA = "pous/streamed-wipe/batch-record/v1"
CLOCK = "CLOCK_MONOTONIC_RAW ns"
"""The clock a session's times are on, as its header names it: Linux's raw monotonic clock, in ns.  Time synchronisation
slews `CLOCK_MONOTONIC`, so an interval on that clock states no rate; the raw clock is never adjusted."""
LABEL_BYTES = DIGEST_BYTES["sha512"]
LABEL_BITS = 8 * LABEL_BYTES
MARK_BYTES = DIGEST_BYTES["sha512"]
"""A start mark: the SHA-512 of the start certificate."""
COMPLETE, LATE, WRONG = "complete", "late", "wrong"

DEPLOYED_FILL_BYTES = 2_600_000_000_000
"""The deployed fill: 2.6 TB, every tier of one server."""
DEPLOYED_LABELS = DEPLOYED_FILL_BYTES // LABEL_BYTES
DEPLOYED_K = 74_000
"""The deployed challenge count: retained prior content at most about 1.02 GB plus the egress, at δ = 2^-40."""


def raw_clock_ns() -> int:
    """A reading of `CLOCK`, the raw monotonic clock, in ns: what a `Challenger` reads unless a clock is injected."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def _sha512(*parts: bytes) -> bytes:
    h = hashlib.sha512()
    for p in parts:
        h.update(p)
    return h.digest()


def wipe_key(secret: bytes, server: bytes, session: bytes) -> bytes:
    """`K`, the labels' key for one server and one wipe, from the auditor's secret."""
    if len(session) != 32 or not server:
        raise ValueError("a session id is 32 bytes and a server id is nonempty")
    return derive(secret, KEY_DOMAIN, {"server": server, "session": session}).value


def label(key: bytes, i: int) -> bytes:
    """`L_i = SHA-512(LABEL_TAG ‖ K ‖ u64 i)`: 75 bytes, one SHA-512 block."""
    if len(key) != 32 or not 0 <= i < 1 << 64:
        raise ValueError("a label key is 32 bytes and an index is a u64")
    return _sha512(LABEL_TAG, key, i.to_bytes(8, "big"))


def labels(key: bytes, start: int, count: int) -> bytes:
    """`L_start ‖ … ‖ L_{start+count-1}`."""
    head = hashlib.sha512(LABEL_TAG + key)
    out = bytearray()
    for i in range(start, start + count):
        h = head.copy()
        h.update(i.to_bytes(8, "big"))
        out += h.digest()
    return bytes(out)


def draw_indices(coin_secret: bytes, server: bytes, session: bytes, n: int, k: int) -> tuple[int, ...]:
    """The session's `k` challenges, each uniform in `[0, n)` and independent (with replacement)."""
    key = derive(coin_secret, COIN_DOMAIN, {"server": server, "session": session, "n": n})
    return tuple(key.uniform(n, j) for j in range(k))


@dataclass(frozen=True, slots=True)
class WipeParams:
    """`labels` (`N`), the `k` challenges revealed `batch` at a time, each batch's deadline `limit_ns`, and the memory
    tiers inside the boundary as `(name, bytes)`, whose total holds the fill: what they declare past it (`outside_bits`)
    is memory the fill doesn't reach, which the bound counts in full."""

    labels: int
    k: int
    batch: int
    limit_ns: int
    tiers: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.labels < 1 or not 1 <= self.batch <= self.k or self.k >= 1 << 32 or self.limit_ns <= 0:
            raise ValueError("a wipe has labels, 1 <= batch <= k < 2^32 challenges and a positive deadline")
        if not self.tiers or any(b <= 0 for _, b in self.tiers) or len({t for t, _ in self.tiers}) != len(self.tiers):
            raise ValueError("a wipe declares its tiers, each named once with a positive size")
        if sum(b for _, b in self.tiers) < self.labels * LABEL_BYTES:
            raise ValueError(f"the tiers total {sum(b for _, b in self.tiers)} bytes, less than the fill's "
                             f"{self.labels * LABEL_BYTES}: the declared tiers hold the fill")

    @property
    def outside_bits(self) -> int:
        """`R`: the declared memory the fill doesn't reach, in bits."""
        return 8 * sum(b for _, b in self.tiers) - self.labels * LABEL_BITS

    @property
    def covered(self) -> bool:
        """Whether the guarantee covers this reveal: the sequential one, one index a batch.  `verdict` refuses any other."""
        return self.batch == 1

    @property
    def batches(self) -> int:
        return -(-self.k // self.batch)


@dataclass(frozen=True, slots=True)
class WipeCommitment:
    """What the challenger publishes when it opens the session: its parameters, for one server and session id."""

    server: bytes
    session: bytes
    params: WipeParams

    def header(self) -> dict[str, Any]:
        p = self.params
        return {"protocol": PROTOCOL, "records_tree": RECORDS_TREE, "server": self.server.hex(),
                "session": self.session.hex(), "labels": p.labels, "label_bytes": LABEL_BYTES, "k": p.k,
                "batch": p.batch, "limit_ns": p.limit_ns, "tiers": [[t, b] for t, b in p.tiers],
                "clock": CLOCK}

    def to_json(self) -> dict[str, Any]:
        """The commitment as the Lean verdict reads it (`MemoryChallenger.StreamedWipe.Commitment.ofJson`): its header."""
        return self.header()


@dataclass(frozen=True, slots=True)
class BatchRecord:
    """One batch: its number, its first challenge, its size, the send and receipt times, the wrong answers, and the
    digest of the indices and the answers as received."""

    b: int
    first: int
    count: int
    send_ns: int
    receipt_ns: int
    wrong: int
    digest: bytes

    @property
    def elapsed_ns(self) -> int:
        return self.receipt_ns - self.send_ns

    def value(self) -> bytes:
        return (self.b.to_bytes(4, "big") + self.first.to_bytes(4, "big") + self.count.to_bytes(4, "big")
                + self.send_ns.to_bytes(8, "big") + self.receipt_ns.to_bytes(8, "big") + self.wrong.to_bytes(4, "big")
                + self.digest)

    def to_json(self) -> dict[str, Any]:
        return {"b": self.b, "first": self.first, "count": self.count, "send_ns": self.send_ns,
                "receipt_ns": self.receipt_ns, "wrong": self.wrong, "digest": self.digest.hex()}


def batch_digest(indices: Sequence[int], answers: bytes) -> bytes:
    return _sha512(b"".join(i.to_bytes(8, "big") for i in indices), answers)


def records_root(session: bytes, batches: Sequence[BatchRecord]) -> bytes:
    """The frame-v3-sha512 root over the batch records."""
    domain = CommitmentDomain(_sha512(RECORD_BINDING, session)[:32], -1, RangeIndexedDomain(0, len(batches)), hash="sha512")
    levels = tree_levels(domain, [domain.leaf(r, r, RECORD_SCHEMA, x.value()) for r, x in enumerate(batches)])
    return levels[-1][0] if levels else domain.empty_root()


@dataclass(frozen=True, slots=True)
class SignedWipe:
    """A closed session: its commitment, how it ended, its start mark (the SHA-512 of the start certificate the
    challenger held before the fill, empty when it had no mark source), the fill's times, the batch records, and the
    one signature over them.  It holds no egress count."""

    commitment: WipeCommitment
    ended: str
    start_mark: bytes
    fill_start_ns: int
    fill_end_ns: int
    batches: tuple[BatchRecord, ...]
    scheme: str
    public_key: bytes
    signature: bytes

    def message(self) -> bytes:
        return message(self.commitment, self.ended, self.start_mark, self.fill_start_ns, self.fill_end_ns, self.batches)

    def to_json(self) -> dict[str, Any]:
        """The signed session as the Lean verdict reads it (`MemoryChallenger.StreamedWipe.Signed.ofJson`)."""
        return {"commitment": self.commitment.to_json(), "ended": self.ended, "start_mark": self.start_mark.hex(),
                "fill_start_ns": self.fill_start_ns, "fill_end_ns": self.fill_end_ns,
                "batches": [r.to_json() for r in self.batches], "scheme": self.scheme,
                "public_key": self.public_key.hex(), "signature": self.signature.hex()}


def mark_of(text: str) -> bytes:
    """A start mark read back from its header field: 128 lowercase hex digits, or the empty string for no mark."""
    if not isinstance(text, str) or len(text) not in (0, 2 * MARK_BYTES) or text.strip("0123456789abcdef"):
        raise ValueError("a start mark is 128 lowercase hex digits, or empty")
    return bytes.fromhex(text)


def message(commitment: WipeCommitment, ended: str, start_mark: bytes, fill_start_ns: int, fill_end_ns: int,
            batches: Sequence[BatchRecord]) -> bytes:
    """What the session's one signature signs: `SHA-512(SESSION_TAG ‖ u64 len ‖ header ‖ records root)`."""
    head = json.dumps({**commitment.header(), "ended": ended, "start_mark": start_mark.hex(),
                       "fill_start_ns": fill_start_ns, "fill_end_ns": fill_end_ns, "batches": len(batches)},
                      sort_keys=True, separators=(",", ":")).encode()
    return _sha512(SESSION_TAG, len(head).to_bytes(8, "big"), head, records_root(commitment.session, batches))


def start_message(commitment: WipeCommitment) -> bytes:
    """The start message, what the challenger hands the network certifier before the fill:
    `SHA-512(START_TAG ‖ u64 len ‖ commitment header)`, the header the commitment's alone (`WipeCommitment.header()`),
    as sorted JSON with no spaces, the way the signed header writes it.  Not signed."""
    head = json.dumps(commitment.header(), sort_keys=True, separators=(",", ":")).encode()
    return _sha512(START_TAG, len(head).to_bytes(8, "big"), head)


def end_message(signed: SignedWipe) -> bytes:
    """The end message, what the challenger hands the network certifier after its last read:
    `SHA-512(END_TAG ‖ u64 len ‖ message ‖ u64 len ‖ signature)`, the session's signed message and its signature.
    Not signed."""
    m = signed.message()
    return _sha512(END_TAG, len(m).to_bytes(8, "big"), m, len(signed.signature).to_bytes(8, "big"), signed.signature)


class Challenger:
    """One streamed-label wipe of `server` under `session`, signed by `signer` (module docstring).  The session id goes
    into `log` before anything else.  `coin_secret` is sampled from the OS when the fill completes unless a test gives
    it.  `mark_source` takes the start message and returns the network certifier's start certificate for it, as bytes;
    with none the session's start mark is empty.  `clock` is `raw_clock_ns` unless given.  The header names `CLOCK`
    whatever `clock` reads, so a `clock` given outside a test reads the raw clock too."""

    def __init__(self, secret: bytes, server: bytes, session: bytes, params: WipeParams, signer: Signer, *,
                 log: SessionLog, clock: Callable[[], int] = raw_clock_ns, resolution_ns: int | None = None,
                 coin_secret: bytes | None = None, mark_source: Callable[[bytes], bytes] | None = None) -> None:
        if resolution_ns is None:
            resolution_ns = time.clock_getres(time.CLOCK_MONOTONIC_RAW) * 1e9 if clock is raw_clock_ns else 1
        if resolution_ns > CLOCK_RESOLUTION_NS:
            raise ValueError(f"the clock ticks every {resolution_ns} ns, coarser than {CLOCK_RESOLUTION_NS} ns")
        self.key = wipe_key(secret, server, session)
        log.claim(session)
        self.commitment = WipeCommitment(server, session, params)
        self.signer, self.clock, self._coin_secret = signer, clock, coin_secret
        self._mark_source = mark_source
        self.start_certificate = b""
        self.start_mark = b""
        self.filled = 0
        self.fill_start_ns: int | None = None
        self.fill_end_ns: int | None = None
        self.indices: tuple[int, ...] | None = None
        self.batches: list[BatchRecord] = []
        self.ended: str | None = None
        self.signed: SignedWipe | None = None
        self._open: tuple[int, int, tuple[int, ...], int] | None = None

    @property
    def params(self) -> WipeParams:
        return self.commitment.params

    def _hold_start_mark(self) -> None:
        """With a mark source: hand it the start message, keep the start certificate's bytes it returns, and set the
        start mark to their SHA-512.  The bytes are not verified.  A source that returns none raises."""
        if self._mark_source is None:
            return
        certificate = self._mark_source(start_message(self.commitment))
        if not isinstance(certificate, (bytes, bytearray)) or not certificate:
            raise ValueError("the mark source returned no start certificate: the fill waits for its bytes")
        self.start_certificate = bytes(certificate)
        self.start_mark = _sha512(self.start_certificate)

    def fill(self, chunk: int) -> Iterator[tuple[int, bytes]]:
        """The labels in order, `chunk` at a time, as `(first index, bytes)`; once only.  The start mark is held
        before the clock is read, so before any label."""
        if self.fill_start_ns is not None:
            raise RuntimeError("the fill streams once")
        if chunk < 1:
            raise ValueError("a chunk holds at least one label")
        self._hold_start_mark()
        self.fill_start_ns = self.clock()
        n = self.params.labels
        for start in range(0, n, chunk):
            count = min(chunk, n - start)
            yield start, labels(self.key, start, count)
            self.filled = start + count
        self.fill_end_ns = self.clock()

    def fill_by(self, send: Callable[[bytes, int], Any]) -> Any:
        """`fill`'s labels streamed by `send(K, N)`, which returns once it has sent the last; once only. Returns what
        `send` returns. The answers are checked against this module's `label` whatever `send` streamed.  The start mark
        is held before the clock is read, so before `send` is called."""
        if self.fill_start_ns is not None:
            raise RuntimeError("the fill streams once")
        self._hold_start_mark()
        self.fill_start_ns = self.clock()
        out = send(self.key, self.params.labels)
        self.filled = self.params.labels
        self.fill_end_ns = self.clock()
        return out

    def challenge(self) -> tuple[int, ...]:
        """Reveal the next batch's indices and read its send time; the coins are flipped at the first batch."""
        if self.fill_end_ns is None:
            raise RuntimeError("the challenges start once the fill is complete")
        if self.ended is not None:
            raise RuntimeError(f"the session has ended ({self.ended}): no batch is retried and no index redrawn")
        if self._open is not None:
            raise RuntimeError("the previous batch is unanswered")
        if self.indices is None:
            secret = self._coin_secret if self._coin_secret is not None else os.urandom(32)
            c = self.commitment
            self.indices = draw_indices(secret, c.server, c.session, self.params.labels, self.params.k)
        b = len(self.batches)
        first = b * self.params.batch
        idx = self.indices[first:first + self.params.batch]
        self._open = (b, first, idx, self.clock())
        return idx

    def receive(self, answers: bytes) -> BatchRecord:
        """The open batch's answers, `LABEL_BYTES` an index in order: the receipt time first, then the check."""
        receipt = self.clock()
        if self._open is None:
            raise RuntimeError("no batch is open")
        b, first, idx, send = self._open
        if len(answers) != LABEL_BYTES * len(idx):
            raise ValueError(f"a batch of {len(idx)} is answered with {LABEL_BYTES * len(idx)} bytes, not {len(answers)}")
        self._open = None
        wrong = sum(answers[LABEL_BYTES * t:LABEL_BYTES * (t + 1)] != label(self.key, i) for t, i in enumerate(idx))
        r = BatchRecord(b, first, len(idx), send, receipt, wrong, batch_digest(idx, answers))
        self.batches.append(r)
        if not 0 <= r.elapsed_ns <= self.params.limit_ns:
            self.ended = LATE
        elif wrong:
            self.ended = WRONG
        elif first + len(idx) == self.params.k:
            self.ended = COMPLETE
        return r

    def run(self, respond: Callable[[tuple[int, ...]], bytes]) -> None:
        """Drive the challenges against `respond(indices)` until the session ends."""
        while self.ended is None:
            self.receive(respond(self.challenge()))

    def close(self) -> SignedWipe:
        """Sign the ended session, once."""
        if self.ended is None:
            raise RuntimeError("the session has not ended")
        if self.signed is None:
            assert self.fill_start_ns is not None and self.fill_end_ns is not None
            batches = tuple(self.batches)
            m = message(self.commitment, self.ended, self.start_mark, self.fill_start_ns, self.fill_end_ns, batches)
            self.signed = SignedWipe(self.commitment, self.ended, self.start_mark, self.fill_start_ns, self.fill_end_ns,
                                     batches, self.signer.scheme, self.signer.public_key,
                                     self.signer.sign(self.commitment.session, m))
        return self.signed

    def coins(self) -> tuple[int, ...]:
        """The sampled indices, public once the session is signed (the verifier's samples)."""
        if self.signed is None or self.indices is None:
            raise RuntimeError("the coins are published after the session, once it is signed")
        return self.indices


class Prover:
    """The honest server: it keeps every label it is sent and answers from them."""

    def __init__(self, params: WipeParams) -> None:
        self.store = bytearray(params.labels * LABEL_BYTES)

    def receive(self, start: int, data: bytes) -> None:
        self.store[LABEL_BYTES * start:LABEL_BYTES * start + len(data)] = data

    def answer(self, indices: Sequence[int]) -> bytes:
        return b"".join(bytes(self.store[LABEL_BYTES * i:LABEL_BYTES * (i + 1)]) for i in indices)


EGRESS_UNIT_BITS = 8 << 20
"""The public egress count's unit: one MiB."""
EGRESS_MAX_UNITS = (1 << 15) - 1
"""The wipe's public unit stays in the least public size class, 16 bits: the verdict's one and the count's 15, so the
count saturates at 32 GiB less one MiB."""


def egress_units(egress_bits: int) -> int:
    """A count of egress bits in whole MiB, rounded up, saturating at `EGRESS_MAX_UNITS`: the unit the wipe's check
    takes its egress in, which the bound reads as `E`.  A saturated count (`EGRESS_MAX_UNITS`) bounds nothing."""
    return min(-(-egress_bits // EGRESS_UNIT_BITS), EGRESS_MAX_UNITS)


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a signed session passes; `reasons` are witness, and only `public()` is published.  It carries no
    egress: the session signs none."""

    accepted: bool
    reasons: tuple[str, ...]

    def public(self) -> dict[str, Any]:
        return {"accepted": self.accepted}


def verdict(signed: SignedWipe, commitment: WipeCommitment, *, trusted: frozenset[bytes] | set[bytes],
            schemes: Mapping[str, Verify] = SCHEMES) -> Verdict:
    """The check of a signed session against the commitment published when it opened.  It fails closed: an unknown
    scheme, an untrusted key or a bad signature rejects whatever the batches say, and a commitment whose batch size the
    guarantee doesn't cover (`WipeParams.covered`) rejects the session however it ran."""
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
    p = commitment.params
    if not p.covered:
        reasons.append(f"batch size {p.batch} is uncovered: the guarantee covers the sequential reveal, batch size 1")
    if signed.ended != COMPLETE or len(signed.batches) != p.batches:
        reasons.append(f"the session ended {signed.ended} after {len(signed.batches)} of {p.batches} batches")
    for b, r in enumerate(signed.batches):
        if r.b != b or r.first != b * p.batch or r.count != min(p.batch, p.k - r.first):
            reasons.append(f"batch {b} is not the session's batch {b}")
        if r.wrong:
            reasons.append(f"batch {b} has {r.wrong} wrong answers")
        if not 0 <= r.elapsed_ns <= p.limit_ns:
            reasons.append(f"batch {b} is past its deadline of {p.limit_ns} ns")
        if b == 0 and r.send_ns < signed.fill_end_ns:
            reasons.append("the first batch was sent before the fill completed")
    return Verdict(not reasons, tuple(reasons))


def _log2_choose(n: int, r: int) -> float:
    if not 0 <= r <= n:
        return -math.inf
    return (math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)) / math.log(2)


def _log2_add(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a if b == -math.inf else a + math.log2(1 + 2 ** (b - a))


def log2_error(n: int, k: int, g: int, memory_bits: int, egress_bits: int, kept_bits: int, label_bits: int = LABEL_BITS) -> float:
    """log2 of the guarantee's bound at `g`: `(g/N)^k + k · C(N, g+1) · 2^(M + E − q − (g+1)·b)`, the probability that a
    prover with `M` bits inside the boundary and `E` bits of egress is accepted and keeps `q` bits of uniform prior
    content (`Pous.Guarantees.StreamedWipe`, before the PRF's advantage)."""
    sampling = k * math.log2(g / n) if g > 0 else -math.inf
    compression = (math.log2(k) + _log2_choose(n, g + 1) + memory_bits + egress_bits - kept_bits
                   - (g + 1) * label_bits) if g + 1 <= n else -math.inf
    return _log2_add(sampling, compression)


def retained_bits(n: int, k: int, memory_bits: int, egress_bits: int, log2_delta: float,
                  label_bits: int = LABEL_BITS) -> tuple[int, int]:
    """`(q, g)`: the fewest bits `q` of uniform prior content for which an accepted wipe keeps them with probability at
    most `δ` (before the PRF's advantage), half of `δ` spent on sampling at the largest such `g`."""
    g = math.floor(n * 2 ** ((log2_delta - 1) / k))
    while g > 0 and k * math.log2(g / n) > log2_delta - 1:
        g -= 1
    q = math.ceil(math.log2(k) + _log2_choose(n, g + 1) + memory_bits + egress_bits - (g + 1) * label_bits
                  - (log2_delta - 1))
    return max(q, 0), g


def challenges_for(n: int, slack_labels: int, log2_delta: float) -> int:
    """The fewest `k` with `((N − D)/N)^k ≤ δ`: the sampling term alone, `D` labels not held."""
    if not 0 < slack_labels < n:
        raise ValueError("the slack is between 0 and N labels")
    return math.ceil(log2_delta / math.log2(1 - slack_labels / n))


__all__ = ["CLOCK", "COIN_DOMAIN", "COMPLETE", "DEPLOYED_FILL_BYTES", "DEPLOYED_K", "DEPLOYED_LABELS", "EGRESS_MAX_UNITS",
           "EGRESS_UNIT_BITS", "END_TAG", "KEY_DOMAIN", "LABEL_BITS", "LABEL_BYTES", "LABEL_TAG", "LATE", "MARK_BYTES",
           "PROTOCOL", "RECORDS_TREE", "RECORD_SCHEMA", "SESSION_TAG", "START_TAG", "WRONG", "BatchRecord", "Challenger",
           "Prover", "SignedWipe", "Verdict", "WipeCommitment", "WipeParams", "batch_digest", "challenges_for",
           "draw_indices", "egress_units", "end_message", "label", "labels", "log2_error", "mark_of", "message",
           "raw_clock_ns", "records_root", "retained_bits", "start_message", "verdict", "wipe_key"]
