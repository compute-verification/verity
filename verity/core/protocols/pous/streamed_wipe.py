"""The streamed-label wipe (experimental): a proof of secure erasure for a memory wipe, run by the memory challenger.
`PROTOCOL.md` ("The streamed-label wipe") is its specification and states the guarantee.

The memory challenger, the auditor's device on a machine the prover doesn't control, fills every memory tier of one
server with labels it derives itself, then reveals `k` indices drawn on live coins after the fill and checks each answer
against its own derivation, under a deadline per batch.  The labels are the auditor's randomness: the answers are
checked in the clear, and nothing of the prover's is opened.

~~~text
K     = derive(secret, KEY_DOMAIN, {server, session})            32 bytes, one key per server and wipe
L_i   = SHA-512(LABEL_TAG ‖ K ‖ u64 i)                           64 bytes, i in [0, N): one compression a label
c_j   = derive(coin_secret, COIN_DOMAIN, {server, session, n}).uniform(N, j)    sampled once the fill is complete
~~~

`secret` is the auditor's for this server and wipe, `coin_secret` is fresh from the OS when the fill completes.  A
session has four steps:

1. **Open.** The challenger claims the session id in its `SessionLog` (one session an id) and publishes the
   `WipeCommitment`: the server, the session, `N`, `k`, the batch size, the deadline a batch and the declared memory tiers,
   whose sizes add up to at least the fill.
2. **Fill.** `fill(chunk)` streams `L_0 … L_{N-1}` in order; the challenger reads its clock at the first and last chunk.
   `fill_by(send)` is the same fill streamed by `send`, a streamer on the challenger's own host given `K`.
3. **Challenge.** Once the fill is complete the challenger flips its coins, then reveals them a batch at a time
   (`challenge`), reads the receipt time before it hashes anything (`receive`), and checks every 64-byte answer against
   `L_{c_j}`.  The first late or wrong batch ends the session: no batch is retried, and no index is resampled.
4. **Sign.** One signature over `SHA-512(SESSION_TAG ‖ u64 len ‖ header ‖ records root)`, where the header is the
   commitment, how the session ended, the fill's times and the egress the network certifier counted over the session
   (`close`), and the records root is the frame-v3-sha512 root of the batch records.

A batch record is `u32 b ‖ u32 first ‖ u32 count ‖ u64 send_ns ‖ u64 receipt_ns ‖ u32 wrong ‖ SHA-512(indices ‖
answers)` (96 bytes).  The guarantee (`PROTOCOL.md`; Lean `Pous.Guarantees.StreamedWipe`) covers the sequential reveal,
batch size 1 (`WipeParams.covered`); a larger batch lets an answer depend on the whole batch, which the proof doesn't
cover yet, so `verdict` refuses it.  `verdict` is what a verifier checks of a signed session; its public output is the
verdict and the egress the network certifier counted, in whole MiB (15 bits), which the bound reads.  Its Lean port decides it
(`MemoryChallenger.StreamedWipe` and `wipe_verdict.judge` in `verity/core/devices/memory_challenger/lean/`, reading the
records' `to_json`), and `verdict` here is the reference that port is difftested against.  `retained_bits` and
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

from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.randomness import derive

from .memory_challenger import CLOCK_RESOLUTION_NS, SCHEMES, SessionLog, Signer, Verify

PROTOCOL = "verity-pous/streamed-wipe/v1"
KEY_DOMAIN = "verity/pous/streamed-wipe-labels/v1"
COIN_DOMAIN = "verity/pous/streamed-wipe-coins/v1"
LABEL_TAG = b"verity-pous/streamed-wipe/label/v1\0"
SESSION_TAG = b"verity-pous/streamed-wipe/session/v1\0"
RECORD_BINDING = b"verity-pous/streamed-wipe/batch-records/v1\0"
RECORDS_TREE = "verity-pous/streamed-wipe/batch-records/frame-v3-sha512/v1"
RECORD_SCHEMA = "pous/streamed-wipe/batch-record/v1"
LABEL_BYTES = 64
LABEL_BITS = 8 * LABEL_BYTES
COMPLETE, LATE, WRONG = "complete", "late", "wrong"

DEPLOYED_FILL_BYTES = 2_600_000_000_000
"""The deployed fill: 2.6 TB, every tier of one server."""
DEPLOYED_LABELS = DEPLOYED_FILL_BYTES // LABEL_BYTES
DEPLOYED_K = 74_000
"""The deployed challenge count: retained prior content at most about 1.02 GB plus the egress, at δ = 2^-40."""


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
                "clock": "CLOCK_MONOTONIC ns"}

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
    """A closed session: its commitment, how it ended, the fill's times, the batch records, the egress the network
    certifier counted over the session in bits, and the one signature over them."""

    commitment: WipeCommitment
    ended: str
    fill_start_ns: int
    fill_end_ns: int
    batches: tuple[BatchRecord, ...]
    egress_bits: int
    scheme: str
    public_key: bytes
    signature: bytes

    def message(self) -> bytes:
        return message(self.commitment, self.ended, self.fill_start_ns, self.fill_end_ns, self.batches, self.egress_bits)

    def to_json(self) -> dict[str, Any]:
        """The signed session as the Lean verdict reads it (`MemoryChallenger.StreamedWipe.Signed.ofJson`)."""
        return {"commitment": self.commitment.to_json(), "ended": self.ended, "fill_start_ns": self.fill_start_ns,
                "fill_end_ns": self.fill_end_ns, "batches": [r.to_json() for r in self.batches],
                "egress_bits": self.egress_bits, "scheme": self.scheme, "public_key": self.public_key.hex(),
                "signature": self.signature.hex()}


def message(commitment: WipeCommitment, ended: str, fill_start_ns: int, fill_end_ns: int,
            batches: Sequence[BatchRecord], egress_bits: int) -> bytes:
    """What the session's one signature signs: `SHA-512(SESSION_TAG ‖ u64 len ‖ header ‖ records root)`."""
    head = json.dumps({**commitment.header(), "ended": ended, "fill_start_ns": fill_start_ns, "fill_end_ns": fill_end_ns,
                       "batches": len(batches), "egress_bits": egress_bits}, sort_keys=True, separators=(",", ":")).encode()
    return _sha512(SESSION_TAG, len(head).to_bytes(8, "big"), head, records_root(commitment.session, batches))


class Challenger:
    """One streamed-label wipe of `server` under `session`, signed by `signer` (module docstring).  The session id goes
    into `log` before anything else.  `coin_secret` is sampled from the OS when the fill completes unless a test gives
    it."""

    def __init__(self, secret: bytes, server: bytes, session: bytes, params: WipeParams, signer: Signer, *,
                 log: SessionLog, clock: Callable[[], int] = time.perf_counter_ns, resolution_ns: int | None = None,
                 coin_secret: bytes | None = None) -> None:
        if resolution_ns is None:
            resolution_ns = time.clock_getres(time.CLOCK_MONOTONIC) * 1e9 if clock is time.perf_counter_ns else 1
        if resolution_ns > CLOCK_RESOLUTION_NS:
            raise ValueError(f"the clock ticks every {resolution_ns} ns, coarser than {CLOCK_RESOLUTION_NS} ns")
        self.key = wipe_key(secret, server, session)
        log.claim(session)
        self.commitment = WipeCommitment(server, session, params)
        self.signer, self.clock, self._coin_secret = signer, clock, coin_secret
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

    def fill(self, chunk: int) -> Iterator[tuple[int, bytes]]:
        """The labels in order, `chunk` at a time, as `(first index, bytes)`; once only."""
        if self.fill_start_ns is not None:
            raise RuntimeError("the fill streams once")
        if chunk < 1:
            raise ValueError("a chunk holds at least one label")
        self.fill_start_ns = self.clock()
        n = self.params.labels
        for start in range(0, n, chunk):
            count = min(chunk, n - start)
            yield start, labels(self.key, start, count)
            self.filled = start + count
        self.fill_end_ns = self.clock()

    def fill_by(self, send: Callable[[bytes, int], Any]) -> Any:
        """`fill`'s labels streamed by `send(K, N)`, which returns once it has sent the last; once only. Returns what
        `send` returns. The answers are checked against this module's `label` whatever `send` streamed."""
        if self.fill_start_ns is not None:
            raise RuntimeError("the fill streams once")
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

    def close(self, egress_bits: int) -> SignedWipe:
        """Sign the ended session, once, with the egress the network certifier counted over it, in bits."""
        if self.ended is None:
            raise RuntimeError("the session has not ended")
        if egress_bits < 0:
            raise ValueError("egress is a count of bits")
        if self.signed is None:
            assert self.fill_start_ns is not None and self.fill_end_ns is not None
            batches = tuple(self.batches)
            m = message(self.commitment, self.ended, self.fill_start_ns, self.fill_end_ns, batches, egress_bits)
            self.signed = SignedWipe(self.commitment, self.ended, self.fill_start_ns, self.fill_end_ns, batches, egress_bits,
                                     self.signer.scheme, self.signer.public_key, self.signer.sign(self.commitment.session, m))
        return self.signed

    def coins(self) -> tuple[int, ...]:
        """The drawn indices, public once the session is signed (the verifier's samples)."""
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
"""The verdict's unit stays in the least public size class, 16 bits: the verdict's one and the count's 15, so the count
saturates at 32 GiB less one MiB."""


def egress_units(egress_bits: int) -> int:
    """The certified egress in whole MiB, rounded up, saturating at `EGRESS_MAX_UNITS`: what the bound reads as `E`."""
    return min(-(-egress_bits // EGRESS_UNIT_BITS), EGRESS_MAX_UNITS)


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a signed session passes, and its egress in MiB; `reasons` are witness, and only `public()` is
    published.  The guarantee's `E` is `egress_mib` MiB, and a saturated count (`EGRESS_MAX_UNITS`) bounds nothing."""

    accepted: bool
    reasons: tuple[str, ...]
    egress_mib: int

    def public(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "egress_mib": self.egress_mib}


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
    return Verdict(not reasons, tuple(reasons), egress_units(signed.egress_bits))


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


__all__ = ["COIN_DOMAIN", "COMPLETE", "DEPLOYED_FILL_BYTES", "DEPLOYED_K", "DEPLOYED_LABELS", "EGRESS_MAX_UNITS",
           "EGRESS_UNIT_BITS", "KEY_DOMAIN", "LABEL_BITS", "LABEL_BYTES", "LABEL_TAG", "LATE", "PROTOCOL",
           "RECORDS_TREE", "RECORD_SCHEMA", "SESSION_TAG", "WRONG", "BatchRecord", "Challenger", "Prover",
           "SignedWipe", "Verdict", "WipeCommitment", "WipeParams", "batch_digest", "challenges_for", "draw_indices",
           "egress_units", "label", "labels", "log2_error", "message", "records_root", "retained_bits", "verdict",
           "wipe_key"]
