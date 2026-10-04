"""Secure erasure: a prover whose `M` bytes of volatile memory pass the audit cannot keep other content through all the
spot reads.

With its disks unplugged, a prover holds only volatile memory, and a fresh fill of all of it, read back fast enough that
no byte can be recomputed, leaves no room for anything else: secure erasure, the full-utilization case of a proof of
memory utilization.  `ErasureVerifier` enforces the accept rule of the Lean model's secure-erasure game over distinct
reads (`Pous.eraseWinsDraw` with `Pous.distinctReads`, `lean/Pous/Protocol/Model/ReadDraw.lean`) on the keyed page fill
(`Pous.PageFill.pageScheme`, `lean/Pous/Protocol/Model/PageFill.lean`):

1. **A fresh key.**  `K = fill_key(source, M, s)`, derived from the verifier's own fresh secret, is the fill challenge.
2. **The fill.**  The prover fills its `M` bytes with `F_K` (`KeyedFill`) and returns `h`.  The fill passes iff
   `h = Digest_K(F_K)`, a keyed ε-almost-universal digest of all `M` bytes, and `h` is in within `T_fill` of the send.
3. **Spot reads.**  Only then does the verifier draw `s` fresh byte indices (`spot_indices`: the subset sampler on a
   second key, which the prover never sees, revealed in a uniformly shuffled order) and reveal them one at a time.
   Read `j` passes iff its answer is `F_K[i_j]` and is in within `T_read` of index `i_j`'s reveal, and read `j + 1` is
   revealed within `T_round` of read `j`'s reveal.

The verifier accepts iff the fill and all `s` reads pass (`ErasureResult.accepted`).  It fails closed:

- an answer past its deadline is rejected without being compared;
- a step with no answer by its deadline is closed as rejected by `expire()`, which `run_erasure` calls by the
  verifier's clock;
- a read not revealed within `T_round` of the last reveal is closed as rejected, unrevealed (`challenge()` raises
  `Lapsed`);
- the first step that fails ends the audit, rejected, and no other index is revealed.

Every deadline runs from the verifier's own clock reading at the send, never from a time the prover reports.  An answer
names the read it answers (`respond(j, value)`), and one for any other read is refused, so a stale answer is never
credited to a later read.

**The deadlines.**  A read shows that a byte was held only if the byte cannot be recomputed in the time the prover has.
Each page of `F_K` is a ten-round Feistel-SHAKE256 permutation (`primitives.Feistel10`) of a zero page under the tweak
`K ‖ u64(page)`, so every byte of it costs nine rounds' absorbs: `page_keccak_f(page_bytes)` sequential Keccak-f of at
least `primitives.KECCAK_F_NS` each (`KeyedFill.recompute_ns`, 1.7352 ms for 1 MiB pages).  The Lean model reads that as
`D = 0` over a whole round, from one read's arrival at the prover to the next's.  Read `j + 1`'s index reaches the
prover no earlier than its reveal and, if its answer is in time, no later than `T_read` after it, so two arrivals are at
most `T_round + T_read` apart on the verifier's clock.  `ErasureVerifier` refuses deadlines unless
`T_read ≤ T_round` and `T_round + T_read < recompute_ns`; `T_round` defaults to the largest, `recompute_ns − 1 − T_read`.

**The guarantee.**  An audit of `M = P` whole pages of 1 MiB with at least 3046 reads at `P = 96112`, or 3430 at any
`1 ≤ P ≤ 2^64`, is covered by `Pous.Guarantees.SecureErasurePages96112` or `Pous.Guarantees.SecureErasurePagesFamily`
(`pinned_guarantee`; `ErasureVerifier.pin`).  Both are every-arrival claims at `ε = 2^-128`: a prover whose memory at
each read's arrival is at most `⌊(18/19)·8M⌋` bits is accepted with probability at most `ε`, and an accepted prover
cannot keep `M/19 + 32` bytes of uniform foreign content (independent of the fill, the pages' permutations and the reads,
and read from its memory alone) through all the reads, except with probability `ε`.  Neither says what a prover holds at any one instant.  `M` must be
the prover's whole volatile memory: memory outside the fill (the rest of the device, its on-chip SRAM, host DRAM) adds to
what a prover can keep.  The verifier refuses an audit outside both pins unless `research=True`.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache

from verity.randomness import Key, Part, derive

from .primitives import (
    FEISTEL_ROUNDS,
    KECCAK_F_NS,
    ROUND_PREFIX_BYTES,
    SHAKE256_RATE,
    Feistel10,
)

FILL_DOMAIN = "verity/pous/erasure-fill/v1"
"""`K`'s domain: the fill key, sent to the prover."""

SPOT_DOMAIN = "verity/pous/erasure-spot/v1"
"""The spot reads' key's domain: kept by the verifier, bound to `K`."""

PAGE_DOMAIN = 0x45
"""The first byte of every round-function input of a fill page."""

PAGE_BYTES = 1 << 20
"""A fill page: 1 MiB, so that recomputing any byte of it takes at least 1.7352 ms."""

PAGE_TWEAK_BYTES = 40
"""A page's tweak, `K ‖ u64(page)` little-endian."""

DIGEST_PRIME = (1 << 127) - 1
"""The digest's field, `GF(2^127 − 1)`."""

DIGEST_WORD_BYTES = 15
"""A digest word: 15 bytes little-endian, below `DIGEST_PRIME`, so two different words are different field elements."""

DIGEST_BYTES = 16

PIN_96112 = "Pous.Guarantees.SecureErasurePages96112"
"""The calibrated fill's guarantee: 96112 pages of 1 MiB, 3046 reads."""

PIN_FAMILY = "Pous.Guarantees.SecureErasurePagesFamily"
"""Every fill of `1 ≤ P ≤ 2^64` pages of 1 MiB, 3430 reads."""

CALIBRATED_PAGES = 96_112
"""The 4 Oct 2026 node-2 calibration's fill on an RTX PRO 6000 (93.86 GiB, `art:3340ecba`)."""

FAMILY_MAX_PAGES = 1 << 64

PINNED_READS = {PIN_96112: 3046, PIN_FAMILY: 3430}
"""The spot reads each pin is proved at."""


def required_reads(pages: int) -> int:
    """The fewest spot reads a pin covers at `pages` pages of 1 MiB: 3046 at 96112 pages, 3430 at any other."""
    return PINNED_READS[PIN_96112 if pages == CALIBRATED_PAGES else PIN_FAMILY]


def pinned_guarantee(size: int, spot_reads: int, page_bytes: int = PAGE_BYTES) -> tuple[str, tuple[str, ...]]:
    """The pin an audit of `size` bytes in pages of `page_bytes` with `spot_reads` reads cites, and why it does not
    cover the audit (empty: it does).  `SecureErasurePages96112` at exactly 96112 pages of 1 MiB, else
    `SecureErasurePagesFamily`.  Both are about whole pages of 1 MiB; the deadlines they need, `D = 0` over every round,
    `ErasureVerifier` enforces for every audit."""
    pages, cut = divmod(size, page_bytes)
    pin = PIN_96112 if page_bytes == PAGE_BYTES and not cut and pages == CALIBRATED_PAGES else PIN_FAMILY
    out = []
    if page_bytes != PAGE_BYTES:
        out.append(f"pages of {page_bytes} bytes; {pin} is about pages of {PAGE_BYTES}")
    if cut:
        out.append(f"M = {size} bytes is not a whole number of {page_bytes}-byte pages; {pin} is about whole pages")
    if pages + (cut > 0) > FAMILY_MAX_PAGES:
        out.append(f"{pages + (cut > 0)} pages; {pin} is about at most 2^64")
    if spot_reads < PINNED_READS[pin]:
        out.append(f"s = {spot_reads} reads is below the {PINNED_READS[pin]} of {pin}")
    return pin, tuple(out)


def page_keccak_f(page_bytes: int) -> int:
    """The sequential Keccak-f[1600] below which no byte of a fill page can be computed.  Every byte of a page is an
    output of round 9 or 10, and each of rounds 1 to 9 absorbs its whole input (prefix, tweak and half a page) before
    it squeezes.  A round's absorb streams with the previous round's squeeze, so a round costs its absorb, as in
    `primitives.pi2_keccak_f`."""
    return (FEISTEL_ROUNDS - 1) * ((ROUND_PREFIX_BYTES + PAGE_TWEAK_BYTES + page_bytes // 2) // SHAKE256_RATE + 1)


@cache
def _feistel(page_bytes: int) -> Feistel10:
    return Feistel10(page_bytes, PAGE_DOMAIN)


def _pages(chunks: Iterable[bytes], size: int, page_bytes: int) -> Iterator[bytes]:
    """`size` bytes, given in order in chunks of any length, as pages of `page_bytes`, the last one cut at `size`."""
    buf, total = b"", 0
    for chunk in chunks:
        total += len(chunk)
        buf += chunk
        while len(buf) >= page_bytes:
            yield buf[:page_bytes]
            buf = buf[page_bytes:]
    if total != size:
        raise ValueError(f"the digest is over {size} bytes, not {total}")
    if buf:
        yield buf


class KeyedFill:
    """`F_K` over `size` bytes, and `Digest_K`.

    Page `p` is `Feistel10(page_bytes, PAGE_DOMAIN).forward(0, K ‖ u64(p))` as `page_bytes` little-endian bytes, the
    last page cut at `size`.  `Digest_K` is the Lean model's `Pous.Digest.digest g z xs` over `GF(2^127 − 1)`, with
    `xs` the `R` pages as blocks of `W = ⌈page_bytes / 15⌉` 15-byte little-endian words (each page zero-padded to
    `15·W` bytes) and `(g, z) = K.uniform(2^127 − 1, "digest-g" | "digest-z")`: `Σ_r P_g(x_r) · z^r` with
    `P_g(x) = Σ_j x_j · g^(W−1−j)`, as 16 little-endian bytes.  Two different `size`-byte strings collide for at most
    a `((W − 1) + (R − 1)) / (2^127 − 1)` fraction of the keys (`Pous.Guarantees.DigestAU`).

    With `size` a whole number of pages this is the Lean model's `Pous.PageFill.pageScheme page_bytes R K` with each
    page's `Feistel10` as an ideal permutation: page `p` is `π_{K ‖ u64(p)}(0)` (`PageFill.page`, `PageFill.tweak`),
    and byte `i` is byte `i mod page_bytes` of page `⌊i / page_bytes⌋`."""

    def __init__(self, key: Key, size: int, page_bytes: int = PAGE_BYTES) -> None:
        if size < 1:
            raise ValueError("a fill has at least one byte")
        if page_bytes < 2 or page_bytes % 2:
            raise ValueError("a page is an even number of bytes, at least 2")
        self.key, self.size, self.page_bytes = key, size, page_bytes
        self.digest_key = (key.uniform(DIGEST_PRIME, "digest-g"), key.uniform(DIGEST_PRIME, "digest-z"))
        self._pages: dict[int, bytes] = {}
        self._digest: bytes | None = None

    @property
    def pages(self) -> int:
        return -(-self.size // self.page_bytes)

    @property
    def recompute_ns(self) -> int:
        """The least time in which any byte of `F_K` can be computed: a page's `page_keccak_f` at `KECCAK_F_NS`."""
        return page_keccak_f(self.page_bytes) * KECCAK_F_NS

    def tweak(self, p: int) -> bytes:
        """Page `p`'s tweak, `K ‖ u64(p)` little-endian."""
        return self.key.value + p.to_bytes(8, "little")

    def page(self, p: int) -> bytes:
        if not 0 <= p < self.pages:
            raise IndexError(f"page {p} is outside 0..{self.pages - 1}")
        whole = _feistel(self.page_bytes).forward(0, self.tweak(p)).to_bytes(self.page_bytes, "little")
        return whole[:self.size - p * self.page_bytes]

    def chunks(self) -> Iterator[bytes]:
        """`F_K`, page by page."""
        return map(self.page, range(self.pages))

    def at(self, i: int) -> int:
        """`F_K[i]`."""
        if not 0 <= i < self.size:
            raise IndexError(f"byte {i} is outside 0..{self.size - 1}")
        p, offset = divmod(i, self.page_bytes)
        if p not in self._pages:
            self._pages[p] = self.page(p)
        return self._pages[p][offset]

    def digest_of(self, chunks: Iterable[bytes]) -> bytes:
        """`Digest_K` of `size` bytes, given in order in chunks of any length."""
        g, z = self.digest_key
        width = -(-self.page_bytes // DIGEST_WORD_BYTES) * DIGEST_WORD_BYTES
        total, zr = 0, 1
        for page in _pages(chunks, self.size, self.page_bytes):
            page, h = page.ljust(width, b"\0"), 0
            for k in range(0, width, DIGEST_WORD_BYTES):
                h = (h * g + int.from_bytes(page[k:k + DIGEST_WORD_BYTES], "little")) % DIGEST_PRIME
            total, zr = (total + h * zr) % DIGEST_PRIME, zr * z % DIGEST_PRIME
        return total.to_bytes(DIGEST_BYTES, "little")

    def digest(self) -> bytes:
        """`Digest_K(F_K)`."""
        if self._digest is None:
            self._digest = self.digest_of(self.chunks())
        return self._digest


def fill_key(source: bytes, size: int, spot_reads: int, context: Mapping[str, Part] | None = None) -> Key:
    """`K`: `source` is the verifier's fresh secret, never the prover's; `context` names what else the audit is bound
    to (a device, a session)."""
    context = dict(context or {})
    if {"memory", "spot-reads"} & set(context):
        raise ValueError("the context's keys 'memory' and 'spot-reads' are the audit's own")
    return derive(source, FILL_DOMAIN, {**context, "memory": size, "spot-reads": spot_reads})


def spot_key(source: bytes, fill: Key) -> Key:
    """The spot reads' key: from the same secret, bound to `K`, never sent."""
    return derive(source, SPOT_DOMAIN, {"fill": fill.value})


def spot_indices(key: Key, spot_reads: int, size: int) -> tuple[int, ...]:
    """`s` distinct byte indices, uniform: `key.subset(s, size, "indices")`, shuffled by `key.uniform(j + 1,
    ("order", j))` for `j` from `s − 1` down to 1, so that no index tells anything about the next but that it differs.
    Every sequence of `s` distinct indices is equally likely: the Lean model's `Pous.distinctReads`, a uniform
    `Fin s ↪ Fin B`."""
    out = list(key.subset(spot_reads, size, "indices"))
    for j in range(spot_reads - 1, 0, -1):
        k = key.uniform(j + 1, ("order", j))
        out[j], out[k] = out[k], out[j]
    return tuple(out)


@dataclass(frozen=True, slots=True)
class FillChallenge:
    """The fill challenge: `K`'s 32 bytes, the memory size `M` and the page size, from which the prover computes
    `F_K` (`KeyedFill(Key(key), size, page_bytes)`)."""

    key: bytes
    size: int
    page_bytes: int


@dataclass(frozen=True, slots=True)
class ReadChallenge:
    """Spot read `j`: the byte at `index`."""

    j: int
    index: int


@dataclass(frozen=True, slots=True)
class DigestReply:
    digest: bytes


@dataclass(frozen=True, slots=True)
class ReadReply:
    j: int
    value: int


@dataclass(frozen=True, slots=True)
class Check:
    """One closed step: the fill (`j` and `index` None) or spot read `j` of byte `index`.  `on_time`: in within the
    step's deadline; `correct`: on time and the expected digest or byte (an answer past the deadline is never
    compared); `answered`: False for a step closed with no answer (`expire()`); `elapsed_ns` on the verifier's clock,
    from the send to the answer or the close.  A read whose reveal lapsed (`index` None, never revealed) is closed
    unanswered, with `elapsed_ns` from the last read's reveal."""

    j: int | None
    index: int | None
    correct: bool
    on_time: bool
    answered: bool
    elapsed_ns: int

    @property
    def passed(self) -> bool:
        return self.correct and self.on_time


@dataclass(frozen=True, slots=True)
class ErasureResult:
    """The fill's check, if the fill was closed, and the reads closed, in order.  `pin` is the guarantee the audit
    cites (`pinned_guarantee`), `gaps` where it does not cover the audit (only with `research=True`)."""

    spot_reads: int
    fill: Check | None
    reads: tuple[Check, ...]
    pin: str | None = None
    gaps: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        """The accept rule: the fill passed, and so did all `s` reads."""
        return (self.fill is not None and self.fill.passed and len(self.reads) == self.spot_reads
                and all(r.passed for r in self.reads))

    @property
    def certified(self) -> bool:
        """`pin` covers this audit: an accepted one may cite it."""
        return self.pin is not None and not self.gaps


class Lapsed(RuntimeError):
    """A step's deadline passed before the call that would have closed it or revealed the next read: the verifier closed
    the step as rejected, and the audit is rejected."""


class ErasureVerifier:
    """The fill, then `s` spot reads, each closed by its answer or its deadline before the next step.

    `fill_challenge()` sends `K` and starts `T_fill`; `receive_digest(h)` closes the fill.  `challenge()` reveals read
    `(j, index)` and starts its `T_read`; `respond(j, value)` closes read `j` and refuses any other.  The next read must
    be revealed by `reveal_by`, `T_round` after the last reveal: past it, `challenge()` closes that read unrevealed as
    rejected and raises `Lapsed`.  `expire()` closes the open step, or a lapsed reveal, as rejected once its deadline
    has passed (`overdue`).  `done` once all `s` reads are closed or a step has failed; `accepted` iff the fill and all
    `s` reads passed.

    `source` is the verifier's fresh secret for this audit (32 random bytes by default): reusing one reuses `K` and the
    indices.  Deadlines are inclusive, in the units of `clock`.  It refuses deadlines unless `T_read ≤ T_round` and
    `T_round + T_read < recompute_ns`, so that no two reads' arrivals at the prover are a page's recompute apart, and
    `T_round` defaults to `recompute_ns − 1 − T_read`.  It refuses an audit that neither pin covers
    (`pinned_guarantee`); `research=True` runs one anyway, and `gaps` says why it is not covered."""

    def __init__(self, size: int, spot_reads: int, *, fill_deadline_ns: int, read_deadline_ns: int,
                 round_deadline_ns: int | None = None, source: bytes | None = None,
                 context: Mapping[str, Part] | None = None, page_bytes: int = PAGE_BYTES,
                 clock: Callable[[], int] = time.perf_counter_ns, research: bool = False) -> None:
        if size < 1 or not 1 <= spot_reads <= size:
            raise ValueError(f"an audit reads from 1 to M = {size} bytes, not {spot_reads}")
        if fill_deadline_ns < 0 or read_deadline_ns < 0 or (round_deadline_ns is not None and round_deadline_ns < 0):
            raise ValueError("deadlines are nonnegative")
        source = secrets.token_bytes(32) if source is None else source
        self.key = fill_key(source, size, spot_reads, context)
        self.fill = KeyedFill(self.key, size, page_bytes)
        recompute = self.fill.recompute_ns
        if read_deadline_ns >= recompute:
            raise ValueError(f"T_read = {read_deadline_ns} ns reaches the {recompute} ns in which any byte "
                             f"of a {page_bytes}-byte page can be recomputed: a read would not show the byte was held")
        if round_deadline_ns is None:
            round_deadline_ns = recompute - 1 - read_deadline_ns
        if round_deadline_ns + read_deadline_ns >= recompute:
            raise ValueError(f"T_round + T_read = {round_deadline_ns + read_deadline_ns} ns reaches the {recompute} ns of "
                             f"a page's recompute: two reads' arrivals could be that far apart, and the guarantees' D = 0 "
                             f"round runs from one arrival to the next")
        if round_deadline_ns < read_deadline_ns:
            raise ValueError(f"T_round = {round_deadline_ns} ns is below T_read = {read_deadline_ns} ns: an answer in "
                             f"time would lapse the next read's reveal (T_read is at most {(recompute - 1) // 2} ns)")
        pin, gaps = pinned_guarantee(size, spot_reads, page_bytes)
        if gaps and not research:
            raise ValueError(f"the audit is outside {pin}: {'; '.join(gaps)} (research=True runs it anyway)")
        self.pin, self.gaps = pin, gaps
        self.size, self.spot_reads, self.clock = size, spot_reads, clock
        self.fill_deadline_ns, self.read_deadline_ns = fill_deadline_ns, read_deadline_ns
        self.round_deadline_ns = round_deadline_ns
        self._spot = spot_key(source, self.key)
        self._indices: tuple[int, ...] | None = None
        self._sent = False
        self._open: tuple[int | None, int | None, int] | None = None
        self._revealed: int | None = None
        self._fill_check: Check | None = None
        self._reads: list[Check] = []

    @property
    def reads(self) -> tuple[Check, ...]:
        return tuple(self._reads)

    @property
    def pending(self) -> int | None:
        """The open read's `j`, or None."""
        return None if self._open is None else self._open[0]

    @property
    def closes_at(self) -> int | None:
        """The open step's send time plus its deadline: on the verifier's clock past it, the step is rejected.  None
        with no step open."""
        if self._open is None:
            return None
        j, _, sent = self._open
        return sent + (self.fill_deadline_ns if j is None else self.read_deadline_ns)

    @property
    def reveal_by(self) -> int | None:
        """The last read's reveal plus `T_round`: the next read must be revealed by then.  None before the first
        reveal, with a step open, or once the audit is over."""
        if self._revealed is None or self._open is not None or self.done:
            return None
        return self._revealed + self.round_deadline_ns

    @property
    def overdue(self) -> bool:
        """The verifier's clock is past `closes_at`, or with no step open past `reveal_by`, so `expire()` closes the
        step as rejected."""
        if self._open is not None:
            return self.clock() > self.closes_at
        return self.reveal_by is not None and self.clock() > self.reveal_by

    @property
    def rejected(self) -> bool:
        """A step failed: the audit rejects, and reveals no other index."""
        closed = [self._fill_check, *self._reads] if self._fill_check is not None else []
        return any(not c.passed for c in closed)

    @property
    def done(self) -> bool:
        return self.rejected or len(self._reads) == self.spot_reads

    @property
    def accepted(self) -> bool:
        return self.result().accepted

    def fill_challenge(self) -> FillChallenge:
        """Send `K` and start the fill's clock."""
        if self._sent:
            raise RuntimeError("the fill challenge is sent")
        self._sent = True
        self._open = (None, None, self.clock())
        return FillChallenge(self.key.value, self.size, self.fill.page_bytes)

    def receive_digest(self, digest: bytes) -> Check:
        stop = self.clock()
        if self._open is None or self._open[0] is not None:
            raise RuntimeError("the fill is not open")
        elapsed = stop - self._open[2]
        on_time = elapsed <= self.fill_deadline_ns
        correct = on_time and type(digest) is bytes and digest == self.fill.digest()
        return self._close(Check(None, None, correct, on_time, True, elapsed))

    def challenge(self) -> tuple[int, int]:
        """Reveal the next read, `(j, index)`, and start its clock.  A step open within its deadline is refused; one
        past it is first closed as rejected, as by `expire()`, and the call raises `Lapsed`.  So does a call past
        `reveal_by`, which closes read `j` unrevealed as rejected."""
        if self._open is not None:
            if not self.overdue:
                raise RuntimeError("the open step is unanswered")
            self.expire()
            raise Lapsed("the open step's deadline passed, so it is closed as rejected: the audit is rejected")
        if self._fill_check is None:
            raise RuntimeError("the fill comes first")
        if self.rejected:
            raise RuntimeError("the audit is rejected: no other index is revealed")
        if self.done:
            raise RuntimeError(f"all {self.spot_reads} reads are closed")
        now, j = self.clock(), len(self._reads)
        if self._revealed is not None and now - self._revealed > self.round_deadline_ns:
            self._close(Check(j, None, False, False, False, now - self._revealed))
            raise Lapsed(f"read {j} would be revealed {now - self._revealed} ns after read {j - 1}, past T_round = "
                         f"{self.round_deadline_ns} ns: it is closed unrevealed, and the audit is rejected")
        if self._indices is None:
            self._indices = spot_indices(self._spot, self.spot_reads, self.size)
        self._revealed = now
        self._open = (j, self._indices[j], now)
        return j, self._indices[j]

    def respond(self, j: int, value: int) -> Check:
        stop = self.clock()
        if self._open is None:
            raise RuntimeError("no step is open")
        k, i, sent = self._open
        if k is None or j != k:
            state = "closed" if k is not None and j < k else "not revealed"
            raise RuntimeError(f"read {j} is {state}: an answer counts only for the open read")
        elapsed = stop - sent
        on_time = elapsed <= self.read_deadline_ns
        correct = on_time and type(value) is int and value == self.fill.at(i)
        return self._close(Check(j, i, correct, on_time, True, elapsed))

    def expire(self) -> Check:
        """Close the open step unanswered once its deadline has passed, or with none open the next read unrevealed
        once `reveal_by` has passed: rejected."""
        now = self.clock()
        if self._open is None:
            by = self.reveal_by
            if by is None:
                raise RuntimeError("no step is open")
            if now <= by:
                raise RuntimeError("the next read's reveal is within T_round")
            return self._close(Check(len(self._reads), None, False, False, False, now - self._revealed))
        if now <= self.closes_at:
            raise RuntimeError("the open step is within its deadline")
        j, i, sent = self._open
        return self._close(Check(j, i, False, False, False, now - sent))

    def _close(self, check: Check) -> Check:
        self._open = None
        if check.j is None:
            self._fill_check = check
        else:
            self._reads.append(check)
        return check

    def result(self) -> ErasureResult:
        return ErasureResult(self.spot_reads, self._fill_check, tuple(self._reads), self.pin, self.gaps)


def run_erasure(verifier: ErasureVerifier, send: Callable[[FillChallenge | ReadChallenge], None],
                receive: Callable[[int], DigestReply | ReadReply | None]) -> ErasureResult:
    """Drive `verifier` through the fill and its reads over a transport.  `send(msg)` sends a challenge;
    `receive(until)` returns the next reply in, or None once the verifier's clock is past `until`, the open step's
    `closes_at`, with none.

    A reply that answers the open step closes it; any other is dropped.  Once the deadline has passed the driver closes
    the open step with `expire()`, so a silent or slow prover never holds the audit open.  A transport that returns
    None before then fails closed: `expire()` refuses, and the audit stops with the step open, not accepted.  Each read
    is revealed as soon as the last is closed; if that comes past `reveal_by` anyway (a slow `send` or `receive`), the
    verifier closes the read unrevealed and the audit is rejected."""
    send(verifier.fill_challenge())
    while verifier.closes_at is not None:
        reply = receive(verifier.closes_at)
        if isinstance(reply, DigestReply):
            verifier.receive_digest(reply.digest)
        elif reply is None or verifier.overdue:
            verifier.expire()
    while not verifier.done:
        try:
            j, i = verifier.challenge()
        except Lapsed:
            break
        send(ReadChallenge(j, i))
        while verifier.pending == j:
            reply = receive(verifier.closes_at)
            if isinstance(reply, ReadReply) and reply.j == j:
                verifier.respond(j, reply.value)
            elif reply is None or verifier.overdue:
                verifier.expire()
    return verifier.result()


__all__ = ["CALIBRATED_PAGES", "PINNED_READS", "PIN_96112", "PIN_FAMILY", "Check", "DigestReply", "ErasureResult",
           "ErasureVerifier", "FillChallenge", "KeyedFill", "Lapsed", "ReadChallenge", "ReadReply", "fill_key",
           "page_keccak_f", "pinned_guarantee", "required_reads", "run_erasure", "spot_indices", "spot_key"]
