"""The bandwidth audit v0 over a PoUS store `E` (P2's `C*`), one window at a time.  Each device's store is set up under
a salt with its identity bound in (`draw_salt`), so devices holding one `W` hold different stores.

1. The verifier draws a fresh nonce from its own randomness, derives the window's key from its own secret, bound to
   `vk`, the tag and the nonce, and sends the seed `σ` (`Window.start`, which starts its clock).
2. Before `T = deadline_ns` (plus `rtt_allowance_ns`) the prover computes `y_j = Tag(σ, j, E_j)` for every block and
   returns the root of the tags' tree (`tag_pass`, then `Window.commit`, which stops the clock).
3. Only then does the verifier draw `s = samples` indices, uniform with replacement, from the same key.  A late
   commitment closes the window rejected and draws none.
4. The prover opens each: the block with its audit path to `vk`'s root, and the tags of its leaf with their path to the
   committed root (`TagCommitment.open`).  The verifier checks both paths and recomputes the tag (`Window.verdict`).
   Openings are not timed: the commitment already binds every tag.

A window is used once: a second start or commitment is refused, a nonce is never drawn twice, and nothing is redrawn.
One window over a store is open at a time, from its seed to its commitment or until `T + RTT` has passed without one:
the verifier refuses to start another before then, so no read serves two windows' ingress.  The verdict publishes only whether the window was accepted (`Verdict.public`).  `Params` refuses `T + RTT` at or past
`regen_floor_ns`, the fastest cooperating P2 root (`2 · ROOT_CALL_NS`), so no block is rebuilt from `W` within a window
(SeqRoot).  What an accepted window certifies, and on what, is `PROTOCOL.md`; `bound` computes it, for a certified tag
only (`tags.Tag.certified`).
"""

from __future__ import annotations

import math
import secrets
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from verity.core.protocols.pous.codec import Geometry
from verity.core.protocols.pous.params import SALT_BITS
from verity.core.protocols.pous.schemes.p2 import ROOT_CALL_NS
from verity.core.protocols.pous.verifier import MerkleTree, VerifierKey, leaf_hash, verify_path
from verity.ml.randomness import Key, derive

from . import tags
from .tags import SEED_BYTES

WINDOW_DOMAIN = "verity/bandwidth/window/v2"
SALT_DOMAIN = "verity/bandwidth/salt/v1"
NONCE_BYTES = 32

REGEN_FLOOR_NS = 2 * ROOT_CALL_NS
"""The fastest 16,448-bit root on the allowed hardware, 8 cores of one chiplet cooperating, 1,708,058 ns
(`pous.schemes.p2.ROOT_CALL_NS` is half of it).  Rebuilding a block of `C*` from `W` takes two roots in sequence, so a
window shorter than one cannot rebuild any block.  The fastest root measured on the RTX PRO 6000 is slower, 18.53 ms
(floor 2.26 ms, art:f26f917e0605e2974f44a9b0f8ae7f4fe4a78b95181deaf8d8c457ce98c64ad3)."""

FAILURES = ("late", "malformed", "store-path", "tag-path", "tag-mismatch")


@dataclass(frozen=True, slots=True)
class Params:
    """A window's parameters: the tag construction, `s`, `T`, the round-trip allowance and the rebuild floor."""

    tag: str
    samples: int
    deadline_ns: int
    rtt_allowance_ns: int = 0
    regen_floor_ns: int = REGEN_FLOOR_NS

    def __post_init__(self) -> None:
        tags.get(self.tag)
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError("a window draws at least one sample")
        if self.deadline_ns <= 0 or self.rtt_allowance_ns < 0:
            raise ValueError("the deadline is positive and the round-trip allowance nonnegative")
        if self.limit_ns >= self.regen_floor_ns:
            raise ValueError(f"T + RTT = {self.limit_ns} ns reaches the rebuild floor of {self.regen_floor_ns} ns: "
                             "a prover could rebuild a block from W within the window")

    @property
    def limit_ns(self) -> int:
        """`T` plus the round-trip allowance: the longest a prover may take, which the bound charges it with."""
        return self.deadline_ns + self.rtt_allowance_ns


def draw_salt(source: bytes, commitment: bytes, device: str) -> bytes:
    """The salt of `device`'s store, which PoUS's setup (`pous.setup`) takes in place of its own `draw_salt`: drawn after
    the commitment to `W` from the verifier's secret (or a later beacon round) as PoUS's is, with the identity the
    device attests bound in.  Devices holding one `W` then hold different stores, so their `vk`, seeds and tags differ."""
    if type(device) is not str or not device:
        raise ValueError("a device's identity is a nonempty str")
    return derive(source, SALT_DOMAIN, {"w": commitment, "device": device}).stream("salt", SALT_BITS // 8)


def window_key(source: bytes, vk: VerifierKey, tag: str, nonce: bytes) -> Key:
    """The key of a window of `tag` over `vk`'s store.  `source` is the verifier's own secret: from a public one the
    samples are known before the commitment, and a prover tags only the blocks they name.  `nonce` is drawn fresh for
    the window from the verifier's own randomness, never counted: a key that a window number fixes repeats in every
    verifier, tag or restart that counts from the same place, and a window whose samples are known is passed by tagging
    only them."""
    if type(nonce) is not bytes or len(nonce) != NONCE_BYTES:
        raise ValueError(f"a window's nonce is {NONCE_BYTES} bytes")
    return derive(source, WINDOW_DOMAIN, {"root": vk.root, "salt": vk.salt, "tag": tags.get(tag).name, "nonce": nonce})


def seed(key: Key) -> bytes:
    """`σ`, sent at the window's start."""
    return key.stream("seed", SEED_BYTES)


def samples(key: Key, blocks: int, s: int) -> tuple[int, ...]:
    """The `s` sampled indices, uniform on `0..blocks-1` with replacement, drawn only after the commitment."""
    return tuple(key.uniform(blocks, ("sample", t)) for t in range(s))


class Store:
    """The prover's `E`: the blocks of a PoUS store with their tree under `vk`'s root (`pous.verifier`)."""

    def __init__(self, params: Geometry, salt: bytes, blocks: Sequence[bytes]) -> None:
        if len(blocks) != params.blocks or any(len(b) != params.block_bytes for b in blocks):
            raise ValueError(f"E is {params.blocks} blocks of {params.block_bytes} bytes")
        self.params, self.salt, self.blocks = params, salt, tuple(bytes(b) for b in blocks)
        self.tree = MerkleTree([leaf_hash(params, salt, i, b) for i, b in enumerate(self.blocks)])

    @classmethod
    def from_segments(cls, params: Geometry, salt: bytes, segments: Iterable[Sequence[bytes]]) -> Store:
        return cls(params, salt, [b for segment in segments for b in segment])

    @property
    def root(self) -> bytes:
        return self.tree.root


@dataclass(frozen=True, slots=True)
class Opening:
    """Sample `index` opened: its block and audit path to `vk`'s root, and its leaf's tags with their path."""

    index: int
    block: bytes
    store_path: tuple[bytes, ...]
    leaf_tags: tuple[bytes, ...]
    tag_path: tuple[bytes, ...]


class TagCommitment:
    """The prover's tags for one window, and the tree whose `root` it commits."""

    def __init__(self, store: Store, tag: str, seed: bytes, ys: Sequence[bytes]) -> None:
        self.store, self.tag, self.seed, self.ys = store, tags.get(tag), seed, tuple(ys)
        if len(self.ys) != store.params.blocks:
            raise ValueError("one tag per block")
        self.tree = MerkleTree(tags.leaves(self.tag, seed, self.ys))

    @property
    def root(self) -> bytes:
        return self.tree.root

    def open(self, j: int) -> Opening:
        per = tags.tags_per_leaf(self.tag)
        c = j // per
        return Opening(j, self.store.blocks[j], self.store.tree.path(j), self.ys[c * per:(c + 1) * per],
                       self.tree.path(c))


def tag_pass(store: Store, tag: str, seed: bytes) -> TagCommitment:
    """The honest prover's pass: every block's tag, committed.  The GPU tagger (`benchmarks/bandwidth`) computes the
    same root."""
    t = tags.get(tag)
    t.check_block(store.params.block_bytes)
    keyed = t.keyed(seed)
    return TagCommitment(store, tag, seed, [keyed(j, b) for j, b in enumerate(store.blocks)])


@dataclass(frozen=True, slots=True)
class Verdict:
    """A window's outcome.  `failure` (one of `FAILURES`), the failing sample's `index` and the commitment's
    `elapsed_ns` stay with the verifier; only `public()` leaves it."""

    accepted: bool
    elapsed_ns: int | None = None
    failure: str | None = None
    index: int | None = None

    def public(self) -> dict[str, bool]:
        return {"accepted": self.accepted}


class Verifier:
    """The verifier of one store `vk`: it opens each window once, under a fresh nonce from `nonces` (its own
    randomness) and its own secret `source`, and one at a time."""

    def __init__(self, vk: VerifierKey, params: Params, source: bytes,
                 clock: Callable[[], int] = time.perf_counter_ns,
                 nonces: Callable[[int], bytes] = secrets.token_bytes) -> None:
        tags.get(params.tag).check_block(vk.params.block_bytes)
        self.vk, self.params, self.source, self.clock, self._nonces = vk, params, source, clock, nonces
        self._used: set[bytes] = set()
        self._open: Window | None = None

    def _claim(self, w: Window) -> bytes:
        o = self._open
        if o is not None and o._live():
            raise RuntimeError("another window over this store is open: it closes at its commitment or at T + RTT")
        nonce = self._nonces(NONCE_BYTES)
        if nonce in self._used:
            raise RuntimeError("the verifier drew a nonce it has used: a window's key is never reused")
        self._used.add(nonce)
        self._open = w
        return nonce

    def window(self) -> Window:
        return Window(self)


class Window:
    """One window: `start()` sends `σ`, `commit(root)` takes the prover's root and returns the samples, and
    `verdict(openings)` checks them, each once and in that order."""

    def __init__(self, verifier: Verifier) -> None:
        self._v = verifier
        self._key: Key | None = None
        self._started: int | None = None
        self._root: bytes | None = None
        self._elapsed: int | None = None
        self._samples: tuple[int, ...] | None = None
        self._verdict: Verdict | None = None

    def _live(self) -> bool:
        """Whether the window holds its store: started, without a commitment, and within `T + RTT` on its clock."""
        return (self._started is not None and self._elapsed is None
                and self._v.clock() - self._started <= self._v.params.limit_ns)

    def start(self) -> bytes:
        """`σ`, and the clock starts; refused, and the window left unstarted, while another over the store is open."""
        if self._key is not None:
            raise RuntimeError("the window has started")
        nonce = self._v._claim(self)
        self._key = window_key(self._v.source, self._v.vk, self._v.params.tag, nonce)
        s = seed(self._key)
        self._started = self._v.clock()
        return s

    def commit(self, root: bytes) -> tuple[int, ...]:
        """Stop the clock on the prover's root; the samples if it is on time, else none and the window is rejected."""
        stop = self._v.clock()
        if self._started is None:
            raise RuntimeError("the window has not started")
        if self._elapsed is not None:
            raise RuntimeError("the window has a commitment")
        self._elapsed = stop - self._started
        if type(root) is not bytes or len(root) != tags.DIGEST_BYTES:
            self._verdict = Verdict(False, self._elapsed, "malformed")
            return ()
        if self._elapsed > self._v.params.limit_ns:
            self._verdict = Verdict(False, self._elapsed, "late")
            return ()
        self._root = root
        self._samples = samples(self._key, self._v.vk.params.blocks, self._v.params.samples)
        return self._samples

    def verdict(self, openings: Sequence[Opening] = ()) -> Verdict:
        if self._elapsed is None:
            raise RuntimeError("the window has no commitment")
        if self._verdict is None:
            self._verdict = self._check(openings)
        return self._verdict

    def _check(self, openings: Sequence[Opening]) -> Verdict:
        vk, params, elapsed = self._v.vk, self._v.params, self._elapsed
        tag = tags.get(params.tag)
        if len(openings) != len(self._samples):
            return Verdict(False, elapsed, "malformed")
        keyed, head = tag.keyed(seed(self._key)), tags.leaf_head(tag, seed(self._key))
        per, blocks = tags.tags_per_leaf(tag), vk.params.blocks
        count = tags.leaf_count(tag, blocks)
        for i, o in zip(self._samples, openings):
            c = i // per
            size = min(per, blocks - c * per)
            if (not isinstance(o, Opening) or o.index != i or type(o.block) is not bytes
                    or len(o.leaf_tags) != size
                    or any(type(y) is not bytes or len(y) != tag.tag_bytes for y in o.leaf_tags)):
                return Verdict(False, elapsed, "malformed", i)
            if not vk.check_path(i, o.block, o.store_path):
                return Verdict(False, elapsed, "store-path", i)
            if not verify_path(self._root, count, c, tags.leaf_hash(head, c, o.leaf_tags), o.tag_path):
                return Verdict(False, elapsed, "tag-path", i)
            if keyed(i, o.block) != o.leaf_tags[i - c * per]:
                return Verdict(False, elapsed, "tag-mismatch", i)
        return Verdict(True, elapsed)


EPSILON = Fraction(1, 64)
SAMPLES = 1849
QUERIES_LOG2 = 64
SLACK_LOG2 = 41
"""The proof's deployed point: `ε_s = 1/64`, `s = 1,849`, at most `2^64` oracle queries in a window, and
`λ = 64·k + 41`, which together keep the error under `2^-40` at `N = 2^19`.  `2^64` is past what any fleet computes in
a window (an RTX PRO 6000 makes about `2^26` BLAKE3 compressions in 1.7 ms), so the query bound needs no assumption."""


def certified_blocks(blocks: int, epsilon: Fraction = EPSILON) -> int:
    """`k = ⌈(1 − ε_s)·N⌉`."""
    return math.ceil((1 - Fraction(epsilon)) * blocks)


def samples_for(blocks: int, error_log2: int, epsilon: Fraction = EPSILON) -> int:
    """The fewest `s` with `((k − 1)/N)^s ≤ 2^error_log2`, the sampling term of the error."""
    k = certified_blocks(blocks, epsilon)
    return math.ceil(error_log2 * math.log(2) / math.log((k - 1) / blocks))


@dataclass(frozen=True, slots=True)
class Bound:
    """`Bandwidth.Guarantees.AuditRO` for one device and window: an accepted window read at least `ingress_bytes` of
    device memory, except with probability at most `error`, the sum of its three terms."""

    blocks: int
    certified_blocks: int
    block_bytes: int
    chip_bytes: int
    side_bytes: int
    slack_bits: int
    ingress_bytes: int
    error_samples: float
    error_guess: float
    error_compression_log2: int

    @property
    def error(self) -> float:
        return self.error_samples + self.error_guess + 2.0 ** self.error_compression_log2


def bound(params: Params, blocks: int, block_bytes: int, *, chip_bytes: int, side_bytes_per_s: float,
          epsilon: Fraction = EPSILON, queries_log2: int = QUERIES_LOG2, slack_log2: int = SLACK_LOG2) -> Bound:
    """`ingress ≥ k·b − S − c − G·b − λ` bits with `k = ⌈(1 − ε_s)·N⌉`, except with probability
    `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ` (`Bandwidth.Guarantees.AuditRO`, PROTOCOL.md §5).  `S` is the chip's storage,
    `c = cap_side·(T + RTT)`, `G = 0` because `Params` refuses a window as long as a rebuild, `q ≤ 2^queries_log2` and
    `λ = queries_log2·k + slack_log2`, so `C(q, k)/2^λ ≤ q^k/2^λ = 2^-slack_log2`.  Only a certified tag has a bound."""
    tag = tags.get(params.tag)
    if not tag.certified:
        raise ValueError(f"{tag.name} is an uncertified fast check: nothing bounds a device's ingress through it")
    k = certified_blocks(blocks, epsilon)
    lam = queries_log2 * k + slack_log2
    side = math.ceil(side_bytes_per_s * params.limit_ns / 1e9)
    ingress_bits = k * 8 * block_bytes - 8 * chip_bytes - 8 * side - lam
    error_samples = math.exp(params.samples * math.log((k - 1) / blocks)) if k > 1 else 0.0
    return Bound(blocks, k, block_bytes, chip_bytes, side, lam, max(0, ingress_bits // 8), error_samples,
                 float(params.samples * tag.guess(block_bytes)), queries_log2 * k - lam)
