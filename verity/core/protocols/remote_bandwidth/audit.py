"""The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`): a verifier with no hardware at the site certifies how
many reads of a node's registered blocks the node made in each window, from receipts chained pass to pass.

1. Registration.  The prover commits to its payload blocks (`payload_tree`: the weights, KV pages and filler, each an
   hm96-sha512/row/v2 leaf) and sends the root.  Only then does the verifier draw the codec seed from its own secret and
   a fresh nonce (`register`), and the prover encodes every block under it (`Store`) with the codec the registration
   names (`codec.DEFAULT`, p1024 masked at k = 3, unless another is named).  The registration also names the receipt
   kind (`receipts.KINDS`: sketch-sha512 by default, or block-sha512 for comparison), whether its receipts bind
   PoUW's tile outputs (`bind`, off by default: PoUW's opt-in join), the block layout its kernels' sketch vectors
   follow (`receipts.LAYOUTS`: store-order/v0, a vector one registered block's bytes; `receipts.plan_of` checks a
   kernel's map), and sketch-sha512's rows (`receipts.ROWS`: 16 by default, or 64; 0 under block-sha512).
   `Registration.digest` binds the node, the store's size, the payload root, the codec seed, the codec's name, the
   receipt kind, `bind`, the layout and the rows.  The verifier keeps one live registration a node (`Registry`): a
   second while one is live is refused and draws nothing, and a registration ends only when the verifier closes it,
   counting its sessions, an unended one as failed.
2. Coins.  Before the session the challenger draws two live coins a window, its challenge `σ_t` and its openings coin,
   and commits to them (`coins.Challenger`).  The prover gets the commitment first (`Session.commitment`).
3. A session of `windows` windows.  The verifier releases `σ_0` (`Session.start`), and at each window's end the next
   window's challenge, or the close after the last (`Session.boundary`), one-way.  A window lasts at least the stated
   `T` (`Params.window_ns`, 60 s by default).
4. Chained passes.  Pass i of window t reads the blocks of its plan (`receipts.Plan`: distinct blocks, the whole store by
   default) and computes their receipts under `seed_(t,i) = H(σ_t, i, root_(t,i−1))` (`receipts.seed`; the root
   before chained to every append received before the pass's own root, `chained`), each of its block alone, or under
   a registration that binds z also of PoUW's tile output `z` before the read (`Tiles`); it commits them in a tree
   bound to the plan and sends the root and the plan, whole or as its change from the previous pass's
   (`Plan.changed`), with the step it covers, as soon as the pass ends (`Session.receive`).  A pass covers one
   step `s` of the served circuit's execution (the engine step), and its plan is the blocks the circuit reads at `s`:
   the steps are public, and strictly increase over the session's passes.  The chain runs across the session's windows: a window's first
   pass follows the root of the previous window's last.  On a boundary message the prover finishes the pass in flight,
   which is its window's last, and starts the next window's chain.  Blocks written after the codec seed (KV pages)
   join the store mid-session: their payloads committed (`append_tree`, `Session.append`) before any pass whose plan
   holds them, and so bound into its seed, at fresh store indices from N up, each encoded under its bound key and its
   append's coin `c_a`, which the verifier draws and sends in reply to the append (`codec.encode(..., bound=True,
   coin=c_a)`, `Store.encode`), so no choice the prover made before committing them depends on their encodings;
   every coin is drawn once, and an append no root follows, or one sent again, rejects the session (`coin`);
   a block released (`Session.release`) is in no later plan, and one the release names withdrawn (released before its
   encode landed, so never written) was in no earlier plan either; and an appended block is in no plan of a window L
   or more after the window it joined (`Params.lifetime`): one kept longer is released and appended again, under a
   fresh index and a fresh coin, so a fresh bound key, and a session that defers residency to the encode
   (`Params.deferred`) has L ≥ 2.
5. Deadline.  Window t's roots must all arrive by `D_t = b_t + A`, where `b_t` is the verifier's clock when it sent the
   boundary message and `A` the allowance (`Params.deadline_ns`: the message's trip down, the pass in flight, its root's
   reduction and its trip up, `timing.deadline`).  A late root, one out of order, one with no step or a step not above
   the previous pass's, a plan that repeats a block, or one that holds an appended block past L rejects the session.
   A window with fewer than `min_passes` roots is rejected.
6. Openings.  Once window t's roots are final (the next window's first root has arrived, or `D_t` has passed), the
   challenger releases the window's openings coin, and the verifier draws `s = samples` distinct receipts uniformly from
   the window's `U_t = Σ_i L_i` with it (`Session.close`).  The prover opens each (`Prover.open`): the block, its receipt
   leaf with its salt and path to the pass's root, the payload's salt and path to the payload root, and, under a
   registration that binds z, PoUW's evidence for the tile before it; a block appended after the seed carries its
   payload's digest `d_j` too, its path going to its append's root.
7. Verification (`Session.verify`), each opening: the leaf's path under the pass's root; the block's words canonical
   and their decoding (under the bound key of the `d_j` it carries and its append's coin, for an appended block) on
   its path under the payload root or its append's; under a registration that binds z, `z` recomputed by PoUW's tile
   check (`Tiles.recompute`); the receipt recomputed from the block (and `z`) under the registration's
   kind (under sketch-sha512, the block's sketch from the pass's signs first).  A window is accepted only if every opening checks,
   and a rejected window ends its session.
8. The record (`Session.record`): per window each pass's length and step, its challenge's release and its deadline on the
   verifier's clock, its openings coin's release, an `IntegrityProfile` over its receipts with `Subset(s)` drawn, and
   each pass's head (the append chain its seed binds) and stale bound (`WindowRecord.heads`, `WindowRecord.stale`);
   and the chain over the session's appends (`Record.appends`);
   `ledger` turns it, with the node's measured capacities and a squaring ceiling at its codec's width (`Rebuild`), into
   each γ term over the session's span and over each window's own (`WindowLedger`), μ_sample read per window and
   pooled over the session's horizon (`pooled`, `floors`).  It credits a receipt of a block appended after the seed
   only on the property's verdict on the circuits that wrote it (`credit_post_seed`, I2's: the verifier runs property/v1
   on their public Program and binds it to the session, `post_seed`, in its Lean definition), and then charges what their carriers may
   hold (μ_carrier); without one, those receipts are certified but not credited.  On the served
   circuit's requirement linked to the passes (`requirement`, a `RequirementLink`: I4's plan and memory-map units and
   the census's layout rule, the memory-map unit's verdict on the record's appends and heads among them, `mapped`),
   each pass's credit is its receipts at the link's D bits each, at most the circuit's reads at its step
   (`Ledger.linked`), less the share ε of it the units' draw leaves unproved, its δ added to the ledger's error
   (`Ledger.error`);
   without one the credit is the node's reads of its store.
   What an accepted window certifies, and on what, is `PROTOCOL.md`.

The verifier's clock is injected (`clock`, nanoseconds, `time.monotonic_ns` by default).  Openings are checked in the
clear here, as the specification of what each zero-knowledge proof unit proves on the deployed path (`PROTOCOL.md` §8).
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import math
import os
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol

from verity.core.protocols.profile import IntegrityProfile, Subset
from verity.ml.commitments import Commitment, CommitmentDomain, MerkleTree, Opening as MerkleOpening, RangeIndexedDomain
from verity.ml.commitments import verify_opening
from verity.ml.randomness import derive, frame

from . import codec, coins, lean_verdict, receipts
from .codec import BLOCK_BYTES, SEED_BYTES, Codec
from .receipts import DIGEST_BYTES, LIMITS, SALT_BYTES, Plan

PROTOCOL_ID = "verity/remote-bandwidth/v0"
CODEC_DOMAIN = "verity/remote-bandwidth/codec/v0"
OPENINGS_DOMAIN = "verity/remote-bandwidth/openings/v0"
PAYLOAD_DOMAIN = b"verity/remote-bandwidth/payload/v0\0"
APPEND_DOMAIN = b"verity/remote-bandwidth/append/v0\0"
APPENDS_DOMAIN = b"verity/remote-bandwidth/appends/v0\0"
CHAINED_DOMAIN = b"verity/remote-bandwidth/chained/v0\0"
REGISTRATION_DOMAIN = b"verity/remote-bandwidth/registration/v0\0"
TILE_DOMAIN = b"verity/remote-bandwidth/stub-tile/v0\0"
PAYLOAD_SCHEMA = receipts.SCHEMA
NONCE_BYTES = 32
NODE_BYTES = 64
QUERIES_LOG2 = 64
SLACK_LOG2 = 41
WINDOW_NS = 60 * 10**9
SAMPLES = 33_273
DELTA = 2.0 ** -40

FAILURES = ("late", "order", "step", "plan", "lifetime", "append", "coin", "release", "few", "malformed",
            "receipt-path", "codec", "payload-path", "tile", "receipt")
STEP_LIMIT = 1 << 64
"""A step is an integer below this, memory-map/v1's."""
LIFETIME = 60
"""L, the lifetime cap, in windows: a block appended after the codec seed is in no plan of a window L or more after the
window it joined (the window of the first root received after its append), so it leaves the credited store within L
windows unless the prover appends it again, which gives it a fresh index and a fresh coin, and so a fresh bound key.
60 windows is an hour at the stated window."""


@dataclass(frozen=True, slots=True)
class Params:
    """A session's parameters: the store's size `N`, the allowance `A` from a window's boundary message to its roots'
    deadline, the windows `W`, the openings `s` per window at `delta` (33,273 at 2^-40 by default, 555 a second of the
    stated window), the fewest passes a window may certify, the stated window `T` (`window_ns`, 60 s by default: no
    boundary message leaves sooner after its window's challenge), the store's codec, receipt kind, whether its
    receipts bind z, its block layout and its sketch rows (None for the kind's default, `receipts.sketch_rows`), the
    registration's, the lifetime cap L on an appended block, in windows (`LIFETIME`), and whether the prover defers an
    appended block's residency to its encode (`deferred`, memory-map/v1's deferred residency), which needs L ≥ 2: a
    block appended in window w is resident from w or w + 1, so its re-key falls at age L − 1, and at L = 1 that is
    before it may be resident."""

    blocks: int
    deadline_ns: int
    windows: int
    samples: int = SAMPLES
    delta: float = DELTA
    min_passes: int = 1
    window_ns: int = WINDOW_NS
    codec: Codec = codec.DEFAULT
    receipt: str = receipts.DEFAULT
    bind: bool = False
    layout: str = receipts.LAYOUT
    rows: int | None = None
    lifetime: int = LIFETIME
    deferred: bool = False

    def __post_init__(self) -> None:
        for name in ("blocks", "deadline_ns", "windows", "samples", "min_passes", "window_ns", "lifetime"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} is a positive integer")
        if type(self.deferred) is not bool:
            raise ValueError("deferred is a bool")
        if self.deferred and self.lifetime < 2:
            raise ValueError(f"a session that defers residency has a lifetime of at least 2 windows, not "
                             f"{self.lifetime}: a block appended in window w is resident from w or w + 1, so its "
                             "re-key falls at age L − 1, which at L = 1 is before it may be resident")
        if not 0 < self.delta < 1:
            raise ValueError("0 < delta < 1")
        if type(self.codec) is not Codec:
            raise ValueError("a session's codec is a Codec")
        receipts.receipt_kind(self.receipt)
        _bind(self.bind)
        receipts.block_layout(self.layout)
        receipts.sketch_rows(self.receipt, self.rows)

    @property
    def coins(self) -> int:
        """2W: each window's challenge and its openings coin."""
        return 2 * self.windows


def _node(node: bytes) -> bytes:
    if type(node) is not bytes or not 1 <= len(node) <= NODE_BYTES:
        raise ValueError(f"a node's id is 1 to {NODE_BYTES} bytes")
    return node


def _bind(bind: object) -> bool:
    if type(bind) is not bool:
        raise ValueError("bind is True or False: whether the receipts bind PoUW's tile outputs")
    return bind


def payload_domain(node: bytes, blocks: int) -> CommitmentDomain:
    binding = hashlib.sha512(PAYLOAD_DOMAIN + frame(_node(node))).digest()
    return CommitmentDomain(binding, 0, RangeIndexedDomain(0, blocks), hash="sha512")


def payload_tree(node: bytes, payloads: Sequence[bytes], salts: Sequence[bytes]) -> MerkleTree:
    """The prover's registration commitment: a frame-v3-sha512 tree with a block a leaf, block j hidden under
    `salts[j]` (`receipts.hidden`)."""
    if any(type(m) is not bytes or len(m) != BLOCK_BYTES for m in payloads):
        raise ValueError(f"a payload block is {BLOCK_BYTES} bytes")
    if len(salts) != len(payloads):
        raise ValueError("a salt per payload block")
    return MerkleTree(payload_domain(node, len(payloads)),
                      {j: receipts.hidden(m, y) for j, (m, y) in enumerate(zip(payloads, salts))}, lambda _j: PAYLOAD_SCHEMA)


def append_domain(registration: bytes, first: int, count: int) -> CommitmentDomain:
    """The domain of an append's payload tree: store indices `first` … `first + count − 1` of the registration whose
    digest is `registration`."""
    if type(registration) is not bytes or len(registration) != DIGEST_BYTES:
        raise ValueError(f"a registration's digest is {DIGEST_BYTES} bytes")
    if type(first) is not int or type(count) is not int or first < 0 or count < 1:
        raise ValueError("an append is of count ≥ 1 blocks from a nonnegative first index")
    binding = hashlib.sha512(APPEND_DOMAIN + registration).digest()
    return CommitmentDomain(binding, 0, RangeIndexedDomain(first, first + count), hash="sha512")


def append_tree(registration: bytes, first: int, digests: Sequence[bytes], salts: Sequence[bytes]) -> MerkleTree:
    """The prover's commitment to the payloads of blocks written after the codec seed, at store indices `first` …,
    from their digests `d_j` (`codec.digest`) alone, so a payload never leaves the device that wrote it: leaves as
    `payload_tree`'s (block j hidden under `salts[j − first]`, its `x_j` the digest), in `append_domain`.  Its root
    commits each block's `d_j` before the append's coin is drawn, and so, with the coin, its bound key and its
    encoding, before any pass whose plan holds it."""
    if not digests or any(type(d) is not bytes or len(d) != DIGEST_BYTES for d in digests):
        raise ValueError(f"an append is the digests of one or more payload blocks, {DIGEST_BYTES} bytes each")
    if len(salts) != len(digests):
        raise ValueError("a salt per payload block")
    return MerkleTree(append_domain(registration, first, len(digests)),
                      {first + u: receipts.hidden_digest(d, y) for u, (d, y) in enumerate(zip(digests, salts))},
                      lambda _j: PAYLOAD_SCHEMA)


def append_head(head: bytes | None, first: int, count: int, root: bytes) -> bytes:
    """The chain over a session's appends, in the order the verifier receives them, after one more:
    SHA512(APPENDS_DOMAIN ‖ head ‖ u64be first ‖ u64be count ‖ root), `head` 64 zero bytes (None) before the first."""
    if type(first) is not int or type(count) is not int or first < 0 or count < 1:
        raise ValueError("an append is of count ≥ 1 blocks from a nonnegative first index")
    if type(root) is not bytes or len(root) != DIGEST_BYTES or (head is not None and (
            type(head) is not bytes or len(head) != DIGEST_BYTES)):
        raise ValueError(f"an append's root and the chain's head are {DIGEST_BYTES} bytes")
    return hashlib.sha512(APPENDS_DOMAIN + (head or bytes(DIGEST_BYTES)) + first.to_bytes(8, "big")
                          + count.to_bytes(8, "big") + root).digest()


def chained(previous: bytes, head: bytes | None) -> bytes:
    """What a pass's seed takes as the root before it (`receipts.seed`'s `previous`): that root itself while no block
    has been appended before the pass's own root (`head` None), else SHA512(CHAINED_DOMAIN ‖ root ‖ head), `head` the
    chain over every append received before the pass's root (`append_head`).  So the seed of a pass comes after every
    block its plan can hold: a block appended after it is in no plan of the pass, and one that carried the pass's
    receipts of other blocks would be a fixed point of SHA-512."""
    if head is None:
        return previous
    if type(previous) is not bytes or len(previous) != DIGEST_BYTES or type(head) is not bytes or len(head) != DIGEST_BYTES:
        raise ValueError(f"the root before a pass and the appends' head are {DIGEST_BYTES} bytes")
    return hashlib.sha512(CHAINED_DOMAIN + previous + head).digest()


@dataclass(frozen=True, slots=True)
class Registration:
    """What the store is: the node, `N`, the payload root, the codec seed drawn after it, the codec it names, the
    receipt kind its sessions use, whether their receipts bind PoUW's tile outputs (`bind`), the block layout its
    kernels' sketch vectors follow (`receipts.LAYOUTS`), and sketch-sha512's rows k_s (0 under block-sha512)."""

    node: bytes
    blocks: int
    payload_root: bytes
    codec_seed: bytes
    codec: Codec
    receipt: str
    bind: bool
    layout: str
    rows: int

    @property
    def digest(self) -> bytes:
        """SHA512(REGISTRATION_DOMAIN ‖ frame(node) ‖ u64be N ‖ payload root ‖ codec seed ‖ frame(codec name) ‖
        frame(receipt kind) ‖ u8 bind ‖ frame(layout) ‖ u32be rows): what a record and a session's coin commitment
        cite, and `root_(0,−1)`, the root a session's first pass follows."""
        return hashlib.sha512(REGISTRATION_DOMAIN + frame(self.node) + self.blocks.to_bytes(8, "big") + self.payload_root
                              + self.codec_seed + frame(self.codec.name) + frame(self.receipt)
                              + bytes((self.bind,)) + frame(self.layout) + self.rows.to_bytes(4, "big")).digest()

    @property
    def key(self) -> codec.Key:
        return codec.key(self.codec_seed, self.codec)


def register(source: bytes, node: bytes, blocks: int, payload_root: bytes, codec: Codec = codec.DEFAULT,
             receipt: str = receipts.DEFAULT, *, bind: bool = False, layout: str = receipts.LAYOUT,
             rows: int | None = None, nonce: bytes | None = None) -> Registration:
    """The verifier's side of registration: the codec seed, drawn from its own secret `source` and a fresh nonce only
    after the payload root has arrived.  Drawn before it, the prover would register `decode(0)` and hold a store of
    zeros; drawn after, each encoded block costs k square roots a word to produce, whatever the payload.  Every block
    registered here is under the seed's key; a block written after the seed takes its bound key under its append's
    coin (`codec.binding`, PROTOCOL.md §3) and joins the store by an append (`Store.append`, `Session.append`).  None
    of the
    receipt kind, `bind`, the block layout and the sketch rows (`receipts.sketch_rows`: the kind's default if None) is
    in the codec seed's context, since the store depends on none of them; the digest binds all four.  Each call draws
    afresh: the verifier draws through its `Registry`, which allows a node one live registration."""
    receipts.store_blocks(blocks)
    if type(codec) is not Codec:
        raise ValueError("a registration names a Codec")
    receipts.receipt_kind(receipt)
    _bind(bind)
    receipts.block_layout(layout)
    rows = receipts.sketch_rows(receipt, rows)
    if blocks > codec.max_blocks:
        raise ValueError(f"a store under {codec.name} is at most {codec.max_blocks} blocks")
    if type(payload_root) is not bytes or len(payload_root) != DIGEST_BYTES:
        raise ValueError(f"the payload root is {DIGEST_BYTES} bytes")
    nonce = secrets.token_bytes(NONCE_BYTES) if nonce is None else nonce
    if type(nonce) is not bytes or len(nonce) != NONCE_BYTES:
        raise ValueError(f"a registration's nonce is {NONCE_BYTES} bytes")
    seed = derive(source, CODEC_DOMAIN, {"node": _node(node), "blocks": blocks, "payload": payload_root,
                                         "codec": codec.name, "nonce": nonce}).stream("codec", SEED_BYTES)
    return Registration(node, blocks, payload_root, seed, codec, receipt, bind, layout, rows)


class Tiles(Protocol):
    """PoUW's side of a receipt, under a registration that binds z (`bind`; under one that does not, a receipt is of
    its block alone, and the prover and the verifier take None for tiles).  `output(seed, j)` is `z_(i,j)`, the noised
    output of the tile the prover computes before reading block j in the pass whose seed is `seed`; `evidence` is what
    PoUW's check reads to recompute it; and `recompute`, the verifier's, returns `z_(i,j)` from that evidence or None
    if PoUW's check fails.  PoUW's noise for a pass is derived from the pass's seed, so `z` differs every pass whatever
    the weights and activations, and a pass reads each block once, with a tile of its own, so no `z` unlocks two
    receipts."""

    def output(self, seed: bytes, j: int) -> bytes: ...

    def evidence(self, seed: bytes, j: int) -> bytes: ...

    def recompute(self, seed: bytes, j: int, evidence: bytes) -> bytes | None: ...


def _tiles(tiles: Tiles | None, bind: bool) -> Tiles | None:
    if (tiles is None) == bind:
        raise ValueError("a registration that binds z takes PoUW's tiles, and one that does not takes None")
    return tiles


@dataclass(frozen=True, slots=True)
class NoiseTiles:
    """A stub of PoUW's tile recompute: `z_(i,j)` is `z_bytes` bytes of SHA-512 in counter mode over
    TILE_DOMAIN ‖ seed ‖ u64be j, and the evidence is empty.  It stands where PoUW's noised product goes; it does no
    tensor work and certifies none."""

    z_bytes: int = 64

    def output(self, seed: bytes, j: int) -> bytes:
        m = TILE_DOMAIN + seed + j.to_bytes(8, "big")
        out = b"".join(hashlib.sha512(m + c.to_bytes(4, "big")).digest() for c in range(-(-self.z_bytes // 64)))
        return out[:self.z_bytes]

    def evidence(self, seed: bytes, j: int) -> bytes:
        return b""

    def recompute(self, seed: bytes, j: int, evidence: bytes) -> bytes | None:
        return self.output(seed, j) if evidence == b"" else None


@dataclass(frozen=True, slots=True)
class Append:
    """Blocks written after the codec seed, the prover's: store indices `first` … `first + count − 1`, their payloads'
    digests `d_j` (kept beside the blocks), their salts, and their payload tree (`append_tree`), whose root the
    verifier receives before any pass whose plan holds one."""

    first: int
    digests: tuple[bytes, ...]
    salts: tuple[bytes, ...]
    tree: MerkleTree

    @property
    def count(self) -> int:
        return len(self.digests)

    @property
    def root(self) -> bytes:
        return self.tree.commitment.root


class Store:
    """The prover's encoded store `E` under a registration, with its payload tree and the payloads' salts, and the
    blocks appended after the codec seed (`append`) at the next fresh store indices, each encoded under its bound key
    and its append's coin once the verifier has sent it (`encode`; None until then, its payload `pending`).  A
    released block is kept, since a window's draws may still open it from a pass before its release."""

    def __init__(self, registration: Registration, payloads: Sequence[bytes], salts: Sequence[bytes]) -> None:
        if len(payloads) != registration.blocks:
            raise ValueError(f"the registration is of {registration.blocks} blocks")
        self.registration = registration
        self.payloads, self.salts = tuple(payloads), tuple(salts)
        self.payload_tree = payload_tree(registration.node, self.payloads, self.salts)
        if self.payload_tree.commitment.root != registration.payload_root:
            raise ValueError("these payloads are not the registered ones")
        k = registration.key
        self.blocks: list[bytes | None] = [codec.encode(m, j, k) for j, m in enumerate(self.payloads)]
        self.appends: list[Append] = []
        self.pending: dict[int, tuple[bytes, ...]] = {}

    @property
    def extent(self) -> int:
        """The store's next fresh index: N, and every block appended since."""
        return self.registration.blocks + sum(a.count for a in self.appends)

    @property
    def head(self) -> bytes | None:
        """The chain over the store's appends (`append_head`), None without one: what the seed of a pass binds once
        the verifier has every one of them (`chained`)."""
        out = None
        for a in self.appends:
            out = append_head(out, a.first, a.count, a.root)
        return out

    def append(self, payloads: Sequence[bytes], salts: Sequence[bytes]) -> Append:
        """Payload blocks written after the codec seed, at the next fresh store indices, committed in one `append_tree`
        by their digests, kept beside them; their encodings wait on the append's coin (`encode`)."""
        reg, first = self.registration, self.extent
        if first + len(payloads) > reg.codec.max_blocks:
            raise ValueError(f"a store under {reg.codec.name} is at most {reg.codec.max_blocks} blocks")
        digests = tuple(codec.digest(m) for m in payloads)
        a = Append(first, digests, tuple(salts), append_tree(reg.digest, first, digests, salts))
        self.blocks.extend([None] * a.count)
        self.pending[first] = tuple(payloads)
        self.appends.append(a)
        return a

    def encode(self, first: int, coin: bytes) -> None:
        """The blocks of the append from `first`, encoded under their bound keys and its coin `c_a`, the verifier's
        reply to its root (`codec.encode(..., bound=True, coin=c_a)`): once, for an append awaiting its coin."""
        payloads = self.pending.get(first)
        if payloads is None:
            raise ValueError(f"no append from block {first} awaits its coin")
        k = self.registration.key
        encoded = [codec.encode(m, first + u, k, bound=True, coin=coin) for u, m in enumerate(payloads)]
        del self.pending[first]
        self.blocks[first:first + len(encoded)] = encoded

    def committed(self, j: int) -> tuple[bytes, tuple[bytes, ...], bytes]:
        """Block j's payload salt, its path to the payload root or its append's, and its digest `d_j` (empty under
        the seed's key)."""
        if j < self.registration.blocks:
            return self.salts[j], self.payload_tree.open(j).path, b""
        a = self.appends[bisect.bisect_right([x.first for x in self.appends], j) - 1]
        return a.salts[j - a.first], a.tree.open(j).path, a.digests[j - a.first]


@dataclass(frozen=True, slots=True)
class Pass:
    """One pass of the prover's: its window, its index in the window, its seed, its plan, the receipts of the plan's
    blocks in plan order, its leaves' salts, and their tree."""

    window: int
    index: int
    seed: bytes
    plan: Plan
    receipts: tuple[bytes, ...]
    salts: tuple[bytes, ...]
    tree: MerkleTree

    @property
    def root(self) -> bytes:
        return self.tree.commitment.root


@dataclass(frozen=True, slots=True)
class Opening:
    """Receipt (i, p) of a window opened, position p of pass i, block j = plan_i[p]: the block, its receipt leaf's
    receipts, salt and path to pass i's root, the payload's salt and path to the payload root (to its append's root,
    for a block appended after the seed), PoUW's evidence for `z_(i,j)` (empty unless the registration binds z), and
    the payload's digest `d_j` that an appended block's bound key is derived from, with its append's coin, which the
    verifier holds (empty under the seed's key)."""

    pass_index: int
    position: int
    block: bytes
    leaf: bytes
    leaf_salt: bytes
    receipt_path: tuple[bytes, ...]
    payload_salt: bytes
    payload_path: tuple[bytes, ...]
    tile: bytes
    digest: bytes = b""

    @property
    def size(self) -> int:
        """Its bytes on the wire, with u32 and u64 headers for the indices and the variable fields (`d_j`'s presence
        follows from j)."""
        return (4 + 8 + len(self.block) + len(self.leaf) + len(self.leaf_salt) + len(self.payload_salt)
                + DIGEST_BYTES * (len(self.receipt_path) + len(self.payload_path)) + len(self.tile) + len(self.digest)
                + 4 * 3)


class Prover:
    """The honest prover: each pass reads its plan's blocks (every block by default), computes their receipts and
    commits them, each leaf under a fresh salt from `entropy`.  `tiles` is PoUW's if the registration binds z, and
    None if it does not."""

    def __init__(self, store: Store, tiles: Tiles | None, commitment: coins.CoinCommitment, *,
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        self.store, self.commitment, self.entropy = store, commitment, entropy
        self.tiles = _tiles(tiles, store.registration.bind)

    def run(self, challenge: coins.CoinOpening, i: int, previous: bytes, plan: Plan | None = None, *,
            head: bytes | None = None) -> Pass:
        """Pass i of the window whose challenge is `challenge`, after the root `previous`, its seed binding `head`, the
        chain over every append the verifier will have received before this pass's root (`chained`; None without
        one, the honest prover's `Store.head`): refused unless the coin is a window's challenge and opens against the
        session's commitment."""
        t = getattr(challenge, "index", None)
        reason = coins.coin_reason(self.commitment, challenge, t)
        if reason is None and t >= self.commitment.count // 2:
            reason = f"coin {t} is an openings coin, not a window's challenge"
        if reason is not None:
            raise ValueError(reason)
        return self.pass_under(t, i, receipts.seed(challenge.coin, i, chained(previous, head)), plan)

    def pass_under(self, window: int, i: int, seed: bytes, plan: Plan | None = None) -> Pass:
        """Pass i of `window`'s receipts and tree under `seed`, however it was derived."""
        plan = plan or Plan.whole(self.store.registration.blocks)
        reg = self.store.registration
        receipter, tiles = receipts.receipter(reg.receipt, seed, reg.rows), self.tiles
        ys = tuple(receipter(j, self.store.blocks[j], None if tiles is None else tiles.output(seed, j)) for j in plan)
        salts = tuple(self.entropy(SALT_BYTES) for _ in range(receipts.leaves(plan.length)))
        return Pass(window, i, seed, plan, ys, salts, receipts.tree(seed, plan, ys, salts))

    def open(self, ps: Pass, p: int) -> Opening:
        j = ps.plan.block(p)
        c = p // receipts.PER_LEAF
        leaf = ps.tree.open(c)
        salt, path, d = self.store.committed(j)
        return Opening(ps.index, p, self.store.blocks[j], b"".join(ps.receipts[receipts.PER_LEAF * c:receipts.PER_LEAF * (c + 1)]),
                       ps.salts[c], leaf.path, salt, path, b"" if self.tiles is None else self.tiles.evidence(ps.seed, j),
                       d)


@dataclass(frozen=True, slots=True)
class Verdict:
    """A window's outcome.  `failure` is one of `FAILURES`, `at` the failing draw (pass, position) when there is one."""

    window: int
    accepted: bool
    failure: str | None = None
    at: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class WindowRecord:
    """What an accepted window established: chained passes of `lengths` reads of distinct registered blocks each, made
    after its challenge left at `released_ns` and committed by `deadline_ns` on the verifier's clock; its openings coin
    released at `opened_ns`; the openings' law (`profile`); of each pass's reads, `post_seed` of blocks appended
    after the codec seed (none if empty), which the ledger credits only on the property's verdict; the step each
    pass covers, `steps`, strictly increasing; each pass's head, `heads`, the chain over every append received
    before its root (`append_head`, None before the first), which its seed binds (`chained`); and each pass's stale
    bound, `stale`: N, the registered store's size, plus the counts of the session's appends that joined window
    `window` − L or before (L `Params.lifetime`), an append joining the window of the first root received after it,
    which is the window of the first pass whose head chains it.  That is the store index below which every appended
    block is past the cap, which a re-key's old index is below at the cap, and the same for every pass of a window;
    `Session._stale_below` takes it as the first index of the earliest append received before the pass's root that
    joined a window above `window` − L, or the store's extent at that root if none did.  All three are empty in a
    record no session made, such as `timing.stated`'s model."""

    window: int
    lengths: tuple[int, ...]
    released_ns: int
    deadline_ns: int
    opened_ns: int
    profile: IntegrityProfile
    post_seed: tuple[int, ...] = ()
    steps: tuple[int, ...] = ()
    heads: tuple[bytes | None, ...] = ()
    stale: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.post_seed == ():
            object.__setattr__(self, "post_seed", (0,) * len(self.lengths))
        if len(self.post_seed) != len(self.lengths) or any(
                type(a) is not int or not 0 <= a <= n for a, n in zip(self.post_seed, self.lengths)):
            raise ValueError("a window's post-seed reads are a count of each pass's, at most its length")
        if self.steps != () and (
                len(self.steps) != len(self.lengths)
                or any(type(s) is not int or not 0 <= s < STEP_LIMIT for s in self.steps)
                or any(a >= b for a, b in zip(self.steps, self.steps[1:]))):
            raise ValueError("a window's steps are each pass's, strictly increasing integers from 0 below 2^64")
        if self.heads != () and (
                len(self.heads) != len(self.lengths)
                or any(h is not None and (type(h) is not bytes or len(h) != DIGEST_BYTES) for h in self.heads)):
            raise ValueError(f"a window's heads are each pass's, None or a {DIGEST_BYTES}-byte append chain")
        if self.stale != () and (
                len(self.stale) != len(self.lengths) or any(type(b) is not int or b < 0 for b in self.stale)):
            raise ValueError("a window's stale bounds are each pass's, a store index")

    @property
    def passes(self) -> int:
        return len(self.lengths)

    @property
    def receipts(self) -> int:
        """U_t, the window's receipts."""
        return sum(self.lengths)

    def credited(self, post_seed: bool) -> tuple[int, ...]:
        """Each pass's credited receipts: all of them if `post_seed` blocks are credited, else the registration's."""
        return self.lengths if post_seed else tuple(n - a for n, a in zip(self.lengths, self.post_seed))

    @property
    def span_ns(self) -> int:
        """`D_t − b_(t−1)`: the verifier-fixed span its reads fall in, from its challenge's release to its deadline.
        Consecutive windows' spans overlap by A, and no read is in two windows' passes."""
        return self.deadline_ns - self.released_ns

    def to_json(self) -> dict:
        return {"deadline_ns": self.deadline_ns, "heads": [None if h is None else h.hex() for h in self.heads],
                "lengths": list(self.lengths), "opened_ns": self.opened_ns, "post_seed": list(self.post_seed),
                "profile": self.profile.to_json(), "released_ns": self.released_ns, "span_ns": self.span_ns,
                "stale": list(self.stale), "steps": list(self.steps), "window": self.window}


@dataclass(frozen=True, slots=True)
class Record:
    """An accepted session: its registration, its coin commitment's root, its store's `blocks`, every window's record,
    and the receipt kind the registration names, whether its receipts bind z, its codec, its block layout and its sketch
    rows (its digest binds all five; the record repeats them, since what a receipt rests on depends on the kind, `bind`
    and the rows, μ_rebuild on the codec, and which bytes a credited block is on the layout), the blocks the
    session appended after the codec seed and released, the lifetime cap it held them to (`Params.lifetime`, None
    in a record no session made), and the chain over its appends in the order received (`append_head`, None without
    one), which, with each pass's head and stale bound (`WindowRecord.heads`, `WindowRecord.stale`), the
    memory-map unit's verdict is on (`RequirementLink.mapped`).  `public` is what acceptance makes public;
    `profile` is the protocol's profile, each window's schedule and openings law (the γ terms need the node's
    capacities: `ledger`)."""

    registration: bytes
    coins: bytes
    blocks: int
    block_bytes: int
    windows: tuple[WindowRecord, ...]
    receipt: str = receipts.DEFAULT
    bind: bool = False
    codec: Codec = codec.DEFAULT
    layout: str = receipts.LAYOUT
    rows: int = receipts.SKETCH_ROWS
    appended: int = 0
    released: int = 0
    lifetime: int | None = None
    appends: bytes | None = None

    @property
    def start_ns(self) -> int:
        return self.windows[0].released_ns

    @property
    def end_ns(self) -> int:
        """The last window's deadline: the earliest the next session over this store may release its first coin, less
        the trusted round-trip floor."""
        return self.windows[-1].deadline_ns

    @property
    def span_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def passes(self) -> int:
        return sum(w.passes for w in self.windows)

    @property
    def receipts(self) -> int:
        return sum(w.receipts for w in self.windows)

    def public(self) -> dict:
        """What acceptance makes public.  `heads[t][i]`, pass i of window t's head, is a hiding commitment like
        `appends`.  `stale[t][i]` is protocol structure: `blocks` plus the sum of `count` over the session's appends
        (first, count, root), in the order received, that joined window t − `lifetime` or before, where an append
        joined the window of the first pass whose head chains it (`append_head`); the same for every i.  So `stale`
        adds nothing beyond the appends' extents and the windows they joined, which the verifier sees in the clear (each
        append's count, the first index it assigns, and the root that follows) and which `appends` commits to.  This
        record lists neither, only their chain and total (`appended`), so read from it alone `stale` shows the blocks
        appended L windows back, window by window.  Whether the extents should be public at all is a separate question
        about `appends` (the counts per window depend on the workload), not about `stale`."""
        return {"appended": self.appended, "appends": None if self.appends is None else self.appends.hex(),
                "bind": self.bind, "block_bytes": self.block_bytes, "blocks": self.blocks,
                "codec": self.codec.name, "codec_params": self.codec.to_json(), "coins": self.coins.hex(),
                "heads": [[None if h is None else h.hex() for h in w.heads] for w in self.windows],
                "layout": self.layout, "lifetime": self.lifetime, "passes": [w.passes for w in self.windows],
                "post_seed": [sum(w.post_seed) for w in self.windows], "protocol": PROTOCOL_ID,
                "receipt": self.receipt, "receipts": [w.receipts for w in self.windows],
                "registration": self.registration.hex(), "released": self.released, "rows": self.rows,
                "span_ns": self.span_ns, "stale": [list(w.stale) for w in self.windows]}

    def profile(self) -> dict:
        return {"windows": [w.to_json() for w in self.windows]}


class SessionError(RuntimeError):
    """A call the session's state refuses: out of turn, after the session ended, or before a window's roots are final."""


@dataclass(slots=True)
class _Window:
    challenge: bytes
    released_ns: int
    roots: list[bytes] = field(default_factory=list)
    plans: list[Plan] = field(default_factory=list)
    post: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    heads: list[bytes | None] = field(default_factory=list)
    stale: list[int] = field(default_factory=list)
    boundary_ns: int | None = None
    final: bool = False
    draws: tuple[tuple[int, int], ...] | None = None
    opened_ns: int | None = None
    verdict: Verdict | None = None


def _merged(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Half-open spans `[a, b)` as the fewest disjoint ones covering the same blocks, in order."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _hits(spans: list[tuple[int, int]], first: int, count: int) -> bool:
    """Whether merged `spans` hold a block of `first … first + count − 1`."""
    r = bisect.bisect_right(spans, (first, math.inf))
    return (r > 0 and spans[r - 1][1] > first) or (r < len(spans) and spans[r][0] < first + count)


def _within(spans: list[tuple[int, int]], first: int, count: int) -> bool:
    """Whether merged `spans` hold every block of `first … first + count − 1`."""
    r = bisect.bisect_right(spans, (first, math.inf))
    return r > 0 and spans[r - 1][1] >= first + count


class Session:
    """The verifier's side of one session, driven by the coins it releases and the roots it receives, on its own clock.
    `challenger` holds the session's coins (`coins.Challenger`), committed before the session.  `not_before_ns`, which
    every session names, is the earliest its first coin may leave on the verifier's clock: the previous session's
    `Record.end_ns` over the same store, less the trusted round-trip floor if there is one, or the clock at
    registration for the store's first session, so that no read serves receipts of two sessions.  `tiles` is PoUW's
    check if the registration binds z, and None if it does not.  Blocks appended after the codec seed (`append`) are
    the session's: they take indices from N up, and a later session over the store starts again from N."""

    def __init__(self, params: Params, registration: Registration, tiles: Tiles | None, challenger: coins.Challenger, *,
                 not_before_ns: int, clock: Callable[[], int] = time.monotonic_ns) -> None:
        if (registration.blocks != params.blocks or registration.codec != params.codec
                or registration.receipt != params.receipt or registration.bind != params.bind
                or registration.layout != params.layout
                or registration.rows != receipts.sketch_rows(params.receipt, params.rows)):
            raise ValueError("the parameters are not the registration's")
        if type(not_before_ns) is not int or not_before_ns < 0:
            raise ValueError("not_before_ns is the previous session's last deadline over the store, or the clock at "
                             "registration for its first: a nonnegative integer")
        c = challenger.commitment
        if c.registration != registration.digest or c.count != params.coins:
            raise ValueError(f"the challenger's coins are not this session's: {params.coins} coins for this registration")
        self.params, self.registration, self.challenger = params, registration, challenger
        self.tiles = _tiles(tiles, registration.bind)
        self.clock, self.not_before_ns = clock, not_before_ns
        self.windows: list[_Window] = []
        self.filling = 0
        self.failed: Verdict | None = None
        self._payload_domain = payload_domain(registration.node, registration.blocks)
        self.extent = registration.blocks
        self.released = 0
        self._appends: list[tuple[int, CommitmentDomain, Commitment, bytes]] = []
        self._roots: set[bytes] = set()
        self._head: bytes | None = None
        self._joined: list[int] = []
        self._gone: list[tuple[int, int]] = []
        self._planned: list[tuple[int, int]] = []
        self._plan = Plan.whole(registration.blocks)
        self._step = -1

    @property
    def commitment(self) -> coins.CoinCommitment:
        """The session's coin commitment, sent to the prover before the first coin."""
        return self.challenger.commitment

    def _release(self, t: int, now: int) -> coins.CoinOpening:
        opening = self.challenger.open(t)
        self.windows.append(_Window(opening.coin, now))
        return opening

    def _live(self) -> None:
        if self.failed is not None:
            raise SessionError(f"the session ended: window {self.failed.window} was rejected ({self.failed.failure})")

    def _fail(self, t: int, failure: str, at: tuple[int, int] | None = None) -> Verdict:
        self.failed = Verdict(t, False, failure, at)
        if t < len(self.windows):
            self.windows[t].verdict = self.failed
        return self.failed

    def deadline(self, t: int) -> int | None:
        """`D_t = b_t + A`, once window t's boundary message has been sent."""
        b = self.windows[t].boundary_ns
        return None if b is None else b + self.params.deadline_ns

    def start(self) -> coins.CoinOpening:
        """`σ_0`, released now: refused before `not_before_ns`."""
        if self.windows:
            raise SessionError("a session starts once")
        now = self.clock()
        if now < self.not_before_ns:
            raise SessionError(f"the session's first coin may leave at {self.not_before_ns} ns, not {now}: the session "
                               "before it over this store may still be reading")
        return self._release(0, now)

    def boundary(self) -> coins.CoinOpening | None:
        """The end of the next unended window: the following window's challenge, or None, the close, after the last.
        Refused sooner than `window_ns` after the window's challenge left."""
        self._live()
        now = self.clock()
        t = next((t for t, w in enumerate(self.windows) if w.boundary_ns is None), None)
        if t is None:
            raise SessionError("every window has ended" if self.windows else "the session has not started")
        if now < self.windows[t].released_ns + self.params.window_ns:
            raise SessionError(f"window {t} is {self.params.window_ns} ns, stated: its boundary may leave at "
                               f"{self.windows[t].released_ns + self.params.window_ns} ns, not {now}")
        self.windows[t].boundary_ns = now
        return self._release(t + 1, now) if t + 1 < self.params.windows else None

    def append(self, root: bytes, count: int) -> tuple[int, bytes] | Verdict:
        """The root of an append's payload tree (`append_tree`), `count` blocks written after the codec seed: they take
        the store's next fresh indices, from the first returned, and a plan received after this may hold them, until
        the lifetime cap: they join the window of the next root received, and no plan of a window `Params.lifetime` or
        more after it may hold them.  The seed of every pass whose root comes after this binds it (`chained`).  The
        reply is the first index and the append's coin `c_a`, drawn now (`coins.Challenger.append_coin`), under which
        the prover encodes the blocks (`Store.encode`) and an opening decodes them.  Its rejection, `append`, if it is
        malformed or would take the store past its codec's `max_blocks`; `coin`, drawing none, if its root was sent
        before (a retry never redraws) or the last window's roots are final, so no root can follow it."""
        self._live()
        if (type(root) is not bytes or len(root) != DIGEST_BYTES or type(count) is not int or count < 1
                or self.extent + count > self.params.codec.max_blocks):
            return self._fail(self.filling, "append")
        if root in self._roots or (len(self.windows) == self.params.windows and self.windows[-1].final):
            return self._fail(self.filling, "coin")
        first, coin = self.extent, self.challenger.append_coin()
        self._appends.append((first, append_domain(self.registration.digest, first, count), Commitment(root, count),
                              coin))
        self._roots.add(root)
        self._head = append_head(self._head, first, count, root)
        self.extent += count
        return first, coin

    def release(self, runs: Sequence[tuple[int, int]], withdrawn: Sequence[tuple[int, int]] = ()) -> Verdict | None:
        """Blocks the prover stops holding, `runs` (first, count) of the store in a plan's canonical form, none
        released before: no plan received after this holds one.  `withdrawn`, empty or in the same form, names the
        appended blocks among them that the prover withdrew (memory-map/v1's `withdrawn`: released before their encode
        landed, so never written), none of which a plan received before this held.  Its rejection, `release`,
        otherwise."""
        self._live()
        try:
            gone = Plan(tuple(runs), self.extent).runs
            out = Plan(tuple(withdrawn), self.extent).runs if len(withdrawn) else ()
        except (TypeError, ValueError):
            return self._fail(self.filling, "release")
        if any(self._released(f, c) for f, c in gone):
            return self._fail(self.filling, "release")
        spans = _merged([(f, f + c) for f, c in gone])
        if any(f < self.params.blocks or not _within(spans, f, c) or _hits(self._planned, f, c) for f, c in out):
            return self._fail(self.filling, "release")
        self._gone = _merged(self._gone + spans)
        self.released += sum(c for _, c in gone)
        return None

    def _released(self, first: int, count: int) -> bool:
        return _hits(self._gone, first, count)

    def _stale_below(self, t: int) -> int:
        """The store index below which every appended block joined a window `Params.lifetime` or more before `t`:
        appends join windows in index order, so the stale ones are a prefix of them."""
        k = bisect.bisect_right(self._joined, t - self.params.lifetime)
        return self._appends[k][0] if k < len(self._appends) else self.extent

    def receive(self, t: int, i: int, root: bytes, runs: Sequence[tuple[int, int]] | None = None, *,
                change: tuple[Sequence[tuple[int, int]], Sequence[tuple[int, int]]] | None = None,
                step: int | None = None) -> Verdict | None:
        """Root `i` of window `t`, its pass's plan as the prover sends it, and the step it covers, as they arrive: `runs`
        (None: the registered store in order; a numpy array of rows (first, count): `Plan.from_array`), or `change`,
        the blocks (added, removed) from the session's previous pass's plan (`Plan.changed`; before its first pass, the
        registered store).  A plan is over the store's blocks so far, registered and appended, and holds none released;
        its rejection, `lifetime`, if it holds an appended block that joined a window `Params.lifetime` or more before
        `t`.  `step` is the step of the served circuit whose reads the pass covers, above the previous pass's: the plan
        is to be the blocks the circuit reads at it, which the plan's proof unit checks against the prover's memory map
        (`PROTOCOL.md` §8); its rejection, `step`, if it is missing or not above.  Returns the rejection they cause, if
        any."""
        self._live()
        if not self.windows:
            raise SessionError("the session has not started")
        now = self.clock()
        if type(t) is not int or type(i) is not int or type(root) is not bytes or len(root) != DIGEST_BYTES:
            return self._fail(self.filling, "malformed")
        if t == self.filling + 1 and i == 0 and t < len(self.windows):
            self.windows[self.filling].final = True
            self.filling = t
        elif t != self.filling or i != len(self.windows[t].roots) or self.windows[t].final:
            return self._fail(self.filling, "order")
        d = self.deadline(t)
        if d is not None and now > d:
            return self._fail(t, "late")
        if type(step) is not int or not self._step < step < STEP_LIMIT:
            return self._fail(t, "step")
        N, extent = self.params.blocks, self.extent
        try:
            if change is not None:
                if runs is not None:
                    raise ValueError("a plan comes whole or as a change, not both")
                added, removed = change
                plan = self._plan.changed(added, removed, extent)
            else:
                plan = (Plan.whole(N) if runs is None else Plan.from_array(runs, extent) if hasattr(runs, "tolist")
                        else Plan(tuple(runs), extent))
        except (TypeError, ValueError):
            return self._fail(t, "plan")
        if self._gone and any(self._released(f, c) for f, c in plan.runs):
            return self._fail(t, "plan")
        self._joined.extend([t] * (len(self._appends) - len(self._joined)))
        if (below := self._stale_below(t)) > N and any(f < below and f + c > N for f, c in plan.runs):
            return self._fail(t, "lifetime")
        if extent > N:
            held = [(max(f, N), f + c) for f, c in (change[0] if change is not None else plan.runs) if f + c > N]
            if held:
                self._planned = _merged(self._planned + held)
        self.windows[t].roots.append(root)
        self.windows[t].plans.append(plan)
        self.windows[t].post.append(0 if extent == N else sum(max(0, f + c - max(f, N)) for f, c in plan.runs))
        self.windows[t].steps.append(step)
        self.windows[t].heads.append(self._head)
        self.windows[t].stale.append(below)
        self._plan, self._step = plan, step
        return None

    def close(self, t: int) -> tuple[tuple[int, int], ...] | Verdict:
        """Window t's draws, from its openings coin released now: `s` distinct receipts (pass, position), uniform over
        its `U_t = Σ_i L_i`, in order.  Refused until its roots are final; its rejection when it has fewer than
        `min_passes` roots, and, at the last window, `coin` if an append's coin went unused: no root followed it."""
        self._live()
        w = self.windows[t] if 0 <= t < len(self.windows) else None
        if w is None or w.draws is not None:
            raise SessionError(f"window {t} is not open to draw")
        if any(v.draws is None for v in self.windows[:t]):
            raise SessionError("windows are drawn in order: a window's first pass chains from the window before")
        d = self.deadline(t)
        now = self.clock()
        if not w.final and (d is None or now <= d):
            raise SessionError(f"window {t}'s roots are not final before its deadline or the next window's first root")
        w.final = True
        if len(w.roots) < self.params.min_passes:
            return self._fail(t, "few")
        if t == self.params.windows - 1 and len(self._joined) < len(self._appends):
            return self._fail(t, "coin")
        coin = self.challenger.open(self.params.windows + t)
        key = derive(coin.coin, OPENINGS_DOMAIN, {"registration": self.registration.digest, "window": t})
        ends = tuple(itertools.accumulate(plan.length for plan in w.plans))
        draws = []
        for u in key.subset(min(self.params.samples, ends[-1]), ends[-1], "openings"):
            i = bisect.bisect_right(ends, u)
            draws.append((i, u - (ends[i - 1] if i else 0)))
        w.draws, w.opened_ns = tuple(draws), now
        return w.draws

    def _previous(self, t: int, i: int) -> bytes:
        """What pass i of window t's seed takes as the root before it: that root, chained to the appends received
        before the pass's own root (`chained`)."""
        if i > 0:
            root = self.windows[t].roots[i - 1]
        else:
            root = self.windows[t - 1].roots[-1] if t > 0 else self.registration.digest
        return chained(root, self.windows[t].heads[i])

    def verify(self, t: int, openings: Sequence[Opening]) -> Verdict:
        """Window t's verdict on the prover's openings of its draws, in the order drawn."""
        self._live()
        w = self.windows[t] if 0 <= t < len(self.windows) else None
        if w is None or w.draws is None or w.verdict is not None:
            raise SessionError(f"window {t} has no draws awaiting openings")
        if len(openings) != len(w.draws):
            return self._fail(t, "malformed")
        key, receipters = self.registration.key, {}
        for (i, p), o in zip(w.draws, openings):
            failure = self._check(t, i, p, o, key, receipters)
            if failure is not None:
                return self._fail(t, failure, (i, p))
        w.verdict = Verdict(t, True)
        return w.verdict

    def _check(self, t: int, i: int, p: int, o: Opening, key: codec.Key, receipters: dict) -> str | None:
        N = self.params.blocks
        if (type(o) is not Opening or o.pass_index != i or o.position != p or type(o.block) is not bytes
                or len(o.block) != BLOCK_BYTES or type(o.leaf) is not bytes or type(o.tile) is not bytes
                or (self.tiles is None and o.tile != b"")
                or any(type(x) is not bytes or len(x) != SALT_BYTES for x in (o.leaf_salt, o.payload_salt))
                or type(o.receipt_path) is not tuple or type(o.payload_path) is not tuple
                or any(type(x) is not bytes or len(x) != DIGEST_BYTES for x in (*o.receipt_path, *o.payload_path))):
            return "malformed"
        plan = self.windows[t].plans[i]
        j = plan.block(p)
        if type(o.digest) is not bytes or len(o.digest) != (0 if j < N else DIGEST_BYTES):
            return "malformed"
        seed = receipts.seed(self.windows[t].challenge, i, self._previous(t, i))
        y = receipts.leaf_receipt(o.leaf, p, plan.length)
        if y is None:
            return "malformed"
        if not receipts.verify_leaf(seed, plan, self.windows[t].roots[i], p, o.leaf, o.leaf_salt, o.receipt_path):
            return "receipt-path"
        if j < N:
            payload = codec.decode(o.block, j, key)
            domain, commitment = self._payload_domain, Commitment(self.registration.payload_root, N)
        else:
            _, domain, commitment, coin = self._appends[bisect.bisect_right(self._appends, j, key=lambda a: a[0]) - 1]
            payload = codec.decode(o.block, j, key, o.digest, coin)
        if payload is None:
            return "codec"
        opening = MerkleOpening(j, receipts.hidden(payload, o.payload_salt), o.payload_path)
        if not verify_opening(domain, commitment, opening, PAYLOAD_SCHEMA, LIMITS):
            return "payload-path"
        z = None
        if self.tiles is not None:
            z = self.tiles.recompute(seed, j, o.tile)
            if type(z) is not bytes or not z:
                return "tile"
        if i not in receipters:
            receipters[i] = receipts.receipter(self.registration.receipt, seed, self.registration.rows)
        if receipters[i](j, o.block, z) != y:
            return "receipt"
        return None

    @property
    def outcome(self) -> str | None:
        """"accepted" once every window is, the failure that rejected it, or None while it runs."""
        if self.failed is not None:
            return self.failed.failure
        if len(self.windows) == self.params.windows and all(w.verdict is not None and w.verdict.accepted
                                                            for w in self.windows):
            return "accepted"
        return None

    def record(self) -> Record:
        """The session's record, once every window is accepted."""
        self._live()
        if len(self.windows) != self.params.windows or any(w.verdict is None or not w.verdict.accepted for w in self.windows):
            raise SessionError("a record needs every window of the session accepted")
        out = []
        for t, w in enumerate(self.windows):
            lengths = tuple(plan.length for plan in w.plans)
            profile = IntegrityProfile(("receipt",), (sum(lengths),), {"receipt": Subset(len(w.draws))},
                                       self.params.delta, drawn={"receipt": len(w.draws)})
            out.append(WindowRecord(t, lengths, w.released_ns, self.deadline(t), w.opened_ns, profile, tuple(w.post),
                                    tuple(w.steps), tuple(w.heads), tuple(w.stale)))
        return Record(self.registration.digest, self.commitment.root, self.params.blocks, BLOCK_BYTES, tuple(out),
                      self.registration.receipt, self.registration.bind, self.registration.codec,
                      self.registration.layout, self.registration.rows, self.extent - self.params.blocks, self.released,
                      self.params.lifetime, self._head)


REGISTRY_SCHEMA = "verity/remote-bandwidth/registry/v0"
UNENDED = "unended"
"""A closed registration's session that was neither accepted nor rejected when it closed: counted as failed."""


class RegistryError(RuntimeError):
    """A call the verifier's registry refuses: a second registration for a node while one is live, a session over a
    registration that isn't its node's live one here, or a close of a node with none live."""


@dataclass(frozen=True, slots=True)
class Closed:
    """A registration the verifier closed: its node, its digest, and each of its sessions' outcomes in the order they
    started: "accepted", the failure that rejected it, or `UNENDED`, which counts as failed."""

    node: bytes
    registration: bytes
    sessions: tuple[str, ...]

    @property
    def failed(self) -> int:
        return sum(o != "accepted" for o in self.sessions)

    def to_json(self) -> dict:
        return {"node": self.node.hex(), "registration": self.registration.hex(), "sessions": list(self.sessions)}


@dataclass(slots=True)
class _Live:
    digest: bytes
    registration: Registration | None
    sessions: list


def _outcome(session: Session | str) -> str:
    if isinstance(session, str):
        return session
    return session.outcome or UNENDED


def _outcomes(value: object) -> list[str]:
    if type(value) is not list or any(type(o) is not str or not o for o in value):
        raise ValueError("a registration's sessions are their outcomes, each a nonempty string")
    return list(value)


class Registry:
    """The verifier's registrations, at most one live a node (the red team's KB-4).  `register` draws a codec seed
    (`register`) only for a node with none live: a second while one is live is refused (`RegistryError`) and counted
    (`refused`), and draws nothing, so a retry never redraws.  A registration ends only when the verifier closes it
    (`close`), which counts each of its sessions as it stands: accepted, the failure that rejected it, or `UNENDED`,
    counted as failed.  So a failed session stays on its registration's record, and every codec seed a node has been
    given is in `closed` or live: `drawn(node)` is R, what an advice form charges as log₂ R.  `to_json` and `from_json`
    keep it across the verifier's processes (never the verifier's secret); a registration read back is live, closable,
    and runs no session: its sessions' outcomes are as they stood."""

    def __init__(self) -> None:
        self._live: dict[bytes, _Live] = {}
        self.closed: list[Closed] = []
        self.refused: dict[bytes, int] = {}

    def live(self, node: bytes) -> bytes | None:
        """The digest of the node's live registration, None without one."""
        x = self._live.get(_node(node))
        return None if x is None else x.digest

    def drawn(self, node: bytes) -> int:
        """R: the registrations the node has had, closed and live, each under a codec seed of its own."""
        node = _node(node)
        return sum(c.node == node for c in self.closed) + (node in self._live)

    def register(self, source: bytes, node: bytes, blocks: int, payload_root: bytes, codec: Codec = codec.DEFAULT,
                 receipt: str = receipts.DEFAULT, **kw) -> Registration:
        """`register`'s draw for a node with no live registration, which it makes live; refused, and counted, for one
        with."""
        node = _node(node)
        if node in self._live:
            self.refused[node] = self.refused.get(node, 0) + 1
            raise RegistryError(f"node {node.hex()} has a live registration, {self._live[node].digest.hex()[:16]}…: a "
                                "second draws no codec seed until the verifier closes the first and counts its sessions")
        registration = register(source, node, blocks, payload_root, codec, receipt, **kw)
        self._live[node] = _Live(registration.digest, registration, [])
        return registration

    def session(self, params: Params, registration: Registration, tiles: Tiles | None, *, not_before_ns: int,
                clock: Callable[[], int] = time.monotonic_ns, entropy: Callable[[int], bytes] = os.urandom,
                index: int | None = None) -> Session:
        """The next session over the node's live registration, its coins `coins.Challenger(digest, k, …)` with k its
        sessions so far (`index`, if given, must be k), kept to be counted at `close`.  Refused for a registration
        that isn't its node's live one, drawn here."""
        x = self._live.get(getattr(registration, "node", None))
        if x is None or x.registration is None or x.registration != registration:
            raise RegistryError("a session runs over its node's live registration, drawn by this registry")
        k = len(x.sessions)
        if index is not None and index != k:
            raise RegistryError(f"the registration's next session is {k}, not {index}")
        session = Session(params, registration, tiles, coins.Challenger(registration.digest, k, params.coins, entropy),
                          not_before_ns=not_before_ns, clock=clock)
        x.sessions.append(session)
        return session

    def close(self, node: bytes) -> Closed:
        """End the node's live registration, its sessions counted as they stand; refused for a node with none live."""
        node = _node(node)
        x = self._live.pop(node, None)
        if x is None:
            raise RegistryError(f"node {node.hex()} has no live registration to close")
        out = Closed(node, x.digest, tuple(_outcome(s) for s in x.sessions))
        self.closed.append(out)
        return out

    def to_json(self) -> dict:
        return {"closed": [c.to_json() for c in self.closed],
                "live": {n.hex(): {"registration": x.digest.hex(), "sessions": [_outcome(s) for s in x.sessions]}
                         for n, x in sorted(self._live.items())},
                "refused": {n.hex(): k for n, k in sorted(self.refused.items())}, "schema": REGISTRY_SCHEMA}

    @classmethod
    def from_json(cls, data: dict) -> Registry:
        if type(data) is not dict or data.get("schema") != REGISTRY_SCHEMA:
            raise ValueError(f"a registry is {REGISTRY_SCHEMA}")

        def digest(value: object) -> bytes:
            out = bytes.fromhex(value) if type(value) is str else b""
            if len(out) != DIGEST_BYTES:
                raise ValueError(f"a registration's digest is {DIGEST_BYTES} bytes")
            return out

        out = cls()
        for n, x in data["live"].items():
            out._live[_node(bytes.fromhex(n))] = _Live(digest(x["registration"]), None, _outcomes(x["sessions"]))
        out.closed = [Closed(_node(bytes.fromhex(c["node"])), digest(c["registration"]), tuple(_outcomes(c["sessions"])))
                      for c in data["closed"]]
        for n, k in data["refused"].items():
            if type(k) is not int or k < 1:
                raise ValueError("a node's refused registrations are a positive count")
            out.refused[_node(bytes.fromhex(n))] = k
        return out


LEDGER_SOURCES = {
    "timing": "the record: the span its passes don't cover at the node's capacity (the round trip and the pass in "
              "flight at the span's ends, the chaining gaps, blocks a plan leaves out), the session's or a window's own "
              "[b_(t−1), D_t]; on a requirement link, each pass's credit is its receipts at D bits each, at most the "
              "circuit's reads at its step, less the link's ε of the session's (the whole of it from any one window)",
    "sample": "the record: its openings' bound on wrong receipts, each window's own at delta (`Ledger.sample`, the "
              "per-window reading), or the session's pooled over its windows at one delta (`Ledger.horizon`, `pooled`)",
    "chip": "S_chip, measured: one pass's worth a block held on chip need not be read again",
    "counting": "AuditRO's slack, a pass's `named_bits` of its receipts (log2 of q^k/k!, which bounds C(q, k); every "
                "certified receipt, whichever reading) and slack_log2",
    "rebuild": "the record's codec and a stated squaring ceiling (`Rebuild`): blocks re-encoded from their payload, "
               "each at least the codec's squarings (WorkHardness(p, k)), in place of reads",
    "host": "a model: the host tier's own shortfall and host DRAM bandwidth PCIe cannot carry",
    "link": "a model, under a registration that binds z: relays meeting z and a block copy over a link (0 otherwise)",
    "carrier": "the property's verdict (`credit_post_seed`): a pass's credited post-seed blocks may carry other "
               "blocks' encodings, one read for two receipts, at most the verdict's carrier bytes and the pass's "
               "post-seed bytes (0 when nothing post-seed is credited)",
}


class PostSeedVerdict(Protocol):
    """The property's verdict on the circuits that wrote a session's post-seed blocks (I2, property/v1: P(c, R_in, L)),
    as `ledger` reads it: whether it `holds`, and `carrier_bytes(read_bytes)`, the most the carriers among a window's
    post-seed blocks can hold when its certified reads are `read_bytes`, R_in·L + (c_ρ·read_bytes·L/E)·b, the side
    capacity the guarantee adds to c.  The verifier computes it (`post_seed`, a `PostSeed`, by its Lean definition):
    property/v1 on the writers' public Program, bound to the one-stage audit's Program, the memory-map units' value rows
    and this session's appends.  `ledger` holds a `PostSeed`'s windows and lifetime cap to the record's
    (`property_reading`), and takes any other verdict as given."""

    @property
    def holds(self) -> bool: ...

    def carrier_bytes(self, read_bytes: int) -> float: ...


class RequirementLink(Protocol):
    """The served circuit's requirement, linked to a session's passes (A1's `Accounting.Requirement`), as `ledger`
    reads it: whether the link `holds`; `density_bits`, D, the fewest bits of the circuit's values that any block a
    plan holds carries; and `read_bits(step)`, the bits the circuit reads at a step (`requirement.Requirement.read_bits`
    in `verity.ml.circuits`).  It holds once the proof units have shown, for every pass, that its plan holds only
    blocks the circuit reads at the pass's step (`PlanMatches`; the plan unit, I4's), each carrying at least D of its
    values' bits (`Packs` under the census's layout rule), and that every free in the memory map, of a registered
    block or a written one, is at its values' last read (the memory-map unit, I4's).  `mapped(record)` is the
    memory-map unit's verdict on this session's record: that W's memory map, the one the plan unit's plans are of,
    makes exactly the appends of the chain `Record.appends` (None without one), each a write at its production step,
    in the map's order, at the index the verifier gave it; that each pass's head (`WindowRecord.heads`) is the chain
    through the appends of the steps below the pass's step, so each append was sent after the passes of the steps up to
    its own and before the rest; a re-key only at the lifetime cap, its old index below the stale bound
    (`WindowRecord.stale`) of the first pass after its step, and of the same payload; and no other (PROTOCOL.md §8,
    "The memory map's choices").  The units are sampled, so the link holds on what their draw established: `epsilon`,
    the most share of the linked credit (Σ min(n·D, read_bits(step)) over the passes) that the units it leaves
    unproved can carry, the passes whose plan unit is wrong and those holding a block a wrong memory-map unit wrote,
    except with probability `delta`.  Those checks are theirs; the ledger takes the link as given, credits 1 − ε of the
    linked credit, adds δ to its error, and refuses a link whose `mapped` isn't True of the record."""

    @property
    def holds(self) -> bool: ...

    @property
    def density_bits(self) -> int: ...

    @property
    def epsilon(self) -> float: ...

    @property
    def delta(self) -> float: ...

    def read_bits(self, step: int) -> int: ...

    def mapped(self, record: Record) -> bool: ...


POST_SEED_BINDINGS = ("program", "values", "mapped", "property")
OPEN_CARRIER = ("open: R_in·L + (c_ρ·read_bytes·L/E)·b reads L as the carriers' lifetime in seconds (R_in is bytes a "
                "second), and the verdict's L is census steps (`property_reading`)")


@dataclass(frozen=True, slots=True)
class StepWindow:
    """A window as property/v1 reads it: census steps `[start, stop)`, the bytes the audit certified as read in it
    (U_t·b) and its span in seconds."""

    start: int
    stop: int
    reads: int
    seconds: Fraction


@dataclass(frozen=True, slots=True)
class PostSeed:
    """The verifier's own `PostSeedVerdict` (`post_seed`, its Lean definition `RemoteAudit.PostSeed.verdict`):
    property/v1 run on the writers' Program W at its own step ends, and the bindings that tie it to the session.  It
    `holds` when nothing is `refused`, each check named in `POST_SEED_BINDINGS`.  It carries W's digest (`program`),
    its census's step count (`census_steps`), the L it was judged at (`lifetime`, in census steps), its `windows` and
    property/v1's own verdict (`checked`); `ledger` holds the lifetime and windows to the record's (binding 4,
    `property_reading`)."""

    refused: tuple[str, ...]
    program: str
    census_steps: int
    lifetime: int
    windows: tuple[StepWindow, ...]
    checked: Mapping

    @property
    def holds(self) -> bool:
        return not self.refused

    def carrier_bytes(self, read_bytes: int) -> float:
        raise NotImplementedError(OPEN_CARRIER)


def property_reading(record: Record, census_steps: int, *, paced: bool = False) -> tuple[tuple[StepWindow, ...], int]:
    """`record`'s windows and lifetime cap as property/v1 reads them over a census of `census_steps` steps (MEM-38):
    pass i, in order, names step s_i (s_0 = −1) and covers the census steps (s_{i−1}, s_i], so a window covers its
    passes' steps, each with its certified reads U_t·b and its seconds; the steps after the last pass are the closing
    window's pass's when the session is `paced`, and an unpaced session whose last pass is before the census's last step
    is refused.  The cap L_w (`Record.lifetime`, windows) is L_steps = max over w of the census steps in windows
    w … w + L_w, rounding up.  A `PostSeed` must carry exactly these (binding 4)."""
    if type(census_steps) is not int or census_steps < 1:
        raise ValueError("a census has one or more steps")
    if not record.windows or any(not w.steps for w in record.windows):
        raise ValueError("a window's passes name no steps: read a session's own record")
    steps = [s for w in record.windows for s in w.steps]
    if any(a >= b for a, b in zip(steps, steps[1:])) or steps[-1] >= census_steps:
        raise ValueError(f"the passes' steps strictly increase below the census's {census_steps}")
    if steps[-1] < census_steps - 1 and not paced:
        raise ValueError("steps after the last pass are uncovered")
    out, start = [], 0
    for i, w in enumerate(record.windows):
        stop = census_steps if i == len(record.windows) - 1 else w.steps[-1] + 1
        out.append(StepWindow(start, stop, w.receipts * record.block_bytes, Fraction(w.span_ns, 10**9)))
        start = stop
    held = max(out[min(i + record.lifetime, len(out) - 1)].stop - w.start for i, w in enumerate(out))
    return tuple(out), held


def post_seed(record: Record, descriptor: Mapping, ends: Sequence[int], *, audited: str, values: Sequence[bytes],
              map_statics: Mapping[str, int], map_public: Sequence[Mapping], map_rows: Sequence[Mapping[str, bytes]],
              link: RequirementLink, c: Mapping[str, object], R_in: object = 100_000, B: int = 128,
              paced: bool = False) -> PostSeed:
    """The property's verdict on the circuits that wrote `record`'s appends (PROTOCOL.md §8, "The verdict's binding"),
    as its Lean definition computes it (`RemoteAudit.PostSeed.verdict`, through `lean_verdict`): property/v1 on the
    writers' Program W (`descriptor`, its descriptor) at its step `ends`, under `record`'s windows and lifetime cap read
    in census steps (`property_reading`, at `paced`), and the bindings, refused by name (`POST_SEED_BINDINGS`):
    ``program``, W's digest is `audited`, the one-stage audit's `program`; ``values``, each memory-map unit's
    ``val{k}`` row (`map_rows[i]`, under its public row `map_public[i]` and `map_statics`' C, K and V) is W's value leaf
    at ``pos[k]`` of `values`; ``mapped``, `link.mapped(record.appends)`; and ``property``, P(c, R_in, L) at B operand
    bits.  The census it judges is W's by construction: the Lean reads W and its ends, never a census.  What the Lean
    refuses (ends that aren't W's steps, a malformed row, steps after an unpaced session's last pass) raises
    ValueError."""
    kind, out = lean_verdict.call("post_seed", {
        "program": dict(descriptor), "ends": list(ends), "audited": audited, "values": [v.hex() for v in values],
        "map": {k: map_statics[k] for k in ("C", "K", "V")},
        "map_public": [{k: p[k] for k in ("first", "count", "present", "pos")} for p in map_public],
        "map_rows": [{r: v.hex() for r, v in rows.items()} for rows in map_rows],
        "mapped": link.mapped(record.appends) is True, "c": {k: str(Fraction(v)) for k, v in c.items()},
        "R_in": str(Fraction(R_in)), "B": B, "block_bytes": record.block_bytes, "lifetime": record.lifetime,
        "paced": paced, "windows": [{"steps": list(w.steps), "receipts": w.receipts, "span_ns": w.span_ns}
                                    for w in record.windows]})
    if kind == "reject":
        raise ValueError(out)
    windows = tuple(StepWindow(a, b, r, Fraction(s)) for a, b, r, s in out["windows"])
    return PostSeed(tuple(out["refused"]), out["program"], out["census_steps"], out["L"], windows, out["verdict"])


class _Terms:
    __slots__ = ()

    @property
    def gamma(self) -> float:
        """γ with the per-window μ_sample, except with probability `windows · delta`."""
        return sum(getattr(self, name) for name in LEDGER_SOURCES)

    @property
    def gamma_horizon(self) -> float:
        """γ with the pooled μ_sample, except with probability `delta`, `floor` holding with it."""
        return self.horizon + sum(getattr(self, name) for name in LEDGER_SOURCES if name != "sample")

    def _json(self) -> dict:
        return {"gamma": self.gamma, "horizon": {"floor": self.floor, "gamma": self.gamma_horizon, "sample": self.horizon},
                "terms": {name: getattr(self, name) for name in LEDGER_SOURCES}}


@dataclass(frozen=True, slots=True)
class WindowLedger(_Terms):
    """Window t's γ terms as fractions of the node's read capacity over its own span `[b_(t−1), D_t]`
    (`WindowRecord.span_ns`): its reads are at least `(1 − gamma)·capacity·span` except with `delta`, and at least
    `(1 − gamma_horizon)·capacity·span` on the session's pooled event, every window's at once.  `floor` is the most
    of its receipts that event leaves wrong, and `horizon` those receipts' bytes."""

    window: int
    span_ns: int
    timing: float
    sample: float
    chip: float
    counting: float
    rebuild: float
    host: float
    link: float
    carrier: float
    horizon: float
    floor: float

    def to_json(self) -> dict:
        return {"span_ns": self.span_ns, "window": self.window, **self._json()}


@dataclass(frozen=True, slots=True)
class Ledger(_Terms):
    """Each γ term as a fraction of the node's read capacity over the session's span; γ is their sum.  μ_sample comes
    both ways: `sample`, each window's openings bounding its own wrong receipts at δ, and `horizon`, the session's
    openings pooled over its windows at one δ (`pooled`).  `floor` is the most of any window's receipts the pooled
    event leaves wrong.  `windows` are the same terms over each window's own span, and `worst` and `worst_horizon`
    the windows of the largest γ either way: the every-window half of the claim is theirs, beside the horizon's.
    `linked`: whether the credit is the served circuit's (capped at its reads, on a `RequirementLink`); without a link
    the credited bytes are the node's reads of its store, not shown to be reads the circuit makes.  `error` is the
    probability the per-window reading fails, every window's δ and AuditRO's 2^-slack_log2 a pass, plus the link's δ;
    `error_horizon` the horizon's, one δ in place of the windows'.  A sketch-sha512 record adds `RateAuditSketch`'s."""

    timing: float
    sample: float
    chip: float
    counting: float
    rebuild: float
    host: float
    link: float
    carrier: float
    horizon: float
    floor: float
    windows: tuple[WindowLedger, ...]
    error: float
    error_horizon: float
    linked: bool = False

    @property
    def worst(self) -> WindowLedger:
        return max(self.windows, key=lambda w: w.gamma)

    @property
    def worst_horizon(self) -> WindowLedger:
        return max(self.windows, key=lambda w: w.gamma_horizon)

    def to_json(self) -> dict:
        w, h = self.worst, self.worst_horizon
        return {**self._json(), "linked": self.linked, "windows": [x.to_json() for x in self.windows],
                "error": {"windows": self.error, "horizon": self.error_horizon},
                "worst": {"gamma": w.gamma, "horizon": {"gamma": h.gamma_horizon, "window": h.window},
                          "window": w.window}}


def _budget(record: Record) -> tuple[list[int], float]:
    delta = record.windows[0].profile.delta
    drawn = [w.profile.drawn["receipt"] for w in record.windows]
    return drawn, -math.expm1(math.log(delta) / sum(drawn)) * sum(drawn)


def floors(record: Record) -> tuple[float, ...]:
    """Each window's share of wrong receipts at most on `pooled`'s event, min(1 − s_t/U_t, d̄·S/s_t): every window's
    at once, except with δ."""
    drawn, budget = _budget(record)
    return tuple(min(1 - s / w.receipts, budget / s) for w, s in zip(record.windows, drawn))


def pooled(record: Record) -> tuple[float, float]:
    """The horizon reading of a record's openings: `(wrong, floor)`, at most `wrong` of the session's receipts wrong
    and at most a share `floor` of any window's, both except with probability δ together.

    Window t draws s_t distinct receipts of its U_t from its own openings coin, released after its roots, so they all
    land on correct ones with probability at most g_t^(s_t), g_t its correct share, however the prover adapted to the
    windows before.  Over the session (`Accounting.Guarantees.HorizonSampling`, `RateSessionDraw`), every draw lands
    on a correct receipt while Σ_t s_t·g_t ≤ (1 − d̄)·S, S = Σ_t s_t, with probability at most (1 − d̄)^S = δ.  So
    except with δ the wrong shares f_t = 1 − g_t have Σ_t s_t·f_t < d̄·S, and each f_t ≤ 1 − s_t/U_t (its s_t draws
    found s_t correct receipts).  `wrong` is the most Σ_t f_t·U_t those allow, and `floor` the most any f_t is:
    min(1 − s_t/U_t, d̄·S/s_t), about ln(1/δ)/s_t.  The Lean states it for windows of one size; at unequal U_t it is the
    same argument with each window's own."""
    drawn, budget = _budget(record)
    floor = max(floors(record))
    wrong = 0.0
    for w, s in sorted(zip(record.windows, drawn), key=lambda x: x[0].receipts / x[1], reverse=True):
        f = min(1 - s / w.receipts, budget / s)
        wrong, budget = wrong + f * w.receipts, max(0.0, budget - f * s)
    return wrong, floor


@dataclass(frozen=True, slots=True)
class Rebuild:
    """μ_rebuild's model: the node runs at most `squarings_per_s` squarings at `codec`'s width (the stated ceiling,
    assumption 5, every GPU and the host together), and a block it re-encodes from its payload rather than reads costs
    at least `codec.encode_squarings` of them (WorkHardness(p, k)).  So rebuilds stand in for at most
    `squarings_per_s · BLOCK_BYTES / encode_squarings` bytes a second of reads, `fraction` of a capacity, in every
    window alike.  `ledger` refuses a model of another codec than the record's."""

    codec: Codec
    squarings_per_s: float

    def __post_init__(self) -> None:
        if type(self.codec) is not Codec:
            raise ValueError("a rebuild model is of a Codec")
        if type(self.squarings_per_s) not in (int, float) or not 0 <= self.squarings_per_s < math.inf:
            raise ValueError("a squaring ceiling is a finite nonnegative rate")

    def fraction(self, capacity_bytes_per_s: float) -> float:
        return self.squarings_per_s * BLOCK_BYTES / (self.codec.encode_squarings * capacity_bytes_per_s)


def named_bits(k: int, queries_log2: int = QUERIES_LOG2) -> int:
    """A whole number of bits at least log2 C(q, k), q = 2^queries_log2: AuditRO's count of which k of a pass's q
    queries named its receipts.  C(q, k) ≤ q^k/k!, so it is k·log2 q − log2 k!, with log2 k! taken below
    `math.lgamma`'s by a relative 2^-40 and a bit, far past its few ulps.  A pass of k receipts charged
    `named_bits(k) + slack_log2` has AuditRO's last term C(q, k)/2^λ at most 2^-slack_log2."""
    if type(k) is not int or type(queries_log2) is not int or k < 0 or queries_log2 < 0:
        raise ValueError("a pass's receipts and its queries' log2 are nonnegative integers")
    factorial = math.lgamma(k + 1) / math.log(2)
    return k * queries_log2 - max(0, math.floor(factorial * (1 - 2.0 ** -40)) - 1)


def ledger(record: Record, *, capacity_bytes_per_s: float, chip_bytes: int, rebuild: Rebuild, host: float = 0.0,
           link: float = 0.0, queries_log2: int = QUERIES_LOG2, slack_log2: int = SLACK_LOG2,
           wrong: Callable[[IntegrityProfile], float] = IntegrityProfile.bound,
           credit_post_seed: PostSeedVerdict | None = None, requirement: RequirementLink | None = None) -> Ledger:
    """γ's terms for an accepted session on a node of `capacity_bytes_per_s` (HBM plus host DRAM) and `chip_bytes` of
    on-chip storage: the read bytes the record credits are at least `(1 − γ)·capacity·span`, except with probability
    `windows · delta`, and at least `(1 − γ_horizon)·capacity·span` except with `delta` (each plus 2^-slack_log2 a
    pass), and each window's over its own span likewise with its own terms (`Ledger.windows`), the horizon's every
    window's at once with the session's.  μ_rebuild is `rebuild`'s fraction of the capacity, a model of the record's
    own codec (name and parameters) at a stated squaring ceiling.  `host` and `link` are the models' fractions, passed
    through.  All three are the same in every window: the record does not bound them.
    `link` is a binding record's only: an unbound receipt is of its block alone, and the record says nothing of where
    its read was made.  `wrong` is the profile's bound on wrong receipts (`IntegrityProfile.bound`; pass an
    approximation for a large store).  A sketch-sha512 record's receipts also rest on `RateAuditSketch`, whose terms
    add to the failure probability, not to γ: `timing.sketch_failure` of the prover's SHA-512 compressions over the
    session (`PROTOCOL.md` §4).
    The record credits the receipts of registered blocks, and those of blocks appended after the codec seed only on
    `credit_post_seed`, the property's verdict on the circuits that wrote them (I2's; None credits none, and a verdict
    that fails is refused, as is a `PostSeed` judged at other windows or another lifetime cap than the record's,
    `property_reading`).  Crediting them charges μ_carrier, each pass the least of the verdict's carrier bytes and
    its post-seed bytes.  `sample`, `counting`, the horizon and the capacity check count every certified receipt.
    On `requirement`, the served circuit's requirement linked to the passes (`RequirementLink`, which must hold, its
    memory-map unit making the record's appends at the passes' heads, `mapped`), a pass's credit is its credited
    receipts at D bits each, at most the bits the circuit reads at the pass's step, so
    the credit is the circuit's reads (`Ledger.linked`); `sample` still charges a whole block for a wrong receipt.
    The link's units are sampled: the session's credit is 1 − ε of that sum, and a window's is its own less ε of the
    session's, since the unproved share may sit in any one window; the link's δ is added to `error` and
    `error_horizon`."""
    if credit_post_seed is not None and credit_post_seed.holds is not True:
        raise ValueError("a verdict that the property fails credits nothing after the seed: pass None")
    if type(credit_post_seed) is PostSeed:
        try:
            reading = property_reading(record, credit_post_seed.census_steps)
        except ValueError as e:
            raise ValueError(f"a verdict over a census this record can't be read in credits nothing after the seed "
                             f"(binding 4): {e}") from None
        if (credit_post_seed.windows, credit_post_seed.lifetime) != reading:
            raise ValueError("a verdict judged at other windows or another lifetime cap than this record's credits "
                             "nothing after the seed (binding 4)")
    density, epsilon, link_delta = 8 * record.block_bytes, 0.0, 0.0
    if requirement is not None:
        if requirement.holds is not True:
            raise ValueError("a requirement link whose proofs don't hold caps nothing: pass None, and the credit is "
                             "the node's reads of its store, not the circuit's")
        if any(len(x) != w.passes for w in record.windows for x in (w.steps, w.heads, w.stale)):
            raise ValueError("a requirement link reads each pass's step, head and stale bound, and this record's "
                             "passes name none: link a session's own record")
        if requirement.mapped(record) is not True:
            raise ValueError("a requirement link whose memory-map unit doesn't make this record's appends (its "
                             "`appends` chain, each pass's head through the appends of the steps below its own, each "
                             "re-key below the next pass's stale bound) caps nothing: frees, re-keys and append order "
                             "chosen after the seed are advice no term charges")
        density = requirement.density_bits
        if type(density) is not int or not 1 <= density <= 8 * record.block_bytes:
            raise ValueError(f"a link's density is a whole number of bits from 1 to a block's {8 * record.block_bytes}")
        epsilon, link_delta = requirement.epsilon, requirement.delta
        if type(epsilon) not in (int, float) or not 0 <= epsilon <= 1:
            raise ValueError("a link's epsilon is the share of its credit its units' draw leaves unproved, 0 to 1")
        if type(link_delta) not in (int, float) or not 0 <= link_delta < 1:
            raise ValueError("a link's delta is the probability its units' draw fails, at least 0 and below 1")
    if capacity_bytes_per_s <= 0 or chip_bytes < 0:
        raise ValueError("the capacity is positive and the chip's storage nonnegative")
    if type(rebuild) is not Rebuild:
        raise ValueError("μ_rebuild is a Rebuild model of the record's codec, not a fraction")
    if rebuild.codec != record.codec:
        raise ValueError(f"the rebuild model is of {rebuild.codec.name} {rebuild.codec.to_json()}, the record's codec "
                         f"{record.codec.name} {record.codec.to_json()}")
    if link and not record.bind:
        raise ValueError("the link term is a registration's that binds z: this record's receipts are of blocks alone")
    mu_rebuild = rebuild.fraction(capacity_bytes_per_s)
    whole = capacity_bytes_per_s * record.span_ns / 1e9
    b = record.block_bytes
    reads = record.receipts * b
    if reads > whole:
        raise ValueError(f"the record certifies {reads} bytes read, more than the capacity allows in its span: the "
                         "capacity is understated")
    wrong_receipts = [math.ceil(wrong(w.profile)) for w in record.windows]
    named = {k: named_bits(k, queries_log2) + slack_log2 for k in {k for w in record.windows for k in w.lengths}}
    credit = credit_post_seed is not None

    def carried(w: WindowRecord) -> float:
        if not credit or not any(w.post_seed):
            return 0.0
        c = credit_post_seed.carrier_bytes(w.receipts * b)
        if type(c) not in (int, float) or not 0 <= c < math.inf:
            raise ValueError("a verdict's carrier bytes are a finite nonnegative number")
        return sum(min(c, a * b) for a in w.post_seed)

    def credited_bytes(w: WindowRecord) -> float:
        if requirement is None:
            return sum(w.credited(credit)) * b
        out = 0
        for n, s in zip(w.credited(credit), w.steps):
            r = requirement.read_bits(s)
            if type(r) is not int or r < 0:
                raise ValueError(f"the bits the circuit reads at step {s} are a nonnegative integer")
            out += min(n * density, r)
        return out / 8

    windows, carriers, credits = [], [carried(w) for w in record.windows], [credited_bytes(w) for w in record.windows]
    unproved = epsilon * sum(credits)
    for w, f, e, cb, cr in zip(record.windows, floors(record), wrong_receipts, carriers, credits):
        whole_t, reads_t = capacity_bytes_per_s * w.span_ns / 1e9, w.receipts * b
        if reads_t > whole_t:
            raise ValueError(f"window {w.window} certifies {reads_t} bytes read, more than the capacity allows in its "
                             "span: the capacity is understated")
        windows.append(WindowLedger(
            w.window, w.span_ns, timing=1 - max(0.0, cr - unproved) / whole_t, sample=e * b / whole_t,
            chip=w.passes * chip_bytes / whole_t, counting=sum(named[k] for k in w.lengths) / 8 / whole_t,
            rebuild=mu_rebuild, host=host, link=link, carrier=cb / whole_t, horizon=f * reads_t / whole_t, floor=f))
    horizon, floor = pooled(record)
    lam_bits = sum(named[k] for w in record.windows for k in w.lengths)
    beside = record.passes * 2.0 ** -slack_log2 + link_delta
    return Ledger(timing=1 - (sum(credits) - unproved) / whole, sample=sum(wrong_receipts) * b / whole,
                  chip=record.passes * chip_bytes / whole, counting=lam_bits / 8 / whole, rebuild=mu_rebuild, host=host,
                  link=link, carrier=sum(carriers) / whole, horizon=horizon * b / whole, floor=floor,
                  windows=tuple(windows), error=sum(w.profile.delta for w in record.windows) + beside,
                  error_horizon=record.windows[0].profile.delta + beside, linked=requirement is not None)
