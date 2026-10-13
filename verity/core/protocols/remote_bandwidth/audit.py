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
   receipt kind, `bind`, the layout and the rows.
2. Coins.  Before the session the challenger draws two live coins a window, its challenge `σ_t` and its openings coin,
   and commits to them (`coins.Challenger`).  The prover gets the commitment first (`Session.commitment`).
3. A session of `windows` windows.  The verifier releases `σ_0` (`Session.start`), and at each window's end the next
   window's challenge, or the close after the last (`Session.boundary`), one-way.  A window lasts at least the stated
   `T` (`Params.window_ns`, 20 s by default).
4. Chained passes.  Pass i of window t reads the blocks of its plan (`receipts.Plan`: distinct blocks, the whole store by
   default) and computes their receipts under `seed_(t,i) = H(σ_t, i, root_(t,i−1))` (`receipts.seed`), each of its
   block alone, or under a registration that binds z also of PoUW's tile output `z` before the read (`Tiles`); it
   commits them in a tree bound to the plan and sends the root and the plan as soon as the pass ends
   (`Session.receive`).  The chain runs across the session's windows: a window's first
   pass follows the root of the previous window's last.  On a boundary message the prover finishes the pass in flight,
   which is its window's last, and starts the next window's chain.
5. Deadline.  Window t's roots must all arrive by `D_t = b_t + A`, where `b_t` is the verifier's clock when it sent the
   boundary message and `A` the allowance (`Params.deadline_ns`: the message's trip down, the pass in flight, its root's
   reduction and its trip up, `timing.deadline`).  A late root, one out of order, or a plan that repeats a block rejects
   the session.  A window with fewer than `min_passes` roots is rejected.
6. Openings.  Once window t's roots are final (the next window's first root has arrived, or `D_t` has passed), the
   challenger releases the window's openings coin, and the verifier draws `s = samples` distinct receipts uniformly from
   the window's `U_t = Σ_i L_i` with it (`Session.close`).  The prover opens each (`Prover.open`): the block, its receipt
   leaf with its salt and path to the pass's root, the payload's salt and path to the payload root, and, under a
   registration that binds z, PoUW's evidence for the tile before it.
7. Verification (`Session.verify`), each opening: the leaf's path under the pass's root; the block's words canonical
   and their decoding's leaf on its path under the payload root; under a registration that binds z, `z` recomputed by
   PoUW's tile check (`Tiles.recompute`); the receipt recomputed from the block (and `z`) under the registration's
   kind (under sketch-sha512, the block's sketch from the pass's signs first).  A window is accepted only if every opening checks,
   and a rejected window ends its session.
8. The record (`Session.record`): per window each pass's length, its challenge's release and its deadline on the
   verifier's clock, its openings coin's release, and an `IntegrityProfile` over its receipts with `Subset(s)` drawn;
   `ledger` turns it, with the node's measured capacities and a squaring ceiling at its codec's width (`Rebuild`), into
   each γ term over the session's span and over each window's own (`WindowLedger`), μ_sample read per window and
   pooled over the session's horizon (`pooled`, `floors`).
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from verity.core.protocols.profile import IntegrityProfile, Subset
from verity.ml.commitments import Commitment, CommitmentDomain, MerkleTree, Opening as MerkleOpening, RangeIndexedDomain
from verity.ml.commitments import verify_opening
from verity.ml.randomness import derive, frame

from . import codec, coins, receipts
from .codec import BLOCK_BYTES, SEED_BYTES, Codec
from .receipts import DIGEST_BYTES, LIMITS, SALT_BYTES, Plan

PROTOCOL_ID = "verity/remote-bandwidth/v0"
CODEC_DOMAIN = "verity/remote-bandwidth/codec/v0"
OPENINGS_DOMAIN = "verity/remote-bandwidth/openings/v0"
PAYLOAD_DOMAIN = b"verity/remote-bandwidth/payload/v0\0"
REGISTRATION_DOMAIN = b"verity/remote-bandwidth/registration/v0\0"
TILE_DOMAIN = b"verity/remote-bandwidth/stub-tile/v0\0"
PAYLOAD_SCHEMA = receipts.SCHEMA
NONCE_BYTES = 32
NODE_BYTES = 64
QUERIES_LOG2 = 64
SLACK_LOG2 = 41
WINDOW_NS = 20 * 10**9

FAILURES = ("late", "order", "plan", "few", "malformed", "receipt-path", "codec", "payload-path", "tile", "receipt")


@dataclass(frozen=True, slots=True)
class Params:
    """A session's parameters: the store's size `N`, the allowance `A` from a window's boundary message to its roots'
    deadline, the windows `W`, the openings `s` per window at `delta`, the fewest passes a window may certify, the
    stated window `T` (`window_ns`, 20 s by default: no boundary message leaves sooner after its window's challenge),
    and the store's codec, receipt kind, whether its receipts bind z, its block layout and its sketch rows (None for
    the kind's default, `receipts.sketch_rows`), the registration's."""

    blocks: int
    deadline_ns: int
    windows: int
    samples: int
    delta: float
    min_passes: int = 1
    window_ns: int = WINDOW_NS
    codec: Codec = codec.DEFAULT
    receipt: str = receipts.DEFAULT
    bind: bool = False
    layout: str = receipts.LAYOUT
    rows: int | None = None

    def __post_init__(self) -> None:
        for name in ("blocks", "deadline_ns", "windows", "samples", "min_passes", "window_ns"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} is a positive integer")
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
    zeros; drawn after, each encoded block costs k square roots a word to produce, whatever the payload.  None of the
    receipt kind, `bind`, the block layout and the sketch rows (`receipts.sketch_rows`: the kind's default if None) is
    in the codec seed's context, since the store depends on none of them; the digest binds all four."""
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


class Store:
    """The prover's encoded store `E` under a registration, with its payload tree and the payloads' salts."""

    def __init__(self, registration: Registration, payloads: Sequence[bytes], salts: Sequence[bytes]) -> None:
        if len(payloads) != registration.blocks:
            raise ValueError(f"the registration is of {registration.blocks} blocks")
        self.registration = registration
        self.payloads, self.salts = tuple(payloads), tuple(salts)
        self.payload_tree = payload_tree(registration.node, self.payloads, self.salts)
        if self.payload_tree.commitment.root != registration.payload_root:
            raise ValueError("these payloads are not the registered ones")
        k = registration.key
        self.blocks = tuple(codec.encode(m, j, k) for j, m in enumerate(self.payloads))


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
    receipts, salt and path to pass i's root, the payload's salt and path to the payload root, and PoUW's evidence for
    `z_(i,j)` (empty unless the registration binds z)."""

    pass_index: int
    position: int
    block: bytes
    leaf: bytes
    leaf_salt: bytes
    receipt_path: tuple[bytes, ...]
    payload_salt: bytes
    payload_path: tuple[bytes, ...]
    tile: bytes

    @property
    def size(self) -> int:
        """Its bytes on the wire, with u32 and u64 headers for the indices and the variable fields."""
        return (4 + 8 + len(self.block) + len(self.leaf) + len(self.leaf_salt) + len(self.payload_salt)
                + DIGEST_BYTES * (len(self.receipt_path) + len(self.payload_path)) + len(self.tile) + 4 * 3)


class Prover:
    """The honest prover: each pass reads its plan's blocks (every block by default), computes their receipts and
    commits them, each leaf under a fresh salt from `entropy`.  `tiles` is PoUW's if the registration binds z, and
    None if it does not."""

    def __init__(self, store: Store, tiles: Tiles | None, commitment: coins.CoinCommitment, *,
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        self.store, self.commitment, self.entropy = store, commitment, entropy
        self.tiles = _tiles(tiles, store.registration.bind)

    def run(self, challenge: coins.CoinOpening, i: int, previous: bytes, plan: Plan | None = None) -> Pass:
        """Pass i of the window whose challenge is `challenge`, after the root `previous`: refused unless the coin is a
        window's challenge and opens against the session's commitment."""
        t = getattr(challenge, "index", None)
        reason = coins.coin_reason(self.commitment, challenge, t)
        if reason is None and t >= self.commitment.count // 2:
            reason = f"coin {t} is an openings coin, not a window's challenge"
        if reason is not None:
            raise ValueError(reason)
        return self.pass_under(t, i, receipts.seed(challenge.coin, i, previous), plan)

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
        return Opening(ps.index, p, self.store.blocks[j], b"".join(ps.receipts[receipts.PER_LEAF * c:receipts.PER_LEAF * (c + 1)]),
                       ps.salts[c], leaf.path, self.store.salts[j], self.store.payload_tree.open(j).path,
                       b"" if self.tiles is None else self.tiles.evidence(ps.seed, j))


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
    released at `opened_ns`; and the openings' law (`profile`)."""

    window: int
    lengths: tuple[int, ...]
    released_ns: int
    deadline_ns: int
    opened_ns: int
    profile: IntegrityProfile

    @property
    def passes(self) -> int:
        return len(self.lengths)

    @property
    def receipts(self) -> int:
        """U_t, the window's receipts."""
        return sum(self.lengths)

    @property
    def span_ns(self) -> int:
        """`D_t − b_(t−1)`: the verifier-fixed span its reads fall in, from its challenge's release to its deadline.
        Consecutive windows' spans overlap by A, and no read is in two windows' passes."""
        return self.deadline_ns - self.released_ns

    def to_json(self) -> dict:
        return {"deadline_ns": self.deadline_ns, "lengths": list(self.lengths), "opened_ns": self.opened_ns,
                "profile": self.profile.to_json(), "released_ns": self.released_ns, "span_ns": self.span_ns,
                "window": self.window}


@dataclass(frozen=True, slots=True)
class Record:
    """An accepted session: its registration, its coin commitment's root, its store's `blocks`, every window's record,
    and the receipt kind the registration names, whether its receipts bind z, its codec, its block layout and its sketch
    rows (its digest binds all five; the record repeats them, since what a receipt rests on depends on the kind, `bind`
    and the rows, μ_rebuild on the codec, and which bytes a credited block is on the layout).  `public` is what
    acceptance makes public;
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
        return {"bind": self.bind, "block_bytes": self.block_bytes, "blocks": self.blocks, "codec": self.codec.name,
                "codec_params": self.codec.to_json(), "coins": self.coins.hex(), "layout": self.layout,
                "passes": [w.passes for w in self.windows], "protocol": PROTOCOL_ID, "receipt": self.receipt,
                "receipts": [w.receipts for w in self.windows], "registration": self.registration.hex(),
                "rows": self.rows, "span_ns": self.span_ns}

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
    boundary_ns: int | None = None
    final: bool = False
    draws: tuple[tuple[int, int], ...] | None = None
    opened_ns: int | None = None
    verdict: Verdict | None = None


class Session:
    """The verifier's side of one session, driven by the coins it releases and the roots it receives, on its own clock.
    `challenger` holds the session's coins (`coins.Challenger`), committed before the session.  `not_before_ns`, which
    every session names, is the earliest its first coin may leave on the verifier's clock: the previous session's
    `Record.end_ns` over the same store, less the trusted round-trip floor if there is one, or the clock at
    registration for the store's first session, so that no read serves receipts of two sessions.  `tiles` is PoUW's
    check if the registration binds z, and None if it does not."""

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

    def receive(self, t: int, i: int, root: bytes, runs: Sequence[tuple[int, int]] | None = None) -> Verdict | None:
        """Root `i` of window `t` and its pass's plan as the prover sends it, `runs` (None: the whole store in order; a
        numpy array of rows (first, count): `Plan.from_array`), as they arrive.  Returns the rejection they cause, if
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
        N = self.params.blocks
        try:
            plan = (Plan.whole(N) if runs is None else Plan.from_array(runs, N) if hasattr(runs, "tolist")
                    else Plan(tuple(runs), N))
        except (TypeError, ValueError):
            return self._fail(t, "plan")
        self.windows[t].roots.append(root)
        self.windows[t].plans.append(plan)
        return None

    def close(self, t: int) -> tuple[tuple[int, int], ...] | Verdict:
        """Window t's draws, from its openings coin released now: `s` distinct receipts (pass, position), uniform over
        its `U_t = Σ_i L_i`, in order.  Refused until its roots are final; its rejection when it has fewer than
        `min_passes` roots."""
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
        if i > 0:
            return self.windows[t].roots[i - 1]
        return self.windows[t - 1].roots[-1] if t > 0 else self.registration.digest

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
        seed = receipts.seed(self.windows[t].challenge, i, self._previous(t, i))
        y = receipts.leaf_receipt(o.leaf, p, plan.length)
        if y is None:
            return "malformed"
        if not receipts.verify_leaf(seed, plan, self.windows[t].roots[i], p, o.leaf, o.leaf_salt, o.receipt_path):
            return "receipt-path"
        payload = codec.decode(o.block, j, key)
        if payload is None:
            return "codec"
        opening = MerkleOpening(j, receipts.hidden(payload, o.payload_salt), o.payload_path)
        if not verify_opening(self._payload_domain, Commitment(self.registration.payload_root, N), opening, PAYLOAD_SCHEMA,
                              LIMITS):
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
            out.append(WindowRecord(t, lengths, w.released_ns, self.deadline(t), w.opened_ns, profile))
        return Record(self.registration.digest, self.commitment.root, self.params.blocks, BLOCK_BYTES, tuple(out),
                      self.registration.receipt, self.registration.bind, self.registration.codec,
                      self.registration.layout, self.registration.rows)


LEDGER_SOURCES = {
    "timing": "the record: the span its passes don't cover at the node's capacity (the round trip and the pass in "
              "flight at the span's ends, the chaining gaps, blocks a plan leaves out), the session's or a window's own "
              "[b_(t−1), D_t]",
    "sample": "the record: its openings' bound on wrong receipts, each window's own at delta (`Ledger.sample`, the "
              "per-window reading), or the session's pooled over its windows at one delta (`Ledger.horizon`, `pooled`)",
    "chip": "S_chip, measured: one pass's worth a block held on chip need not be read again",
    "counting": "AuditRO's slack, queries_log2 bits a receipt (each certified one, whichever reading) and slack_log2 a "
                "pass",
    "rebuild": "the record's codec and a stated squaring ceiling (`Rebuild`): blocks re-encoded from their payload, "
               "each at least the codec's squarings (WorkHardness(p, k)), in place of reads",
    "host": "a model: the host tier's own shortfall and host DRAM bandwidth PCIe cannot carry",
    "link": "a model, under a registration that binds z: relays meeting z and a block copy over a link (0 otherwise)",
}


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
    the windows of the largest γ either way: the every-window half of the claim is theirs, beside the horizon's."""

    timing: float
    sample: float
    chip: float
    counting: float
    rebuild: float
    host: float
    link: float
    horizon: float
    floor: float
    windows: tuple[WindowLedger, ...]

    @property
    def worst(self) -> WindowLedger:
        return max(self.windows, key=lambda w: w.gamma)

    @property
    def worst_horizon(self) -> WindowLedger:
        return max(self.windows, key=lambda w: w.gamma_horizon)

    def to_json(self) -> dict:
        w, h = self.worst, self.worst_horizon
        return {**self._json(), "windows": [x.to_json() for x in self.windows],
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


def ledger(record: Record, *, capacity_bytes_per_s: float, chip_bytes: int, rebuild: Rebuild, host: float = 0.0,
           link: float = 0.0, queries_log2: int = QUERIES_LOG2, slack_log2: int = SLACK_LOG2,
           wrong: Callable[[IntegrityProfile], float] = IntegrityProfile.bound) -> Ledger:
    """γ's terms for an accepted session on a node of `capacity_bytes_per_s` (HBM plus host DRAM) and `chip_bytes` of
    on-chip storage: the read bytes the record certifies are at least `(1 − γ)·capacity·span`, except with probability
    `windows · delta`, and at least `(1 − γ_horizon)·capacity·span` except with `delta` (each plus 2^-slack_log2 a
    pass), and each window's over its own span likewise with its own terms (`Ledger.windows`), the horizon's every
    window's at once with the session's.  μ_rebuild is `rebuild`'s fraction of the capacity, a model of the record's
    own codec (name and parameters) at a stated squaring ceiling.  `host` and `link` are the models' fractions, passed
    through.  All three are the same in every window: the record does not bound them.
    `link` is a binding record's only: an unbound receipt is of its block alone, and the record says nothing of where
    its read was made.  `wrong` is the profile's bound on wrong receipts (`IntegrityProfile.bound`; pass an
    approximation for a large store).  A sketch-sha512 record's receipts also rest on `RateAuditSketch`, whose terms
    add to the failure probability, not to γ: `timing.sketch_failure` of the prover's SHA-512 compressions over the
    session (`PROTOCOL.md` §4)."""
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
    windows = []
    for w, f, e in zip(record.windows, floors(record), wrong_receipts):
        whole_t, reads_t = capacity_bytes_per_s * w.span_ns / 1e9, w.receipts * b
        if reads_t > whole_t:
            raise ValueError(f"window {w.window} certifies {reads_t} bytes read, more than the capacity allows in its "
                             "span: the capacity is understated")
        windows.append(WindowLedger(
            w.window, w.span_ns, timing=1 - reads_t / whole_t, sample=e * b / whole_t, chip=w.passes * chip_bytes / whole_t,
            counting=(queries_log2 * w.receipts + slack_log2 * w.passes) / 8 / whole_t, rebuild=mu_rebuild, host=host,
            link=link, horizon=f * reads_t / whole_t, floor=f))
    horizon, floor = pooled(record)
    lam_bits = queries_log2 * record.receipts + slack_log2 * record.passes
    return Ledger(timing=1 - reads / whole, sample=sum(wrong_receipts) * b / whole,
                  chip=record.passes * chip_bytes / whole, counting=lam_bits / 8 / whole, rebuild=mu_rebuild, host=host,
                  link=link, horizon=horizon * b / whole, floor=floor, windows=tuple(windows))
