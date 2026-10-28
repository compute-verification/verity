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
2. Coins.  Before the session the proof service's challenger draws two live coins a window, its challenge `σ_t` and
   its openings coin, and a coin an append, and commits to them (`coins.Coins`: session k's epoch of the one challenger,
   on its own host when it is remote).  The prover gets the commitment first (`Session.commitment`).  Each coin opens
   once, after the root the verifier recorded for it, which the challenger's signed log keeps
   (`Session.released_reason`).
3. A session of `windows` windows.  The verifier releases `σ_0` (`Session.start`), and at each window's end the next
   window's challenge, or the close after the last (`Session.boundary`), one-way.  A window lasts at least the stated
   `T` (`Params.window_ns`, 60 s by default).
4. Chained passes.  Pass i of window t reads the blocks of its plan (`receipts.Plan`: distinct blocks, the whole store by
   default) and computes their receipts under `seed_(t,i) = H(σ_t, i, root_(t,i−1))` (`receipts.seed`; the root
   before chained to every append received before the pass's own root, `chained`), each of its block alone, or under
   a registration that binds z also of PoUW's tile output `z` before the read (`Tiles`); it commits them in a tree
   bound to the plan and sends the root and the plan, whole or as its change from the previous pass's
   (`Plan.changed`), with the step it covers, as soon as the pass ends (`Session.receive`).  A pass covers one
   step `s` of the served circuit's execution (the engine step), and its plan holds every complete block the circuit
   reads at `s`, filler beside them allowed (`PlanMatches`): the steps are public, and strictly increase over the
   session's passes.  The chain runs across the session's windows: a window's first
   pass follows the root of the previous window's last.  On a boundary message the prover finishes the pass in flight,
   which is its window's last, and starts the next window's chain.  Blocks written after the codec seed (KV pages)
   join the store mid-session: their payloads committed (`append_tree`, `Session.append`) before any pass whose plan
   holds them, and so bound into its seed, at fresh store indices from N up, each encoded under its bound key and its
   append's coin `c_a`, which the verifier draws and sends in reply to the append (`codec.encode(..., bound=True,
   coin=c_a)`, `Store.encode`), so no choice the prover made before committing them depends on their encodings;
   every coin is drawn once, and an append no root follows, or one sent again, rejects the session (`coin`);
   a block released (`Session.release`) is in no later plan, and one the release names withdrawn (released before its
   encode landed, so never written) was in no earlier plan either; and an appended block is in no plan of a window L
   or more after the window it joined (`Params.lifetime`, the windows of T in 30 s, `lifetime_windows`): one kept
   longer is released and appended again, under a
   fresh index and a fresh coin, so a fresh bound key, and a session that defers residency to the encode
   (`Params.deferred`) has L ≥ 2.
5. Deadline.  Window t's roots must all arrive by `D_t = b_t + A`, where `b_t` is the verifier's clock when it sent the
   boundary message and `A` the allowance (`Params.deadline_ns`: the message's trip down, the pass in flight, its root's
   reduction and its trip up, `timing.deadline`).  A late root, one out of order, one with no step or a step not above
   the previous pass's, a plan that repeats a block, or one that holds an appended block past L rejects the session.
   A window with fewer than `min_passes` roots is rejected.
6. Openings.  Once window t's roots are final (the next window's first root has arrived, `D_t` has passed, or, in a
   paced session, the window holds its k roots and its boundary has left), the
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
   only on the property's verdict on the circuits that wrote it (`credit_post_seed`, I2's) and the session's schedule
   check (`schedule`, `Session.scheduled`: every pass's plan the published schedule's at its step, the schedule within
   the cap on its bits the session's parameters committed before its first coin, `Params.schedule_cap`,
   `schedule.py`), and then charges what their carriers may hold (μ_carrier); without both, those receipts are
   certified but not credited.  On the served
   circuit's requirement linked to the passes (`requirement`, a `RequirementLink`: I4's plan and memory-map units and
   the census's layout rule, the memory-map unit's verdict on the record's appends and heads among them, `mapped`),
   each pass is credited the bits the circuit reads from complete blocks at its step, less a block for each planned
   block it credits no receipt of (`linked_bits`, `Accounting.Guarantees.RequirementMoved`; `Ledger.linked`), less the
   share ε of it the units' draw leaves unproved, its δ added to the ledger's error (`Ledger.error`);
   without one the credit is the node's reads of its store.
   What an accepted window certifies, and on what, is `PROTOCOL.md`.

The verifier's clock is injected (`clock`, nanoseconds, `time.monotonic_ns` by default).  Openings are checked in the
clear here, as the specification of what each zero-knowledge proof block proves on the deployed path (`PROTOCOL.md` §8).
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
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, Protocol

from verity.core.protocols.profile import IntegrityProfile, Subset
from verity.ml.commitments import Commitment, CommitmentDomain, MerkleTree, Opening as MerkleOpening, RangeIndexedDomain
from verity.ml.commitments import verify_opening
from verity.ml.randomness import derive, frame

from . import codec, coins, receipts
from . import pooled as _pooled
from . import schedule as _schedule
from .codec import BLOCK_BYTES, P2_LAYOUT, SEED_BYTES, Codec, P2Codec
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
SLACK_LOG2 = 67
"""AuditRO's slack, the bits a pass is charged beyond `named_bits`: its last term C(q, k)/2^λ is then at most 2^-67 a
pass, which γ's error and the memory line's δ add once a pass.  Up to 2^17 passes they add at most 2^-50, 2^-10 of
`DELTA`: an hour, six times the stated 10-minute horizon, of the node's 41.2 ms passes over 130,023,424 blocks
(532.6 GB, gate runs 8 and 9's store) at `timing.READ_BYTES_PER_S` is 87,326 passes."""
WINDOW_NS = 60 * 10**9
SAMPLES = 33_273
DELTA = 2.0 ** -40

FAILURES = ("late", "order", "step", "plan", "lifetime", "append", "coin", "release", "few", "malformed",
            "receipt-path", "codec", "payload-path", "tile", "receipt")
RECEIPT_LIMIT = 1 << 64
"""A window's receipts, its passes' lengths summed, stay below this: `Session.receive` rejects with `plan` a pass that
would take them to it, so each draw's sampler word lies in its stream's first SHA-512 block (`Key._uniform` at
`U < 2^448`)."""
STEP_LIMIT = 1 << 64
"""A step is an integer below this, memory-map/v1's."""
LIFETIME_NS = 30 * 10**9
"""D, the lifetime cap's duration: a block appended after the codec seed is in no plan of a window L or more after the
window it joined (the window of the first root received after its append), L = `lifetime_windows(T)` the whole windows
of T in D, so it leaves the credited store within L windows unless the prover appends it again, which gives it a fresh
index and a fresh coin, and so a fresh bound key.  A carrier of a declared circuit's may work on a block from its coin
until it leaves, so what the carriers may hold grows with the passes in L windows
(`Accounting.Guarantees.DeclaredCircuitRequirement`'s `C`): over 30 s of 61 ms passes it is under 8% of the store at
both served models' caps (`PROTOCOL.md` §4, "The lifetime cap"), where 60 windows was an hour at the stated window."""


def lifetime_windows(window_ns: int, deferred: bool = False, life_ns: int = LIFETIME_NS) -> int:
    """L, the lifetime cap in windows of `window_ns` (a paced session's: its groups, at their nominal length): the whole
    windows in `life_ns`, at least 1, and at least 2 for a session that defers residency (`Params.deferred`), so at the
    stated 60 s window it is 1, or 2 deferred, and at 1 s it is 30."""
    for name, v in (("window_ns", window_ns), ("life_ns", life_ns)):
        if type(v) is not int or v < 1:
            raise ValueError(f"the lifetime's {name} is a positive integer of nanoseconds, not {v!r}")
    return max(2 if deferred else 1, life_ns // window_ns)


@dataclass(frozen=True, slots=True)
class Pace:
    """A paced session's schedule (`PROTOCOL.md` §7, "Pacing"): each window is a group of about `k` passes under a live
    coin of its own, the last the closing window, whose reads back the credited span's end (`ledger.memory`).  Window
    t's boundary message is due once t has k − 1 roots, so that the pass in flight is its last (at k = 1, once window
    t − 1 has its root, and window 0's at once), and leaves at the latest at max(its challenge + (k − 1)·A, D_(t−1)),
    never sooner than `Params.window_ns` after its challenge (`Session.next_boundary_ns`).  When its coins leave
    follows its roots, so is a value: its γ is declared per fixed window of `window_ns`, the stated T, from its first
    challenge (`ledger.windows`).

    `in_flight` f is the windows in flight: at 2 (the default) the schedule above, the next coin out while the
    window's last pass runs.  At 1, window t's boundary message is due once t has its k roots and leaves after the
    last of them arrived, at the latest at max(its challenge + k·A, D_(t−1)); a root of window t after its boundary
    is out of turn (`order`), so every coin leaves after the root before it (`coins.released_reason`), and the
    prover idles a round trip between windows.  `to_json` names `in_flight` only at 1."""

    k: int
    window_ns: int = WINDOW_NS
    in_flight: int = 2

    def __post_init__(self) -> None:
        for name in ("k", "window_ns"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"a pace's {name} is a positive integer")
        if type(self.in_flight) is not int or self.in_flight not in (1, 2):
            raise ValueError(f"a pace's in_flight is 1 or 2, not {self.in_flight!r}")

    def to_json(self) -> dict:
        return {"k": self.k, "window_ns": self.window_ns} | ({"in_flight": 1} if self.in_flight == 1 else {})


@dataclass(frozen=True, slots=True)
class Params:
    """A session's parameters: the store's size `N`, the allowance `A` from a window's boundary message to its roots'
    deadline, the windows `W`, the openings `s` per window at `delta` (`SAMPLES` at `DELTA` by default), the fewest
    passes a window may certify, the stated window `T` (`window_ns`, 60 s by default: no boundary message leaves sooner
    after its window's challenge), the store's codec, receipt kind, whether its receipts bind z, its block layout and its sketch rows (None for the kind's default, `receipts.sketch_rows`), the
    registration's, the lifetime cap L on an appended block, in windows (None: `lifetime_windows` of the stated
    window, `LIFETIME_NS`), and whether the prover defers an
    appended block's residency to its encode (`deferred`, memory-map/v1's deferred residency), which needs L ≥ 2: a
    block appended in window w is resident from w or w + 1, so its re-key falls at age L − 1, and at L = 1 that is
    before it may be resident.  `pace` paces the session (`Pace`, None for a stated window each): its windows are
    then groups of at most k passes, the last the closing window, and L counts groups: by default as many as the
    pace's stated window would hold of `LIFETIME_NS`, the fewest the life allows, which the verifier raises to the
    groups of their nominal length (`lifetime_windows(T // groups)`)."""

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
    lifetime: int | None = None
    deferred: bool = False
    pace: Pace | None = None
    groups_per_coin: int = 1
    """g (1 by default: an openings coin a window) draws the openings of g consecutive windows from one coin, released
    after the last root of the last of them: windows cg … min((c + 1)g, W) − 1 are draw c, whose `samples` receipts
    are drawn uniformly over their receipts together (`Session.close`), and the pooled bound reads each draw as one
    window (`pooled_windows`)."""
    schedule_cap: int | None = None
    """a, the most bits of the schedule the prover will publish (`schedule.check`), which it commits with the session's
    parameters, so before the session's first coin: the schedule check refuses a schedule over it, and the ledger
    credits it as the layout's steering (`Ledger.steering`, the `a` of `SteeredLayout`) at every coin alike.  None (the
    default) commits none, and the session credits no appended block (`Session.scheduled` refuses)."""

    def __post_init__(self) -> None:
        if type(self.deferred) is not bool:
            raise ValueError("deferred is a bool")
        if self.schedule_cap is not None and (type(self.schedule_cap) is not int or self.schedule_cap < 0):
            raise ValueError("a schedule's cap is a nonnegative integer of bits, or None for none")
        if self.pace is not None and (type(self.pace) is not Pace or self.windows < 2):
            raise ValueError(f"a paced session is a Pace's, with at least one window and the closing window, not "
                             f"{self.windows} windows under {self.pace!r}")
        if self.lifetime is None:
            stated = self.window_ns if self.pace is None else self.pace.window_ns
            if type(stated) is int and stated >= 1:
                object.__setattr__(self, "lifetime", lifetime_windows(stated, self.deferred))
        for name in ("blocks", "deadline_ns", "windows", "samples", "min_passes", "window_ns", "lifetime"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} is a positive integer")
        if self.deferred and self.lifetime < 2:
            raise ValueError(f"a session that defers residency has a lifetime of at least 2 windows, not "
                             f"{self.lifetime}: a block appended in window w is resident from w or w + 1, so its "
                             "re-key falls at age L − 1, which at L = 1 is before it may be resident")
        if not 0 < self.delta < 1:
            raise ValueError("0 < delta < 1")
        if type(self.codec) not in (Codec, P2Codec):
            raise ValueError("a session's codec is a Codec, or a P2 store's P2Codec")
        receipts.receipt_kind(self.receipt)
        _bind(self.bind)
        if type(self.codec) is Codec:
            receipts.block_layout(self.layout)
        elif self.layout != P2_LAYOUT or self.bind or self.receipt != receipts.SKETCH_SHA512:
            raise ValueError(f"a P2 store's session reads {P2_LAYOUT} units under sketch-sha512, of the block alone")
        receipts.sketch_rows(self.receipt, self.rows)
        if type(self.groups_per_coin) is not int or self.groups_per_coin < 1:
            raise ValueError("groups_per_coin is a positive integer")

    @property
    def coins(self) -> int:
        """W + ⌈W/g⌉: each window's challenge and each draw's openings coin, 2W at g = 1."""
        return self.windows + coins.openings_coins(self.windows, self.groups_per_coin)

    def draw_of(self, t: int) -> range:
        """The windows window t is drawn with: its draw's, t // g · g up to the draw's last."""
        g = self.groups_per_coin
        return range(t // g * g, min((t // g + 1) * g, self.windows))


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

    def __init__(self, store: Store, tiles: Tiles | None, commitment: coins.LiveCommitment, *,
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        self.store, self.commitment, self.entropy = store, commitment, entropy
        self.tiles = _tiles(tiles, store.registration.bind)

    def run(self, challenge: coins.LiveOpening, i: int, previous: bytes, plan: Plan | None = None, *,
            head: bytes | None = None) -> Pass:
        """Pass i of the window whose challenge is `challenge`, after the root `previous`, its seed binding `head`, the
        chain over every append the verifier will have received before this pass's root (`chained`; None without
        one, the honest prover's `Store.head`): refused unless the coin is a window's challenge and opens against the
        session's commitment."""
        t = getattr(challenge, "index", None)
        reason = coins.coin_reason(self.commitment, challenge, coins.CHALLENGE, t)
        if reason is not None:
            raise ValueError(reason)
        return self.pass_under(t, i, receipts.seed(coins.coin(challenge), i, chained(previous, head)), plan)

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
    after its challenge left at `released_ns` and committed by `deadline_ns` on the verifier's clock, each pass's root
    arriving at `arrived_ns` on that clock (one a pass, empty in a record no session made); its openings coin
    released at `opened_ns`; the openings' law (`profile`); of each pass's reads, `post_seed` of blocks appended
    after the codec seed (none if empty), which the ledger credits only on the property's verdict; the step each
    pass covers, `steps`, strictly increasing; each pass's head, `heads`, the chain over every append received
    before its root (`append_head`, None before the first), which its seed binds (`chained`); and each pass's stale
    bound, `stale`: N, the registered store's size, plus the counts of the session's appends that joined window
    `window` − L or before (L `Params.lifetime`), an append joining the window of the first root received after it,
    which is the window of the first pass whose head chains it.  That is the store index below which every appended
    block is past the cap, which a re-key's old index is below at the cap, and the same for every pass of a window;
    `Session._stale_below` takes it as the first index of the earliest append received before the pass's root that
    joined a window above `window` − L, or the store's extent at that root if none did.  `stale` is a value: the
    appends' counts it sums follow the workload, so the verifier keeps it in its own record and never publishes it
    (`Record.public` has no bound), and the memory-map unit takes it inside its proof, recomputing it from the
    committed appends.  All three are empty in a record no session made, such as `timing.stated`'s model."""

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
    arrived_ns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.arrived_ns != () and (
                len(self.arrived_ns) != len(self.lengths) or any(type(a) is not int for a in self.arrived_ns)):
            raise ValueError("a window's arrivals are each pass's root's, an integer time each")
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


def window_profile(lengths: Sequence[int], drawn: int, delta: float) -> IntegrityProfile:
    """A window's openings law (`WindowRecord.profile`): `drawn` distinct receipts uniform over its passes' `lengths`
    receipts, at `delta`."""
    return IntegrityProfile(("receipt",), (sum(lengths),), {"receipt": Subset(drawn)}, delta, drawn={"receipt": drawn})


@dataclass(frozen=True, slots=True)
class Record:
    """An accepted session: its registration, its coin commitment's root, its store's `blocks`, every window's record,
    its pace (`Params.pace`, None unpaced), and the receipt kind the registration names, whether its receipts bind z,
    its codec, its block layout and its sketch rows (its digest binds all five; the record repeats them, since what a
    receipt rests on depends on the kind, `bind` and the rows, μ_rebuild on the codec, and which bytes a credited block
    is on the layout), the blocks the session appended after the codec seed and released, the lifetime cap it held
    them to (`Params.lifetime`, None
    in a record no session made), and the chain over its appends in the order received (`append_head`, None without
    one), which, with each pass's head (`WindowRecord.heads`), the memory-map unit's verdict is on
    (`RequirementLink.mapped`), and the receipts every draw requested (`Params.samples`, None in a record no session
    made): a draw of U receipts drew min(`samples`, U), so `samples` is the k_t of the draw's rejection Σ_t k_t/2^64
    (`ledger.rejection`), which a draw opening all its receipts doesn't show.  `samples` is also the session's draw cap
    s and `append_coins` the append coins its coin commitment reserved, which bound every window of the session,
    whatever its outcome (`ledger.binding`; None in a record no session made).  `public` is what acceptance makes
    public; `profile` is the protocol's profile, each window's schedule and openings law (the γ terms need the node's
    capacities: `ledger`).  At `groups_per_coin` g > 1 a window's `profile` counts its share of its draw's openings,
    and the law is the draw's: `samples` receipts uniform over its g windows' receipts together (`pooled_windows`).
    Given how many land in a window, they are uniform over its receipts, so its own reading holds at its share's
    count (none at a share of none).  A record whose pace has one window in flight (`Pace.in_flight`) releases each
    window's challenge after every root of the window before arrived, and one that doesn't is refused."""

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
    append_coins: int | None = field(default=None, kw_only=True)
    lifetime: int | None = None
    appends: bytes | None = None
    pace: Pace | None = None
    groups_per_coin: int = 1
    samples: int | None = None

    def __post_init__(self) -> None:
        if self.pace is not None and self.pace.in_flight == 1 and any(
                a > w.released_ns for v, w in zip(self.windows, self.windows[1:]) for a in v.arrived_ns):
            raise ValueError("at one window in flight a window's challenge leaves after the window before's last root")
        if self.samples is None:
            return
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError("a record's samples, the receipts every draw requested, is a positive integer")
        g = self.groups_per_coin
        for a in range(0, len(self.windows), g):
            ws = self.windows[a:a + g]
            u, drawn = sum(w.receipts for w in ws), sum(w.profile.drawn["receipt"] for w in ws)
            if drawn != min(self.samples, u):
                what = f"window {ws[0].window}" if g == 1 else f"the draw of windows {ws[0].window}…{ws[-1].window}"
                raise ValueError(f"{what} drew {drawn} of its {u} receipts, not min({self.samples}, {u}) at the "
                                 "record's samples")

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
        `appends`.  No stale bound is here: `WindowRecord.stale[i]` is `blocks` plus the counts of the appends that
        joined window t − `lifetime` or before, counts that follow the workload, so it is a value, which the memory-map
        unit recomputes inside its proof from the committed appends and the verifier never publishes.  It names
        `groups_per_coin` only when a coin draws over more than one window."""
        shared = {} if self.groups_per_coin == 1 else {"groups_per_coin": self.groups_per_coin}
        return shared | {"appended": self.appended, "appends": None if self.appends is None else self.appends.hex(),
                "bind": self.bind, "block_bytes": self.block_bytes, "blocks": self.blocks,
                "codec": self.codec.name, "codec_params": self.codec.to_json(), "coins": self.coins.hex(),
                "heads": [[None if h is None else h.hex() for h in w.heads] for w in self.windows],
                "layout": self.layout, "lifetime": self.lifetime,
                "pace": None if self.pace is None else self.pace.to_json(), "passes": [w.passes for w in self.windows],
                "post_seed": [sum(w.post_seed) for w in self.windows], "protocol": PROTOCOL_ID,
                "receipt": self.receipt, "receipts": [w.receipts for w in self.windows],
                "registration": self.registration.hex(), "released": self.released, "rows": self.rows,
                "samples": self.samples, "span_ns": self.span_ns}

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
    arrived: list[int] = field(default_factory=list)
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
    `coins` are the session's coins through the one challenger (`coins.Coins`), committed before the session, each
    asked for after the root it follows: σ_0 after the session's window root, σ_t after the last root received,
    window t's openings coin after its last root and an append's coin after its root.  `not_before_ns`, which
    every session names, is the earliest its first coin may leave on the verifier's clock: the previous session's
    `Record.end_ns` over the same store, less the trusted round-trip floor if there is one, or the clock at
    registration for the store's first session, so that no read serves receipts of two sessions.  `tiles` is PoUW's
    check if the registration binds z, and None if it does not.  Blocks appended after the codec seed (`append`) are
    the session's: they take indices from N up, and a later session over the store starts again from N.  A session
    over PoUS's P2 store is `p2.Session`, whose blocks are units of `block_bytes` (`codecs`, the codec it takes)."""

    codecs: tuple[type, ...] = (Codec,)
    block_bytes: int = BLOCK_BYTES

    def __init__(self, params: Params, registration: Registration, tiles: Tiles | None, coins_: coins.Coins, *,
                 not_before_ns: int, clock: Callable[[], int] = time.monotonic_ns) -> None:
        if type(params.codec) not in self.codecs:
            raise ValueError(f"a {type(self).__name__} is over a store under {[c.__name__ for c in self.codecs]}, not "
                             f"{type(params.codec).__name__}: a P2 store's session is p2.Session")
        if (registration.blocks != params.blocks or registration.codec != params.codec
                or registration.receipt != params.receipt or registration.bind != params.bind
                or registration.layout != params.layout
                or registration.rows != receipts.sketch_rows(params.receipt, params.rows)):
            raise ValueError("the parameters are not the registration's")
        if type(not_before_ns) is not int or not_before_ns < 0:
            raise ValueError("not_before_ns is the previous session's last deadline over the store, or the clock at "
                             "registration for its first: a nonnegative integer")
        if (type(coins_) is not coins.Coins or coins_.registration != registration.digest
                or coins_.windows != params.windows or coins_.per_coin != params.groups_per_coin):
            raise ValueError(f"the challenger's coins are not this session's: {params.windows} windows' coins, "
                             f"{params.groups_per_coin} a draw, for this registration")
        self.params, self.registration, self.coins = params, registration, coins_
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
        self._latest = coins_.window

    @property
    def commitment(self) -> coins.LiveCommitment:
        """The session's coin commitment, sent to the prover before the first coin."""
        return self.coins.commitment

    def _release(self, t: int, now: int) -> coins.LiveOpening:
        """σ_t, after the last root received (R before any).  `now`, read before the coin leaves (and before the
        challenger is asked, unless it was asked with the coin before it, `coins.Coins.ahead`), is its release in the
        record: the coin leaves later, so the record's span is the longer."""
        opening = self.coins.challenge(t, self._latest)
        self.windows.append(_Window(coins.coin(opening), now))
        return opening

    def released_reason(self, signed: coins.LiveCommitment, received: dict[str, Sequence[bytes]]) -> str | None:
        """Why this session's coins are not the challenger's as it signed them (`coins.released_reason`), the commitment
        `signed` and the roots `received` read from the challenger's side; None when they are."""
        return coins.released_reason(self.coins, [w.roots for w in self.windows],
                                     [a[2].root for a in self._appends], signed, received,
                                     2 if self.params.pace is None else self.params.pace.in_flight)

    def trusted_reason(self, directory: Any, heads: Sequence[Any], key: coins.ChallengerKey, pins: Any, main: Any,
                       verify: coins.Verify, period: str | None = None) -> str | None:
        """Why this session's coins are not those of the one challenger session the auditor's ledger opens for its
        window `{root: R}` (`coins.trusted`: the ledger's published `heads` and the session's files in `directory`,
        under the pinned `key`, run from a commit the host's pins log `pins` records on the verifier's first-parent
        list of main `main`, inside the audit's `period` when it names one), held to them as `released_reason` holds
        them; None when they are.  So coins from a second challenger session, opened for another root or refused by the
        ledger as a second open for R, are refused, and so is a session with no pins log or copy of main, or whose
        host ran any other tree."""
        try:
            signed, received = coins.trusted(self.coins, directory, heads, key, pins, main, verify, period)
        except coins.CoinError as e:
            return f"the challenger's session for this session's window is not trusted: {e}"
        return self.released_reason(signed, received)

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

    def start(self) -> coins.LiveOpening:
        """`σ_0`, released now: refused before `not_before_ns`."""
        if self.windows:
            raise SessionError("a session starts once")
        now = self.clock()
        if now < self.not_before_ns:
            raise SessionError(f"the session's first coin may leave at {self.not_before_ns} ns, not {now}: the session "
                               "before it over this store may still be reading")
        return self._release(0, now)

    def boundary(self) -> coins.LiveOpening | None:
        """The end of the next unended window: the following window's challenge, or None, the close, after the last.
        Refused sooner than `window_ns` after the window's challenge left, and at one window in flight
        (`Pace.in_flight`) before its last root arrived."""
        self._live()
        now = self.clock()
        t = next((t for t, w in enumerate(self.windows) if w.boundary_ns is None), None)
        if t is None:
            raise SessionError("every window has ended" if self.windows else "the session has not started")
        if now < self.windows[t].released_ns + self.params.window_ns:
            raise SessionError(f"window {t} is {self.params.window_ns} ns, stated: its boundary may leave at "
                               f"{self.windows[t].released_ns + self.params.window_ns} ns, not {now}")
        if self._one_in_flight and any(now < a for a in self.windows[t].arrived):
            raise SessionError(f"window {t}'s boundary leaves after its last root at one window in flight, not at {now}")
        self.windows[t].boundary_ns = now
        return self._release(t + 1, now) if t + 1 < self.params.windows else None

    def next_boundary_ns(self) -> int | None:
        """When the next unended window's boundary message (`boundary`) is to leave on the verifier's clock, given the
        roots received so far: `window_ns` after its challenge, or in a paced session (`Pace`) then once it is due,
        and at the latest at max(its challenge + (k − 1)·A, the window before's deadline), at one window in flight
        max(its challenge + k·A, that deadline).  None once every window has ended or before the session starts."""
        t = next((t for t, w in enumerate(self.windows) if w.boundary_ns is None), None)
        if t is None:
            return None
        w, pace = self.windows[t], self.params.pace
        soonest = w.released_ns + self.params.window_ns
        if pace is None:
            return soonest
        k = pace.k
        if self._one_in_flight:
            if len(w.roots) >= k:
                return max(soonest, w.arrived[-1])
            return max(soonest, w.released_ns + k * self.params.deadline_ns, self.deadline(t - 1) if t else 0)
        if len(w.roots) >= k - 1 and (k > 1 or t == 0 or len(self.windows[t - 1].roots) >= 1):
            return soonest
        return max(soonest, w.released_ns + (k - 1) * self.params.deadline_ns, self.deadline(t - 1) if t else 0)

    @property
    def _one_in_flight(self) -> bool:
        return self.params.pace is not None and self.params.pace.in_flight == 1

    def append(self, root: bytes, count: int) -> tuple[int, bytes] | Verdict:
        """The root of an append's payload tree (`append_tree`), `count` blocks written after the codec seed: they take
        the store's next fresh indices, from the first returned, and a plan received after this may hold them, until
        the lifetime cap: they join the window of the next root received, and no plan of a window `Params.lifetime` or
        more after it may hold them.  The seed of every pass whose root comes after this binds it (`chained`).  The
        reply is the first index and the append's coin `c_a`, released now after the append's root (`coins.Coins.append`),
        under which the prover encodes the blocks (`Store.encode`) and an opening decodes them.  Its rejection, `append`,
        if it is malformed or would take the store past its codec's `max_blocks`; `coin`, releasing none, if its root was
        sent before (a retry never redraws), the last window's roots are final, so no root can follow it, or the session's
        append coins are all released (or it reserved none: `coins.families`)."""
        self._live()
        if (type(root) is not bytes or len(root) != DIGEST_BYTES or type(count) is not int or count < 1
                or self.extent + count > self.params.codec.max_blocks):
            return self._fail(self.filling, "append")
        if (root in self._roots or (len(self.windows) == self.params.windows and self.windows[-1].final)
                or self.coins.appends == (self.coins.commitment.family(coins.APPEND) or 0)):
            return self._fail(self.filling, "coin")
        first, coin = self.extent, coins.coin(self.coins.append(root))
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
        registered store).  A plan is over the store's blocks so far, registered and appended, holds none released,
        and keeps its window's receipts below `RECEIPT_LIMIT`; its rejection, `lifetime`, if it holds an appended block
        that joined a window `Params.lifetime` or more before `t`.  `step` is the step of the served circuit whose reads the pass covers, above the previous pass's: the plan
        is to hold every complete block the circuit reads at it, filler beside them allowed (`PlanMatches`), which the
        plan's proof block checks against the prover's memory map (`PROTOCOL.md` §8); its rejection, `step`, if it is
        missing or not above.  A paced session's window holds at
        most k roots, and a later one is out of turn (`order`), as is, at one window in flight (`Pace.in_flight`), a
        root that passes every other check after its window's boundary.  The root's arrival, the clock now, goes in
        the record.
        Returns the rejection they cause, if any."""
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
        if self.params.pace is not None and i >= self.params.pace.k:
            return self._fail(t, "order")
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
        if sum(p.length for p in self.windows[t].plans) + plan.length >= RECEIPT_LIMIT:
            return self._fail(t, "plan")
        self._joined.extend([t] * (len(self._appends) - len(self._joined)))
        if (below := self._stale_below(t)) > N and any(f < below and f + c > N for f, c in plan.runs):
            return self._fail(t, "lifetime")
        if self._one_in_flight and self.windows[t].boundary_ns is not None:
            return self._fail(t, "order")
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
        self.windows[t].arrived.append(now)
        self._plan, self._step, self._latest = plan, step, root
        return None

    def close(self, t: int) -> tuple[tuple[int, int], ...] | Verdict:
        """Window t's draws, from its openings coin released now: `s` distinct receipts (pass, position), uniform over
        its `U_t = Σ_i L_i`, in order.  Refused until its roots are final: the next window's first root has arrived,
        `D_t` has passed, or, in a paced session, the window holds its k roots, past which `receive` takes none, and
        its boundary has left, so its record has its deadline; the challenger's log fixes a paced window's end at its
        last root.  Its rejection when it has fewer than
        `min_passes` roots, and, at the last window, `coin` if an append's coin went unused: no root followed it.

        At `groups_per_coin` g > 1 the openings coin is draw c's, c = t // g (`Params.draw_of`): only the draw's last
        window closes, and that makes every window of the draw final (an earlier one's deadline is before t's), each
        rejected `few` in order; its `s` receipts are drawn over the draw's passes in order, numbered across its
        windows, under the key of window c, and each window keeps its own share, (pass, position) in it, which `verify`
        checks.  A window whose share is none has nothing to open: the close accepts it, and `verify` refuses it.  At
        g = 1 that is the window's own draw, never none."""
        self._live()
        w = self.windows[t] if 0 <= t < len(self.windows) else None
        if w is None or w.draws is not None:
            raise SessionError(f"window {t} is not open to draw")
        group = self.params.draw_of(t)
        if t != group[-1]:
            raise SessionError(f"window {t} is drawn with its draw's last window, {group[-1]}")
        if any(v.draws is None for v in self.windows[:group[0]]):
            raise SessionError("windows are drawn in order: a window's first pass chains from the window before")
        d = self.deadline(t)
        now = self.clock()
        full = self.params.pace is not None and d is not None and len(w.roots) >= self.params.pace.k
        if not w.final and not full and (d is None or now <= d):
            raise SessionError(f"window {t}'s roots are not final before its deadline, the next window's first root "
                               "or, paced, its k-th root after its boundary")
        ws = self.windows[group[0]:t + 1]
        for v in ws:
            v.final = True
        for u, v in zip(group, ws):
            if len(v.roots) < self.params.min_passes:
                return self._fail(u, "few")
        if t == self.params.windows - 1 and len(self._joined) < len(self._appends):
            return self._fail(t, "coin")
        c = t // self.params.groups_per_coin
        coin = coins.coin(self.coins.openings(c, w.roots[-1]))
        key = derive(coin, OPENINGS_DOMAIN, {"registration": self.registration.digest, "window": c})
        ends = tuple(itertools.accumulate(plan.length for v in ws for plan in v.plans))
        draws = []
        for u in key.subset(min(self.params.samples, ends[-1]), ends[-1], "openings"):
            i = bisect.bisect_right(ends, u)
            draws.append((i, u - (ends[i - 1] if i else 0)))
        base = 0
        for u, v in zip(group, ws):
            n = len(v.plans)
            v.draws, v.opened_ns = tuple((i - base, p) for i, p in draws if base <= i < base + n), now
            if not v.draws:
                v.verdict = Verdict(u, True)
            base += n
        return tuple(draws)

    def _previous(self, t: int, i: int) -> bytes:
        """What pass i of window t's seed takes as the root before it: that root, chained to the appends received
        before the pass's own root (`chained`)."""
        if i > 0:
            root = self.windows[t].roots[i - 1]
        else:
            root = self.windows[t - 1].roots[-1] if t > 0 else self.registration.digest
        return chained(root, self.windows[t].heads[i])

    def verify(self, t: int, openings: Sequence[Opening]) -> Verdict:
        """Window t's verdict on the prover's openings of its draws, in the order drawn: `malformed` on a window with
        no draws, whatever its openings."""
        self._live()
        w = self.windows[t] if 0 <= t < len(self.windows) else None
        if w is None or w.draws is None or w.verdict is not None:
            raise SessionError(f"window {t} has no draws awaiting openings")
        if not w.draws or len(openings) != len(w.draws):
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

    @property
    def appended(self) -> tuple[tuple[int, int], ...]:
        """The session's appends as it numbered them, `(first, count)`, in order."""
        return tuple((first, commitment.count) for first, _, commitment, _ in self._appends)

    def scheduled(self, schedule: _schedule.Schedule) -> Scheduled:
        """The schedule check (`schedule.check`, of record in `lean/RemoteAudit/Schedule.lean`) on this session's own
        passes, each plan as received and the step it named, and its appends, at the cap its parameters committed
        before its first coin (`Params.schedule_cap`): its verdict, which `ledger` requires before it credits an
        appended block, or `schedule.Refused` and why not.  A session whose parameters commit no cap has none
        (`SessionError`)."""
        cap = self.params.schedule_cap
        if cap is None:
            raise SessionError("the session's parameters commit no cap on its schedule's bits before its coins "
                               "(`Params.schedule_cap`), so no schedule bounds its layout's steering and no appended "
                               "block is credited")
        passes = [(s, plan.runs) for w in self.windows for s, plan in zip(w.steps, w.plans, strict=True)]
        _schedule.check(schedule, self.params.blocks, cap, passes, self.appended)
        return Scheduled(self.registration.digest, self.commitment.root, cap)

    def record(self) -> Record:
        """The session's record, once every window is accepted, with its bounds (`bounded`)."""
        return self.bounded(self._record())

    def bounded(self, record: Record) -> Record:
        """`record` with the bounds every window of this session obeys, whatever its outcome: its draw cap and the
        append coins its coin commitment reserved (`Record.samples`, `Record.append_coins`, `ledger.binding`)."""
        return replace(record, samples=self.params.samples, append_coins=self.commitment.family(coins.APPEND) or 0)

    def _record(self) -> Record:
        """The session's record without its bounds."""
        self._live()
        if len(self.windows) != self.params.windows or any(w.verdict is None or not w.verdict.accepted for w in self.windows):
            raise SessionError("a record needs every window of the session accepted")
        out = []
        for t, w in enumerate(self.windows):
            lengths = tuple(plan.length for plan in w.plans)
            profile = window_profile(lengths, len(w.draws), self.params.delta)
            out.append(WindowRecord(t, lengths, w.released_ns, self.deadline(t), w.opened_ns, profile, tuple(w.post),
                                    tuple(w.steps), tuple(w.heads), tuple(w.stale), tuple(w.arrived)))
        return Record(self.registration.digest, self.commitment.root, self.params.blocks, self.block_bytes, tuple(out),
                      self.registration.receipt, self.registration.bind, self.registration.codec,
                      self.registration.layout, self.registration.rows, self.extent - self.params.blocks, self.released,
                      self.params.lifetime, self._head, self.params.pace, self.params.groups_per_coin,
                      self.params.samples)


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
        with.  One that would bind PoUW's tile outputs (`bind`) draws nothing either (ValueError): the verifier's
        definition, `RemoteAudit.Registry.register`, covers receipts of a block alone and refuses it."""
        node = _node(node)
        if node in self._live:
            self.refused[node] = self.refused.get(node, 0) + 1
            raise RegistryError(f"node {node.hex()} has a live registration, {self._live[node].digest.hex()[:16]}…: a "
                                "second draws no codec seed until the verifier closes the first and counts its sessions")
        if _bind(kw.get("bind", False)):
            raise ValueError("a registration that binds PoUW's tile outputs is not covered: the verifier's definition "
                             "(RemoteAudit, in Lean) covers receipts of a block alone")
        registration = register(source, node, blocks, payload_root, codec, receipt, **kw)
        self._live[node] = _Live(registration.digest, registration, [])
        return registration

    def session(self, params: Params, registration: Registration, tiles: Tiles | None, *, not_before_ns: int,
                clock: Callable[[], int] = time.monotonic_ns, entropy: Callable[[int], bytes] = os.urandom,
                index: int | None = None, challenger: Callable[[int], object] | None = None) -> Session:
        """The next session over the node's live registration, k its sessions so far (`index`, if given, must be k),
        kept to be counted at `close`.  Its coins are epoch k of `challenger(k)`, the one challenger's session for it
        (`coins.Coins`), or without one a challenger in this process drawing from `entropy` (`coins.local`).  Refused
        for a registration that isn't its node's live one, drawn here."""
        x = self._live.get(getattr(registration, "node", None))
        if x is None or x.registration is None or x.registration != registration:
            raise RegistryError("a session runs over its node's live registration, drawn by this registry")
        k = len(x.sessions)
        if index is not None and index != k:
            raise RegistryError(f"the registration's next session is {k}, not {index}")
        g = params.groups_per_coin
        c = (coins.local(registration.digest, k, params.windows, entropy, per_coin=g) if challenger is None
             else coins.Coins(challenger(k), registration.digest, k, params.windows, per_coin=g))
        session = Session(params, registration, tiles, c, not_before_ns=not_before_ns, clock=clock)
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
              "[b_(t−1), D_t]; on a requirement link, each pass's credit is the circuit's reads from complete blocks "
              "at its step, less a block for each planned block it credits no receipt of (`linked_bits`, "
              "RequirementMoved), less the link's ε of the session's (the whole of it from any one window)",
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
    capacity the guarantee adds to c.  That the verdict is of the circuits that wrote this session's appends is I2's
    to bind and check, not the ledger's."""

    @property
    def holds(self) -> bool: ...

    def carrier_bytes(self, read_bytes: int) -> float: ...


class RequirementLink(Protocol):
    """The served circuit's requirement, linked to a session's passes (A1's `Accounting.Requirement`), as `ledger`
    reads it: whether the link `holds`, and `read_bits(step)`, the bits the circuit reads at a step from complete blocks
    (`creditedReadBits`: the bits of `requirement.credited_reads` in `verity.ml.circuits`; for a store of registered
    blocks only, the weights while KV isn't registered, the bits the step reads of registered values).  It holds once
    the proof blocks have shown, for every pass, that its plan holds every complete block the circuit reads at the
    pass's step and the layout packs the values read into blocks (`PlanMatches`; the plan unit, I4's: PlanMatch, or
    PlanShard for a registered store, with `Packs` under the census's layout rule), and that every free in the memory
    map, of a registered block or a written one, is at its values' last structural reader on the public schedule, a
    gather that may select one included (the memory-map unit, I4's; `memory_map.py`).  The link
    is one-sided: a plan may hold blocks the circuit doesn't read at its step (filler, a gathered table's unselected
    rows, a paused request's pages), which carry none of its reads, so a planned block need carry none of its bits,
    and the ledger credits each pass the circuit's reads, never its receipts (`linked_bits`, `RequirementMoved`).  A
    plan unit that lets a plan leave blocks out (PlanShard at GAPS > 0) makes each shard's count public, and so how
    many of a shard's blocks each plan holds; at GAPS = 0 every count is its shard's size.  `mapped(record)` is the
    memory-map unit's verdict on this session's record: that W's memory map, the one the plan unit's plans are of,
    makes exactly the appends of the chain `Record.appends` (None without one), each a write at its production step,
    in the map's order, at the index the verifier gave it; that each pass's head (`WindowRecord.heads`) is the chain
    through the appends of the steps below the pass's step, so each append was sent after the passes of the steps up to
    its own and before the rest; a re-key only at the lifetime cap, its old index below the stale bound
    (`WindowRecord.stale`, which the unit recomputes from those appends) of the first pass after its step, and of the
    same payload; and no other (PROTOCOL.md §8, "The memory map's choices").  The units are sampled, so the link
    holds on what their draw established: `epsilon`, the most share of the linked credit (`linked_bits` summed over
    the passes) that the units it leaves unproved can carry, the passes whose plan unit is wrong and those holding a
    block a wrong memory-map unit wrote, except with probability `delta`.  Those checks are theirs; the ledger takes
    the link as given, credits 1 − ε of the linked credit, adds δ to its error, and refuses a link whose `mapped`
    isn't True of the record."""

    @property
    def holds(self) -> bool: ...

    @property
    def epsilon(self) -> float: ...

    @property
    def delta(self) -> float: ...

    def read_bits(self, step: int) -> int: ...

    def mapped(self, record: Record) -> bool: ...


@dataclass(frozen=True, slots=True)
class Scheduled:
    """The schedule check's verdict on a session (`Session.scheduled`, `schedule.check`): every pass's plan was the
    published schedule's at the pass's step, the appends were the schedule's, and the schedule was at most `cap` bits,
    the cap the session's parameters committed before its first coin (`Params.schedule_cap`).  So each pass's layout is
    the verifier's fixed function of a schedule of at most `cap` bits (`Accounting.ScheduleDeterministic`).  It names
    its session by its registration and its coin commitment's root, as the record does (`Record.registration`,
    `Record.coins`).  `cap` bounds the `a` that `Accounting.Assumptions.SteeredLayout` reads for this session: fewer
    than 2^(cap+1) schedules encode in at most `cap` bits, and the cap was fixed before the coins, so on every coin the
    check accepts the layout is one of those schedules' plans.  The bits of the schedule published after the coins
    would not bound it: their length can follow the coins."""

    registration: bytes
    coins: bytes
    cap: int

    def of(self, record: Record) -> bool:
        return self.registration == record.registration and self.coins == record.coins


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
    `linked`: whether the credit is the served circuit's (its reads at each pass's step, on a `RequirementLink`:
    `linked_bits`); without a link
    the credited bytes are the node's reads of its store, not shown to be reads the circuit makes.  `steering`: the
    cap on the schedule's bits that the session committed before its coins, which the schedule the passes' layout was
    checked against is within, when appended blocks are credited (`Scheduled.cap`, `Params.schedule_cap`; the layout is
    then `SteeredLayout` at cap + 1), None when none are.  `error` is the
    probability the per-window reading fails, every window's δ and AuditRO's 2^-slack_log2 a pass, plus the link's δ;
    `error_horizon` the horizon's, one δ in place of the windows'.  A sketch-sha512 record adds `SketchBound`'s."""

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
    steering: int | None = None

    @property
    def worst(self) -> WindowLedger:
        return max(self.windows, key=lambda w: w.gamma)

    @property
    def worst_horizon(self) -> WindowLedger:
        return max(self.windows, key=lambda w: w.gamma_horizon)

    def to_json(self) -> dict:
        w, h = self.worst, self.worst_horizon
        return {**self._json(), "linked": self.linked, "steering_bits": self.steering,
                "windows": [x.to_json() for x in self.windows],
                "error": {"windows": self.error, "horizon": self.error_horizon},
                "worst": {"gamma": w.gamma, "horizon": {"gamma": h.gamma_horizon, "window": h.window},
                          "window": w.window}}


def pooled_windows(record: Record) -> tuple[list[tuple[int, int]], Fraction]:
    """The pooled bound's inputs (`pooled.bound`): each draw's receipts U and draws s, a window's own at
    `groups_per_coin` g = 1 and its g windows' together above, and δ, the float's exact value."""
    g, ws = record.groups_per_coin, record.windows
    return ([(sum(w.receipts for w in ws[a:a + g]), sum(w.profile.drawn["receipt"] for w in ws[a:a + g]))
             for a in range(0, len(ws), g)], Fraction(ws[0].profile.delta))


def pooled_bound(record: Record) -> _pooled.Bound:
    """The record's pooled bound in exact rationals (`pooled.bound`, the Lean's `RemoteAudit.Pooled`)."""
    return _pooled.bound(*pooled_windows(record))


def _up(q: Fraction) -> float:
    """The least float at least `q`."""
    x = float(q)
    return math.nextafter(x, math.inf) if Fraction(x) < q else x


def floors(record: Record) -> tuple[float, ...]:
    """Each window's share of wrong receipts at most on `pooled`'s event, min(1 − s_t/U_t, d̄·S/s_t): every window's
    at once, except with δ.  Each is `pooled_bound`'s, rounded up.  At `groups_per_coin` g > 1 the bound's floor f is
    its draw's, over the draw's U receipts, so a window of U_t of them has at most min(1, f·U/U_t)."""
    g, us = record.groups_per_coin, [w.receipts for w in record.windows]
    fs = pooled_bound(record).floors
    return tuple(_up(min(Fraction(1), fs[t // g] * sum(us[t // g * g:(t // g + 1) * g]) / u)) for t, u in enumerate(us))


def pooled(record: Record) -> tuple[float, float]:
    """The horizon reading of a record's openings: `(wrong, floor)`, at most `wrong` of the session's receipts wrong
    and at most a share `floor` of any window's, both except with probability δ together.

    Window t draws s_t distinct receipts of its U_t from its own openings coin, released after its roots, so they all
    land on correct ones with probability at most g_t^(s_t), g_t its correct share, however the prover adapted to the
    windows before.  Over the session (`Accounting.Guarantees.HorizonSampling`, `RateSessionDraw`), every draw lands
    on a correct receipt while Σ_t s_t·g_t ≤ (1 − d̄)·S, S = Σ_t s_t, with probability at most (1 − d̄)^S = δ.  So
    except with δ the wrong shares f_t = 1 − g_t have Σ_t s_t·f_t < d̄·S, and each f_t ≤ 1 − s_t/U_t (its s_t draws
    found s_t correct receipts).  `wrong` is the most Σ_t f_t·U_t those allow, and `floor` the most any f_t is:
    min(1 − s_t/U_t, d̄·S/s_t), about ln(1/δ)/s_t.  The Lean states `wrong` at each window's own U_t, on the deployed
    draw, at δ + Σ_t s_t/2^64 + the PRF advantage of each window's distinguisher, one a window that the session fixes,
    + the coin commitment's hiding terms + the receipts' binding term (`Accounting.Guarantees.Remote.PooledDraw`,
    `ledger.draw`).

    Both are `pooled_bound`'s exact values rounded up: d̄·S from above by a certified rational (`pooled.budget`), and
    `wrong` the greedy over the windows by U_t/s_t falling, which `Properties.RemoteAudit.PooledCovers` proves at least
    the bound `PooledDraw` reads."""
    b = pooled_bound(record)
    return _up(b.wrong), _up(b.floor)


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
            raise ValueError("a rebuild model is of a Codec: μ_rebuild over a P2 store waits on the squaring ceiling at "
                             "P2's encode (SquaringCeilingP2, drafted, not stated)")
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


def linked_bits(window: WindowRecord, requirement: RequirementLink, block_bytes: int,
                post_seed: bool) -> tuple[int, ...]:
    """Each pass's credit on a requirement link, in bits, its post-seed receipts credited or not (`post_seed`).

    `Accounting.Guarantees.RequirementMoved`: when the pass's plan covers the complete blocks the circuit reads at its
    step (`PlanMatches`, the link's to hold), a pass that moved the planned blocks M moved every bit the circuit reads
    from complete blocks at that step, r = `read_bits(step)`, but at most a block's b bits for each planned block
    outside M.  A pass computes one receipt a planned block, so one whose receipts are all credited covers its plan,
    and is credited r: a credited receipt that wasn't a read from node memory (wrong, or answered from the chip, a
    rebuild, the host or a link) is a planned block outside M, which `sample` and the other terms already charge a
    block each.  One credited n of its plan's L receipts (its post-seed ones uncredited, without the property's
    verdict) is credited r − (L − n)·b, at least 0: the same lemma, its L − n uncredited blocks outside M.  A planned
    block the circuit doesn't read at the step (filler) adds nothing to r and costs nothing.  `PlanMatches` packs the
    values read into the planned blocks (`Packs`), so r is at most L·b: a link reading more is refused, since it can't
    hold."""
    bits = 8 * block_bytes
    out = []
    for length, n, s in zip(window.lengths, window.credited(post_seed), window.steps, strict=True):
        r = requirement.read_bits(s)
        if type(r) is not int or r < 0:
            raise ValueError(f"the bits the circuit reads at step {s} are a nonnegative integer")
        if r > length * bits:
            raise ValueError(f"the circuit reads {r} bits from complete blocks at step {s}, more than its pass's plan "
                             f"of {length} blocks of {bits} bits holds: PlanMatches packs the values read into the "
                             "planned blocks, so this link doesn't hold; read_bits is the step's creditedReadBits")
        out.append(max(0, r - (length - n) * bits))
    return tuple(out)


def ledger(record: Record, *, capacity_bytes_per_s: float, chip_bytes: int, rebuild: Rebuild, host: float = 0.0,
           link: float = 0.0, queries_log2: int = QUERIES_LOG2, slack_log2: int = SLACK_LOG2,
           wrong: Callable[[IntegrityProfile], float] = IntegrityProfile.bound,
           credit_post_seed: PostSeedVerdict | None = None, requirement: RequirementLink | None = None,
           schedule: Scheduled | None = None) -> Ledger:
    """γ's terms for an accepted session on a node of `capacity_bytes_per_s` (HBM plus host DRAM) and `chip_bytes` of
    on-chip storage: the read bytes the record credits are at least `(1 − γ)·capacity·span`, except with probability
    `windows · delta`, and at least `(1 − γ_horizon)·capacity·span` except with `delta` (each plus 2^-slack_log2 a
    pass), and each window's over its own span likewise with its own terms (`Ledger.windows`), the horizon's every
    window's at once with the session's.  μ_rebuild is `rebuild`'s fraction of the capacity, a model of the record's
    own codec (name and parameters) at a stated squaring ceiling.  `host` and `link` are the models' fractions, passed
    through.  All three are the same in every window: the record does not bound them.
    `link` is a binding record's only: an unbound receipt is of its block alone, and the record says nothing of where
    its read was made.  `wrong` is the profile's bound on wrong receipts (`IntegrityProfile.bound`; pass an
    approximation for a large store).  A sketch-sha512 record's receipts also rest on `SketchBound`
    (`Accounting.Guarantees.Remote.SketchBoundDeployed`), whose terms add to the failure probability, not to γ:
    `timing.sketch_failure` of the prover's SHA-512 compressions over the session (`PROTOCOL.md` §4).
    The record credits the receipts of registered blocks, and those of blocks appended after the codec seed only on
    `credit_post_seed`, the property's verdict on the circuits that wrote them (I2's; None credits none, and a verdict
    that fails is refused), and only with this session's schedule check (`schedule`, `Session.scheduled`): every
    pass's plan the published schedule's at its step and the schedule within the cap on its bits the session committed
    before its coins, so the layout the appended blocks are read at is the verifier's function of the schedule, steered
    through at most the cap's bits on every coin (`Ledger.steering`, the cap; `SteeredLayout` at cap + 1).
    Without it, or with another session's, crediting them is refused.  Crediting them charges μ_carrier, each pass the
    least of the verdict's carrier bytes and its post-seed bytes.  `sample`, `counting`, the horizon and the capacity check count every certified receipt.
    On `requirement`, the served circuit's requirement linked to the passes (`RequirementLink`, which must hold, its
    memory-map unit making the record's appends at the passes' heads, `mapped`), a pass's credit is the bits the
    circuit reads from complete blocks at its step, less a block for each planned block it credits no receipt of
    (`linked_bits`, by `RequirementMoved`), so the credit is the circuit's reads (`Ledger.linked`), and a plan topped
    up with filler earns nothing for it; `sample` still charges a whole block for a wrong receipt, as
    `RequirementMoved` charges a planned block the pass didn't move.
    The link's units are sampled: the session's credit is 1 − ε of that sum, and a window's is its own less ε of the
    session's, since the unproved share may sit in any one window; the link's δ is added to `error` and
    `error_horizon`."""
    if credit_post_seed is not None and credit_post_seed.holds is not True:
        raise ValueError("a verdict that the property fails credits nothing after the seed: pass None")
    if credit_post_seed is not None and (type(schedule) is not Scheduled or not schedule.of(record)):
        raise ValueError("appended blocks are credited only on this session's schedule check (`Session.scheduled`): "
                         "without it the layout they are read at may follow the coins and the encodings, and no "
                         "SteeredLayout bound a has content")
    epsilon, link_delta = 0.0, 0.0
    if requirement is not None:
        if requirement.holds is not True:
            raise ValueError("a requirement link whose proofs don't hold credits nothing: pass None, and the credit is "
                             "the node's reads of its store, not the circuit's")
        if any(len(x) != w.passes for w in record.windows for x in (w.steps, w.heads)):
            raise ValueError("a requirement link reads each pass's step and head, and this record's passes name none: "
                             "link a session's own record")
        if requirement.mapped(record) is not True:
            raise ValueError("a requirement link whose memory-map unit doesn't make this record's appends (its "
                             "`appends` chain, each pass's head through the appends of the steps below its own, each "
                             "re-key below the next pass's stale bound) credits nothing: frees, re-keys and append order "
                             "chosen after the seed are advice no term charges")
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
        return sum(linked_bits(w, requirement, b, credit)) / 8

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
                  error_horizon=record.windows[0].profile.delta + beside, linked=requirement is not None,
                  steering=schedule.cap if credit else None)
