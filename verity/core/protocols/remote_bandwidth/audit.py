"""The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`): a verifier with no hardware at the site certifies how
many reads of a node's registered blocks the node made in each window, from receipts chained pass to pass.

1. Registration.  The prover commits to its payload blocks (`payload_tree`: the weights, KV pages and filler, each an
   hm96-sha512/row/v2 leaf) and sends the root.  Only then does the verifier draw the codec seed from its own secret and
   a fresh nonce (`register`), and the prover encodes every block under it (`Store`) with the codec the registration
   names (`codec.DEFAULT`, p1024 masked at k = 3, unless another is named).  `Registration.digest` binds the node, the
   store's size, the payload root, the codec seed and the codec's name.
2. Coins.  Before the session the challenger draws two live coins a window, its challenge `σ_t` and its openings coin,
   and commits to them (`coins.Challenger`).  The prover gets the commitment first (`Session.commitment`).
3. A session of `windows` windows.  The verifier releases `σ_0` (`Session.start`), and at each window's end the next
   window's challenge, or the close after the last (`Session.boundary`), one-way.
4. Chained passes.  Pass i of window t reads the blocks of its plan (`receipts.Plan`: distinct blocks, the whole store by
   default) and computes their receipts under `seed_(t,i) = H(σ_t, i, root_(t,i−1))` (`receipts.seed`), each bound to
   PoUW's tile output `z` before the read (`Tiles`); it commits them in a tree bound to the plan and sends the root and
   the plan as soon as the pass ends (`Session.receive`).  The chain runs across the session's windows: a window's first
   pass follows the root of the previous window's last.  On a boundary message the prover finishes the pass in flight,
   which is its window's last, and starts the next window's chain.
5. Deadline.  Window t's roots must all arrive by `D_t = b_t + A`, where `b_t` is the verifier's clock when it sent the
   boundary message and `A` the allowance (`Params.deadline_ns`: the message's trip down, the pass in flight, its root's
   reduction and its trip up, `timing.deadline`).  A late root, one out of order, or a plan that repeats a block rejects
   the session.  A window with fewer than `min_passes` roots is rejected.
6. Openings.  Once window t's roots are final (the next window's first root has arrived, or `D_t` has passed), the
   challenger releases the window's openings coin, and the verifier draws `s = samples` distinct receipts uniformly from
   the window's `U_t = Σ_i L_i` with it (`Session.close`).  The prover opens each (`Prover.open`): the block, its receipt
   leaf with its salt and path to the pass's root, the payload's salt and path to the payload root, and PoUW's evidence
   for the tile before it.
7. Verification (`Session.verify`), each opening: the leaf's path under the pass's root; the block's words canonical
   and their decoding's leaf on its path under the payload root; `z` recomputed by PoUW's tile check
   (`Tiles.recompute`); the receipt recomputed from the block and `z`.  A window is accepted only if every opening
   checks, and a rejected window ends its session.
8. The record (`Session.record`): per window each pass's length, its challenge's release and its deadline on the
   verifier's clock, its openings coin's release, and an `IntegrityProfile` over its receipts with `Subset(s)` drawn;
   `ledger` turns it, with the node's measured capacities, into each γ term, μ_sample read per window and pooled over
   the session's horizon (`pooled`).  What an accepted window certifies, and on what, is `PROTOCOL.md`.

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

FAILURES = ("late", "order", "plan", "few", "malformed", "receipt-path", "codec", "payload-path", "tile", "receipt")


@dataclass(frozen=True, slots=True)
class Params:
    """A session's parameters: the store's size `N`, the allowance `A` from a window's boundary message to its roots'
    deadline, the windows `W`, the openings `s` per window at `delta`, the fewest passes a window may certify, and the
    store's codec, the registration's."""

    blocks: int
    deadline_ns: int
    windows: int
    samples: int
    delta: float
    min_passes: int = 1
    codec: Codec = codec.DEFAULT

    def __post_init__(self) -> None:
        for name in ("blocks", "deadline_ns", "windows", "samples", "min_passes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} is a positive integer")
        if not 0 < self.delta < 1:
            raise ValueError("0 < delta < 1")
        if type(self.codec) is not Codec:
            raise ValueError("a session's codec is a Codec")

    @property
    def coins(self) -> int:
        """2W: each window's challenge and its openings coin."""
        return 2 * self.windows


def _node(node: bytes) -> bytes:
    if type(node) is not bytes or not 1 <= len(node) <= NODE_BYTES:
        raise ValueError(f"a node's id is 1 to {NODE_BYTES} bytes")
    return node


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
    """What the store is: the node, `N`, the payload root, the codec seed drawn after it, and the codec it names."""

    node: bytes
    blocks: int
    payload_root: bytes
    codec_seed: bytes
    codec: Codec

    @property
    def digest(self) -> bytes:
        """SHA512(REGISTRATION_DOMAIN ‖ frame(node) ‖ u64be N ‖ payload root ‖ codec seed ‖ frame(codec name)): what a
        record and a session's coin commitment cite, and `root_(0,−1)`, the root a session's first pass follows."""
        return hashlib.sha512(REGISTRATION_DOMAIN + frame(self.node) + self.blocks.to_bytes(8, "big") + self.payload_root
                              + self.codec_seed + frame(self.codec.name)).digest()

    @property
    def key(self) -> codec.Key:
        return codec.key(self.codec_seed, self.codec)


def register(source: bytes, node: bytes, blocks: int, payload_root: bytes, codec: Codec = codec.DEFAULT, *,
             nonce: bytes | None = None) -> Registration:
    """The verifier's side of registration: the codec seed, drawn from its own secret `source` and a fresh nonce only
    after the payload root has arrived.  Drawn before it, the prover would register `decode(0)` and hold a store of
    zeros; drawn after, each encoded block costs k square roots a word to produce, whatever the payload."""
    receipts.store_blocks(blocks)
    if type(codec) is not Codec:
        raise ValueError("a registration names a Codec")
    if blocks > codec.max_blocks:
        raise ValueError(f"a store under {codec.name} is at most {codec.max_blocks} blocks")
    if type(payload_root) is not bytes or len(payload_root) != DIGEST_BYTES:
        raise ValueError(f"the payload root is {DIGEST_BYTES} bytes")
    nonce = secrets.token_bytes(NONCE_BYTES) if nonce is None else nonce
    if type(nonce) is not bytes or len(nonce) != NONCE_BYTES:
        raise ValueError(f"a registration's nonce is {NONCE_BYTES} bytes")
    seed = derive(source, CODEC_DOMAIN, {"node": _node(node), "blocks": blocks, "payload": payload_root,
                                         "codec": codec.name, "nonce": nonce}).stream("codec", SEED_BYTES)
    return Registration(node, blocks, payload_root, seed, codec)


class Tiles(Protocol):
    """PoUW's side of a receipt.  `output(seed, j)` is `z_(i,j)`, the noised output of the tile the prover computes
    before reading block j in the pass whose seed is `seed`; `evidence` is what PoUW's check reads to recompute it; and
    `recompute`, the verifier's, returns `z_(i,j)` from that evidence or None if PoUW's check fails.  PoUW's noise for a
    pass is derived from the pass's seed, so `z` differs every pass whatever the weights and activations, and a pass
    reads each block once, with a tile of its own, so no `z` unlocks two receipts."""

    def output(self, seed: bytes, j: int) -> bytes: ...

    def evidence(self, seed: bytes, j: int) -> bytes: ...

    def recompute(self, seed: bytes, j: int, evidence: bytes) -> bytes | None: ...


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
    `z_(i,j)`."""

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
    commits them, each leaf under a fresh salt from `entropy`."""

    def __init__(self, store: Store, tiles: Tiles, commitment: coins.CoinCommitment, *,
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        self.store, self.tiles, self.commitment, self.entropy = store, tiles, commitment, entropy

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
        receipter = receipts.Receipter(seed)
        ys = tuple(receipter(j, self.store.blocks[j], self.tiles.output(seed, j)) for j in plan)
        salts = tuple(self.entropy(SALT_BYTES) for _ in range(receipts.leaves(plan.length)))
        return Pass(window, i, seed, plan, ys, salts, receipts.tree(seed, plan, ys, salts))

    def open(self, ps: Pass, p: int) -> Opening:
        j = ps.plan.block(p)
        c = p // receipts.PER_LEAF
        leaf = ps.tree.open(c)
        return Opening(ps.index, p, self.store.blocks[j], b"".join(ps.receipts[receipts.PER_LEAF * c:receipts.PER_LEAF * (c + 1)]),
                       ps.salts[c], leaf.path, self.store.salts[j], self.store.payload_tree.open(j).path,
                       self.tiles.evidence(ps.seed, j))


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

    def to_json(self) -> dict:
        return {"deadline_ns": self.deadline_ns, "lengths": list(self.lengths), "opened_ns": self.opened_ns,
                "profile": self.profile.to_json(), "released_ns": self.released_ns, "window": self.window}


@dataclass(frozen=True, slots=True)
class Record:
    """An accepted session: its registration, its coin commitment's root, its store's `blocks`, and every window's
    record.  `public` is what acceptance makes public; `profile` is the protocol's profile, each window's schedule and
    openings law (the γ terms need the node's capacities: `ledger`)."""

    registration: bytes
    coins: bytes
    blocks: int
    block_bytes: int
    windows: tuple[WindowRecord, ...]

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
        return {"block_bytes": self.block_bytes, "blocks": self.blocks, "coins": self.coins.hex(),
                "passes": [w.passes for w in self.windows], "protocol": PROTOCOL_ID,
                "receipts": [w.receipts for w in self.windows], "registration": self.registration.hex(),
                "span_ns": self.span_ns}

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
    `challenger` holds the session's coins (`coins.Challenger`), committed before the session.  `not_before_ns` is the
    earliest its first coin may leave: the previous session's `Record.end_ns` over the same store, less the trusted
    round-trip floor if there is one, so that no read serves receipts of both sessions."""

    def __init__(self, params: Params, registration: Registration, tiles: Tiles, challenger: coins.Challenger, *,
                 clock: Callable[[], int] = time.monotonic_ns, not_before_ns: int | None = None) -> None:
        if registration.blocks != params.blocks or registration.codec != params.codec:
            raise ValueError("the parameters are not the registration's")
        c = challenger.commitment
        if c.registration != registration.digest or c.count != params.coins:
            raise ValueError(f"the challenger's coins are not this session's: {params.coins} coins for this registration")
        self.params, self.registration, self.tiles, self.challenger = params, registration, tiles, challenger
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
        if self.not_before_ns is not None and now < self.not_before_ns:
            raise SessionError(f"the session's first coin may leave at {self.not_before_ns} ns, not {now}: the session "
                               "before it over this store may still be reading")
        return self._release(0, now)

    def boundary(self) -> coins.CoinOpening | None:
        """The end of the next unended window: the following window's challenge, or None, the close, after the last."""
        self._live()
        now = self.clock()
        t = next((t for t, w in enumerate(self.windows) if w.boundary_ns is None), None)
        if t is None:
            raise SessionError("every window has ended" if self.windows else "the session has not started")
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
        key = self.registration.key
        for (i, p), o in zip(w.draws, openings):
            failure = self._check(t, i, p, o, key)
            if failure is not None:
                return self._fail(t, failure, (i, p))
        w.verdict = Verdict(t, True)
        return w.verdict

    def _check(self, t: int, i: int, p: int, o: Opening, key: codec.Key) -> str | None:
        N = self.params.blocks
        if (type(o) is not Opening or o.pass_index != i or o.position != p or type(o.block) is not bytes
                or len(o.block) != BLOCK_BYTES or type(o.leaf) is not bytes or type(o.tile) is not bytes
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
        z = self.tiles.recompute(seed, j, o.tile)
        if type(z) is not bytes or not z:
            return "tile"
        if receipts.Receipter(seed)(j, o.block, z) != y:
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
        return Record(self.registration.digest, self.commitment.root, self.params.blocks, BLOCK_BYTES, tuple(out))


LEDGER_SOURCES = {
    "timing": "the record: the span its passes don't cover at the node's capacity (the round trip and the pass in "
              "flight at the session's ends, the chaining gaps, blocks a plan leaves out)",
    "sample": "the record: its openings' bound on wrong receipts, each window's own at delta (`Ledger.sample`, the "
              "per-window reading), or the session's pooled over its windows at one delta (`Ledger.horizon`, `pooled`)",
    "chip": "S_chip, measured: one pass's worth a block held on chip need not be read again",
    "counting": "AuditRO's slack, queries_log2 bits a receipt (each certified one, whichever reading) and slack_log2 a "
                "pass",
    "rebuild": "a model: re-encoding blocks from their payload with leftover compute (WorkHardness(p, k))",
    "host": "a model: the host tier's own shortfall and host DRAM bandwidth PCIe cannot carry",
    "link": "a model: relays meeting z and a block copy over a link",
}


@dataclass(frozen=True, slots=True)
class Ledger:
    """Each γ term as a fraction of the node's read capacity over the session's span; γ is their sum.  μ_sample comes
    both ways: `sample`, each window's openings bounding its own wrong receipts at δ, and `horizon`, the session's
    openings pooled over its windows at one δ (`pooled`).  `floor` is the hybrid claim's other half, from the same
    pooled event: no window has more than that share of its receipts wrong."""

    timing: float
    sample: float
    chip: float
    counting: float
    rebuild: float
    host: float
    link: float
    horizon: float
    floor: float

    @property
    def gamma(self) -> float:
        """γ with the per-window μ_sample, except with probability `windows · delta`."""
        return sum(getattr(self, name) for name in LEDGER_SOURCES)

    @property
    def gamma_horizon(self) -> float:
        """γ with the pooled μ_sample, except with probability `delta`, `floor` holding with it."""
        return self.horizon + sum(getattr(self, name) for name in LEDGER_SOURCES if name != "sample")

    def to_json(self) -> dict:
        terms = {name: getattr(self, name) for name in LEDGER_SOURCES}
        return {"gamma": self.gamma, "horizon": {"floor": self.floor, "gamma": self.gamma_horizon, "sample": self.horizon},
                "terms": terms}


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
    delta = record.windows[0].profile.delta
    drawn = [w.profile.drawn["receipt"] for w in record.windows]
    budget = -math.expm1(math.log(delta) / sum(drawn)) * sum(drawn)
    floor = max(min(1 - s / w.receipts, budget / s) for w, s in zip(record.windows, drawn))
    wrong = 0.0
    for w, s in sorted(zip(record.windows, drawn), key=lambda x: x[0].receipts / x[1], reverse=True):
        f = min(1 - s / w.receipts, budget / s)
        wrong, budget = wrong + f * w.receipts, max(0.0, budget - f * s)
    return wrong, floor


def ledger(record: Record, *, capacity_bytes_per_s: float, chip_bytes: int, rebuild: float = 0.0, host: float = 0.0,
           link: float = 0.0, queries_log2: int = QUERIES_LOG2, slack_log2: int = SLACK_LOG2,
           wrong: Callable[[IntegrityProfile], float] = IntegrityProfile.bound) -> Ledger:
    """γ's terms for an accepted session on a node of `capacity_bytes_per_s` (HBM plus host DRAM) and `chip_bytes` of
    on-chip storage: the read bytes the record certifies are at least `(1 − γ)·capacity·span`, except with probability
    `windows · delta`, and at least `(1 − γ_horizon)·capacity·span` except with `delta` (each plus 2^-slack_log2 a
    pass).  `rebuild`, `host` and `link` are the models' fractions, passed through: the record does not bound them.
    `wrong` is the profile's bound on wrong receipts (`IntegrityProfile.bound`; pass an approximation for a large
    store)."""
    if capacity_bytes_per_s <= 0 or chip_bytes < 0:
        raise ValueError("the capacity is positive and the chip's storage nonnegative")
    whole = capacity_bytes_per_s * record.span_ns / 1e9
    b = record.block_bytes
    reads = record.receipts * b
    if reads > whole:
        raise ValueError(f"the record certifies {reads} bytes read, more than the capacity allows in its span: the "
                         "capacity is understated")
    wrong_receipts = sum(math.ceil(wrong(w.profile)) for w in record.windows)
    horizon, floor = pooled(record)
    lam_bits = queries_log2 * record.receipts + slack_log2 * record.passes
    return Ledger(timing=1 - reads / whole, sample=wrong_receipts * b / whole,
                  chip=record.passes * chip_bytes / whole, counting=lam_bits / 8 / whole, rebuild=rebuild, host=host,
                  link=link, horizon=horizon * b / whole, floor=floor)
