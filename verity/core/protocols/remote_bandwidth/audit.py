"""The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`): a verifier with no hardware at the site certifies how
many reads of a node's registered blocks the node made, pass by pass, from receipts keyed by the challenger's live coins.

1. Registration.  The prover commits to its payload blocks (`payload_tree`: the weights, KV pages and filler, each an
   hm96-sha512/row/v2 leaf) and sends the root.  Only then does the verifier draw the codec seed from its own secret and
   a fresh nonce (`register`), and the prover encodes every block under it (`Store`).  `Registration.digest` binds the
   node, the store's size, the payload root, the codec seed and k.
2. Coins.  Before the session the challenger draws the session's coins and commits to them (`coins.Challenger`): one a
   pass and one for each window's openings.  The prover gets the commitment first (`Session.commitment`).
3. The stream.  The verifier releases pass k's coin `c_k` on schedule, `Δ` after the one before (`Session.release`),
   one-way.  Pass k reads the blocks of its plan (`receipts.Plan`: distinct blocks, the whole store by default) and
   computes their receipts under `seed_k = H(c_k, k)` (`receipts.seed`), each bound to PoUW's tile output `z` before the
   read (`Tiles`).  It commits them in a tree bound to the plan and sends the root and the plan (`Session.receive`).
   The schedule alone orders the passes: nothing in a pass depends on another pass's root.
4. Deadlines.  Pass k's root must arrive by `D_k = r_k + A`, where `r_k` is the verifier's clock when `c_k` left and `A` is
   the allowance (`Params.deadline_ns`: the round trip, the pass and its root's reduction, `timing.deadline`).  A late
   root, a root for a coin not yet released, a second root for a pass, or a plan that repeats a block rejects the
   session.
5. Openings.  Window t is passes tP … tP + P − 1.  Once their roots are all in, or the last one's deadline has passed
   (and then a missing root rejects the window), the challenger releases the window's openings coin, and the verifier
   draws `s = samples` distinct receipts uniformly from the window's `U_t = Σ_k L_k` with it (`Session.close`).  The
   prover opens each (`Prover.open`): the block, its receipt leaf with its salt and path to the pass's root, the
   payload's salt and path to the payload root, and PoUW's evidence for the tile before it.
6. Verification (`Session.verify`), each opening: the leaf's path under the pass's root; the block's words canonical
   and their decoding's leaf on its path under the payload root; `z` recomputed by PoUW's tile check
   (`Tiles.recompute`); the receipt recomputed from the block and `z`.  A window is accepted only if every opening
   checks, and a rejected window ends its session.
7. The record (`Session.record`): per window, each pass's length, its coin's release and its deadline on the verifier's
   clock, and an `IntegrityProfile` over its receipts with `Subset(s)` drawn.  `ledger` turns it into each γ term with
   the node's measured capacities and the trusted round-trip floor.  `PROTOCOL.md` says what an accepted window
   certifies, and on what.

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
from .codec import BLOCK_BYTES
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

FAILURES = ("late", "early", "duplicate", "plan", "missing", "malformed", "receipt-path", "codec", "payload-path", "tile",
            "receipt")


@dataclass(frozen=True, slots=True)
class Params:
    """A session's parameters: the store's size `N`, the passes `P` a window, the coin interval `Δ`, the allowance `A`
    from a coin's release to its root's deadline, the windows `W`, the openings `s` per window at `delta`, and the
    codec's k."""

    blocks: int
    passes: int
    interval_ns: int
    deadline_ns: int
    windows: int
    samples: int
    delta: float
    rounds: int = codec.MIN_ROUNDS

    def __post_init__(self) -> None:
        for name in ("blocks", "passes", "interval_ns", "deadline_ns", "windows", "samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} is a positive integer")
        if not 0 < self.delta < 1:
            raise ValueError("0 < delta < 1")
        codec.key(bytes(codec.SEED_BYTES), self.rounds)

    @property
    def coins(self) -> int:
        """K = W·(P + 1): a coin a pass and one for each window's openings."""
        return self.windows * (self.passes + 1)


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
    """What the store is: the node, `N`, the payload root, the codec seed drawn after it, and k."""

    node: bytes
    blocks: int
    payload_root: bytes
    codec_seed: bytes
    rounds: int

    @property
    def digest(self) -> bytes:
        """SHA512(REGISTRATION_DOMAIN ‖ frame(node) ‖ u64be N ‖ payload root ‖ codec seed ‖ u8 k): what a record and a
        session's coin commitment cite."""
        return hashlib.sha512(REGISTRATION_DOMAIN + frame(self.node) + self.blocks.to_bytes(8, "big") + self.payload_root
                              + self.codec_seed + self.rounds.to_bytes(1, "big")).digest()

    @property
    def key(self) -> codec.Key:
        return codec.key(self.codec_seed, self.rounds)


def register(source: bytes, node: bytes, blocks: int, payload_root: bytes, rounds: int = codec.MIN_ROUNDS, *,
             nonce: bytes | None = None) -> Registration:
    """The verifier's side of registration: the codec seed, drawn from its own secret `source` and a fresh nonce only
    after the payload root has arrived.  Drawn before it, the prover would register `decode(0)` and hold a store of
    zeros; drawn after, each encoded block costs k square roots a word to produce, whatever the payload."""
    if type(blocks) is not int or blocks < 1:
        raise ValueError("a store has at least one block")
    if type(payload_root) is not bytes or len(payload_root) != DIGEST_BYTES:
        raise ValueError(f"the payload root is {DIGEST_BYTES} bytes")
    nonce = secrets.token_bytes(NONCE_BYTES) if nonce is None else nonce
    if type(nonce) is not bytes or len(nonce) != NONCE_BYTES:
        raise ValueError(f"a registration's nonce is {NONCE_BYTES} bytes")
    codec.key(bytes(codec.SEED_BYTES), rounds)
    seed = derive(source, CODEC_DOMAIN, {"node": _node(node), "blocks": blocks, "payload": payload_root,
                                         "nonce": nonce}).stream("codec", codec.SEED_BYTES)
    return Registration(node, blocks, payload_root, seed, rounds)


class Tiles(Protocol):
    """PoUW's side of a receipt.  `output(seed, j)` is `z_(k,j)`, the noised output of the tile the prover computes
    before reading block j in the pass whose seed is `seed`; `evidence` is what PoUW's check reads to recompute it; and
    `recompute`, the verifier's, returns `z_(k,j)` from that evidence or None if PoUW's check fails.  PoUW's noise for a
    pass is derived from the pass's seed, so `z` differs every pass whatever the weights and activations, and a pass
    reads each block once, with a tile of its own, so no `z` unlocks two receipts."""

    def output(self, seed: bytes, j: int) -> bytes: ...

    def evidence(self, seed: bytes, j: int) -> bytes: ...

    def recompute(self, seed: bytes, j: int, evidence: bytes) -> bytes | None: ...


@dataclass(frozen=True, slots=True)
class NoiseTiles:
    """A stub of PoUW's tile recompute: `z_(k,j)` is `z_bytes` bytes of SHA-512 in counter mode over
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
    """One pass of the prover's: its index in the session, its seed, its plan, the receipts of the plan's blocks in plan
    order, its leaves' salts, and their tree."""

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
    """Receipt (k, p) of a window opened, position p of pass k, block j = plan_k[p]: the block, its receipt leaf's
    receipts, salt and path to pass k's root, the payload's salt and path to the payload root, and PoUW's evidence for
    `z_(k,j)`."""

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
        return (8 + 8 + len(self.block) + len(self.leaf) + len(self.leaf_salt) + len(self.payload_salt)
                + DIGEST_BYTES * (len(self.receipt_path) + len(self.payload_path)) + len(self.tile) + 4 * 3)


class Prover:
    """The honest prover: each pass reads its plan's blocks (every block by default) once its coin is there, computes
    their receipts and commits them, each leaf under a fresh salt from `entropy`."""

    def __init__(self, store: Store, tiles: Tiles, commitment: coins.CoinCommitment, *,
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        self.store, self.tiles, self.commitment, self.entropy = store, tiles, commitment, entropy

    def run(self, coin: coins.CoinOpening, plan: Plan | None = None) -> Pass:
        """The pass of coin `coin`, refused unless the coin opens against the session's commitment."""
        reason = coins.coin_reason(self.commitment, coin, getattr(coin, "index", None))
        if reason is not None:
            raise ValueError(reason)
        return self.pass_under(coin.index, receipts.seed(coin.coin, coin.index), plan)

    def pass_under(self, k: int, seed: bytes, plan: Plan | None = None) -> Pass:
        """Pass k's receipts and tree under `seed`, whatever its coin."""
        plan = plan or Plan.whole(self.store.registration.blocks)
        receipter = receipts.Receipter(seed)
        ys = tuple(receipter(j, self.store.blocks[j], self.tiles.output(seed, j)) for j in plan)
        salts = tuple(self.entropy(SALT_BYTES) for _ in range(receipts.leaves(plan.length)))
        return Pass(k, seed, plan, ys, salts, receipts.tree(seed, plan, ys, salts))

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
    """What an accepted window established: passes of `lengths` reads of distinct registered blocks each, pass i's
    receipts made after its coin left at `released[i]` and committed by its root's arrival, by `deadlines[i]`, on the
    verifier's clock; the openings coin released at `opened_ns`; and the openings' law (`profile`)."""

    window: int
    lengths: tuple[int, ...]
    released: tuple[int, ...]
    deadlines: tuple[int, ...]
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
        return {"deadlines": list(self.deadlines), "lengths": list(self.lengths), "opened_ns": self.opened_ns,
                "profile": self.profile.to_json(), "released": list(self.released), "window": self.window}


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
        return self.windows[0].released[0]

    @property
    def end_ns(self) -> int:
        return max(max(w.deadlines) for w in self.windows)

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
    """A call the session's state refuses: out of turn, after the session ended, or before a window's roots are in."""


@dataclass(slots=True)
class _Window:
    draws: tuple[tuple[int, int], ...] | None = None
    opened_ns: int | None = None
    verdict: Verdict | None = None


class Session:
    """The verifier's side of one session, driven by the coins it releases and the roots it receives, on its own clock.
    `challenger` holds the session's coins (`coins.Challenger`), committed before the session."""

    def __init__(self, params: Params, registration: Registration, tiles: Tiles, challenger: coins.Challenger, *,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        if registration.blocks != params.blocks or registration.rounds != params.rounds:
            raise ValueError("the parameters are not the registration's")
        c = challenger.commitment
        if c.registration != registration.digest or c.count != params.coins:
            raise ValueError(f"the challenger's coins are not this session's: {params.coins} coins for this registration")
        self.params, self.registration, self.tiles, self.challenger = params, registration, tiles, challenger
        self.clock = clock
        self.coins: list[bytes] = []
        self.released: list[int] = []
        self.roots: dict[int, bytes] = {}
        self.plans: dict[int, Plan] = {}
        self.windows = [_Window() for _ in range(params.windows)]
        self.failed: Verdict | None = None
        self._payload_domain = payload_domain(registration.node, registration.blocks)

    @property
    def commitment(self) -> coins.CoinCommitment:
        """The session's coin commitment, sent to the prover before the first coin."""
        return self.challenger.commitment

    def _live(self) -> None:
        if self.failed is not None:
            raise SessionError(f"the session ended: window {self.failed.window} was rejected ({self.failed.failure})")

    def _window(self, k: object) -> int:
        return min(max(k // self.params.passes, 0), self.params.windows - 1) if type(k) is int else 0

    def _fail(self, t: int, failure: str, at: tuple[int, int] | None = None) -> Verdict:
        self.failed = Verdict(t, False, failure, at)
        self.windows[t].verdict = self.failed
        return self.failed

    def deadline(self, k: int) -> int:
        """`D_k = r_k + A`, once pass k's coin has been released."""
        return self.released[k] + self.params.deadline_ns

    def release(self) -> coins.CoinOpening:
        """The next pass's coin, released now: refused before its scheduled time `r_0 + k·Δ`."""
        self._live()
        k, now = len(self.released), self.clock()
        if k == self.params.windows * self.params.passes:
            raise SessionError("every pass's coin has been released")
        if k and now < self.released[0] + k * self.params.interval_ns:
            raise SessionError(f"pass {k}'s coin is due at {self.released[0] + k * self.params.interval_ns} ns, not {now}")
        opening = self.challenger.open(k)
        self.coins.append(opening.coin)
        self.released.append(now)
        return opening

    def receive(self, k: int, root: bytes, runs: Sequence[tuple[int, int]] | None = None) -> Verdict | None:
        """Pass k's root and its plan as the prover sends it, `runs` (None: the whole store in order), as they arrive.
        Returns the rejection they cause, if any."""
        self._live()
        now = self.clock()
        if type(k) is not int or type(root) is not bytes or len(root) != DIGEST_BYTES:
            return self._fail(self._window(k), "malformed")
        if not 0 <= k < len(self.released):
            return self._fail(self._window(k), "early")
        if k in self.roots:
            return self._fail(self._window(k), "duplicate")
        if now > self.deadline(k):
            return self._fail(self._window(k), "late")
        try:
            plan = Plan.whole(self.params.blocks) if runs is None else Plan(tuple(runs), self.params.blocks)
        except (TypeError, ValueError):
            return self._fail(self._window(k), "plan")
        self.roots[k], self.plans[k] = root, plan
        return None

    def _passes(self, t: int) -> range:
        return range(t * self.params.passes, (t + 1) * self.params.passes)

    def close(self, t: int) -> tuple[tuple[int, int], ...] | Verdict:
        """Window t's draws, from its openings coin released now: `s` distinct receipts (pass, position), uniform over
        its `U_t = Σ_k L_k`, in order.  Refused until its roots are all in or its last deadline has passed; its rejection
        when a root is missing then."""
        self._live()
        if not 0 <= t < self.params.windows or self.windows[t].draws is not None:
            raise SessionError(f"window {t} is not open to draw")
        ks = self._passes(t)
        if len(self.released) < ks.stop:
            raise SessionError(f"window {t}'s coins are not all released")
        now = self.clock()
        if any(k not in self.roots for k in ks):
            if now <= max(self.deadline(k) for k in ks):
                raise SessionError(f"window {t}'s roots are not in before its last deadline")
            return self._fail(t, "missing")
        coin = self.challenger.open(self.params.windows * self.params.passes + t)
        key = derive(coin.coin, OPENINGS_DOMAIN, {"registration": self.registration.digest, "window": t})
        ends = tuple(itertools.accumulate(self.plans[k].length for k in ks))
        draws = []
        for u in key.subset(min(self.params.samples, ends[-1]), ends[-1], "openings"):
            i = bisect.bisect_right(ends, u)
            draws.append((ks[i], u - (ends[i - 1] if i else 0)))
        w = self.windows[t]
        w.draws, w.opened_ns = tuple(draws), now
        return w.draws

    def verify(self, t: int, openings: Sequence[Opening]) -> Verdict:
        """Window t's verdict on the prover's openings of its draws, in the order drawn."""
        self._live()
        w = self.windows[t] if 0 <= t < self.params.windows else None
        if w is None or w.draws is None or w.verdict is not None:
            raise SessionError(f"window {t} has no draws awaiting openings")
        if len(openings) != len(w.draws):
            return self._fail(t, "malformed")
        key = self.registration.key
        for (k, p), o in zip(w.draws, openings):
            failure = self._check(k, p, o, key)
            if failure is not None:
                return self._fail(t, failure, (k, p))
        w.verdict = Verdict(t, True)
        return w.verdict

    def _check(self, k: int, p: int, o: Opening, key: codec.Key) -> str | None:
        N = self.params.blocks
        if (type(o) is not Opening or o.pass_index != k or o.position != p or type(o.block) is not bytes
                or len(o.block) != BLOCK_BYTES or type(o.leaf) is not bytes or type(o.tile) is not bytes
                or any(type(x) is not bytes or len(x) != SALT_BYTES for x in (o.leaf_salt, o.payload_salt))
                or type(o.receipt_path) is not tuple or type(o.payload_path) is not tuple
                or any(type(x) is not bytes or len(x) != DIGEST_BYTES for x in (*o.receipt_path, *o.payload_path))):
            return "malformed"
        plan = self.plans[k]
        j = plan.block(p)
        seed = receipts.seed(self.coins[k], k)
        y = receipts.leaf_receipt(o.leaf, p, plan.length)
        if y is None:
            return "malformed"
        if not receipts.verify_leaf(seed, plan, self.roots[k], p, o.leaf, o.leaf_salt, o.receipt_path):
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
        if any(w.verdict is None or not w.verdict.accepted for w in self.windows):
            raise SessionError("a record needs every window of the session accepted")
        out = []
        for t, w in enumerate(self.windows):
            ks = self._passes(t)
            lengths = tuple(self.plans[k].length for k in ks)
            profile = IntegrityProfile(("receipt",), (sum(lengths),), {"receipt": Subset(len(w.draws))},
                                       self.params.delta, drawn={"receipt": len(w.draws)})
            out.append(WindowRecord(t, lengths, tuple(self.released[k] for k in ks),
                                    tuple(self.deadline(k) for k in ks), w.opened_ns, profile))
        return Record(self.registration.digest, self.commitment.root, self.params.blocks, BLOCK_BYTES, tuple(out))


def overlap_ns(released: Sequence[int], deadlines: Sequence[int], rtt_floor_ns: int) -> int:
    """`Σ_k |I_k| − |∪_k I_k|` for `I_k = [r_k, D_k − RTT_floor]`, pass k's window for reads shifted by the coin's one-way
    floor: the time in which one read may serve receipts of two or more passes.  Zero when consecutive passes' windows
    don't overlap, `Δ ≥ A − RTT_floor`."""
    if rtt_floor_ns < 0:
        raise ValueError("a round-trip floor is nonnegative")
    spans = sorted((r, d - rtt_floor_ns) for r, d in zip(released, deadlines, strict=True) if d - rtt_floor_ns > r)
    total = sum(e - s for s, e in spans)
    union, reach = 0, None
    for s, e in spans:
        if reach is None or s > reach:
            union += e - s
            reach = e
        elif e > reach:
            union += e - reach
            reach = e
    return total - union


LEDGER_SOURCES = {
    "timing": "the record: the span its passes don't cover at the node's capacity (the round trip at the ends, the "
              "coin interval a pass doesn't fill, blocks a plan leaves out)",
    "overlap": "the record and the trusted round-trip floor: reads that may serve passes whose windows overlap",
    "sample": "the record: each window's openings law, its bound on wrong receipts at delta",
    "chip": "S_chip, measured: one pass's worth a block held on chip need not be read again",
    "counting": "AuditRO's slack, queries_log2 bits a certified receipt and slack_log2 a pass",
    "rebuild": "a model: re-encoding blocks from their payload with leftover compute (WorkHardness(p, k))",
    "host": "a model: the host tier's own shortfall and host DRAM bandwidth PCIe cannot carry",
    "link": "a model: relays meeting z and a block copy over a link",
}


@dataclass(frozen=True, slots=True)
class Ledger:
    """Each γ term as a fraction of the node's read capacity over the session's span; γ is their sum."""

    timing: float
    overlap: float
    sample: float
    chip: float
    counting: float
    rebuild: float
    host: float
    link: float

    @property
    def gamma(self) -> float:
        return sum(getattr(self, name) for name in LEDGER_SOURCES)

    def to_json(self) -> dict:
        terms = {name: getattr(self, name) for name in LEDGER_SOURCES}
        return {"gamma": self.gamma, "terms": terms}


def ledger(record: Record, *, capacity_bytes_per_s: float, chip_bytes: int, rtt_floor_ns: int = 0, rebuild: float = 0.0,
           host: float = 0.0, link: float = 0.0, queries_log2: int = QUERIES_LOG2, slack_log2: int = SLACK_LOG2,
           wrong: Callable[[IntegrityProfile], float] = IntegrityProfile.bound) -> Ledger:
    """γ's terms for an accepted session on a node of `capacity_bytes_per_s` (HBM plus host DRAM) and `chip_bytes` of
    on-chip storage, `rtt_floor_ns` the round trip between the challenger and the node that no message can beat (0, the
    default, trusts none): the read bytes the record certifies are at least `(1 − γ)·capacity·span`, except with
    probability `windows · delta` (plus 2^-slack_log2 a pass).  `rebuild`, `host` and `link` are the models' fractions,
    passed through: the record does not bound them.  `wrong` is the profile's bound on wrong receipts
    (`IntegrityProfile.bound`; pass an approximation for a large store)."""
    if capacity_bytes_per_s <= 0 or chip_bytes < 0:
        raise ValueError("the capacity is positive and the chip's storage nonnegative")
    span = record.span_ns
    whole = capacity_bytes_per_s * span / 1e9
    b = record.block_bytes
    reads = record.receipts * b
    shared = overlap_ns([r for w in record.windows for r in w.released], [d for w in record.windows for d in w.deadlines],
                        rtt_floor_ns)
    if reads > whole * (1 + shared / span):
        raise ValueError(f"the record certifies {reads} bytes read, more than the capacity allows in its span: the "
                         "capacity is understated")
    wrong_receipts = [math.ceil(wrong(w.profile)) for w in record.windows]
    certified = [w.receipts - m for w, m in zip(record.windows, wrong_receipts)]
    lam_bits = sum(queries_log2 * k + slack_log2 * w.passes for w, k in zip(record.windows, certified))
    return Ledger(timing=1 - reads / whole, overlap=shared / span, sample=sum(wrong_receipts) * b / whole,
                  chip=record.passes * chip_bytes / whole, counting=lam_bits / 8 / whole, rebuild=rebuild, host=host,
                  link=link)
