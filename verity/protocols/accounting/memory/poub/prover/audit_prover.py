"""PoUB's honest prover (`verity/remote-bandwidth/v0`, `verity.protocols.accounting.memory.poub.verifier.audit`): each pass's
receipts over its store, committed in the pass's tree; the window's sampled receipts opened; and, at the session's end,
every challenge it was sent checked against the coin commitment. The store, the passes and the openings are the
audit's (`Store`, `Pass`, `audit.Opening`).
"""

from __future__ import annotations

import bisect
import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from verity.commitments import MerkleTree
from verity.protocols.accounting.memory.poub.verifier.rota import REKEY_MAX, Origin, Schedule
from verity.protocols.accounting.memory.poub.verifier import audit, codec, coins, receipts
from verity.protocols.accounting.memory.poub.verifier.audit import (
    REKEY_DOMAIN,
    Registration,
    append_head,
    append_tree,
    payload_tree,
)
from verity.protocols.accounting.memory.poub.verifier.receipts import SALT_BYTES, Plan


def rekey_root(registration: bytes, first: int, count: int, origin: Origin) -> bytes:
    """`rekeyRoot` of `Poub/Session.lean`: SHA512(REKEY_DOMAIN ‖ registration ‖ u64be first ‖ u64be count ‖ u8 0 ‖
    u64be lo) for `("registered", lo)`, or ‖ u8 1 ‖ u64be k for `("append", k)`, k a slot."""
    tag = {"registered": b"\0", "append": b"\1"}[origin[0]]
    return hashlib.sha512(REKEY_DOMAIN + registration + first.to_bytes(8, "big") + count.to_bytes(8, "big") + tag
                          + origin[1].to_bytes(8, "big")).digest()


@dataclass(frozen=True, slots=True)
class Append:
    """Blocks written after the codec seed, the prover's: store indices `first` … `first + count − 1`, their payloads'
    digests `d_j` (kept beside the blocks), their salts, and their payload tree (`append_tree`), whose root the
    verifier receives before any pass whose plan holds one.  A re-key (`origin` set) has no tree: its digests and salts
    are its origin's blocks', and its root is its declaration (`rekey_root`)."""

    first: int
    digests: tuple[bytes, ...]
    salts: tuple[bytes, ...]
    tree: MerkleTree | None
    origin: Origin | None = None
    declaration: bytes | None = None
    slot: int | None = None

    @property
    def count(self) -> int:
        return len(self.digests)

    @property
    def root(self) -> bytes:
        return self.declaration if self.tree is None else self.tree.commitment.root


class Store:
    """The prover's encoded store `E` under a registration, with its payload tree and the payloads' salts, and the
    blocks appended after the codec seed (`append`) at the next fresh store indices, each encoded under its bound key
    and its append's coin once the verifier has sent it (`encode`; None until then, its payload `pending`).  A
    released block is kept, since a window's samples may still open it from a pass before its release.  Each block's
    payload (`payload`) is the witness check 3's unit takes (`witness`)."""

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
        self.coin_of: dict[int, bytes] = {}

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

    def append(self, payloads: Sequence[bytes], salts: Sequence[bytes], slot: int | None = None) -> Append:
        """Payload blocks written after the codec seed, at the next fresh store indices, committed in one `append_tree`
        by their digests, kept beside them; their encodings wait on the append's coin (`encode`).  Under refresh it
        fills `slot`, a fresh slot of the session's `Schedule`."""
        reg, first = self.registration, self.extent
        if first + len(payloads) > reg.codec.max_blocks:
            raise ValueError(f"a store under {reg.codec.name} is at most {reg.codec.max_blocks} blocks")
        digests = tuple(codec.digest(m) for m in payloads)
        a = Append(first, digests, tuple(salts), append_tree(reg.digest, first, digests, salts), slot=slot)
        self.blocks.extend([None] * a.count)
        self.pending[first] = tuple(payloads)
        self.appends.append(a)
        return a

    def slotted(self, first: int, slot: int) -> Append:
        """The append of pages from block `first`, filling `slot` once it is sent."""
        i = next(i for i, a in enumerate(self.appends) if a.first == first)
        self.appends[i] = replace(self.appends[i], slot=slot)
        return self.appends[i]

    def in_slot(self, k: int) -> Append | None:
        """The append filling slot k, if any."""
        return next((a for a in self.appends if a.slot == k), None)

    def rekey(self, origin: Origin, count: int, slot: int | None = None) -> Append:
        """A refresh re-key, in `slot`, of `origin` whole: registered run [lo, lo + count), or the fresh append in
        slot k, of its count; at the next fresh store indices, the same payloads, encoded under the re-key's own coin
        once it is sent (`encode`), committed by no tree of their own."""
        reg, first, n = self.registration, self.extent, self.registration.blocks
        src = self.in_slot(origin[1]) if origin[0] == "append" and len(origin) == 2 else None
        if (origin[0] == "registered" and len(origin) == 2 and type(count) is int and count >= 1
                and 0 <= origin[1] and origin[1] + count <= n):
            at = origin[1]
            payloads = self.payloads[at:at + count]
            digests, salts = tuple(codec.digest(m) for m in payloads), self.salts[at:at + count]
        elif src is not None and src.origin is None and count == src.count:
            payloads = tuple(self.payload(src.first + u) for u in range(count))
            digests, salts = src.digests, src.salts
        else:
            raise ValueError(f"no re-key of {count} blocks of {origin!r}: a registered run inside the store's {n} "
                             "blocks, or a fresh append's slot of that count, never of a re-key")
        if first + count > reg.codec.max_blocks:
            raise ValueError(f"a store under {reg.codec.name} is at most {reg.codec.max_blocks} blocks")
        a = Append(first, digests, tuple(salts), None, origin, rekey_root(reg.digest, first, count, origin), slot)
        self.blocks.extend([None] * count)
        self.pending[first] = tuple(payloads)
        self.appends.append(a)
        return a

    def origin_position(self, j: int) -> int | None:
        """The store index block j's payload was first committed at, if j is a re-key's block; None otherwise."""
        if j < self.registration.blocks:
            return None
        a = self._append_of(j)
        if a.origin is None:
            return None
        start = a.origin[1] if a.origin[0] == "registered" else self.in_slot(a.origin[1]).first
        return start + j - a.first

    def encode(self, first: int, coin: bytes) -> None:
        """The blocks of the append from `first`, encoded under their bound keys and its coin `c_a`, the verifier's
        reply to its root (`codec.encode(..., bound=True, coin=c_a)`): once, for an append awaiting its coin."""
        payloads = self.pending.get(first)
        if payloads is None:
            raise ValueError(f"no append from block {first} awaits its coin")
        k = self.registration.key
        encoded = [codec.encode(m, first + u, k, bound=True, coin=coin) for u, m in enumerate(payloads)]
        del self.pending[first]
        self.coin_of[first] = coin
        self.blocks[first:first + len(encoded)] = encoded

    def _append_of(self, j: int) -> Append:
        return self.appends[bisect.bisect_right([x.first for x in self.appends], j) - 1]

    def committed(self, j: int) -> tuple[bytes, tuple[bytes, ...], bytes]:
        """Block j's payload salt, its path to the payload root or its append's, and its digest `d_j` (empty under
        the seed's key); a re-key's block's salt and path are its origin's leaf's."""
        if j < self.registration.blocks:
            return self.salts[j], self.payload_tree.open(j).path, b""
        a = self._append_of(j)
        d = a.digests[j - a.first]
        if (at := self.origin_position(j)) is not None:
            if at < self.registration.blocks:
                return self.salts[at], self.payload_tree.open(at).path, d
            return a.salts[j - a.first], self._append_of(at).tree.open(at).path, d
        return a.salts[j - a.first], a.tree.open(j).path, d

    def payload(self, j: int) -> bytes:
        """Block j's payload: registered, or appended, its append's while it awaits its coin and then its block decoded
        under its digest and that coin, so the store keeps no second copy of what it appends."""
        if j < self.registration.blocks:
            return self.payloads[j]
        a = self._append_of(j)
        if a.first in self.pending:
            return self.pending[a.first][j - a.first]
        m = codec.decode(self.blocks[j], j, self.registration.key, a.digests[j - a.first], self.coin_of[a.first])
        if m is None:
            raise ValueError(f"block {j} decodes to no payload of its digest under its append's coin")
        return m


class Refresh:
    """The honest prover's refresh on the session's `Schedule`: the re-keys of window t's slots it makes before the
    window's first root (`due`), the fresh slot each append of pages takes when it is sent (`sent`), the indices each
    re-key's encode moves or withdraws (`appended`), and the plan of a window (`plan`).  A chunk is named by the store
    index its payload was first committed at.  A re-key slot is filled when a chunk of its origin is still kept, and
    re-keys its origin whole: the blocks of chunks no longer kept are withdrawn."""

    def __init__(self, store: Store, schedule: Schedule) -> None:
        if schedule.blocks != store.registration.blocks:
            raise ValueError(f"the schedule is of {schedule.blocks} registered blocks, the store of "
                             f"{store.registration.blocks}")
        self.store, self.schedule = store, schedule
        self.held: dict[int, int] = {j: j for j in range(schedule.blocks)}
        self.last = -1
        self.unsent = 0
        self.filled: dict[int, int] = {}
        self._plan: tuple[tuple, Plan] | None = None

    def appended(self, a: Append) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """`a`, made: a fresh append's blocks become chunks; a re-key's encode moves each kept chunk of its origin to
        its index.  The indices that no longer hold a chunk, which the prover releases, and those of `a` that hold
        none, withdrawn."""
        if a.origin is None:
            if a.count > REKEY_MAX:
                raise ValueError(f"an append of pages under refresh is at most {REKEY_MAX} blocks, not {a.count}")
            self.unsent += 1
            self.held.update((j, j) for j in range(a.first, a.first + a.count))
            return (), ()
        released, withdrawn = [], []
        for j in range(a.first, a.first + a.count):
            p = self.store.origin_position(j)
            if p in self.held:
                released.append(self.held[p])
                self.held[p] = j
            else:
                withdrawn.append(j)
        return tuple(sorted(released)), tuple(withdrawn)

    def fresh_left(self, t: int) -> bool:
        """Whether pages made now in window t still find a fresh slot of t, beside the appends of pages unsent."""
        return self.filled.get(t, 0) + self.unsent < self.schedule.fresh_slots

    def sent(self, a: Append, t: int) -> Append:
        """`a`, sent in window t: an append of pages takes t's next fresh slot (`Store.slotted`); refused once t's F
        are used, or for a re-key whose slot is not t's."""
        s = self.schedule
        if a.origin is None:
            k = next((k for i in range(s.fresh_slots) if (k := s.fresh_slot(t, i)) > self.last), None)
            if k is None:
                raise ValueError(f"window {t}'s {s.fresh_slots} fresh slots are filled: pages wait for the next window")
            self.unsent -= 1
            self.filled[t] = self.filled.get(t, 0) + 1
            a = self.store.slotted(a.first, k)
        elif a.slot <= self.last or s.slot_at(a.slot)[0] != t:
            raise ValueError(f"the re-key in slot {a.slot} is sent in window {t}, after slot {self.last}")
        self.last = a.slot
        return a

    def freed(self, p: int) -> int:
        """Chunk p is no longer kept: the index it is held at, which the prover releases."""
        return self.held.pop(p)

    def due(self, t: int) -> tuple[tuple[int, Origin, int], ...]:
        """The re-keys to make before window t's first root, as (slot, origin, count): each re-key slot of t whose
        origin has a chunk still kept, in slot order."""
        s, out = self.schedule, []
        for m, k in enumerate(s.slots(t)):
            kind = s.slot_in(t, m)
            if kind[0] == "registered":
                origin, first, count = ("registered", kind[1]), kind[1], kind[2]
            elif kind[0] == "rekey" and (src := self.store.in_slot(kind[1])) is not None:
                origin, first, count = ("append", kind[1]), src.first, src.count
            else:
                continue
            if any(p in self.held for p in range(first, first + count)):
                out.append((k, origin, count))
        return tuple(out)

    def _valid(self, j: int, t: int) -> bool:
        s = self.schedule
        if j < s.blocks:
            return t < s.lifetime
        a = self.store._append_of(j)
        joined = s.slot_at(a.slot)[0]
        born = (joined if a.origin is None else None if a.origin[0] == "registered" else s.slot_at(a.origin[1])[0])
        return joined <= t < joined + s.lifetime and (born is None or born + s.lifetime <= t)

    def plan(self, t: int, *, pending: bool = False) -> Plan:
        """The plan of window t: every kept chunk at the index it is held at, where a plan of t may hold that index and
        it is encoded, or, with `pending`, is a re-key's still awaiting the coin it is encoded under before the pass
        reads it.  An append of pages is never planned: its blocks' first encode is their first re-key's."""
        # Appends only grow, and between two appends the kept chunks and the appends awaiting coins only shrink.
        key = (t, pending, len(self.store.appends), len(self.store.pending), len(self.held))
        if self._plan is not None and self._plan[0] == key:
            return self._plan[1]
        out = []
        for j in self.held.values():
            if j >= self.schedule.blocks and self.store._append_of(j).slot is None:
                continue
            if not self._valid(j, t) or (self.store.blocks[j] is None
                                         and not (pending and self.store._append_of(j).first in self.store.pending)):
                continue
            out.append(j)
        runs: list[list[int]] = []
        for j in sorted(out):
            if runs and runs[-1][0] + runs[-1][1] == j:
                runs[-1][1] += 1
            else:
                runs.append([j, 1])
        if not runs:
            raise ValueError(f"window {t} holds no block a plan may read: no kept chunk is valid in it")
        plan = Plan(tuple((f, c) for f, c in runs), self.store.extent)
        self._plan = (key, plan)
        return plan


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



class Prover:
    """The honest prover: each pass reads its plan's blocks (every block by default), computes their receipts and
    commits them, each leaf under a fresh salt from `entropy`.  `tiles` is PoUW's if the registration binds z, and
    None if it does not.  It keeps every challenge it is sent (`received`) and checks them at the session's end
    against the commitment it gets only then (`end`)."""

    def __init__(self, store: Store, tiles: audit.Tiles | None, *,
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        self.store, self.entropy = store, entropy
        self.tiles = audit._tiles(tiles, store.registration.bind)
        self.received: dict[int, coins.Released] = {}

    def run(self, challenge: coins.Released | coins.LiveOpening, i: int, previous: bytes, plan: Plan | None = None,
            *, head: bytes | None = None) -> Pass:
        """Pass i of the window whose challenge is `challenge` (`coins.released`), after the root `previous`, its seed
        binding `head`, the chain over every append the verifier will have received before this pass's root
        (`audit.chained`; None without one, the honest prover's `Store.head`): refused unless the coin is a window's
        challenge, the one it was sent for that window before if any."""
        try:
            got = coins.released(challenge)
        except coins.CoinError as e:
            raise ValueError(str(e)) from None
        if got.label != coins.CHALLENGE or got.index < 0:
            raise ValueError(f"not a window's challenge: {got.label} coin {got.index}")
        if self.received.setdefault(got.index, got) != got:
            raise ValueError(f"window {got.index}'s challenge was sent again, another coin")
        return self.pass_under(got.index, i, receipts.seed(coins.coin(got), i, audit.chained(previous, head)), plan)

    def end(self, commitment: coins.LiveCommitment, openings: Sequence[coins.LiveOpening]) -> None:
        """The session's end: `commitment` and every released coin's opening, as the verifier sends them
        (`AuditRun.disclose`), refused (ValueError) unless every challenge this prover was sent opens against it with
        the value and the root it was sent with (`coins.received_reason`)."""
        reason = coins.received_reason(commitment, tuple(self.received[t] for t in sorted(self.received)), openings)
        if reason is not None:
            raise ValueError(reason)

    def pass_under(self, window: int, i: int, seed: bytes, plan: Plan | None = None) -> Pass:
        """Pass i of `window`'s receipts and tree under `seed`, however it was derived."""
        plan = plan or Plan.whole(self.store.registration.blocks)
        reg = self.store.registration
        receipter, tiles = receipts.receipter(reg.receipt, seed, reg.rows), self.tiles
        ys = tuple(receipter(j, self.store.blocks[j], None if tiles is None else tiles.output(seed, j)) for j in plan)
        salts = tuple(self.entropy(SALT_BYTES) for _ in range(receipts.leaves(plan.length)))
        return Pass(window, i, seed, plan, ys, salts, receipts.tree(seed, plan, ys, salts))

    def open(self, ps: Pass, p: int) -> audit.Opening:
        j = ps.plan.block(p)
        c = p // receipts.PER_LEAF
        leaf = ps.tree.open(c)
        salt, path, d = self.store.committed(j)
        return audit.Opening(ps.index, p, self.store.blocks[j],
                             b"".join(ps.receipts[receipts.PER_LEAF * c:receipts.PER_LEAF * (c + 1)]),
                             ps.salts[c], leaf.path, salt, path,
                             b"" if self.tiles is None else self.tiles.evidence(ps.seed, j), d,
                             self.store.origin_position(j))


__all__ = ["Append", "Pass", "Prover", "Refresh", "Store"]
