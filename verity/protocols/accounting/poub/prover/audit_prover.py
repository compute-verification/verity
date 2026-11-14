"""PoUB's honest prover (`verity/remote-bandwidth/v0`, `verity.protocols.accounting.poub.verifier.audit`): each pass's
receipts over its store, committed in the pass's tree; the window's sampled receipts opened; and, at the session's end,
every challenge it was sent checked against the coin commitment. The store, the passes and the openings are the
audit's (`Store`, `Pass`, `audit.Opening`).
"""

from __future__ import annotations

import bisect
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from verity.commitments import MerkleTree
from verity.protocols.accounting.poub.verifier import audit, codec, coins, receipts
from verity.protocols.accounting.poub.verifier.audit import Registration, append_head, append_tree, payload_tree
from verity.protocols.accounting.poub.verifier.receipts import SALT_BYTES, Plan


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
        self.coin_of[first] = coin
        self.blocks[first:first + len(encoded)] = encoded

    def _append_of(self, j: int) -> Append:
        return self.appends[bisect.bisect_right([x.first for x in self.appends], j) - 1]

    def committed(self, j: int) -> tuple[bytes, tuple[bytes, ...], bytes]:
        """Block j's payload salt, its path to the payload root or its append's, and its digest `d_j` (empty under
        the seed's key)."""
        if j < self.registration.blocks:
            return self.salts[j], self.payload_tree.open(j).path, b""
        a = self._append_of(j)
        return a.salts[j - a.first], a.tree.open(j).path, a.digests[j - a.first]

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
                             b"" if self.tiles is None else self.tiles.evidence(ps.seed, j), d)


__all__ = ["Append", "Pass", "Prover", "Store"]
