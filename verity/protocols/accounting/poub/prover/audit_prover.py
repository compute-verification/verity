"""PoUB's honest prover (`verity/remote-bandwidth/v0`, `verity.protocols.accounting.poub.verifier.audit`): each pass's
receipts over its store, committed in the pass's tree; the window's sampled receipts opened; and, at the session's end,
every challenge it was sent checked against the coin commitment. The store, the passes and the openings are the
audit's (`audit.Store`, `audit.Pass`, `audit.Opening`).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from verity.protocols.accounting.poub.verifier import audit, coins, receipts
from verity.protocols.accounting.poub.verifier.receipts import SALT_BYTES, Plan


class Prover:
    """The honest prover: each pass reads its plan's blocks (every block by default), computes their receipts and
    commits them, each leaf under a fresh salt from `entropy`.  `tiles` is PoUW's if the registration binds z, and
    None if it does not.  It keeps every challenge it is sent (`received`) and checks them at the session's end
    against the commitment it gets only then (`end`)."""

    def __init__(self, store: audit.Store, tiles: audit.Tiles | None, *,
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        self.store, self.entropy = store, entropy
        self.tiles = audit._tiles(tiles, store.registration.bind)
        self.received: dict[int, coins.Released] = {}

    def run(self, challenge: coins.Released | coins.LiveOpening, i: int, previous: bytes, plan: Plan | None = None,
            *, head: bytes | None = None) -> audit.Pass:
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

    def pass_under(self, window: int, i: int, seed: bytes, plan: Plan | None = None) -> audit.Pass:
        """Pass i of `window`'s receipts and tree under `seed`, however it was derived."""
        plan = plan or Plan.whole(self.store.registration.blocks)
        reg = self.store.registration
        receipter, tiles = receipts.receipter(reg.receipt, seed, reg.rows), self.tiles
        ys = tuple(receipter(j, self.store.blocks[j], None if tiles is None else tiles.output(seed, j)) for j in plan)
        salts = tuple(self.entropy(SALT_BYTES) for _ in range(receipts.leaves(plan.length)))
        return audit.Pass(window, i, seed, plan, ys, salts, receipts.tree(seed, plan, ys, salts))

    def open(self, ps: audit.Pass, p: int) -> audit.Opening:
        j = ps.plan.block(p)
        c = p // receipts.PER_LEAF
        leaf = ps.tree.open(c)
        salt, path, d = self.store.committed(j)
        return audit.Opening(ps.index, p, self.store.blocks[j],
                             b"".join(ps.receipts[receipts.PER_LEAF * c:receipts.PER_LEAF * (c + 1)]),
                             ps.salts[c], leaf.path, salt, path,
                             b"" if self.tiles is None else self.tiles.evidence(ps.seed, j), d)


__all__ = ["Prover"]
