"""The honest prover over PoUS's P2 store (`p2-unit4/v0`, `verity.protocols.accounting.poub.verifier.p2`): each pass's
receipts from its plan's units, or from receipts the decode computed, committed in the audit's tree, and each opening's
four blocks with their paths to the registration's root.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from verity.protocols.accounting.poub.prover import audit_prover
from verity.protocols.accounting.poub.verifier import p2, receipts
from verity.protocols.accounting.poub.verifier.receipts import PER_LEAF, SALT_BYTES, Plan
from verity.protocols.accounting.pous.verifier.verifier import MerkleTree, leaf_hash


class Store:
    """The prover's C* under a P2 registration: its N blocks and PoUS's vk tree over them, whose root must be the
    registration's."""

    def __init__(self, registration: p2.Registration, blocks: Sequence[bytes]) -> None:
        params = registration.params
        if len(blocks) != params.blocks or any(type(b) is not bytes or len(b) != p2.BLOCK_BYTES for b in blocks):
            raise ValueError(f"the registration's C* is {params.blocks} blocks of {p2.BLOCK_BYTES} bytes")
        self.registration, self.blocks = registration, tuple(blocks)
        self.tree = MerkleTree([leaf_hash(params, registration.salt, j, b) for j, b in enumerate(self.blocks)])
        if self.tree.root != registration.root:
            raise ValueError("these blocks are not the registered C*: their vk root is not the registration's")

    def unit(self, g: int) -> bytes:
        return b"".join(self.blocks[p2.UNIT_BLOCKS * g:p2.UNIT_BLOCKS * (g + 1)])



class Prover(audit_prover.Prover):
    """The honest prover over a P2 store: each pass's receipts from its plan's units (`pass_under`), or from receipts
    the decode computed (`commit`), committed in the audit's tree."""

    def __init__(self, store: Store, *, entropy: Callable[[int], bytes] = os.urandom) -> None:
        super().__init__(store, None, entropy=entropy)

    def pass_under(self, window: int, i: int, seed: bytes, plan: Plan | None = None) -> audit_prover.Pass:
        plan = plan or Plan.whole(self.store.registration.blocks)
        sketcher = p2.Sketcher(seed, self.store.registration.rows)
        return self.commit(window, i, seed, plan, [sketcher(g, self.store.unit(g)) for g in plan])

    def commit(self, window: int, i: int, seed: bytes, plan: Plan, ys: Sequence[bytes]) -> audit_prover.Pass:
        """Pass i of `window` under `seed` from its receipts `ys` in plan order, however they were computed."""
        salts = tuple(self.entropy(SALT_BYTES) for _ in range(receipts.leaves(plan.length)))
        return audit_prover.Pass(window, i, seed, plan, tuple(ys), salts, receipts.tree(seed, plan, ys, salts))

    def open(self, ps: audit_prover.Pass, p: int) -> p2.Opening:
        g, c = ps.plan.block(p), p // PER_LEAF
        js = range(p2.UNIT_BLOCKS * g, p2.UNIT_BLOCKS * (g + 1))
        return p2.Opening(ps.index, p, tuple(self.store.blocks[j] for j in js), tuple(self.store.tree.path(j) for j in js),
                          b"".join(ps.receipts[PER_LEAF * c:PER_LEAF * (c + 1)]), ps.salts[c], ps.tree.open(c).path)


__all__ = ["Prover", "Store"]
