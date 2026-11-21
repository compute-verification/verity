"""PoUS's setup as a sampled proof about the store, the prover's side: a block's place and its inputs to the setup unit.

The store C* holds W's registered rows row-aligned (`setup_circuit`'s layout): block u is slot u mod n of row ⌊u/n⌋,
n = ⌈N/P⌉ slots a row of N bytes.  The setup's C (`window.Setup`) is one instance of the setup unit (`setup_unit`) a
block of C*, over its block, the setup's salt, its place `u64be(s) ‖ u64be(v)`, the row of W that holds its payload and
its slot, one proof block each: the salt the one value the verifier supplies, the place and the slot constants of the
unit in C; the proof
layer draws blocks after the store's commitment is registered (its `Draw`), and the auditor's check reads the door's
verdict on the drawn ones in place of the store (the verifier's `setup_proof.check`).

`SetupGives` (`security/Proofs/Properties/Pous/SetupInProofs.lean`) is the lemma over this: its `W` is `padded`'s, so its
`slice q W u` is `slot_payload(row, slot)` at `place(u)`, and its `BlockOk` is what the unit computes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity.circuits.defs import bind
from verity.protocols.accounting.memory.pous.verifier.schemes import p2 as P2
from verity.protocols.accounting.memory.pous.verifier.setup_proof import SALT, SALT_MARK

from .setup_circuit import PousP2SetupCheck, slot_payload, slots

def payload_bytes(w: int) -> int:
    """P: the bytes of W a block of `w` bits carries."""
    return (w - 16) // 8


def prefix_head(w: int, c0: int) -> bytes:
    """`expand_prefix`'s bytes before the salt at P2's point `w`, `c0` (`schemes.p2.prefix_head`)."""
    return P2.prefix_head(P2.P2Params(w, c0, 1))


def setup_unit(w: int, c0: int, row_bytes: int) -> Any:
    """The setup unit at P2's point `w`, `c0` over rows of `row_bytes` bytes: `PousP2SetupCheck_v2{W, C0, N, HEAD}`,
    `HEAD` `prefix_head`'s hex."""
    return bind(PousP2SetupCheck, W=w, C0=c0, N=row_bytes, HEAD=prefix_head(w, c0).hex())


def place(w: int, row_bytes: int, u: int) -> tuple[int, int]:
    """Block `u`'s row of W and its slot in that row."""
    n = slots(row_bytes, payload_bytes(w))
    return u // n, u % n


def padded(rows: Sequence[bytes], w: int) -> bytes:
    """W as `SetupGives` reads it: each row's slots in order, the last zero past the row's end, so its slice at block u
    is `slot_payload` at `place(u)`."""
    p = payload_bytes(w)
    return b"".join(slot_payload(row, k, p) for row in rows for k in range(slots(len(row), p)))


def at(blocks_per_segment: int, u: int) -> bytes:
    """Block `u`'s place in its base, `u64be(s) ‖ u64be(v)`: block v of segment s."""
    s, v = divmod(u, blocks_per_segment)
    return s.to_bytes(8, "big") + v.to_bytes(8, "big")


def unit_inputs(w: int, salt: bytes, blocks_per_segment: int, block: bytes, row: bytes, u: int,
                row_bytes: int) -> tuple[bytes, bytes, bytes, bytes, int]:
    """Block `u`'s five inputs to the unit, `(block, salt, at, row, slot)`: its block of C*, the setup's salt, its place
    (`at`), the row of W that holds its payload and its slot."""
    if len(row) != row_bytes:
        raise ValueError(f"block {u}'s row is {len(row)} bytes, not the {row_bytes} the unit reads")
    return block, salt, at(blocks_per_segment, u), row, place(w, row_bytes, u)[1]


__all__ = ["SALT", "SALT_MARK", "at", "padded", "payload_bytes", "place", "setup_unit", "unit_inputs"]
