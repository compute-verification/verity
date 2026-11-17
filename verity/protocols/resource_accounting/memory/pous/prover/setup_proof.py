"""PoUS's setup as a sampled proof about the store, the prover's side: a sampled block's place and inputs to the setup
unit, which `window.SetupIntegration` records into the window the proof system proves.

The store C* holds W's registered rows row-aligned (`setup_circuit`'s layout): block u is slot u mod n of row ⌊u/n⌋,
n = ⌈N/P⌉ slots a row of N bytes.  The auditor's challenger samples `SAMPLE` of the store's B blocks after the store's
commitment is registered (the verifier's `setup_proof.sampling_design`); each sampled block is one proof block, an
instance of the setup unit (`setup_unit`) over its block of C*, its base, the row of W that holds its payload and its
slot; the auditor's check reads the proof system's verdict in place of the store (`PousCheck.SetupInProofs`).

`SetupGives` (`security/Properties/Pous/SetupInProofs.lean`) is the lemma over this: its `W` is `padded`'s, so its
`slice q W u` is `slot_payload(row, slot)` at `place(u)`, and its `BlockOk` is what the unit computes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity.circuits.defs import bind
from verity.protocols.resource_accounting.memory.pous.verifier.setup_proof import base

from .decode_circuit import P2_BASE_BYTES
from .setup_circuit import PousP2SetupCheck, slot_payload, slots

GOAL = "memory-accounting/pous-setup-as-proof"


def payload_bytes(w: int) -> int:
    """P: the bytes of W a block of `w` bits carries."""
    return (w - 16) // 8


def setup_unit(w: int, c0: int, row_bytes: int) -> Any:
    """The setup unit at P2's point `w`, `c0` over rows of `row_bytes` bytes: `PousP2SetupCheck_v1{W, C0, L=144, N}`."""
    return bind(PousP2SetupCheck, W=w, C0=c0, L=P2_BASE_BYTES, N=row_bytes)


def place(w: int, row_bytes: int, u: int) -> tuple[int, int]:
    """Block `u`'s row of W and its slot in that row."""
    n = slots(row_bytes, payload_bytes(w))
    return u // n, u % n


def padded(rows: Sequence[bytes], w: int) -> bytes:
    """W as `SetupGives` reads it: each row's slots in order, the last zero past the row's end, so its slice at block u
    is `slot_payload` at `place(u)`."""
    p = payload_bytes(w)
    return b"".join(slot_payload(row, k, p) for row in rows for k in range(slots(len(row), p)))


def unit_inputs(w: int, prefix: bytes, blocks_per_segment: int, block: bytes, row: bytes, u: int,
                row_bytes: int) -> tuple[bytes, bytes, bytes, int]:
    """Block `u`'s four inputs to the unit, `(block, base, row, slot)`: its block of C*, its expansion base `prefix ‖
    u64be(s) ‖ u64be(v)` (block v of segment s, `prefix` the verifier's `expand_prefix(params, salt)`), the row of W
    that holds its payload and its slot."""
    if len(row) != row_bytes:
        raise ValueError(f"block {u}'s row is {len(row)} bytes, not the {row_bytes} the unit reads")
    return block, base(prefix, blocks_per_segment, u), row, place(w, row_bytes, u)[1]


__all__ = ["GOAL", "padded", "payload_bytes", "place", "setup_unit", "unit_inputs"]
