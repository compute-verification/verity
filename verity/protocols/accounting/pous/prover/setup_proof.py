"""PoUS's setup as a sampled proof about the store, the prover's side: the statement SPOT samples and proves in place of
the verifier reading the whole store.

The store C* holds W's registered rows row-aligned (`setup_circuit`'s layout): block u is slot u mod n of row ⌊u/n⌋,
n = ⌈N/P⌉ slots a row of N bytes.  Each of the store's B blocks is one proof block, an instance of the setup unit
(`setup_unit`) over its block of C*, its base, the row of W that holds its payload and its slot.  The setup states what
SPOT needs (`spec`): the store's blocks and W's rows committed, the sampling design (`law`, from the verifier's side,
`rate:8828/B`), the unit as the check each sampled block passes, and the public items.  The challenger draws the sample
on its live coins after the store's commitment is registered; the prover proves the sampled blocks (`prove`) behind the
ZK server, staged in `iv`; the auditor's check reads SPOT's outcome in place of the store
(`verity/protocols/accounting/pous/verifier/setup_proof.py`, `PousCheck.SetupInProofs`).

`SetupGives` (`security/Properties/Pous/SetupInProofs.lean`) is the lemma over this: its `W` is `padded`'s, so its
`slice q W u` is `slot_payload(row, slot)` at `place(u)`, and its `BlockOk` is what the unit computes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity.circuits.defs import bind
from verity.circuits.program import Program
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2
from verity.protocols.proofs.spot.verifier import service as S
from verity.protocols.proofs.spot.verifier.one_stage import partition as P

from .decode_circuit import P2_BASE_BYTES
from .setup_circuit import PousP2SetupCheck, slot_payload, slots

GOAL = "memory-accounting/pous-setup-as-proof"
#: the store's blocks, committed once the salt is drawn
STORE = "store"
#: W's rows, registered once per deployment and read by position
WEIGHTS = "weights"
#: the unit's one output bit: the block decodes to its row's slot
OK = "ok"
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2

PUBLIC = (
    S.Public("population", "structure", "one proof block a block of the store: B, the store's size, which the "
             "parameters fix"),
    S.Public(f"leaves:{STORE}", "structure", "B leaves, one a block of C*"),
    S.Public(f"domain:{STORE}", "structure", "the store's domain, which binds the check and the deployment"),
    S.Public(STORE, "commitment", "the store's blocks under one salted root, registered before the challenger's coins"),
    S.Public(f"leaves:{WEIGHTS}", "structure", "W's rows, the registered weights' count"),
    S.Public(f"domain:{WEIGHTS}", "structure", "the weights' domain"),
    S.Public(f"words:{WEIGHTS}", "structure", "a row's words, the served row width"),
    S.Public(f"word_bits:{WEIGHTS}", "structure", "a word's bits"),
    S.Public(OK, "output", "each sampled block decodes to its row's slot of W, or not"),
)


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
    s, v = divmod(u, blocks_per_segment)
    return block, prefix + s.to_bytes(8, "big") + v.to_bytes(8, "big"), row, place(w, row_bytes, u)[1]


def spec(w: int, c0: int, row_bytes: int, law: str | None, context: bytes) -> S.Spec:
    """The setup's SPOT `Spec`: the store's blocks (phase `epoch`, after the salt) and W's registered rows committed,
    the sampling design `law` (`setup_proof.law` on the verifier's side; None, every block), the unit as the check
    with its one output bit public, and `PUBLIC`."""
    unit = setup_unit(w, c0, row_bytes)
    check = S.Check(P.program_sha512(Program(unit)), P.template_instance_query(unit), public_outputs=(OK,))
    selection = S.All() if law is None else S.Law(law)
    values = (S.Value(STORE, SCHEMA, "epoch"), S.Value(WEIGHTS, None, "epoch", registered=True))
    return S.Spec(values, selection, check, PUBLIC, context)


def prove(spec: S.Spec, sample: Sequence[int], inputs: Sequence[tuple[bytes, bytes, bytes, int]]) -> Any:
    """The sampled blocks' statements over the unit, each block's `inputs` its openings, staged in `iv` and proved
    behind the ZK server."""
    raise NotImplementedError(GOAL)


__all__ = ["GOAL", "OK", "PUBLIC", "SCHEMA", "STORE", "WEIGHTS", "padded", "payload_bytes", "place", "prove",
           "setup_unit", "spec", "unit_inputs"]
