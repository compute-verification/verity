"""PoUS's setup unit as a Boolean circuit: one sampled block of the store C decoded and held to the registered row of W
that holds its payload, as one output bit.

    PousP2SetupCheck_v1{W, C0, L, N}  (block: Array<W>, base: Array<8L>, row: Array<8N>, slot: Array<S>) -> Array<1>
    PousP2SlotMatch_v1{P, N}          (decoded: Array<8P + 1>, row: Array<8N>, slot: Array<S>) -> Array<1>

Every array is of `Value<1>`, byte strings and integers in `decode_circuit`'s orders: a byte string is bits in byte order,
bit t being bit t mod 8 of byte ⌊t/8⌋, and the slot is an integer, least significant bit first.  P = (W − 16)/8 is the
decode's payload bytes (2,054 at P2's point), n = ⌈N/P⌉ is the row's slots and S = max(1, ⌈log₂ n⌉) the slot's bits.

**The layout it reads.** W's registered rows are row-aligned in the store: each N-byte row starts at a block boundary and
fills ⌈N/P⌉ blocks, P of its bytes a block, its last block zero past the row's end, as `TensorLayout` places a tensor
(`verity/protocols/accounting/pous/verifier/layout.py`) with each row in the place of a tensor.  Slot k of a row is
then its bytes [kP, kP + P), zero past N (`slot_payload`).

**The unit.** ``block`` is C's row j (2,056 bytes at P2's point), block v of segment s, and ``base`` the decode's
expansion base ``expand_prefix(params, salt) ‖ u64be(s) ‖ u64be(v)`` (`decode_circuit`); ``row`` is W's registered row
that holds block j's payload and ``slot`` that payload's slot in it.  The one bit is 1 iff

    ok ∧ slot < n ∧ payload = slot_payload(row, slot)

with (payload, ok) `PousP2Decode_v2`'s output: so 1 iff block j decodes to the row's bytes at that slot, zero past the
row's end.  The unit hashes nothing beyond the decode's keys: which row of W and which slot block j's are, and that
``block`` and ``row`` are the committed rows, are the statement's around it.

**The slot.** One Program serves every block of a row width: the slot is an input, and `PousP2SlotMatch_v1` selects
its P bytes by a multiplexer tree over the row's n slots, one AND a bit a merge, (n − 1)·8P in all; then one AND tree
over the 8P bits' agreement with the payload, ok and slot < n (at most 8P + 1 ANDs), the bound itself at most S − 1
more, and none when n is a power of two.  Its counts at P2's payload are pinned in
`verity/protocols/accounting/pous/verifier/tests/test_pous_setup_circuit.py`.
"""
from __future__ import annotations

from verity.circuits.boolean.forms import ONE, ZERO, and_reduce
from verity.circuits.defs import CompositeDefinition, bind

from .circuit import _check, _traced_body, bits
from .decode_circuit import PousP2Decode, _field_sig

__all__ = ["PousP2SetupCheck", "PousP2SlotMatch", "slot_bits", "slot_payload", "slots"]


def slots(N: int, P: int) -> int:
    """How many blocks an N-byte row fills at P payload bytes a block."""
    return -(-N // P)


def slot_bits(N: int, P: int) -> int:
    """The slot's width: enough bits for every slot of the row, and at least one."""
    return max(1, (slots(N, P) - 1).bit_length())


def slot_payload(row: bytes, slot: int, P: int) -> bytes | None:
    """The reference: the P payload bytes slot ``slot`` of ``row`` holds in the store (its bytes from slot·P, zero past
    the row's end), or None for a slot past the row's last."""
    if not 0 <= slot < slots(len(row), P):
        return None
    return row[slot * P:(slot + 1) * P].ljust(P, b"\0")


# --- builders over forms ---------------------------------------------------------------------------------------------------

def below(C, x, n: int):
    """x < n for an unsigned x, least significant bit first."""
    if n >> len(x):
        return ONE
    lt = ZERO
    for i, f in enumerate(x):
        lt = C.AND(f, lt.inv()).inv() if n >> i & 1 else C.AND(f.inv(), lt)
    return lt


def select(C, leaves, slot):
    """Leaf ``slot`` of the leaves (each a list of forms), by one merge a slot bit; past the last leaf, any of them."""
    for s in slot:
        leaves = [[a ^ C.AND(s, a ^ b) for a, b in zip(leaves[i], leaves[i + 1])] if i + 1 < len(leaves) else leaves[i]
                  for i in range(0, len(leaves), 2)]
    return leaves[0]


def slot_match(C, P: int, N: int, decoded, row, slot):
    n = slots(N, P)
    padded = list(row) + [ZERO] * (8 * (n * P - N))
    want = select(C, [padded[8 * P * k:8 * P * (k + 1)] for k in range(n)], slot)
    same = [(a ^ b).inv() for a, b in zip(decoded[:8 * P], want)]
    return [and_reduce(C, same + [decoded[8 * P], below(C, slot, n)])]


# --- the Definitions -------------------------------------------------------------------------------------------------------

def _match_sig(S):
    _check("PousP2SlotMatch", P=S.P, N=S.N)
    return (("decoded", bits(8 * S.P + 1)), ("row", bits(8 * S.N)), ("slot", bits(slot_bits(S.N, S.P)))), bits(1)


PousP2SlotMatch = CompositeDefinition(
    "PousP2SlotMatch", 1, ("P", "N"), _match_sig,
    _traced_body(lambda S: ("slot-match", S.P, S.N), lambda S: lambda C, d, r, s: slot_match(C, S.P, S.N, d, r, s),
                 _match_sig),
    conformance="exact-model-tested",
    doc="1 iff the decode's ok bit is set, the slot is one of the N-byte row's ⌈N/P⌉, and the decode's P payload bytes "
        "are the row's bytes at that slot, zero past the row's end")


def _setup_sig(S):
    W = _field_sig("PousP2SetupCheck", S)
    _check("PousP2SetupCheck", N=S.N)
    bind(PousP2Decode, W=W, C0=S.C0, L=S.L).signature()
    return (("block", bits(W)), ("base", bits(8 * S.L)), ("row", bits(8 * S.N)),
            ("slot", bits(slot_bits(S.N, (W - 16) // 8)))), bits(1)


def _setup_body(B, S, block, base, row, slot):
    decoded = B.call(bind(PousP2Decode, W=S.W, C0=S.C0, L=S.L), block, base)
    return B.call(bind(PousP2SlotMatch, P=(S.W - 16) // 8, N=S.N), decoded, row, slot)


PousP2SetupCheck = CompositeDefinition(
    "PousP2SetupCheck", 1, ("W", "C0", "L", "N"), _setup_sig, _setup_body, conformance="exact-model-tested",
    doc="PoUS's setup unit: 1 iff the block decodes (P2 v4, keys from the L-byte expansion base) to the N-byte registered "
        "row's payload bytes at the slot, zero past the row's end; P2's point is W = 16448, C0 = 21065, L = 144")
