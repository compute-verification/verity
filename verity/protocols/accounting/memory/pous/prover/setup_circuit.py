"""PoUS's setup unit as a Boolean circuit: one block of the store C decoded and held to the registered row of W that
holds its payload, as one ok word.

    PousP2SetupCheck_v2{W, C0, N, HEAD}  (block: Array<W/16, Array<16>>, salt: Array<12, Array<16>>, at: Array<8, Array<16>>,
                                          row: Array<N/2, Array<16>>, slot: Array<16>) -> Array<16>
    PousP2SetupBase_v1{W, C0, HEAD}      (salt: Array<192>, at: Array<128>) -> Array<1152>
    PousP2SlotMatch_v1{P, N}       (decoded: Array<8P + 1>, row: Array<8N>, slot: Array<16>) -> Array<1>
    PousBitWord_v1                 (bit: Array<1>) -> Array<16>

Every leaf is a `Value<1>`, byte strings and integers in `decode_circuit`'s orders: a byte string is bits in byte order,
bit t being bit t mod 8 of byte ⌊t/8⌋, and the slot is an integer, least significant bit first.  The setup unit's ports
are u16 words of those bits, a window's rows (`verity.commitments.periods.interface.layout`), so N is even, and its body
hands the decode and the slot match the same bits flat.  P = (W − 16)/8 is the decode's payload bytes (2,054 at P2's
point), n = ⌈N/P⌉ is the row's slots, at most 2^16, and S = max(1, ⌈log₂ n⌉) the slot's bits the selection reads.

**The layout it reads.** W's registered rows are row-aligned in the store: each N-byte row starts at a block boundary and
fills ⌈N/P⌉ blocks, P of its bytes a block, its last block zero past the row's end, as `TensorLayout` places a tensor
(`verity/protocols/accounting/memory/pous/verifier/layout.py`) with each row in the place of a tensor.  Slot k of a row is
then its bytes [kP, kP + P), zero past N (`slot_payload`).

**The unit.** ``block`` is C's row j (2,056 bytes at P2's point), block v of segment s; ``salt`` is the setup's salt
(`pous.setup.salt/v1`, 24 bytes, the one value the verifier supplies); ``at`` is ``u64be(s) ‖ u64be(v)`` and ``slot``
that payload's slot in ``row``, W's registered row that holds block j's payload: both constants of the unit in C. The
unit builds its decode's expansion base ``expand_prefix(params, salt) ‖ at`` inside (`PousP2SetupBase_v1`: ``HEAD``,
`expand_prefix`'s bytes before the salt, hex, bound by the prover's `setup_proof.prefix_head` from the verifier's scheme,
then the salt, zero-padded to 128 bytes; wiring and constants, no AND).  The salt's length is the one ``HEAD``'s last
frame header gives.  Its word is 1 iff

    ok ∧ slot < n ∧ payload = slot_payload(row, slot)

with (payload, ok) `PousP2Decode_v2`'s output, and 0 otherwise: so 1 iff block j decodes to the row's bytes at that
slot, zero past the row's end, an output the verifier holds to 1.  The unit hashes nothing beyond the decode's keys:
which row of W and which slot block j's are, and that ``block`` and ``row`` are the committed rows, are the statement's
around it.

**The slot.** One Program serves every block of a row width: the slot is a port, and `PousP2SlotMatch_v1` selects
its P bytes by a multiplexer tree over the row's n slots, one AND a bit a merge, (n − 1)·8P in all; then one AND tree
over the 8P bits' agreement with the payload, ok and slot < n (at most 8P + 1 ANDs), the bound itself at most 15
more over the slot's 16 bits.  Its counts at P2's payload are pinned in
`tests/verity/protocols/accounting/memory/pous/verifier/test_pous_setup_circuit.py`.
"""
from __future__ import annotations

from verity.circuits.boolean.forms import ONE, ZERO, and_reduce
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import Coll
from verity.circuits.types import Array

from .circuit import _check, _traced_body, bits
from .decode_circuit import P2_BASE_BYTES, PousP2Decode, _field_sig

__all__ = ["AT_BYTES", "SLOT_BITS", "PousBitWord", "PousP2SetupBase", "PousP2SetupCheck", "PousP2SlotMatch",
           "salt_bytes", "slot_bits", "slot_payload", "slots"]

#: the slot's width: one u16 word
SLOT_BITS = 16
#: a block's place in the base, u64be(s) ‖ u64be(v)
AT_BYTES = 16


def slots(N: int, P: int) -> int:
    """How many blocks an N-byte row fills at P payload bytes a block."""
    return -(-N // P)


def slot_bits(N: int, P: int) -> int:
    """The slot's bits the selection reads: enough for every slot of the row, and at least one."""
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
    want = select(C, [padded[8 * P * k:8 * P * (k + 1)] for k in range(n)], slot[:slot_bits(N, P)])
    same = [(a ^ b).inv() for a, b in zip(decoded[:8 * P], want)]
    return [and_reduce(C, same + [decoded[8 * P], below(C, slot, n)])]


# --- the Definitions -------------------------------------------------------------------------------------------------------

def _match_sig(S):
    _check("PousP2SlotMatch", P=S.P, N=S.N)
    if slots(S.N, S.P) > 1 << SLOT_BITS:
        raise ValueError(f"PousP2SlotMatch's row has at most {1 << SLOT_BITS} slots, not {slots(S.N, S.P)} (N = {S.N}, "
                         f"P = {S.P})")
    return (("decoded", bits(8 * S.P + 1)), ("row", bits(8 * S.N)), ("slot", bits(SLOT_BITS))), bits(1)


PousP2SlotMatch = CompositeDefinition(
    "PousP2SlotMatch", 1, ("P", "N"), _match_sig,
    _traced_body(lambda S: ("slot-match", S.P, S.N), lambda S: lambda C, d, r, s: slot_match(C, S.P, S.N, d, r, s),
                 _match_sig),
    conformance="exact-model-tested",
    doc="1 iff the decode's ok bit is set, the slot is one of the N-byte row's ⌈N/P⌉, and the decode's P payload bytes "
        "are the row's bytes at that slot, zero past the row's end")


def salt_bytes(head: str) -> int:
    """The salt's bytes: the length `HEAD`'s last frame header (``b"b" ‖ u64be(len)``) gives."""
    h = bytes.fromhex(head)
    if len(h) < 9 or h[-9:-8] != b"b":
        raise ValueError("PousP2SetupBase's HEAD ends in the salt's frame header, b'b' ‖ u64be(len)")
    return int.from_bytes(h[-8:], "big")


def _base_sig(S):
    _field_sig("PousP2SetupBase", S)
    if not isinstance(S.HEAD, str):
        raise ValueError("PousP2SetupBase's HEAD is expand_prefix's bytes before the salt, in hex")
    n = salt_bytes(S.HEAD)
    if len(S.HEAD) // 2 + n > P2_BASE_BYTES - AT_BYTES:
        raise ValueError(f"PousP2SetupBase's prefix at W = {S.W}, C0 = {S.C0} is past one 128-byte block")
    return (("salt", bits(8 * n)), ("at", bits(8 * AT_BYTES))), bits(8 * P2_BASE_BYTES)


def setup_base(head: str, salt, at):
    """The base `expand_prefix(params, salt) ‖ at` as forms: `head`'s and the zero pad's bits constants, the salt's and
    `at`'s wired through."""
    h = bytes.fromhex(head)
    pad = bytes(P2_BASE_BYTES - AT_BYTES - len(h) - salt_bytes(head))
    const = lambda b: [ONE if x >> k & 1 else ZERO for x in b for k in range(8)]  # noqa: E731
    return const(h) + list(salt) + const(pad) + list(at)


PousP2SetupBase = CompositeDefinition(
    "PousP2SetupBase", 1, ("W", "C0", "HEAD"), _base_sig,
    _traced_body(lambda S: ("setup-base", S.W, S.C0, S.HEAD), lambda S: lambda C, s, a: setup_base(S.HEAD, s, a),
                 _base_sig),
    conformance="exact-model-tested",
    doc="PoUS's setup base expand_prefix(params, salt) ‖ at at P2's point W, C0: HEAD's bytes, the salt's, a zero pad "
        "and the block's place's bits, no gate")


def _word_sig(S):
    return (("bit", bits(1)),), bits(16)


PousBitWord = CompositeDefinition(
    "PousBitWord", 1, (), _word_sig,
    _traced_body(lambda S: ("bit-word",), lambda S: lambda C, b: list(b) + [ZERO] * 15, _word_sig),
    conformance="exact-model-tested", doc="one bit as a u16 word, the bit its least significant: no gate")


def _setup_sig(S):
    W = _field_sig("PousP2SetupCheck", S)
    _check("PousP2SetupCheck", N=S.N)
    bind(PousP2Decode, W=W, C0=S.C0, L=P2_BASE_BYTES).signature()
    bind(PousP2SetupBase, W=W, C0=S.C0, HEAD=S.HEAD).signature()
    n = 8 * salt_bytes(S.HEAD)
    if W % 16 or S.N % 2:
        raise ValueError(f"PousP2SetupCheck's ports are u16 words, so W/16 and N/2 are whole: not W = {W}, N = {S.N}")
    bind(PousP2SlotMatch, P=(W - 16) // 8, N=S.N).signature()
    return (("block", _words(W)), ("salt", _words(n)), ("at", _words(8 * AT_BYTES)), ("row", _words(8 * S.N)),
            ("slot", bits(SLOT_BITS))), bits(16)


def _words(n: int):
    return Array(n // 16, bits(16))


def _setup_body(B, S, block, salt, at, row, slot):
    flat = lambda x, n: Coll(bits(n), x.refs)  # noqa: E731
    base = B.call(bind(PousP2SetupBase, W=S.W, C0=S.C0, HEAD=S.HEAD), flat(salt, 8 * salt_bytes(S.HEAD)),
                  flat(at, 8 * AT_BYTES))
    decoded = B.call(bind(PousP2Decode, W=S.W, C0=S.C0, L=P2_BASE_BYTES), flat(block, S.W), base)
    ok = B.call(bind(PousP2SlotMatch, P=(S.W - 16) // 8, N=S.N), decoded, flat(row, 8 * S.N), slot)
    return B.call(bind(PousBitWord), ok)


PousP2SetupCheck = CompositeDefinition(
    "PousP2SetupCheck", 2, ("W", "C0", "N", "HEAD"), _setup_sig, _setup_body, conformance="exact-model-tested",
    doc="PoUS's setup unit: the word 1 iff the block decodes (P2 v4, keys from expand_prefix(params, salt) ‖ at, built "
        "inside from the salt) to the N-byte registered row's payload bytes at the slot, zero past the row's end, else "
        "0; P2's point is W = 16448, C0 = 21065")
