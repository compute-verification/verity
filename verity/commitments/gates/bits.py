"""What Boolean Definitions over bits and SHA-512 words share: their types, constants and refs, bits from integers, bytes
and digests, a hiding leaf's value, and a Definition's AND count."""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.census import BooleanWork, NoBooleanWork
from verity.circuits.refs import Coll
from verity.circuits.types import Array
from verity.commitments import hm96
from verity.commitments.gates.sha512 import words_be
from verity.commitments.rowleaf import sha512_row_v2_digest

BIT = BL.BIT
BYTE = Array(8, BIT)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
SALT = Array(hm96.SHA512.salt_bytes // 8, WORD)


def _same(x, want) -> object:
    """1 iff bit x is `want`: a bit of the circuit, or the constant 0 or 1."""
    if isinstance(want, int):
        return x if want else x.inv()
    return (x ^ want).inv()


class _Consts:
    """The constant bits a body reads, one node each."""

    def __init__(self, B):
        self.B, self.ref = B, {}

    def bits(self, value: int, w: int) -> list:
        for b in sorted({value >> j & 1 for j in range(w)} - self.ref.keys()):
            self.ref[b] = self.B.call(BL.constant(b)).refs[0]
        return [self.ref[value >> j & 1] for j in range(w)]


def _refs(c: Coll) -> list:
    return [c.refs[k] for k in range(c.refs.count)]


def _coll(t, refs: list) -> Coll:
    return Coll(t, TR._refs(list(refs)))


def hiding_value(row: bytes, salt: bytes) -> bytes:
    """The 64 bytes a row's frame-v3-sha512 leaf binds: hm96's tree leaf over the row's sha512/row/v2 digest."""
    inst, key = hm96.SHA512, hm96.SHA512.default_key
    return inst.tree_leaf(key, inst.commit_string(key, sha512_row_v2_digest(row, 8 * len(row)), salt))


def _digest_bits(d: bytes) -> list[int]:
    return [(w >> j) & 1 for w in words_be(d) for j in range(64)]


def _int_bits(x: int, w: int) -> list[int]:
    return [(x >> j) & 1 for j in range(w)]


def _bytes_bits(data: bytes) -> list[int]:
    return _int_bits(int.from_bytes(data, "little"), 8 * len(data))


def _ands(work: BooleanWork, fn) -> int:
    got = work.of(fn)
    if isinstance(got, str):
        raise NoBooleanWork({fn.id: f"it reaches {got}, which is not a Boolean gate, a constant bit or an input"})
    return sum(a for _g, a in got.values())


def ands(fn) -> int:
    """A Boolean Definition's AND gates: the ANDs of its Boolean work as its census counts it (`census.BooleanWork`),
    every facet's."""
    return _ands(BooleanWork(), fn)


__all__ = ["BIT", "BYTE", "DIGEST", "SALT", "WORD", "ands", "hiding_value"]
